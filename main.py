# main.py

import os
import sys
import json
import time
import warnings
import argparse
import logging
import urllib3
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.reasoners.networkx_symbolic_reasoning_metrics import (
    extract_reasoning_pattern,
    get_reasoning_metrics
)
from src.reasoners.neural_retriever import NeuralRetriever
from src.integrators.hybrid_integrator import HybridIntegrator
from src.utils.dimension_manager import DimensionalityManager
from src.utils.rule_extractor import RuleExtractor
from src.queries.query_logger import QueryLogger
from src.resources.resource_manager import ResourceManager
from src.feedback.feedback_manager import FeedbackManager
from src.feedback.feedback_handler import FeedbackHandler
from src.config.config_loader import ConfigLoader
from src.queries.query_expander import QueryExpander
from src.utils.evaluation import Evaluation
from src.app import App
from src.system.system_control_manager import SystemControlManager, UnifiedResponseAggregator
from src.utils.metrics_collector import MetricsCollector
from src.utils.device_manager import DeviceManager
from src.utils.progress import tqdm, ProgressManager
from src.utils.output_capture import capture_output  # Import the capture_output context manager
from src.ablation_study import run_ablation_study

urllib3.disable_warnings()
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_hotpotqa(hotpotqa_path, max_samples=None):
    """
    Loads a portion of the HotpotQA dataset.
    Each sample includes a query, ground-truth answer,
    combined context, and a 'type' = 'ground_truth_available_hotpotqa'.
    """
    dataset = []
    with open(hotpotqa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    count = 0
    for example in data:
        question = example['question']
        answer = example['answer']
        supporting_facts = example['supporting_facts']

        # Flatten the context
        context_str = []
        for title, sents in example.get('context', []):
            combined_sents = " ".join(sents)
            context_str.append(f"{title}: {combined_sents}")
        context_str = "\n".join(context_str)

        dataset.append({
            "query_id": example.get('_id', f"hotpot_q_{count}"),
            "query": question,
            "answer": answer,
            "context": context_str,
            "type": "ground_truth_available_hotpotqa",
            "supporting_facts": supporting_facts
        })
        count += 1
        if max_samples and count >= max_samples:
            break
    return dataset


def load_drop_dataset(drop_path, max_samples=None):
    """
    Loads a portion of the DROP dataset.
    """
    dataset = []
    with open(drop_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_loaded_qas = 0
    for passage_id, passage_content in data.items():
        passage_text = passage_content['passage']
        for qa_pair in passage_content['qa_pairs']:
            question = qa_pair['question']
            answer_obj = qa_pair['answer']

            dataset.append({
                "query_id": qa_pair.get("query_id", f"{passage_id}_{total_loaded_qas}"),
                "query": question,
                "context": passage_text,
                "answer": answer_obj,
                "type": "ground_truth_available_drop"
            })
            total_loaded_qas += 1
            if max_samples and total_loaded_qas >= max_samples:
                return dataset
        if max_samples and total_loaded_qas >= max_samples:
            break
    return dataset


def run_hysym_system(samples=200, dataset_type='hotpotqa'):
    """Main execution function for the HySym-RAG system."""
    print("\n=== Initializing HySym-RAG System ===")

    ProgressManager.SHOW_PROGRESS = False
    for lib in ['transformers', 'sentence_transformers', 'urllib3.connectionpool']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('DimensionalityManager').setLevel(logging.DEBUG)
    logging.getLogger('src.utils.dimension_manager').setLevel(logging.WARNING)
    logging.getLogger('src.knowledge_integrator').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner').setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    # 1. Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config("src/config/config.yaml")
    model_name = config["model_name"]
    hotpotqa_path = config.get("hotpotqa_dataset_path", "data/hotpot_dev_distractor_v1.json")
    drop_path = config.get("drop_dataset_path", "data/drop_dataset_dev.json")

    # 2. DeviceManager
    device = DeviceManager.get_device()

    # 3. ResourceManager
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager(
        config_path="src/config/resource_config.yaml",
        enable_performance_tracking=True,
        history_window_size=100
    )

    # 4. DimensionalityManager
    print("Initializing Dimensionality Manager...")
    dimensionality_manager = DimensionalityManager(
        target_dim=config['alignment']['target_dim'],
        device=device
    )
    query_adapter = nn.Linear(384, 768).to(device)
    dimensionality_manager.register_adapter('query', query_adapter)

    # 5. Ensure rules.json exists
    rules_path = "data/rules.json"
    if not os.path.exists(rules_path):
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    print(f"Loading existing rules from {rules_path} (initially empty or minimal).")

    # 6. GraphSymbolicReasoner
    print("Initializing Graph-Based Symbolic Reasoner...")
    try:
        symbolic = GraphSymbolicReasoner(
            rules_file=rules_path,
            match_threshold=0.1,
            max_hops=5,
            embedding_model='all-MiniLM-L6-v2',
            device=device,
            dim_manager=dimensionality_manager
        )
        count_rules = len(symbolic.rules) if hasattr(symbolic, 'rules') else 0
        print(f"Loaded {count_rules} rules successfully" if count_rules else "Warning: No rules loaded in symbolic reasoner")
    except Exception as e:
        print(f"Error initializing symbolic reasoner: {e}")
        print("Continuing with empty rule set...")
        symbolic = GraphSymbolicReasoner(
            rules_file=rules_path,
            match_threshold=0.1,
            max_hops=5,
            embedding_model='all-MiniLM-L6-v2',
            device=device,
            dim_manager=dimensionality_manager
        )

    # 7. NeuralRetriever
    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(
        model_name,
        use_quantization=False,
        device=device
    )

    # 8. Support components
    print("Initializing support components...")
    query_logger = QueryLogger()
    feedback_manager = FeedbackManager()
    print("Initializing QueryExpander...")
    expander = QueryExpander(complexity_config="src/config/complexity_rules.yaml")
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()

    print("Loading evaluation dataset...")
    test_queries = []
    ground_truths = {}

    if dataset_type.lower() == 'drop':
        if os.path.exists(drop_path):
            print(f"Loading DROP dataset from {drop_path}...")
            test_queries = load_drop_dataset(drop_path, max_samples=samples)
            ground_truths = {s["query_id"]: s["answer"] for s in test_queries}
        else:
            print(f"ERROR: DROP dataset not found at {drop_path}")
            return {"error": "DROP dataset not found"}

    elif dataset_type.lower() == 'hotpotqa':
        if os.path.exists(hotpotqa_path):
            print(f"Loading HotpotQA dataset from {hotpotqa_path}...")
            test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
            for sample in test_queries:
                ground_truths[sample["query_id"]] = sample["answer"]
                new_rules = rule_extractor.extract_hotpot_facts(sample["context"], min_confidence=0.7)
                if new_rules:
                    try:
                        symbolic.add_dynamic_rules(new_rules)
                    except AttributeError as e:
                        logger.warning(f"Could not track new rules automatically: {e}")
        else:
            print(f"ERROR: HotpotQA dataset not found at {hotpotqa_path}")
            return {"error": "HotpotQA dataset not found"}

    else:
        print(f"ERROR: Unknown dataset_type '{dataset_type}'. Use 'hotpotqa' or 'drop'.")
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        print("No test queries loaded. Exiting.")
        return {"error": "No test queries loaded"}

    evaluator = Evaluation(dataset_type=dataset_type)

    # 9. Create HybridIntegrator (note: resource_manager is no longer passed here)
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(
        symbolic_reasoner=symbolic,
        neural_retriever=neural,
        query_expander=expander,
        dim_manager=dimensionality_manager,
        dataset_type=dataset_type
    )

    # 10. SystemControlManager
    print("Initializing System Control Components...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    metrics_collector = MetricsCollector()
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=2,
        max_query_time=30
    )

    # 11. Application initialization skipped for batch run

    # 12. Optional general knowledge base
    general_context = ""
    kb_path = "data/small_knowledge_base.txt"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as kb_file:
            general_context = kb_file.read()

    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    for q_info in test_queries:
        query_id = q_info["query_id"]
        query = q_info["query"]
        the_answer_obj = q_info.get("answer")
        local_context = q_info.get("context", general_context)
        forced_path = q_info.get("forced_path")
        data_type = q_info.get("type", "")
        supporting_facts = q_info.get("supporting_facts")

        print(f"\nProcessing Query ID: {query_id}, Query: {query[:100]}...")
        print(f"Query Type: {data_type}")
        if forced_path:
            print(f"Forced Path: {forced_path}")
        print("-" * 50)

        prediction_val = "Error: No prediction generated"
        final_answer_dict = {"result": "Error in processing"}

        try:
            initial_time = time.time()
            complexity = expander.get_query_complexity(query)
            print(f"Query Complexity Score: {complexity:.4f}")

            initial_metrics = resource_manager.check_resources()
            final_answer_dict = system_manager.process_query_with_fallback(
                query,
                local_context,
                forced_path=forced_path,
                query_complexity=complexity,
                query_id=query_id
            )
            processing_time_seconds = time.time() - initial_time

            final_metrics = resource_manager.check_resources()
            resource_delta = {k: final_metrics[k] - initial_metrics[k] for k in final_metrics}

            # Extract prediction
            pred = final_answer_dict.get('result', '')
            prediction_val = pred[0] if isinstance(pred, tuple) else pred

            reasoning_path_val = final_answer_dict.get('reasoning_path', 'unknown')

            metrics_collector.collect_query_metrics(
                query=query,
                prediction=str(prediction_val),
                ground_truth=str(the_answer_obj),
                reasoning_path=reasoning_path_val,
                processing_time=processing_time_seconds,
                resource_usage=resource_delta,
                complexity_score=complexity
            )

            # Optional per-component timing logging
            if isinstance(final_answer_dict, dict):
                metrics_collector.component_metrics['symbolic']['execution_time'].append(
                    final_answer_dict.get('symbolic_time', 0.0)
                )
                metrics_collector.component_metrics['neural']['execution_time'].append(
                    final_answer_dict.get('neural_time', 0.0)
                )

            query_logger.log_query(
                query=query,
                result=final_answer_dict,
                source=reasoning_path_val,
                complexity=complexity,
                resource_usage=resource_delta,
                processing_time=processing_time_seconds
            )

            # Print results
            print("\nProcessing Results:")
            print("-" * 20)
            print(f"Prediction: {prediction_val}")
            print(f"Reasoning Path: {reasoning_path_val}")
            print("\nResource Usage:")
            print(f"CPU Delta: {resource_delta['cpu'] * 100:.1f}%")
            print(f"Memory Delta: {resource_delta['memory'] * 100:.1f}%")
            if 'gpu' in resource_delta:
                print(f"GPU Delta: {resource_delta['gpu'] * 100:.1f}%")
            print("-" * 20)

            # Evaluation
            if data_type.startswith("ground_truth_available") and the_answer_obj is not None:
                chain_dict = {}
                eval_metrics = evaluator.evaluate(
                    predictions={query_id: prediction_val},
                    ground_truths={query_id: the_answer_obj},
                    supporting_facts={query_id: supporting_facts} if supporting_facts else None,
                    reasoning_chain=chain_dict
                )
                print("\nEvaluation Metrics:")
                print(f"Exact Match: {eval_metrics.get('average_exact_match', 0.0):.2f}")
                print(f"F1 Score: {eval_metrics.get('average_f1', 0.0):.2f}")
                if dataset_type.lower() == 'hotpotqa':
                    print(f"ROUGE-L: {eval_metrics.get('average_rougeL', 0.0):.2f}")
                    print(f"BLEU: {eval_metrics.get('average_bleu', 0.0):.2f}")

        except KeyError as e:
            print(f"Error: Missing key during processing query {query_id} - {e}")
            logger.exception(f"KeyError for query_id {query_id}")
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            logger.exception(f"Exception for query_id {query_id}")

    print("\nSkipping ablation study in this focused dataset run.")
    print("\nSkipping comparison experiment in this focused dataset run.")

    print("\n=== System Performance Summary ===")
    perf = system_manager.get_performance_metrics()
    if perf.get('total_queries', 0) > 0:
        print(f"- Total Queries: {perf['total_queries']}")
        # Guard against missing key
        if 'avg_response_time' in perf:
            print(f"- Avg Response Time: {perf['avg_response_time']:.2f}s")
        else:
            print(f"- Avg Response Time: N/A")
        print(f"- Success Rate: {perf.get('success_rate', 0.0):.1f}%")
    else:
        print("- No queries processed successfully.")

    print("\nResource Utilization (Final State):")
    final_res = resource_manager.check_resources()
    print(f"- CPU Usage: {final_res['cpu'] * 100:.1f}%")
    print(f"- Memory Usage: {final_res['memory'] * 100:.1f}%")
    if 'gpu' in final_res:
        print(f"- GPU Usage: {final_res['gpu'] * 100:.1f}%")

    print("\nReasoning Path Distribution:")
    path_stats = system_manager.get_reasoning_path_stats()
    total_q = perf.get('total_queries', 0)
    if total_q > 0:
        for path, stats in path_stats.items():
            cnt = stats.get('count', 0)
            pct = (cnt / total_q) * 100
            print(f"- {path}: {cnt} queries ({pct:.1f}%)")
    else:
        print("- No queries to show distribution.")

    print("\n=== Comprehensive Academic Analysis ===")
    academic_report = metrics_collector.generate_academic_report()

    print("\nPerformance Analysis:")
    perf_metrics = academic_report.get('performance_metrics', {})
    if 'processing_time' in perf_metrics:
        proc = perf_metrics['processing_time']
        print(f"- Avg Processing Time: {proc.get('mean', 0.0):.2f}s")
        if 'percentile_95' in proc:
            print(f"- 95th Percentile Time: {proc['percentile_95']:.2f}s")

    print("\nReasoning Analysis:")
    ra = academic_report.get('reasoning_analysis', {})
    cc = ra.get('chain_characteristics', {})
    print(f"- Avg Chain Length: {cc.get('avg_length', 0.0):.2f}")
    print(f"- Avg Confidence: {cc.get('avg_confidence', 0.0):.2f}")

    print("\nResource Efficiency:")
    em = academic_report.get('efficiency_metrics', {})
    for resource, metrics_data in em.items():
        if resource != 'trends':
            mean_u = metrics_data.get('mean_usage', 0.0) * 100
            peak_u = metrics_data.get('peak_usage', 0.0) * 100
            score = metrics_data.get('efficiency_score', 0.0)
            print(f"- {resource.capitalize()}: Mean {mean_u:.1f}%, Peak {peak_u:.1f}%, Score {score:.2f}")

    print("\nStatistical Analysis:")
    sa = academic_report.get('statistical_analysis', {})
    for metric, stats_data in sa.items():
        if isinstance(stats_data, dict) and 'p_value' in stats_data:
            print(f"- {metric}: p-value: {stats_data['p_value']:.3f}, effect size: {stats_data.get('effect_size', 0.0):.2f}")

    print("\n=== End of Run ===")
    print(json.dumps(academic_report, indent=2, default=str))
    return academic_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HySym-RAG system with output capture')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'drop'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--log-dir', default='logs', help='Directory to save log files')
    parser.add_argument('--no-output-capture', action='store_true', help='Disable output capture to file')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to process')

    args = parser.parse_args()

    if args.no_output_capture:
        run_hysym_system(samples=args.samples, dataset_type=args.dataset)
    else:
        dataset_log_dir = os.path.join(args.log_dir, args.dataset)
        os.makedirs(dataset_log_dir, exist_ok=True)
        with capture_output(output_dir=dataset_log_dir) as output_path:
            print(f"Output being saved to: {output_path}")
            run_hysym_system(samples=args.samples, dataset_type=args.dataset)
