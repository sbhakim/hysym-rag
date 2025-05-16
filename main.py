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
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Transformers library not found. Please install it: pip install transformers")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Sentence-transformers library not found. Please install it: pip install sentence-transformers")
    sys.exit(1)

try:
    from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
    from src.reasoners.networkx_symbolic_reasoner_drop import GraphSymbolicReasonerDrop
    from src.reasoners.neural_retriever import NeuralRetriever
    from src.integrators.hybrid_integrator import HybridIntegrator
    from src.utils.dimension_manager import DimensionalityManager
    from src.utils.rule_extractor import RuleExtractor
    from src.queries.query_logger import QueryLogger
    from src.resources.resource_manager import ResourceManager
    from src.feedback.feedback_manager import FeedbackManager
    from src.config.config_loader import ConfigLoader
    from src.queries.query_expander import QueryExpander
    from src.utils.evaluation import Evaluation
    from src.system.system_control_manager import SystemControlManager, UnifiedResponseAggregator
    from src.utils.metrics_collector import MetricsCollector
    from src.utils.device_manager import DeviceManager
    from src.utils.progress import tqdm, ProgressManager
    from src.utils.output_capture import capture_output
except ImportError as e:
    print(f"Error importing HySym-RAG components: {e}")
    print("Please ensure main.py is run from the project root directory or PYTHONPATH is set correctly.")
    sys.exit(1)

urllib3.disable_warnings()
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_hotpotqa(hotpotqa_path, max_samples=None):
    """
    Loads a portion of the HotpotQA dataset.
    Each sample includes a query, ground-truth answer,
    combined context, and a 'type' = 'ground_truth_available_hotpotqa'.
    """
    if not os.path.exists(hotpotqa_path):
        logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
        return []
    dataset = []
    try:
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse HotpotQA JSON from {hotpotqa_path}: {e}")
        return []

    count = 0
    for example in data:
        if not all(k in example for k in ['question', 'answer', 'supporting_facts', 'context', '_id']):
            logger.warning(f"Skipping invalid HotpotQA example (missing keys): {example.get('_id', 'Unknown ID')}")
            continue

        question = example['question']
        answer = example['answer']
        supporting_facts = example['supporting_facts']

        context_str_parts = []
        for title, sents in example.get('context', []):
            if isinstance(title, str) and isinstance(sents, list):
                combined_sents = " ".join(str(s) for s in sents)
                context_str_parts.append(f"{title}: {combined_sents}")
        context_str = "\n".join(context_str_parts)

        dataset.append({
            "query_id": example['_id'],
            "query": question,
            "answer": answer,
            "context": context_str,
            "type": "ground_truth_available_hotpotqa",
            "supporting_facts": supporting_facts
        })
        count += 1
        if max_samples and count >= max_samples:
            logger.info(f"Loaded {count} HotpotQA samples (max requested: {max_samples}).")
            break
    logger.info(f"Finished loading HotpotQA. Total samples: {len(dataset)}.")
    return dataset

def load_drop_dataset(drop_path, max_samples=None):
    """
    Loads a portion of the DROP dataset.
    """
    if not os.path.exists(drop_path):
        logger.error(f"DROP dataset file not found at: {drop_path}")
        return []
    dataset = []
    try:
        with open(drop_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse DROP JSON from {drop_path}: {e}")
        return []

    total_loaded_qas = 0
    for passage_id, passage_content in data.items():
        if not isinstance(passage_content, dict) or 'passage' not in passage_content or 'qa_pairs' not in passage_content:
            logger.warning(f"Skipping invalid passage structure for passage_id: {passage_id}")
            continue

        passage_text = passage_content['passage']
        if not isinstance(passage_content['qa_pairs'], list):
            logger.warning(f"Invalid qa_pairs format (not a list) for passage_id: {passage_id}")
            continue

        for qa_pair_idx, qa_pair in enumerate(passage_content['qa_pairs']):
            if not isinstance(qa_pair, dict) or 'question' not in qa_pair or 'answer' not in qa_pair:
                logger.warning(f"Skipping invalid qa_pair structure in passage_id {passage_id}, index {qa_pair_idx}")
                continue

            question = qa_pair['question']
            answer_obj = qa_pair['answer']
            query_id = qa_pair.get("query_id", f"{passage_id}_{qa_pair_idx}")

            dataset.append({
                "query_id": query_id,
                "query": question,
                "context": passage_text,
                "answer": answer_obj,
                "type": "ground_truth_available_drop"
            })
            total_loaded_qas += 1
            if max_samples and total_loaded_qas >= max_samples:
                logger.info(f"Loaded {total_loaded_qas} DROP samples (max requested: {max_samples}).")
                return dataset

        if max_samples and total_loaded_qas >= max_samples:
            break

    logger.info(f"Finished loading DROP. Total samples: {len(dataset)}.")
    return dataset

def run_hysym_system(samples=200, dataset_type='hotpotqa'):
    """Main execution function for the HySym-RAG system."""
    print(f"\n=== Initializing HySym-RAG System for Dataset: {dataset_type.upper()} ===")

    ProgressManager.SHOW_PROGRESS = False
    for lib in ['transformers', 'sentence_transformers', 'urllib3.connectionpool', 'h5py', 'numexpr']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('src.utils.dimension_manager').setLevel(logging.INFO)
    logging.getLogger('src.integrators.hybrid_integrator').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_base').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_drop').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.neural_retriever').setLevel(logging.INFO)
    logging.getLogger('src.system.system_control_manager').setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    # 1. Load configuration
    print("Loading configuration...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "src", "config", "config.yaml")
        config = ConfigLoader.load_config(config_path)
        model_name = config["model_name"]
        project_root = os.path.dirname(script_dir)
        hotpotqa_path = config.get("hotpotqa_dataset_path", os.path.join(project_root, "data", "hotpot_dev_distractor_v1.json"))
        drop_path = config.get("drop_dataset_path", os.path.join(project_root, "data", "drop_dataset_dev.json"))
        rules_path_default = os.path.join(project_root, "data", "rules.json")
        hotpotqa_rules_path = config.get("hotpotqa_rules_file", rules_path_default)
        drop_rules_path = config.get("drop_rules_file", rules_path_default)
        kb_path_default = os.path.join(project_root, "data", "small_knowledge_base.txt")
        kb_path = config.get("knowledge_base", kb_path_default)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {"error": f"Configuration loading failed: {e}"}

    # 2. DeviceManager
    device = DeviceManager.get_device()
    print(f"Using device: {device}")

    # 3. ResourceManager
    print("Initializing Resource Manager...")
    resource_config_path = os.path.join(script_dir, "src", "config", "resource_config.yaml")
    resource_manager = ResourceManager(
        config_path=resource_config_path,
        enable_performance_tracking=True,
        history_window_size=100
    )

    # 4. DimensionalityManager
    print("Initializing Dimensionality Manager...")
    alignment_config = config.get('alignment', {'target_dim': 768})
    target_dim = alignment_config.get('target_dim', 768)
    dimensionality_manager = DimensionalityManager(target_dim=target_dim, device=device)

    # 5. Select and Ensure Rules File
    rules_path = drop_rules_path if dataset_type.lower() == 'drop' else hotpotqa_rules_path
    print(f"Using rules file: {rules_path}")
    if not os.path.exists(rules_path):
        logger.warning(f"Selected rules file {rules_path} does not exist. Creating empty file.")
        try:
            os.makedirs(os.path.dirname(rules_path), exist_ok=True)
            with open(rules_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"Failed to create empty rules file at {rules_path}: {e}")
            return {"error": f"Failed to ensure rules file exists at {rules_path}"}

    # 6. Initialize RuleExtractor
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()

    # 7. Load Evaluation Dataset and Extract Rules
    print(f"Loading evaluation dataset: {dataset_type.upper()}...")
    test_queries = []
    ground_truths = {}

    if dataset_type.lower() == 'drop':
        if os.path.exists(drop_path):
            print(f"Loading DROP dataset from {drop_path}...")
            test_queries = load_drop_dataset(drop_path, max_samples=samples)
            ground_truths = {s["query_id"]: s["answer"] for s in test_queries if "query_id" in s and "answer" in s}

            # Extract DROP rules dynamically
            questions = [qa['query'] for qa in test_queries]
            passages = [qa['context'] for qa in test_queries]
            print("Extracting DROP-specific rules...")
            try:
                dynamic_rules = rule_extractor.extract_rules_from_drop(
                    drop_json_path=drop_path,
                    questions=questions,
                    passages=passages,
                    min_support=5
                )
                rules_path = os.path.join(project_root, "data", "rules_drop.json")
                # Ensure the directory exists before writing
                os.makedirs(os.path.dirname(rules_path), exist_ok=True)
                with open(rules_path, "w", encoding="utf-8") as f:
                    json.dump(dynamic_rules, f, indent=2)
                print(f"Saved {len(dynamic_rules)} DROP rules to {rules_path}")

                # Check if zero rules were extracted
                if not dynamic_rules:
                    logger.warning("No rules were extracted for DROP dataset. Proceeding with an empty rule set.")
                    # Create an empty rule file to prevent loading errors
                    with open(rules_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
                    print("Created empty rules file to proceed with processing.")
            except Exception as e:
                logger.error(f"Failed to extract DROP rules: {e}")
                return {"error": f"DROP rule extraction failed: {e}"}
        else:
            logger.error(f"DROP dataset not found at specified path: {drop_path}")
            return {"error": "DROP dataset not found"}

    elif dataset_type.lower() == 'hotpotqa':
        if os.path.exists(hotpotqa_path):
            print(f"Loading HotpotQA dataset from {hotpotqa_path}...")
            test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
            for sample in test_queries:
                ground_truths[sample["query_id"]] = sample["answer"]
        else:
            logger.error(f"HotpotQA dataset not found at specified path: {hotpotqa_path}")
            return {"error": "HotpotQA dataset not found"}
    else:
        logger.error(f"Unknown dataset_type '{dataset_type}'. Cannot load data.")
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        logger.error("No test queries loaded. Exiting.")
        return {"error": "No test queries loaded"}
    print(f"Loaded {len(test_queries)} test queries.")

    # 8. GraphSymbolicReasoner (Instantiate based on dataset type)
    print(f"Initializing Symbolic Reasoner for {dataset_type.upper()}...")
    symbolic = None
    embedding_model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
    try:
        if dataset_type.lower() == 'drop':
            symbolic = GraphSymbolicReasonerDrop(
                rules_file=rules_path,
                match_threshold=0.1,
                max_hops=3,
                embedding_model=embedding_model_name,
                device=device,
                dim_manager=dimensionality_manager
            )
        else:
            symbolic = GraphSymbolicReasoner(
                rules_file=rules_path,
                match_threshold=0.1,
                max_hops=5,
                embedding_model=embedding_model_name,
                device=device,
                dim_manager=dimensionality_manager
            )
        count_rules = len(getattr(symbolic, 'rules', []))
        print(f"Symbolic Reasoner loaded {count_rules} rules successfully.")
        if count_rules == 0:
            logger.warning(f"Symbolic reasoner initialized, but no rules were loaded from {rules_path}. Symbolic path may be ineffective.")
    except Exception as e:
        logger.error(f"Fatal error initializing symbolic reasoner: {e}", exc_info=True)
        return {"error": f"Symbolic reasoner initialization failed: {e}"}

    # 9. NeuralRetriever
    print("Initializing Neural Retriever...")
    try:
        neural = NeuralRetriever(
            model_name,
            use_quantization=False,
            device=device
        )
    except Exception as e:
        logger.error(f"Fatal error initializing neural retriever: {e}", exc_info=True)
        return {"error": f"Neural retriever initialization failed: {e}"}

    # 10. Support components
    print("Initializing support components...")
    query_logger = QueryLogger()
    feedback_manager = FeedbackManager()
    complexity_config_path = os.path.join(script_dir, "src", "config", "complexity_rules.yaml")
    if not os.path.exists(complexity_config_path):
        logger.warning(f"Complexity config {complexity_config_path} not found. QueryExpander may use defaults.")
        complexity_config_path = None
    expander = QueryExpander(complexity_config=complexity_config_path)

    # 11. Initialize Evaluation utility
    evaluator = Evaluation(dataset_type=dataset_type)

    # 12. Create HybridIntegrator
    print("Creating Hybrid Integrator...")
    try:
        integrator = HybridIntegrator(
            symbolic_reasoner=symbolic,
            neural_retriever=neural,
            query_expander=expander,
            dim_manager=dimensionality_manager,
            dataset_type=dataset_type
        )
    except Exception as e:
        logger.error(f"Fatal error initializing Hybrid Integrator: {e}", exc_info=True)
        return {"error": f"Hybrid Integrator initialization failed: {e}"}

    # 13. SystemControlManager
    print("Initializing System Control Manager...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    metrics_collector = MetricsCollector(dataset_type=dataset_type)
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=2,
        max_query_time=30
    )

    # 14. Optional general knowledge base
    general_context = ""
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as kb_file:
                general_context = kb_file.read()
            print(f"Loaded general knowledge base from {kb_path}")
        except Exception as e:
            logger.warning(f"Could not load general knowledge base from {kb_path}: {e}")

    # Main Query Processing Loop
    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    results_list = []

    query_iterator = tqdm(test_queries, desc="Processing Queries", disable=not ProgressManager.SHOW_PROGRESS)

    for q_info in query_iterator:
        query_id = q_info.get("query_id")
        query = q_info.get("query")
        the_answer_obj = q_info.get("answer")
        local_context = q_info.get("context", general_context)
        forced_path = q_info.get("forced_path")
        data_type = q_info.get("type", "")
        supporting_facts = q_info.get("supporting_facts")

        if not query_id or not query:
            logger.warning(f"Skipping query due to missing ID or text: {q_info}")
            continue

        print(f"\nProcessing Query ID: {query_id}, Query: {query[:100]}...")
        logger.debug(f"Query Type: {data_type}, Forced Path: {forced_path}")
        print("-" * 50)

        final_response_dict: Dict[str, Any] = {}
        reasoning_path_val: str = 'error'

        try:
            complexity = expander.get_query_complexity(query)
            logger.info(f"Query Complexity Score: {complexity:.4f}")

            response_dict, path_string = system_manager.process_query_with_fallback(
                query=query,
                context=local_context,
                query_id=query_id,
                forced_path=forced_path,
                query_complexity=complexity,
                supporting_facts=supporting_facts,
                dataset_type=dataset_type
            )

            final_response_dict = response_dict
            reasoning_path_val = path_string

            if reasoning_path_val == 'fallback_error':
                logger.error(f"Query {query_id} failed processing after retries. Error: {final_response_dict.get('error')}")
                prediction_val = {'error': final_response_dict.get('error', 'Unknown processing error')} if dataset_type=='drop' else "Error: Processing Failed"
            else:
                logger.info(f"Query {query_id} processed via path: {reasoning_path_val}")
                prediction_val = final_response_dict.get('result', 'Error: Result key missing')

            results_list.append(final_response_dict)

            print("\nProcessing Results:")
            print("-" * 20)
            if isinstance(prediction_val, dict):
                print(f"Prediction (DROP): {prediction_val}")
            else:
                print(f"Prediction (Text): {str(prediction_val)[:200]}...")
            print(f"Reasoning Path: {reasoning_path_val}")
            print(f"Processing Time: {final_response_dict.get('processing_time', 0.0):.3f}s")
            print(f"Status: {final_response_dict.get('status', 'unknown')}")

            res_delta = final_response_dict.get('resource_usage', {})
            if res_delta:
                print("\nResource Usage (Delta):")
                for k, v in res_delta.items():
                    print(f"- {k.capitalize()}: {v*100:+.1f}%")
            print("-" * 20)

            if data_type.startswith("ground_truth_available") and the_answer_obj is not None and reasoning_path_val != 'fallback_error':
                if query_id in ground_truths:
                    gt = ground_truths[query_id]
                    sf = {query_id: supporting_facts} if supporting_facts and dataset_type != 'drop' else None
                    rc = {query_id: final_response_dict.get('debug_info', {}).get('steps')} if 'debug_info' in final_response_dict else None

                    eval_metrics = evaluator.evaluate(
                        predictions={query_id: prediction_val},
                        ground_truths={query_id: gt},
                        supporting_facts=sf,
                        reasoning_chain=rc
                    )
                    print("\nEvaluation Metrics:")
                    print(f"  Exact Match: {eval_metrics.get('average_exact_match', 0.0):.3f}")
                    print(f"  F1 Score: {eval_metrics.get('average_f1', 0.0):.3f}")
                    if dataset_type != 'drop':
                        print(f"  ROUGE-L: {eval_metrics.get('average_rougeL', 0.0):.3f}")
                        print(f"  BLEU: {eval_metrics.get('average_bleu', 0.0):.3f}")
                        print(f"  Semantic Sim: {eval_metrics.get('average_semantic_similarity', 0.0):.3f}")
                else:
                    logger.warning(f"Ground truth not found for query_id {query_id} during evaluation.")

        except Exception as e:
            print(f"\n--- ERROR processing query {query_id} ---")
            logger.exception(f"Unhandled exception for query_id {query_id} in main loop: {e}")
            print(f"Error: {e}")
            print("-" * 30)

    print("\n" + "="*15 + " System Performance Summary " + "="*15)
    try:
        perf = system_manager.get_performance_metrics()
        if perf.get('total_queries', 0) > 0:
            print(f"- Total Queries Processed: {perf['total_queries']}")
            print(f"- Successful Queries: {perf['successful_queries']}")
            print(f"- Success Rate: {perf.get('success_rate', 0.0):.1f}%")
            print(f"- Error Count: {perf.get('error_count', 0)}")
            print(f"- Avg Successful Response Time: {perf.get('avg_response_time', 0.0):.3f}s")
        else:
            print("- No queries processed.")

        print("\nResource Utilization (Final State):")
        final_res = resource_manager.check_resources()
        cpu_final = final_res.get('cpu', 0.0) or 0.0
        mem_final = final_res.get('memory', 0.0) or 0.0
        print(f"- CPU Usage: {cpu_final * 100:.1f}%")
        print(f"- Memory Usage: {mem_final * 100:.1f}%")
        if 'gpu' in final_res:
            gpu_final = final_res.get('gpu', 0.0) or 0.0
            print(f"- GPU Usage: {gpu_final * 100:.1f}%")

        print("\nReasoning Path Distribution (Execution Counts):")
        path_stats = system_manager.get_reasoning_path_stats()
        if path_stats:
            for path, stats in path_stats.items():
                count = stats.get('count', 0)
                success = stats.get('success', 0)
                avg_time = stats.get('avg_time', 0.0)
                perc_total = stats.get('percentage', 0.0)
                print(f"- {path}: Count={count} ({perc_total:.1f}%), Success={success}, AvgTime={avg_time:.3f}s")
        else:
            print("- No path statistics available.")

    except Exception as report_err:
        logger.error(f"Error generating performance summary: {report_err}", exc_info=True)

    print("\n" + "="*15 + " Comprehensive Academic Analysis " + "="*15)
    try:
        academic_report = metrics_collector.generate_academic_report()

        print("\nPerformance Analysis (from Metrics Collector):")
        perf_metrics = academic_report.get('performance_metrics', {})
        if 'processing_time' in perf_metrics:
            proc = perf_metrics['processing_time']
            print(f"- Avg Processing Time: {proc.get('mean', 0.0):.3f}s (Std: {proc.get('std', 0.0):.3f})")
            if 'percentile_95' in proc:
                print(f"- 95th Percentile Time: {proc.get('percentile_95', 0.0):.3f}s")
        else:
            print("- No processing time metrics collected.")

        print("\nReasoning Analysis:")
        ra = academic_report.get('reasoning_analysis', {})
        cc = ra.get('chain_characteristics', {})
        print(f"- Avg Chain Length: {cc.get('avg_length', 0.0):.2f}")
        print(f"- Avg Confidence: {cc.get('avg_confidence', 0.0):.3f}")
        print(f"- Path Distribution: {ra.get('path_distribution', {})}")

        print("\nResource Efficiency:")
        em = academic_report.get('efficiency_metrics', {})
        for resource, metrics_data in em.items():
            if resource != 'trends' and isinstance(metrics_data, dict):
                mean_u = metrics_data.get('mean_usage', 0.0) * 100
                peak_u = metrics_data.get('peak_usage', 0.0) * 100
                score = metrics_data.get('efficiency_score', 0.0)
                print(f"- {resource.capitalize()}: Mean Delta {mean_u:+.1f}%, Peak Delta {peak_u:.1f}%, Score {score:.2f}")
        print(f"- Efficiency Trends: {em.get('trends', {})}")

        print("\nStatistical Analysis:")
        sa = academic_report.get('statistical_analysis', {})
        for metric, stats_data in sa.get('tests',{}).items():
            if isinstance(stats_data, dict) and 'p_value' in stats_data:
                print(f"- {metric}: p-value={stats_data['p_value']:.3g}, effect_size={stats_data.get('effect_size', 0.0):.2f}")
        print(f"- Correlations: {sa.get('correlations', {})}")
        print(f"- Regression: {sa.get('regression_analysis', {})}")

        print("\n" + "="*15 + " End of Run " + "="*15)
        report_file_path = os.path.join(dataset_log_dir if 'dataset_log_dir' in locals() else 'logs', f"academic_report_{dataset_type}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_file_path, 'w', encoding='utf-8') as rf:
                json.dump(academic_report, rf, indent=2, default=str)
            print(f"Academic report saved to: {report_file_path}")
        except Exception as save_err:
            print(f"Warning: Could not save academic report to {report_file_path}: {save_err}")

        return academic_report

    except Exception as acad_err:
        logger.error(f"Error generating academic report: {acad_err}", exc_info=True)
        return {"error": f"Academic report generation failed: {acad_err}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HySym-RAG system with output capture')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'drop'],
                        help='Dataset to use for evaluation (hotpotqa or drop)')
    parser.add_argument('--log-dir', default='logs', help='Directory to save log files')
    parser.add_argument('--no-output-capture', action='store_true', help='Disable output capture to file')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG)
        print("--- DEBUG Logging Enabled ---")

    log_directory = args.log_dir
    dataset_log_dir = os.path.join(log_directory, args.dataset)

    if args.no_output_capture:
        print("--- Running without output capture ---")
        run_hysym_system(samples=args.samples, dataset_type=args.dataset)
    else:
        try:
            os.makedirs(dataset_log_dir, exist_ok=True)
            with capture_output(output_dir=dataset_log_dir) as output_path:
                print(f"Output being saved to: {output_path}")
                run_hysym_system(samples=args.samples, dataset_type=args.dataset)
        except Exception as e:
            print(f"ERROR: Failed to set up output capture or run system: {e}", file=sys.stderr)
            logger.error(f"Failed to set up output capture or run system: {e}", exc_info=True)