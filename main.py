# main.py

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
from src.ablation_study import run_ablation_study
from src.utils.progress import tqdm, ProgressManager
from src.utils.output_capture import capture_output  # Import the capture_output context manager

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os
import json
import torch
import warnings
import time
from collections import defaultdict
import torch.nn as nn
import logging
import urllib3
import sys  # Import sys
import argparse

urllib3.disable_warnings()

# Suppress specific spaCy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False  # Globally disable progress bars

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_hotpotqa(hotpotqa_path, max_samples=None):
    """
    Loads a portion of the HotpotQA dataset.
    Each sample includes a query, ground-truth answer,
    combined context, and a 'type' = 'ground_truth_available'.
    """
    dataset = []
    with open(hotpotqa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # a single large JSON array

    count = 0
    for example in data:
        question = example['question']
        answer = example['answer']
        supporting_facts = example['supporting_facts']  # Retrieve supporting facts

        # Flatten the context
        context_str = []
        for ctx_item in example.get('context', []):
            title, sents = ctx_item[0], ctx_item[1]
            combined_sents = " ".join(sents)
            context_str.append(f"{title}: {combined_sents}")
        context_str = "\n".join(context_str)

        dataset.append({
            "query_id": example.get('_id', f"hotpot_q_{count}"),  # Use existing ID or generate one
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
        data = json.load(f)  # data is a dict of passage_id: passage_data

    total_loaded_qas = 0
    for passage_id, passage_content in data.items():
        passage_text = passage_content['passage']
        for qa_pair in passage_content['qa_pairs']:
            question = qa_pair['question']
            answer_obj = qa_pair['answer']  # DROP answers are structured

            dataset.append({
                "query_id": qa_pair.get("query_id", f"{passage_id}_{total_loaded_qas}"),  # Unique ID
                "query": question,
                "context": passage_text,
                "answer": answer_obj,  # Store the structured answer object
                "type": "ground_truth_available_drop"
            })
            total_loaded_qas += 1
            if max_samples and total_loaded_qas >= max_samples:
                return dataset
        if max_samples and total_loaded_qas >= max_samples:  # Check after processing all QAs for a passage
            break
    return dataset


def run_hysym_system(samples=200, dataset_type='hotpotqa'):  # Added dataset_type argument
    """Main execution function for the HySym-RAG system."""

    print("\n=== Initializing HySym-RAG System ===")

    ProgressManager.SHOW_PROGRESS = False  # Globally disable progress bars
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('DimensionalityManager').setLevel(logging.DEBUG)
    logging.getLogger('src.utils.dimension_manager').setLevel(logging.WARNING)  # Reduce dimension manager logs
    logging.getLogger('src.knowledge_integrator').setLevel(logging.INFO)  # Reduce knowledge integrator logs
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner').setLevel(
        logging.INFO)  # Reduce symbolic reasoner logs
    logger.setLevel(logging.DEBUG)

    # Disable HTTP connection pool warnings
    requests_log = logging.getLogger("urllib3.connectionpool")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = False

    # 1. Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config("src/config/config.yaml")
    model_name = config["model_name"]
    # Get dataset paths from config or use defaults
    hotpotqa_path_config = config.get("hotpotqa_dataset_path", "data/hotpot_dev_distractor_v1.json")
    drop_path_config = config.get("drop_dataset_path", "data/drop_dataset_dev.json")

    # 2. Acquire a unified device from DeviceManager
    device = DeviceManager.get_device()

    # 3. Initialize Resource Manager
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager(
        config_path="src/config/resource_config.yaml",
        enable_performance_tracking=True,
        history_window_size=100
    )

    # 4. Initialize Dimensionality Manager (FIRST - Before other components!)
    print("Initializing Dimensionality Manager...")
    dimensionality_manager = DimensionalityManager(target_dim=config['alignment']['target_dim'], device=device)
    # Register a query adapter: project from 384 to 768 dimensions
    query_adapter = nn.Linear(384, 768).to(device)
    dimensionality_manager.register_adapter('query', query_adapter)

    # 5. Ensure that 'data/rules.json' exists (empty or minimal)
    # TODO: Consider separate rule files for different datasets or reasoning types (e.g., rules_drop.json)
    rules_path = "data/rules.json"
    if not os.path.exists(rules_path):
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    print(f"Loading existing rules from {rules_path} (initially empty or minimal).")

    # 6. Initialize the Graph-Based Symbolic Reasoner (Pass dim_manager)
    print("Initializing Graph-Based Symbolic Reasoner...")
    try:
        # Adjusted match_threshold to 0.1 to leverage enhanced rule matching improvements
        symbolic = GraphSymbolicReasoner(
            rules_file=rules_path,
            match_threshold=0.1,  # This threshold might need tuning for DROP
            max_hops=5,
            embedding_model='all-MiniLM-L6-v2',
            device=device,
            dim_manager=dimensionality_manager  # PASS dim_manager HERE
        )
        if hasattr(symbolic, 'rules') and symbolic.rules:
            print(f"Loaded {len(symbolic.rules)} rules successfully")
        else:
            print("Warning: No rules loaded in symbolic reasoner")
    except Exception as e:
        print(f"Error initializing symbolic reasoner: {str(e)}")
        print("Continuing with empty rule set...")
        symbolic = GraphSymbolicReasoner(
            rules_file=rules_path,
            match_threshold=0.1,
            max_hops=5,
            embedding_model='all-MiniLM-L6-v2',
            device=device,
            dim_manager=dimensionality_manager  # PASS dim_manager HERE (in fallback as well)
        )

    # 7. Initialize the Neural Retriever
    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(
        model_name,
        use_quantization=False,  # Consider quantization for larger models or resource constraints
        device=device
    )

    # 8. Additional components
    print("Initializing support components...")
    query_logger = QueryLogger()
    feedback_manager = FeedbackManager()
    print("Initializing QueryExpander...")
    expander = QueryExpander(
        complexity_config="src/config/complexity_rules.yaml"
        # TODO: QueryExpander might need specific rules/logic for DROP question types
    )
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()  # TODO: RuleExtractor needs significant changes for DROP if generating rules from context

    print("Loading evaluation dataset...")
    test_queries = []
    ground_truths = {}  # Use query_id as key

    if dataset_type.lower() == 'drop':
        if os.path.exists(drop_path_config):
            print(f"Loading DROP dataset from {drop_path_config}...")
            test_queries = load_drop_dataset(drop_path_config, max_samples=samples)
            for sample in test_queries:
                ground_truths[sample["query_id"]] = sample["answer"]
            # Rule extraction for DROP from context is less likely; rules are more operational
        else:
            print(f"ERROR: DROP dataset not found at {drop_path_config}")
            # Decide if to exit or use a fallback
            return {"error": "DROP dataset not found"}
    elif dataset_type.lower() == 'hotpotqa':
        if os.path.exists(hotpotqa_path_config):
            print(f"Loading HotpotQA dataset from {hotpotqa_path_config}...")
            test_queries = load_hotpotqa(hotpotqa_path_config, max_samples=samples)
            for sample in test_queries:
                ground_truths[sample["query_id"]] = sample["answer"]
                # Dynamic rule extraction for HotpotQA
                new_rules = rule_extractor.extract_hotpot_facts(sample["context"], min_confidence=0.7)
                if new_rules:
                    try:
                        symbolic.add_dynamic_rules(new_rules)
                    except AttributeError as e:
                        logger.warning(f"Could not track new rules automatically (missing method?): {str(e)}")
        else:
            print(f"ERROR: HotpotQA dataset not found at {hotpotqa_path_config}")
            return {"error": "HotpotQA dataset not found"}
    else:
        print(f"ERROR: Unknown dataset_type '{dataset_type}'. Please use 'hotpotqa' or 'drop'.")
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        print("No test queries loaded. Exiting.")
        return {"error": "No test queries loaded"}

    evaluator = Evaluation(dataset_type=dataset_type)  # Pass dataset_type to evaluator

    # 9. Create Hybrid Integrator (Pass dim_manager)
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(
        symbolic_reasoner=symbolic,
        neural_retriever=neural,
        resource_manager=resource_manager,
        query_expander=expander,
        dim_manager=dimensionality_manager,
        dataset_type=dataset_type  # Pass dataset_type
    )

    # 10. Initialize System Control Components
    print("Initializing System Control Components...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    metrics_collector = MetricsCollector()
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=2,
        max_query_time=30  # Increased for potentially complex DROP queries
    )

    # 11. Initialize Application
    print("Initializing Application...")
    # feedback_handler = FeedbackHandler(feedback_manager) # Assuming feedback is not primary for this script
    # app = App( # App might not be used directly in this batch processing script
    #     symbolic=symbolic,
    #     neural=neural,
    #     logger=logger, # This should be the global logger
    #     feedback=resource_manager, # Check if this is the correct feedback component
    #     evaluator=evaluator,
    #     expander=expander,
    #     ground_truths=ground_truths,
    #     system_manager=system_manager
    # )

    # 12. Load general knowledge base for neural context (if available and needed)
    # For DROP, the context is given with each query, so a global KB might be less relevant.
    general_context = ""
    kb_path = "data/small_knowledge_base.txt"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as kb_file:
            general_context = kb_file.read()

    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    for q_info in test_queries:
        query_id = q_info["query_id"]
        query = q_info["query"]
        the_answer_obj = q_info.get("answer", None)  # This is the structured answer for DROP
        local_context = q_info.get("context", general_context)  # Use specific context from dataset
        forced_path = q_info.get("forced_path", None)
        data_type = q_info.get("type", "ground_truth_available")  # e.g., ground_truth_available_drop
        supporting_facts = q_info.get("supporting_facts", None)  # Mostly for HotpotQA

        print(f"\nProcessing Query ID: {query_id}, Query: {query[:100]}...")  # Log query ID
        print(f"Query Type: {data_type}")
        if forced_path:
            print(f"Forced Path: {forced_path}")
        print("-" * 50)

        prediction_val = "Error: No prediction generated"  # Default in case of early failure
        final_answer_dict = {"result": "Error in processing"}  # Default structure

        try:
            initial_time = time.time()
            complexity = expander.get_query_complexity(query)  # This might need tuning for DROP
            print(f"Query Complexity Score: {complexity:.4f}")

            initial_metrics = resource_manager.check_resources()

            final_answer_dict = system_manager.process_query_with_fallback(
                query,
                local_context,
                forced_path=forced_path,
                query_complexity=complexity,
                query_id=query_id  # Pass query_id for context in HybridIntegrator if needed
            )
            processing_time_seconds = time.time() - initial_time

            final_metrics = resource_manager.check_resources()
            resource_delta = {
                key: final_metrics[key] - initial_metrics[key]
                for key in final_metrics
            }

            # The structure of final_answer_dict['result'] will be critical for DROP evaluation
            # It needs to be in a format that your Evaluation class can compare against DROP's structured answers
            prediction_val = final_answer_dict.get('result', '')
            if isinstance(prediction_val, tuple) and len(
                    prediction_val) > 0:  # Handle (answer, source) tuple from HybridIntegrator
                prediction_val = prediction_val[0]

            reasoning_path_val = final_answer_dict.get('reasoning_path', 'unknown')
            # Pattern extraction might be less relevant or need adaptation for DROP's discrete reasoning
            # pattern_dict = extract_reasoning_pattern(
            #     query,
            #     final_answer_dict.get('reasoning_path_details', []), # Assuming detailed path is available
            #     symbolic.rules # This assumes rules are relevant in the same way as HotpotQA
            # )
            # pattern_type_val = pattern_dict.get('pattern_type', 'unknown')
            pattern_type_val = reasoning_path_val  # Simplified for now

            metrics_collector.collect_query_metrics(
                query=query,  # Consider logging query_id too
                prediction=str(prediction_val),  # Ensure prediction is string for some basic logging
                ground_truth=str(the_answer_obj),  # Log string representation for now
                reasoning_path=pattern_type_val,
                processing_time=processing_time_seconds,
                resource_usage=resource_delta,
                complexity_score=complexity
            )

            if isinstance(final_answer_dict, dict):
                metrics_collector.component_metrics['symbolic']['execution_time'].append(
                    final_answer_dict.get('symbolic_time', 0.0)
                )
                metrics_collector.component_metrics['neural']['execution_time'].append(
                    final_answer_dict.get('neural_time', 0.0)
                )

            query_logger.log_query(
                query=query,  # Consider logging query_id
                result=final_answer_dict,
                source=final_answer_dict.get('reasoning_path', "hybrid"),
                complexity=complexity,
                resource_usage=resource_delta,
                processing_time=processing_time_seconds
            )

            print("\nProcessing Results:")
            print("-" * 20)
            # Outputting the raw prediction_val which should be structured for DROP by the HybridIntegrator
            print(f"Prediction: {prediction_val}")
            print(f"Reasoning Path: {final_answer_dict.get('reasoning_path', 'unknown')}")
            print("\nResource Usage:")
            print(f"CPU Delta: {resource_delta['cpu'] * 100:.1f}%")
            print(f"Memory Delta: {resource_delta['memory'] * 100:.1f}%")
            if 'gpu' in resource_delta:
                print(f"GPU Delta: {resource_delta['gpu'] * 100:.1f}%")
            print("-" * 20)

            if data_type.startswith("ground_truth_available") and the_answer_obj is not None:
                # chain_dict might not be applicable for DROP in the same way
                # chain_dict = extract_reasoning_pattern(
                #     query,
                #     final_answer_dict.get('reasoning_path_details', []),
                #     symbolic.rules
                # )
                chain_dict = {}

                eval_metrics = evaluator.evaluate(
                    predictions={query_id: prediction_val},  # Use query_id and the structured prediction
                    ground_truths={query_id: the_answer_obj},  # Pass the structured answer object
                    # supporting_facts is for HotpotQA, remove or adapt for DROP if needed
                    supporting_facts={query_id: supporting_facts} if supporting_facts else None,
                    reasoning_chain=chain_dict  # This might need to be generated differently for DROP
                )
                print("\nEvaluation Metrics (specific to dataset type):")
                # The evaluator will print dataset-specific metrics (EM/F1 for DROP)
                # For generic logging here:
                print(f"Exact Match (if applicable): {eval_metrics.get('average_exact_match', 'N/A'):.2f}")
                print(f"F1 Score (if applicable): {eval_metrics.get('average_f1', 'N/A'):.2f}")
                if dataset_type.lower() == 'hotpotqa':
                    print(f"Similarity Score: {eval_metrics.get('average_semantic_similarity', 0.0):.2f}")
                    print(f"ROUGE-L Score: {eval_metrics.get('average_rougeL', 0.0):.2f}")
                    print(f"BLEU Score: {eval_metrics.get('average_bleu', 0.0):.2f}")
                # Reasoning analysis might be different for DROP
                # if 'reasoning_analysis' in eval_metrics:
                #     print("\nReasoning Analysis:")
                #     print(f"Pattern Type: {eval_metrics['reasoning_analysis'].get('pattern_type', 'unknown')}")
                #     print(f"Chain Length: {eval_metrics['reasoning_analysis'].get('chain_length', 0)}")
                #     print(f"Pattern Confidence: {eval_metrics['reasoning_analysis'].get('pattern_confidence', 0.0):.2f}")

        except KeyError as e:
            print(f"Error: Missing key during processing query {query_id} - {str(e)}")
            logger.exception(f"KeyError for query_id {query_id}")
        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")
            logger.exception(f"Exception for query_id {query_id}")

    # Ablation study might need adaptation for DROP or be run separately
    # For now, we assume it uses the default context (general_context) or its own query list.
    # If you want ablation on DROP, the test_queries there would need to be DROP queries.
    print("\nSkipping ablation study in this focused dataset run. Modify if needed.")
    # ablation_results = run_ablation_study(
    #     rules_path=rules_path,
    #     device=device,
    #     neural=neural,
    #     expander=expander,
    #     aggregator=aggregator,
    #     resource_manager=resource_manager,
    #     system_manager=system_manager,
    #     dimensionality_manager=dimensionality_manager,
    #     context=general_context # This context might not be appropriate for all ablation configs if they need specific datasets
    # )
    # metrics_collector.ablation_results = ablation_results.get('ablation_results', ablation_results) if ablation_results else {}

    # Comparison experiment is generic, may or may not be relevant for a focused DROP run
    print("\nSkipping comparison experiment in this focused dataset run. Modify if needed.")
    # ... (your existing comparison code) ...

    print("\n=== System Performance Summary ===")
    performance_stats = system_manager.get_performance_metrics()  # This will reflect the current run
    print("\nOverall Performance:")
    if performance_stats['total_queries'] > 0:
        print(f"- Total Queries: {performance_stats['total_queries']}")
        print(f"- Average Response Time: {performance_stats['avg_response_time']:.2f}s")
        print(
            f"- Success Rate: {performance_stats['success_rate']:.1f}% (Note: success definition depends on evaluator)")
    else:
        print("- No queries processed successfully to calculate overall performance.")

    print("\nResource Utilization (Final State):")
    final_resources = resource_manager.check_resources()
    print(f"- CPU Usage: {final_resources['cpu'] * 100:.1f}%")
    print(f"- Memory Usage: {final_resources['memory'] * 100:.1f}%")
    if 'gpu' in final_resources:
        print(f"- GPU Usage: {final_resources['gpu'] * 100:.1f}%")

    print("\nReasoning Path Distribution:")
    path_stats = system_manager.get_reasoning_path_stats()
    total_processed_queries = performance_stats['total_queries']
    if total_processed_queries > 0:
        for path, stats in path_stats.items():
            count = stats.get('count', 0)
            percentage = (count / total_processed_queries) * 100 if total_processed_queries > 0 else 0
            print(f"- {path}: {count} queries ({percentage:.1f}%)")
    else:
        print("- No queries processed to show path distribution.")

    print("\n=== Comprehensive Academic Analysis (from MetricsCollector) ===")
    academic_report = metrics_collector.generate_academic_report()  # Reflects data collected during this run

    print("\nPerformance Analysis:")
    if 'performance_metrics' in academic_report:
        perf = academic_report['performance_metrics']
        if 'processing_time' in perf and 'mean' in perf['processing_time']:
            print(f"- Average Processing Time: {perf['processing_time']['mean']:.2f}s")
        if 'processing_time' in perf and 'percentile_95' in perf['processing_time']:
            print(f"- 95th Percentile Time: {perf['processing_time']['percentile_95']:.2f}s")

    # Reasoning analysis might need adjustment based on what metrics_collector.reasoning_metrics contains for DROP
    print("\nReasoning Analysis (Note: may need adaptation for DROP specific metrics):")
    if 'reasoning_analysis' in academic_report:
        ra = academic_report['reasoning_analysis']
        cc = ra.get('chain_characteristics', {})
        print(f"- Average Chain Length: {cc.get('avg_length', 0.0):.2f}")
        print(f"- Average Confidence: {cc.get('avg_confidence', 0.0):.2f}")
        print(f"- Average Inference Depth: {cc.get('avg_inference_depth', 0.0):.2f}")

    print("\nResource Efficiency:")
    if 'efficiency_metrics' in academic_report:
        em = academic_report['efficiency_metrics']
        for resource, metrics_data in em.items():  # Renamed 'metrics' to 'metrics_data' to avoid conflict
            if resource != 'trends':
                efficiency_score_val = metrics_data.get('efficiency_score')
                if efficiency_score_val is None:
                    efficiency_score_val = 0.0
                print(f"- {resource.capitalize()}:")
                print(f"  * Mean Usage: {metrics_data.get('mean_usage', 0.0) * 100:.1f}%")
                print(f"  * Peak Usage: {metrics_data.get('peak_usage', 0.0) * 100:.1f}%")
                print(f"  * Efficiency Score: {efficiency_score_val:.2f}")

    print("\nStatistical Analysis:")
    if 'statistical_analysis' in academic_report:
        sa = academic_report['statistical_analysis']
        print("Significance Tests:")
        for metric, stats_data in sa.items():  # Renamed 'stats' to 'stats_data'
            if isinstance(stats_data, dict) and 'p_value' in stats_data:
                print(f"- {metric}:")
                print(f"  * p-value: {stats_data['p_value']:.3f}")
                print(f"  * effect size: {stats_data.get('effect_size', 0.0):.2f}")

    # This summary might be less relevant for DROP if symbolic.reasoning_metrics isn't populated in a similar way
    # print("\n=== System Performance Summary (Extended from SymbolicReasoner) ===")
    # if hasattr(symbolic, 'reasoning_metrics') and symbolic.reasoning_metrics.get('path_lengths'): # Check if metrics exist
    #     pattern_metrics = get_reasoning_metrics(symbolic.reasoning_metrics)
    #     print(f"- Average Chain Length: {pattern_metrics['path_analysis']['average_length']:.2f}")
    #     print(f"- Average Confidence: {pattern_metrics['confidence_analysis']['mean_confidence']:.2f}")
    #     print("\nPattern Distribution:")
    #     for length, frequency in pattern_metrics['path_analysis']['path_distribution'].items():
    #         print(f"- {length}-hop paths: {frequency * 100:.1f}%")
    # else:
    #     print("- Symbolic reasoner metrics not available or not applicable for this run.")

    print("\n=== End of Run ===")
    print("\n=== Academic Evaluation Results (JSON) ===")
    print(json.dumps(academic_report, indent=2, default=str))  # Added default=str for non-serializable items
    return academic_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HySym-RAG system with output capture')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'drop'],
                        help='Dataset to use for evaluation (hotpotqa or drop)')
    parser.add_argument('--log-dir', default='logs', help='Directory to save log files')
    parser.add_argument('--no-output-capture', action='store_true', help='Disable output capture to file')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to process')  # Reduced default for quicker testing

    args = parser.parse_args()

    # Execution with conditional output capture
    if args.no_output_capture:
        # Execute without output capture
        run_hysym_system(samples=args.samples, dataset_type=args.dataset)
    else:
        # Execute with output capture
        # Ensure log directory for the specific dataset run exists
        dataset_log_dir = os.path.join(args.log_dir, args.dataset)
        os.makedirs(dataset_log_dir, exist_ok=True)

        with capture_output(output_dir=dataset_log_dir) as output_path:  # Save to dataset-specific subfolder
            print(f"Output being saved to: {output_path}")
            run_hysym_system(samples=args.samples, dataset_type=args.dataset)