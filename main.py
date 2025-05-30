# main.py

import os
import sys
import json
import time
import warnings
import argparse
import logging
import urllib3  # type: ignore
import yaml # For execute_ablation_study (will be moved)
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except ImportError:
    print("Transformers library not found. Please install it: pip install transformers")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    print("Sentence-transformers library not found. Please install it: pip install sentence-transformers")
    sys.exit(1)

# HySym-RAG component imports
try:
    from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
    from src.reasoners.networkx_symbolic_reasoner_drop import GraphSymbolicReasonerDrop
    from src.reasoners.neural_retriever import NeuralRetriever
    from src.integrators.hybrid_integrator import HybridIntegrator
    from src.utils.dimension_manager import DimensionalityManager
    from src.utils.rule_extractor import RuleExtractor
    from src.queries.query_logger import QueryLogger
    from src.resources.resource_manager import ResourceManager
    from src.config.config_loader import ConfigLoader
    from src.queries.query_expander import QueryExpander
    from src.utils.evaluation import Evaluation
    from src.system import SystemControlManager, UnifiedResponseAggregator
    from src.utils.metrics_collector import MetricsCollector
    from src.utils.sample_debugger import SampleDebugger
    from src.utils.device_manager import DeviceManager
    from src.utils.progress import tqdm, ProgressManager
    from src.utils.output_capture import capture_output
    from src.ablation_study import setup_and_orchestrate_ablation # MODIFIED IMPORT
    from src.utils.data_loaders import load_hotpotqa, load_drop_dataset # MODIFIED IMPORT

except ImportError as e:
    print(f"Error importing HySym-RAG components: {e}")
    print("Please ensure main.py is run from the project root directory or PYTHONPATH is set correctly.")
    sys.exit(1)

urllib3.disable_warnings()  # type: ignore
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Dataset loading functions (load_hotpotqa, load_drop_dataset) MOVED to src/utils/data_loaders.py


def run_hysym_system(samples: int = 200, dataset_type: str = 'hotpotqa', args: Optional[argparse.Namespace] = None) -> \
Dict[str, Any]:
    """Main execution function for the HySym-RAG system for standard evaluation runs."""
    print(f"\n=== Initializing HySym-RAG System for Dataset: {dataset_type.upper()} ===")

    # Configure library logging levels
    for lib_name in ['transformers', 'sentence_transformers', 'urllib3.connectionpool', 'h5py', 'numexpr', 'spacy']:
        logging.getLogger(lib_name).setLevel(logging.WARNING)

    # Configure project-specific logger levels
    logging.getLogger('src.utils.dimension_manager').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.integrators.hybrid_integrator').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_base').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_drop').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.reasoners.neural_retriever').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.system.system_control_manager').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.system.response_aggregator').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.system.system_logic_helpers').setLevel(
        logging.INFO if not (args and args.debug) else logging.DEBUG)

    # 1. Load configuration
    print("Loading configuration...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "src", "config", "config.yaml")
        if not os.path.exists(config_path):
            logger.error(f"Main configuration file not found at: {config_path}")
            return {"error": f"Main configuration file not found: {config_path}"}
        config = ConfigLoader.load_config(config_path)

        model_name = config.get("model_name")
        if not model_name:
            logger.error("model_name not found in configuration.")
            return {"error": "model_name missing from configuration."}

        project_root = script_dir
        default_data_dir = os.path.join(project_root, "data")
        hotpotqa_path = config.get("hotpotqa_dataset_path",
                                   os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        drop_path = config.get("drop_dataset_path", os.path.join(default_data_dir, "drop_dataset_dev.json"))
        rules_path_default_name = "rules.json"  # Default if specific rule files aren't found
        hotpotqa_rules_path = config.get("hotpotqa_rules_file", os.path.join(default_data_dir, "rules_hotpotqa.json"))
        drop_rules_path = config.get("drop_rules_file",
                                     os.path.join(default_data_dir, "rules_drop.json"))  # Static/fallback
        kb_path_default = os.path.join(default_data_dir, "small_knowledge_base.txt")
        kb_path = config.get("knowledge_base", kb_path_default)
        drop_few_shot_examples_path = config.get("drop_few_shot_examples_path",
                                                 os.path.join(default_data_dir, "drop_few_shot_examples.json"))
        use_drop_few_shots_flag = bool(config.get("use_drop_few_shots", 0))  # Default to False if not specified
        logger.info(f"Configuration - Use DROP Few-Shot Examples: {use_drop_few_shots_flag}")

    except Exception as e:
        logger.exception(f"Failed to load configuration: {e}")
        return {"error": f"Configuration loading failed: {e}"}

    # 2. DeviceManager
    device = DeviceManager.get_device()
    print(f"Using device: {device}")

    # 3. ResourceManager
    print("Initializing Resource Manager...")
    resource_config_path = os.path.join(script_dir, "src", "config", "resource_config.yaml")
    resource_manager = ResourceManager(
        config_path=resource_config_path,
        enable_performance_tracking=True,  # As per standard run
        history_window_size=100  # As per standard run
    )

    # 4. DimensionalityManager
    print("Initializing Dimensionality Manager...")
    alignment_config = config.get('alignment', {})
    target_dim = alignment_config.get('target_dim', 768)  # Default from config.yaml
    dimensionality_manager = DimensionalityManager(target_dim=target_dim, device=device)

    # 5. Select and Ensure Rules File
    current_rules_path = ""
    if dataset_type.lower() == 'drop':
        current_rules_path = drop_rules_path  # Start with static/fallback for DROP
        if not os.path.exists(current_rules_path):
            logger.warning(
                f"Configured DROP rules file {current_rules_path} not found, checking default general rules.")
            current_rules_path = os.path.join(default_data_dir, rules_path_default_name)
    else:  # hotpotqa
        current_rules_path = hotpotqa_rules_path
        if not os.path.exists(current_rules_path):
            logger.warning(
                f"Configured HotpotQA rules file {current_rules_path} not found, checking default general rules.")
            current_rules_path = os.path.join(default_data_dir, rules_path_default_name)

    print(f"Initially selected rules file: {current_rules_path}")
    if not os.path.exists(current_rules_path):
        logger.warning(f"Rules file {current_rules_path} does not exist. Creating empty file.")
        try:
            os.makedirs(os.path.dirname(current_rules_path), exist_ok=True)
            with open(current_rules_path, "w", encoding="utf-8") as f:
                json.dump([], f)  # Create an empty JSON array
        except Exception as e:
            logger.error(f"Failed to create empty rules file at {current_rules_path}: {e}")
            return {"error": f"Failed to ensure rules file exists at {current_rules_path}"}

    # 6. Initialize RuleExtractor
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()  # Assuming default initialization is fine

    # 7. Load Evaluation Dataset and Potentially Extract/Override Rules
    print(f"Loading evaluation dataset: {dataset_type.upper()}...")
    test_queries: List[Dict[str, Any]] = []
    ground_truths: Dict[str, Any] = {}  # Store ground truth answers by query_id

    if dataset_type.lower() == 'drop':
        if not os.path.exists(drop_path):
            logger.error(f"DROP dataset file not found at: {drop_path}")
            return {"error": f"DROP dataset not found at {drop_path}"}
        print(f"Loading DROP dataset from {drop_path}...")
        test_queries = load_drop_dataset(drop_path, max_samples=samples)
        for s in test_queries:  # Populate ground_truths for DROP
            if "query_id" in s and "answer" in s:
                ground_truths[s["query_id"]] = s["answer"]

        # Dynamic rule extraction for DROP
        questions_for_rules = [qa['query'] for qa in test_queries if qa.get('query')]
        passages_for_rules = [qa['context'] for qa in test_queries if qa.get('context')]
        if questions_for_rules and passages_for_rules:
            print("Extracting DROP-specific rules dynamically...")
            try:
                dynamic_rules = rule_extractor.extract_rules_from_drop(
                    drop_json_path=drop_path,  # Path to the full DROP dataset for rule extraction
                    questions=questions_for_rules,  # Questions from the loaded samples
                    passages=passages_for_rules,   # Passages from the loaded samples
                    min_support=config.get('drop_rule_min_support', 5)  # Configurable min_support
                )
                # Save and switch to dynamic rules if extraction was successful
                dynamic_rules_path = os.path.join(default_data_dir, "rules_drop_dynamic.json")
                os.makedirs(os.path.dirname(dynamic_rules_path), exist_ok=True)
                with open(dynamic_rules_path, "w", encoding="utf-8") as f:
                    json.dump(dynamic_rules, f, indent=2)
                print(f"Saved {len(dynamic_rules)} dynamically extracted DROP rules to {dynamic_rules_path}")
                current_rules_path = dynamic_rules_path  # Switch to using dynamic rules
                logger.info(f"Switched to use dynamically extracted rules: {current_rules_path}")
            except Exception as e:
                logger.error(
                    f"Failed to extract/save dynamic DROP rules: {e}. Using pre-configured: {current_rules_path}")
        else:
            logger.warning(
                "Not enough data from loaded samples for dynamic DROP rule extraction. Using pre-configured rules.")

    elif dataset_type.lower() == 'hotpotqa':
        if not os.path.exists(hotpotqa_path):
            logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
            return {"error": f"HotpotQA dataset not found at {hotpotqa_path}"}
        print(f"Loading HotpotQA dataset from {hotpotqa_path}...")
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
        for sample_item in test_queries:  # Populate ground_truths for HotpotQA
             ground_truths[sample_item["query_id"]] = sample_item["answer"]
    else:
        logger.error(f"Unknown dataset_type '{dataset_type}'. Cannot load data.")
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        logger.error("No test queries loaded. Exiting.")
        return {"error": "No test queries loaded"}
    print(f"Loaded {len(test_queries)} test queries. Using rules from: {current_rules_path}")

    # Initialize SampleDebugger
    NUM_RANDOM_SAMPLES_TO_PRINT = 2  # Can be made configurable
    sample_debugger = SampleDebugger(num_samples_to_print=NUM_RANDOM_SAMPLES_TO_PRINT)
    all_query_ids_for_sampling = [q_info.get("query_id") for q_info in test_queries if q_info.get("query_id")]
    if all_query_ids_for_sampling:
        sample_debugger.select_random_query_ids(all_query_ids_for_sampling)

    # 8. GraphSymbolicReasoner (Symbolic Path)
    print(f"Initializing Symbolic Reasoner for {dataset_type.upper()} using rules from: {current_rules_path}...")
    symbolic_reasoner: Optional[Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop]] = None
    embedding_model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')  # Default embedding model
    try:
        if dataset_type.lower() == 'drop':
            symbolic_reasoner = GraphSymbolicReasonerDrop(
                rules_file=current_rules_path,
                match_threshold=config.get('symbolic_match_threshold_drop', 0.1),  # from config.yaml
                max_hops=config.get('symbolic_max_hops_drop', 3),  # from config.yaml
                embedding_model=embedding_model_name,
                device=device,
                dim_manager=dimensionality_manager
            )
        else:  # hotpotqa
            symbolic_reasoner = GraphSymbolicReasoner(
                rules_file=current_rules_path,
                match_threshold=config.get('symbolic_match_threshold_hotpot', 0.1),  # from config.yaml
                max_hops=config.get('symbolic_max_hops_hotpot', 5),  # from config.yaml
                embedding_model=embedding_model_name,
                device=device,
                dim_manager=dimensionality_manager
            )
        count_rules = len(getattr(symbolic_reasoner, 'rules', []))
        print(f"Symbolic Reasoner loaded {count_rules} rules successfully from {current_rules_path}.")
        if count_rules == 0:
            logger.warning(f"Symbolic reasoner initialized, but no rules were loaded from {current_rules_path}.")
    except Exception as e:
        logger.exception(f"Fatal error initializing symbolic reasoner: {e}")
        return {"error": f"Symbolic reasoner initialization failed: {e}"}


    # 9. NeuralRetriever (Neural Path)
    print("Initializing Neural Retriever...")
    try:
        nr_few_shot_path = None
        if dataset_type.lower() == 'drop' and use_drop_few_shots_flag:  # Check the flag from config
            nr_few_shot_path = drop_few_shot_examples_path
            if not os.path.exists(nr_few_shot_path):
                logger.warning(
                    f"Few-shot examples file for DROP ('{nr_few_shot_path}') not found. NR will not use them.")
                nr_few_shot_path = None  # Disable if file not found
        elif dataset_type.lower() == 'drop' and not use_drop_few_shots_flag:
            logger.info("Few-shot examples for DROP are disabled by configuration.")

        neural_retriever = NeuralRetriever(
            model_name,  # Main model_name from config
            use_quantization=config.get('neural_use_quantization', False),  # from config.yaml
            max_context_length=config.get('neural_max_context_length', 2048),  # from config.yaml
            chunk_size=config.get('neural_chunk_size', 512),  # from config.yaml
            overlap=config.get('neural_overlap', 128),  # from config.yaml
            device=device,
            few_shot_examples_path=nr_few_shot_path  # Pass the resolved path
        )
    except Exception as e:
        logger.exception(f"Fatal error initializing neural retriever: {e}")
        return {"error": f"Neural retriever initialization failed: {e}"}

    # 10. Support components
    print("Initializing support components...")
    log_dir_query = args.log_dir if args and hasattr(args, 'log_dir') else 'logs'
    query_logger = QueryLogger(log_dir=os.path.join(log_dir_query, dataset_type))
    complexity_config_yaml_path = os.path.join(script_dir, "src", "config", "complexity_rules.yaml")
    query_expander = QueryExpander(
        complexity_config=complexity_config_yaml_path if os.path.exists(complexity_config_yaml_path) else None)

    # 11. Initialize Evaluation utility
    evaluator = Evaluation(dataset_type=dataset_type)  # Pass dataset_type

    # 12. Create HybridIntegrator
    print("Creating Hybrid Integrator...")
    try:
        hybrid_integrator = HybridIntegrator(
            symbolic_reasoner=symbolic_reasoner,
            neural_retriever=neural_retriever,
            query_expander=query_expander,  # Pass the initialized query_expander
            dim_manager=dimensionality_manager,  # Pass the dim_manager
            dataset_type=dataset_type  # Pass dataset_type
        )
    except Exception as e:
        logger.exception(f"Fatal error initializing Hybrid Integrator: {e}")
        return {"error": f"Hybrid Integrator initialization failed: {e}"}

    # 13. SystemControlManager
    print("Initializing System Control Manager...")
    response_aggregator = UnifiedResponseAggregator(include_explanations=True)  # Standard run includes explanations
    log_dir_metrics = args.log_dir if args and hasattr(args, 'log_dir') else 'logs'
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=os.path.join(log_dir_metrics, dataset_type, "metrics_collection")  # Standard run metrics
    )
    system_manager = SystemControlManager(
        hybrid_integrator=hybrid_integrator,
        resource_manager=resource_manager,
        aggregator=response_aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=config.get('error_retry_limit', 2),  # from config.yaml
        max_query_time=config.get('max_query_time', 30.0)  # from config.yaml
    )

    # 14. Optional general knowledge base
    general_context = ""
    if kb_path and os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as kb_file:
                general_context = kb_file.read()
            if general_context:
                print(f"Loaded general knowledge base from {kb_path} ({len(general_context)} chars)")
        except Exception as e:
            logger.warning(f"Could not load general knowledge base from {kb_path}: {e}")
    elif kb_path:  # Path specified but doesn't exist
        logger.warning(f"General knowledge base file specified but not found: {kb_path}")

    # Main Query Processing Loop
    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    results_list: List[Dict[str, Any]] = []  # To store final responses from SCM

    query_iterator = tqdm(test_queries, desc="Processing Queries", unit="query",
                          disable=not ProgressManager.SHOW_PROGRESS)

    for q_info in query_iterator:
        query_id_val = q_info.get("query_id")
        query_text_val = q_info.get("query")
        gt_answer_obj = q_info.get("answer")  # Ground truth answer for this query
        local_context_val = q_info.get("context", general_context)  # Use specific context or general KB
        forced_path_val = q_info.get("forced_path")  # For debugging specific paths
        query_type_val = q_info.get("type", "")  # e.g., "ground_truth_available_hotpotqa"
        supporting_facts_val = q_info.get("supporting_facts")  # For HotpotQA

        if not query_id_val or not query_text_val:
            logger.warning(f"Skipping query due to missing ID or text: {q_info}")
            continue

        if ProgressManager.SHOW_PROGRESS:
            query_iterator.set_description(f"Processing QID: {str(query_id_val)[:8]}")

        logger.info(f"Processing Query ID: {query_id_val}, Query: '{query_text_val[:100]}...'")
        final_response_obj: Dict[str, Any] = {}  # This will be the structured response from SCM
        actual_reasoning_path_val: str = 'error_before_processing'

        try:
            complexity_score_val = query_expander.get_query_complexity(query_text_val)
            logger.info(f"Query '{query_id_val}' Complexity Score: {complexity_score_val:.4f}")

            # Process query using SystemControlManager
            final_response_obj, actual_reasoning_path_val = system_manager.process_query_with_fallback(
                query=query_text_val,
                context=local_context_val,
                query_id=query_id_val,
                forced_path=forced_path_val,
                query_complexity=complexity_score_val,
                supporting_facts=supporting_facts_val,  # Pass supporting facts here
                dataset_type=dataset_type  # Pass dataset_type here
            )

            prediction_val = final_response_obj.get('result')  # This is the actual answer payload
            results_list.append(final_response_obj)  # Store the SCM's full response object

            # Print results
            print("\n" + "-" * 10 + f" Results for QID: {query_id_val} " + "-" * 10)
            if dataset_type == 'drop' and isinstance(prediction_val, dict):
                print(f"  Prediction (DROP): {prediction_val}")
            else:
                print(f"  Prediction (Text): {str(prediction_val)[:300]}...")
            print(f"  Reasoning Path: {actual_reasoning_path_val}")
            print(f"  Overall Processing Time: {final_response_obj.get('processing_time', 0.0):.3f}s")
            print(f"  Status: {final_response_obj.get('status', 'unknown')}")
            res_delta = final_response_obj.get('resource_usage', {})
            if res_delta:
                delta_str_parts = [f"{k.capitalize()}: {v * 100:+.1f}%" for k, v in res_delta.items()]
                if delta_str_parts:
                    print(f"  Resource Delta: {', '.join(delta_str_parts)}")
            explanation_text = final_response_obj.get('explanation')
            if explanation_text:
                print(f"  Explanation: {explanation_text}")
            print("-" * (30 + len(str(query_id_val))))

            # Evaluation
            if query_type_val.startswith("ground_truth_available") and \
                    gt_answer_obj is not None and \
                    final_response_obj.get('status') == 'success':  # Only evaluate if SCM reported success
                if query_id_val in ground_truths:  # Redundant check, gt_answer_obj should suffice
                    gt_answer_for_eval = ground_truths[query_id_val]
                    eval_metrics_dict = evaluator.evaluate(
                        predictions={query_id_val: prediction_val},  # Pass the actual answer payload
                        ground_truths={query_id_val: gt_answer_for_eval},  # Pass the GT object
                    )
                    print("  Evaluation Metrics:")
                    print(f"    Exact Match: {eval_metrics_dict.get('average_exact_match', 0.0):.3f}")
                    print(f"    F1 Score: {eval_metrics_dict.get('average_f1', 0.0):.3f}")
                    if dataset_type != 'drop':  # ROUGE-L only for text-based
                        print(f"    ROUGE-L: {eval_metrics_dict.get('average_rougeL', 0.0):.3f}")
                    final_response_obj['evaluation_metrics'] = eval_metrics_dict  # Store in results

                    # Debug selected samples
                    sample_debugger.print_debug_if_selected(
                        query_id=query_id_val,
                        query_text=query_text_val,
                        ground_truth_answer=gt_answer_for_eval,
                        system_prediction_value=prediction_val,  # Pass the payload
                        actual_reasoning_path=actual_reasoning_path_val,
                        eval_metrics=eval_metrics_dict,
                        dataset_type=dataset_type
                    )
                else:
                    logger.warning(
                        f"Ground truth not found for query_id {query_id_val} during evaluation phase (this shouldn't happen if populated correctly).")
            elif final_response_obj.get('status') != 'success':
                logger.warning(
                    f"Skipping evaluation for QID {query_id_val} due to SCM status: '{final_response_obj.get('status')}'.")


        except Exception as e:
            print(f"\n--- ERROR in main processing loop for query {query_id_val} ---")
            logger.exception(f"Unhandled exception for query_id {query_id_val} in main loop: {e}")
            results_list.append({
                'query_id': query_id_val,
                'query': query_text_val,
                'status': 'critical_error_in_main_loop',
                'error': str(e),
                'reasoning_path': actual_reasoning_path_val  # Path taken before crash if available
            })
            print(f"Error: {e}")
            print("-" * 30)

    # After processing all queries
    print("\n" + "=" * 20 + " System Performance Summary (from SystemControlManager) " + "=" * 20)
    try:
        scm_perf_summary = system_manager.get_performance_metrics()
        if scm_perf_summary.get('total_queries', 0) > 0:
            print(f"- Total Queries Processed by SCM: {scm_perf_summary['total_queries']}")
            print(f"- SCM Successful Queries: {scm_perf_summary['successful_queries']}")
            print(f"- SCM Success Rate: {scm_perf_summary.get('success_rate', 0.0):.1f}%")
            print(f"- SCM Error Count (retries/internal): {scm_perf_summary.get('error_count', 0)}")
            print(
                f"- SCM Avg Successful Response Time: {scm_perf_summary.get('avg_successful_response_time_sec', 0.0):.3f}s")
        else:
            print("- No queries were processed by SystemControlManager according to its metrics.")

        print("\nSCM Reasoning Path Distribution (Execution Counts):")
        scm_path_stats = system_manager.get_reasoning_path_stats()
        if scm_path_stats:
            for path, path_data in scm_path_stats.items():
                count = path_data.get('execution_count', 0)
                success_count = path_data.get('success_count', 0)
                avg_time = path_data.get('avg_time_sec', 0.0)
                perc_total = path_data.get('percentage_of_total_queries', 0.0)
                print(
                    f"- Path '{path}': Executed={count} ({perc_total:.1f}%), Succeeded={success_count}, AvgTime={avg_time:.3f}s")
        else:
            print("- No path execution statistics available from SystemControlManager.")

        print("\nResource Utilization (Final State via ResourceManager):")
        final_res_usage = resource_manager.check_resources()  # Get current usage
        print(f"- CPU Usage: {(final_res_usage.get('cpu', 0.0) or 0.0) * 100:.1f}%")
        print(f"- Memory Usage: {(final_res_usage.get('memory', 0.0) or 0.0) * 100:.1f}%")
        if 'gpu' in final_res_usage and final_res_usage.get('gpu') is not None:
            print(f"- GPU Usage: {(final_res_usage.get('gpu', 0.0) or 0.0) * 100:.1f}%")
    except Exception as report_err:
        logger.error(f"Error generating SystemControlManager performance summary: {report_err}", exc_info=True)

    print("\n" + "=" * 20 + " Comprehensive Academic Analysis (from MetricsCollector) " + "=" * 20)
    # Ensure log_dir for academic report is correctly formed
    current_dataset_log_dir = os.path.join(args.log_dir if args and hasattr(args, 'log_dir') else 'logs', dataset_type)
    os.makedirs(current_dataset_log_dir, exist_ok=True)  # Ensure it exists

    try:
        academic_report = metrics_collector.generate_academic_report()  # Generate report from SCM's collector
        if not academic_report or academic_report.get("experiment_summary", {}).get("total_queries", 0) == 0:
            print("No data collected by MetricsCollector to generate an academic report.")
            return {"status": "No data for academic report"}  # Return status if no data

        # Print Performance Metrics from Academic Report
        perf_metrics = academic_report.get('performance_metrics', {})
        if 'processing_time' in perf_metrics and isinstance(perf_metrics['processing_time'], dict) and perf_metrics[
            'processing_time'].get('count', 0) > 0:
            proc_time_stats = perf_metrics['processing_time']
            print("\nPerformance Metrics (from Metrics Collector):")
            print(
                f"- Avg Processing Time: {proc_time_stats.get('mean', 0.0):.3f}s (Std: {proc_time_stats.get('std', 0.0):.3f}, Median: {proc_time_stats.get('median', 0.0):.3f})")
            if 'percentile_95' in proc_time_stats:
                print(f"- 95th Percentile Time: {proc_time_stats.get('percentile_95', 0.0):.3f}s")
        else:
            print("- No processing time metrics collected by MetricsCollector or count is zero.")

        # Print Reasoning Analysis
        print("\nReasoning Analysis (from Metrics Collector):")
        ra = academic_report.get('reasoning_analysis', {})
        cc = ra.get('chain_characteristics', {})
        print(f"- Avg Chain Length: {cc.get('avg_length', 0.0):.2f}")
        print(f"- Avg Confidence (Fusion): {cc.get('avg_confidence', 0.0):.3f}")
        print(f"- Path Distribution (MetricsCollector): {ra.get('path_distribution', {})}")

        # Print Resource Efficiency
        print("\nResource Efficiency (from Metrics Collector):")
        em_metrics = academic_report.get('efficiency_metrics', {})
        for resource, metrics_val in em_metrics.items():
            if resource != 'trends' and isinstance(metrics_val, dict):
                mean_u = metrics_val.get('mean_usage', 0.0) * 100
                peak_u = metrics_val.get('peak_usage', 0.0) * 100
                score = metrics_val.get('efficiency_score', None)
                # --- Corrected Line Below ---
                # Previous problematic line:
                # print(
                #     f"- {resource.capitalize()}: Mean Delta {mean_u:+.1f}%, Peak Delta {peak_u:.1f}%, Score {score:.2f if score is not None else 'N/A'}")

                # Corrected approach:
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else 'N/A'
                print(
                    f"- {resource.capitalize()}: Mean Delta {mean_u:+.1f}%, Peak Delta {peak_u:.1f}%, Score {score_str}")
                # --- End of Correction ---
        if 'trends' in em_metrics:
            print(f"- Efficiency Trends: {em_metrics.get('trends', {})}")

        # Print Statistical Analysis
        print("\nStatistical Analysis (from Metrics Collector):")
        sa_metrics = academic_report.get('statistical_analysis', {})
        if 'tests' in sa_metrics:
            for metric, stats_val in sa_metrics.get('tests', {}).items():
                if isinstance(stats_val, dict) and 'p_value' in stats_val:
                    print(
                        f"- {metric}: p-value={stats_val['p_value']:.3g}, effect_size={stats_val.get('effect_size', 0.0):.2f}")
        print(f"- Correlations: {sa_metrics.get('correlations', {})}")
        print(f"- Regression: {sa_metrics.get('regression_analysis', {})}")

        # Save Academic Report
        report_file_name = f"academic_report_{dataset_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report_file_path = os.path.join(current_dataset_log_dir, report_file_name)  # Save in dataset-specific log dir
        try:
            with open(report_file_path, 'w', encoding='utf-8') as rf:
                json.dump(academic_report, rf, indent=2, default=str)  # Use default=str for non-serializable
            print(f"\nAcademic report saved to: {report_file_path}")
        except Exception as save_err:
            print(f"Warning: Could not save academic report to {report_file_path}: {save_err}")

        print("\n" + "=" * 20 + " End of Run " + "=" * 20)
        return academic_report  # Return the generated report

    except Exception as acad_err:
        logger.error(f"Error generating academic report: {acad_err}", exc_info=True)
        return {"error": f"Academic report generation failed: {acad_err}"}


def execute_ablation_study(args: argparse.Namespace):
    """ Wrapper function in main.py to call the main ablation orchestration. """
    print(f"\n--- main.py: execute_ablation_study for Dataset: {args.dataset.upper()} ---")
    # The actual setup, component initialization, and result processing
    # is now handled by setup_and_orchestrate_ablation in src.ablation_study
    try:
        # Call the refactored function from ablation_study.py
        # It will handle loading configs, datasets, initializing components,
        # running ablations, and printing the summary.
        setup_and_orchestrate_ablation(args)
    except Exception as e:
        logger.error(f"Call to setup_and_orchestrate_ablation failed: {e}", exc_info=True)
        # Optionally, return an error status or re-raise
        # For now, just logging, as the function itself should handle internal errors gracefully.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run HySym-RAG system with output capture and optional ablation study.')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'drop'],
                        help='Dataset to use for evaluation (hotpotqa or drop)')
    parser.add_argument('--log-dir', default='logs', help='Directory to save log files')
    parser.add_argument('--no-output-capture', action='store_true', help='Disable output capture to file')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to process (for standard run or ablation)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging for main and src components')
    parser.add_argument('--show-progress', action='store_true', help='Show tqdm progress bars')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run the ablation study instead of a standard evaluation.')
    parser.add_argument('--ablation-config', type=str, default='src/config/ablation_config.yaml',
                        help='Path to the YAML file defining ablation configurations.')
    parser.add_argument('--ablation-name', type=str, default=None,
                        help='Name of a specific ablation configuration to run (e.g., "1. Baseline Hybrid (Dynamic Rules, Few-Shots)"). If not specified, runs all.')

    parsed_args = parser.parse_args()

    if parsed_args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG)  # Set all loggers in 'src' package to DEBUG
        print("--- DEBUG Logging Enabled for main.py and 'src' package ---")
    else:
        logger.setLevel(logging.INFO)
        # Keep specific src loggers at INFO unless debug is on, or manage them within functions

    ProgressManager.SHOW_PROGRESS = parsed_args.show_progress
    # Ensure the base log directory for the specific dataset exists
    dataset_log_dir = os.path.join(parsed_args.log_dir, parsed_args.dataset)
    os.makedirs(dataset_log_dir, exist_ok=True)

    if parsed_args.run_ablation:
        print(f"--- Starting Ablation Study for {parsed_args.dataset.upper()} ---")
        if parsed_args.no_output_capture:
            execute_ablation_study(parsed_args) # Calls the new simplified execute_ablation_study
        else:
            # Specific log directory for this ablation run's stdout/stderr
            ablation_output_dir = os.path.join(dataset_log_dir, "ablation_run_logs")
            os.makedirs(ablation_output_dir, exist_ok=True)
            try:
                with capture_output(output_dir=ablation_output_dir) as output_path:
                    print(f"Ablation study output is being captured to: {output_path}")
                    execute_ablation_study(parsed_args) # Calls the new simplified execute_ablation_study
            except Exception as e:
                print(f"ERROR: Failed to set up output capture or run ablation study: {e}", file=sys.stderr)
                logger.error(f"Failed to set up output capture or run ablation study: {e}", exc_info=True)
    else:  # Standard run
        if parsed_args.no_output_capture:
            print("--- Running standard HySym-RAG evaluation without output capture. ---")
            run_hysym_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
        else:
            # Standard run logs go into dataset_log_dir directly
            try:
                with capture_output(output_dir=dataset_log_dir) as output_path:
                    print(f"Output from this run is being captured to: {output_path}")
                    run_hysym_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
            except Exception as e:
                print(f"ERROR: Failed to set up output capture or run system: {e}", file=sys.stderr)
                logger.error(f"Failed to set up output capture or run system: {e}", exc_info=True)