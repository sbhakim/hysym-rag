# main.py

import os
import sys
import json
import time
import warnings
import argparse
import logging
import urllib3  # type: ignore
import yaml     # Added for loading ablation_config.yaml
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Union
import torch

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
    # FeedbackManager might be part of a different workflow, not directly in ablation baseline setup
    # from src.feedback.feedback_manager import FeedbackManager
    from src.config.config_loader import ConfigLoader
    from src.queries.query_expander import QueryExpander
    from src.utils.evaluation import Evaluation
    from src.system import SystemControlManager, UnifiedResponseAggregator
    from src.utils.metrics_collector import MetricsCollector
    from src.utils.sample_debugger import SampleDebugger
    from src.utils.device_manager import DeviceManager
    from src.utils.progress import tqdm, ProgressManager
    from src.utils.output_capture import capture_output
    from src.ablation_study import run_ablation_study # Added import for ablation study
except ImportError as e:
    print(f"Error importing HySym-RAG components: {e}")
    print("Please ensure main.py is run from the project root directory or PYTHONPATH is set correctly.")
    sys.exit(1)

urllib3.disable_warnings()  # type: ignore
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False # Default, can be overridden by args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Default to INFO, can be overridden by --debug for main, or specific debug for ablation
# logger.setLevel(logging.DEBUG) # Set by --debug arg if present


def load_hotpotqa(hotpotqa_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the HotpotQA dataset.
    Each sample includes a query, ground-truth answer,
    combined context, and a 'type' = 'ground_truth_available_hotpotqa'.
    """
    if not os.path.exists(hotpotqa_path):
        logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
        return []
    dataset: List[Dict[str, Any]] = []
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


def load_drop_dataset(drop_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the DROP dataset.
    """
    if not os.path.exists(drop_path):
        logger.error(f"DROP dataset file not found at: {drop_path}")
        return []
    dataset: List[Dict[str, Any]] = []
    try:
        with open(drop_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse DROP JSON from {drop_path}: {e}")
        return []

    total_loaded_qas = 0
    for passage_id, passage_content in data.items():
        if not isinstance(passage_content,
                          dict) or 'passage' not in passage_content or 'qa_pairs' not in passage_content:
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
            query_id = qa_pair.get("query_id", f"{passage_id}-{qa_pair_idx}") # Ensure unique QID

            dataset.append({
                "query_id": query_id,
                "query": question,
                "context": passage_text,
                "answer": answer_obj, # This is the structured answer for DROP
                "type": "ground_truth_available_drop"
            })
            total_loaded_qas += 1
            if max_samples and total_loaded_qas >= max_samples:
                logger.info(f"Loaded {total_loaded_qas} DROP samples (max requested: {max_samples}).")
                return dataset

        if max_samples and total_loaded_qas >= max_samples: # Check after outer loop as well
            break

    logger.info(f"Finished loading DROP. Total samples: {len(dataset)}.")
    return dataset


def run_hysym_system(samples: int = 200, dataset_type: str = 'hotpotqa', args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Main execution function for the HySym-RAG system for standard evaluation runs."""
    print(f"\n=== Initializing HySym-RAG System for Dataset: {dataset_type.upper()} ===")

    # Configure library logging levels
    for lib_name in ['transformers', 'sentence_transformers', 'urllib3.connectionpool', 'h5py', 'numexpr', 'spacy']:
        logging.getLogger(lib_name).setLevel(logging.WARNING)

    # Configure project-specific logger levels (can be overridden by --debug)
    # (These are already top-level in the original code, keeping for consistency)
    logging.getLogger('src.utils.dimension_manager').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.integrators.hybrid_integrator').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_base').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_drop').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.reasoners.neural_retriever').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.system.system_control_manager').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.system.response_aggregator').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)
    logging.getLogger('src.system.system_logic_helpers').setLevel(logging.INFO if not (args and args.debug) else logging.DEBUG)


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
        hotpotqa_path = config.get("hotpotqa_dataset_path", os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        drop_path = config.get("drop_dataset_path", os.path.join(default_data_dir, "drop_dataset_dev.json"))
        rules_path_default_name = "rules.json" # Fallback name if specific paths don't exist
        hotpotqa_rules_path = config.get("hotpotqa_rules_file", os.path.join(default_data_dir, "rules_hotpotqa.json"))
        drop_rules_path = config.get("drop_rules_file", os.path.join(default_data_dir, "rules_drop.json"))
        kb_path_default = os.path.join(default_data_dir, "small_knowledge_base.txt")
        kb_path = config.get("knowledge_base", kb_path_default)
        drop_few_shot_examples_path = config.get("drop_few_shot_examples_path", os.path.join(default_data_dir, "drop_few_shot_examples.json"))
        use_drop_few_shots_flag = bool(config.get("use_drop_few_shots", 0))
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
        enable_performance_tracking=True,
        history_window_size=100
    )

    # 4. DimensionalityManager
    print("Initializing Dimensionality Manager...")
    alignment_config = config.get('alignment', {})
    target_dim = alignment_config.get('target_dim', 768)
    dimensionality_manager = DimensionalityManager(target_dim=target_dim, device=device)

    # 5. Select and Ensure Rules File (This logic determines the 'rules_path' for the run)
    # For DROP, this might be overridden by dynamically generated rules later.
    current_rules_path = ""
    if dataset_type.lower() == 'drop':
        current_rules_path = drop_rules_path
        if not os.path.exists(current_rules_path):
            logger.warning(f"Configured DROP rules file {current_rules_path} not found, checking default.")
            current_rules_path = os.path.join(default_data_dir, rules_path_default_name)
    else:  # hotpotqa or other
        current_rules_path = hotpotqa_rules_path
        if not os.path.exists(current_rules_path):
            logger.warning(f"Configured HotpotQA rules file {current_rules_path} not found, checking default.")
            current_rules_path = os.path.join(default_data_dir, rules_path_default_name)

    print(f"Initially selected rules file: {current_rules_path}")
    if not os.path.exists(current_rules_path):
        logger.warning(f"Rules file {current_rules_path} does not exist. Creating empty file.")
        try:
            os.makedirs(os.path.dirname(current_rules_path), exist_ok=True)
            with open(current_rules_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"Failed to create empty rules file at {current_rules_path}: {e}")
            return {"error": f"Failed to ensure rules file exists at {current_rules_path}"}

    # 6. Initialize RuleExtractor
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()

    # 7. Load Evaluation Dataset and Potentially Extract/Override Rules
    print(f"Loading evaluation dataset: {dataset_type.upper()}...")
    test_queries: List[Dict[str, Any]] = []
    ground_truths: Dict[str, Any] = {} # query_id -> answer object

    if dataset_type.lower() == 'drop':
        if not os.path.exists(drop_path):
            logger.error(f"DROP dataset file not found at: {drop_path}")
            return {"error": f"DROP dataset not found at {drop_path}"}
        print(f"Loading DROP dataset from {drop_path}...")
        test_queries = load_drop_dataset(drop_path, max_samples=samples)
        for s in test_queries: # Populate ground_truths for DROP
             if "query_id" in s and "answer" in s:
                ground_truths[s["query_id"]] = s["answer"]

        # Dynamic rule extraction for DROP
        questions_for_rules = [qa['query'] for qa in test_queries if qa.get('query')]
        passages_for_rules = [qa['context'] for qa in test_queries if qa.get('context')]
        if questions_for_rules and passages_for_rules:
            print("Extracting DROP-specific rules dynamically...")
            try:
                dynamic_rules = rule_extractor.extract_rules_from_drop(
                    drop_json_path=drop_path, # Pass full dataset for comprehensive rule extraction
                    questions=questions_for_rules, # Can be a subset for speed if needed
                    passages=passages_for_rules,
                    min_support=config.get('drop_rule_min_support', 5)
                )
                dynamic_rules_path = os.path.join(default_data_dir, "rules_drop_dynamic.json")
                os.makedirs(os.path.dirname(dynamic_rules_path), exist_ok=True)
                with open(dynamic_rules_path, "w", encoding="utf-8") as f:
                    json.dump(dynamic_rules, f, indent=2)
                print(f"Saved {len(dynamic_rules)} dynamically extracted DROP rules to {dynamic_rules_path}")
                current_rules_path = dynamic_rules_path # Override to use dynamic rules
                logger.info(f"Switched to use dynamically extracted rules: {current_rules_path}")
            except Exception as e:
                logger.error(f"Failed to extract/save dynamic DROP rules: {e}. Using pre-configured: {current_rules_path}")
        else:
            logger.warning("Not enough data from loaded samples for dynamic DROP rule extraction. Using pre-configured rules.")

    elif dataset_type.lower() == 'hotpotqa':
        if not os.path.exists(hotpotqa_path):
            logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
            return {"error": f"HotpotQA dataset not found at {hotpotqa_path}"}
        print(f"Loading HotpotQA dataset from {hotpotqa_path}...")
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
        for sample_item in test_queries: # Populate ground_truths for HotpotQA
            ground_truths[sample_item["query_id"]] = sample_item["answer"]
        # current_rules_path is already set to hotpotqa_rules_path
    else:
        logger.error(f"Unknown dataset_type '{dataset_type}'. Cannot load data.")
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        logger.error("No test queries loaded. Exiting.")
        return {"error": "No test queries loaded"}
    print(f"Loaded {len(test_queries)} test queries. Using rules from: {current_rules_path}")


    # Initialize SampleDebugger
    NUM_RANDOM_SAMPLES_TO_PRINT = 2
    sample_debugger = SampleDebugger(num_samples_to_print=NUM_RANDOM_SAMPLES_TO_PRINT)
    all_query_ids_for_sampling = [q_info.get("query_id") for q_info in test_queries if q_info.get("query_id")]
    if all_query_ids_for_sampling:
        sample_debugger.select_random_query_ids(all_query_ids_for_sampling)

    # 8. GraphSymbolicReasoner (Symbolic Path)
    print(f"Initializing Symbolic Reasoner for {dataset_type.upper()} using rules from: {current_rules_path}...")
    symbolic_reasoner: Optional[Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop]] = None
    embedding_model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
    try:
        if dataset_type.lower() == 'drop':
            symbolic_reasoner = GraphSymbolicReasonerDrop(
                rules_file=current_rules_path,
                match_threshold=config.get('symbolic_match_threshold_drop', 0.1),
                max_hops=config.get('symbolic_max_hops_drop', 3),
                embedding_model=embedding_model_name, device=device, dim_manager=dimensionality_manager
            )
        else: # hotpotqa
            symbolic_reasoner = GraphSymbolicReasoner(
                rules_file=current_rules_path,
                match_threshold=config.get('symbolic_match_threshold_hotpot', 0.1),
                max_hops=config.get('symbolic_max_hops_hotpot', 5),
                embedding_model=embedding_model_name, device=device, dim_manager=dimensionality_manager
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
        if dataset_type.lower() == 'drop' and use_drop_few_shots_flag:
            nr_few_shot_path = drop_few_shot_examples_path
            if not os.path.exists(nr_few_shot_path):
                logger.warning(f"Few-shot examples file for DROP ('{nr_few_shot_path}') not found. NR will not use them.")
                nr_few_shot_path = None
        elif dataset_type.lower() == 'drop' and not use_drop_few_shots_flag:
             logger.info("Few-shot examples for DROP are disabled by configuration.")

        neural_retriever = NeuralRetriever(
            model_name,
            use_quantization=config.get('neural_use_quantization', False),
            max_context_length=config.get('neural_max_context_length', 2048),
            chunk_size=config.get('neural_chunk_size', 512),
            overlap=config.get('neural_overlap', 128),
            device=device,
            few_shot_examples_path=nr_few_shot_path
        )
    except Exception as e:
        logger.exception(f"Fatal error initializing neural retriever: {e}")
        return {"error": f"Neural retriever initialization failed: {e}"}

    # 10. Support components
    print("Initializing support components...")
    log_dir_query = args.log_dir if args and hasattr(args, 'log_dir') else 'logs'
    query_logger = QueryLogger(log_dir=os.path.join(log_dir_query, dataset_type))
    complexity_config_yaml_path = os.path.join(script_dir, "src", "config", "complexity_rules.yaml")
    query_expander = QueryExpander(complexity_config=complexity_config_yaml_path if os.path.exists(complexity_config_yaml_path) else None)

    # 11. Initialize Evaluation utility
    evaluator = Evaluation(dataset_type=dataset_type)

    # 12. Create HybridIntegrator
    print("Creating Hybrid Integrator...")
    try:
        hybrid_integrator = HybridIntegrator(
            symbolic_reasoner=symbolic_reasoner,
            neural_retriever=neural_retriever,
            query_expander=query_expander,
            dim_manager=dimensionality_manager,
            dataset_type=dataset_type # Pass dataset_type
        )
    except Exception as e:
        logger.exception(f"Fatal error initializing Hybrid Integrator: {e}")
        return {"error": f"Hybrid Integrator initialization failed: {e}"}

    # 13. SystemControlManager
    print("Initializing System Control Manager...")
    response_aggregator = UnifiedResponseAggregator(include_explanations=True)
    log_dir_metrics = args.log_dir if args and hasattr(args, 'log_dir') else 'logs'
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=os.path.join(log_dir_metrics, dataset_type, "metrics_collection")
    )
    system_manager = SystemControlManager(
        hybrid_integrator=hybrid_integrator,
        resource_manager=resource_manager,
        aggregator=response_aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=config.get('error_retry_limit', 2),
        max_query_time=config.get('max_query_time', 30.0)
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
    elif kb_path:
        logger.warning(f"General knowledge base file specified but not found: {kb_path}")

    # Main Query Processing Loop (Copied and adapted from original)
    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    results_list: List[Dict[str, Any]] = []

    query_iterator = tqdm(test_queries, desc="Processing Queries", unit="query",
                          disable=not ProgressManager.SHOW_PROGRESS)

    for q_info in query_iterator:
        query_id_val = q_info.get("query_id")
        query_text_val = q_info.get("query")
        gt_answer_obj = q_info.get("answer") # Ground truth object
        local_context_val = q_info.get("context", general_context)
        forced_path_val = q_info.get("forced_path")
        query_type_val = q_info.get("type", "") # e.g., 'ground_truth_available_drop'
        supporting_facts_val = q_info.get("supporting_facts")

        if not query_id_val or not query_text_val:
            logger.warning(f"Skipping query due to missing ID or text: {q_info}")
            continue

        if ProgressManager.SHOW_PROGRESS:
            query_iterator.set_description(f"Processing QID: {str(query_id_val)[:8]}")

        logger.info(f"Processing Query ID: {query_id_val}, Query: '{query_text_val[:100]}...'")
        final_response_obj: Dict[str, Any] = {}
        actual_reasoning_path_val: str = 'error_before_processing'

        try:
            complexity_score_val = query_expander.get_query_complexity(query_text_val)
            logger.info(f"Query '{query_id_val}' Complexity Score: {complexity_score_val:.4f}")

            # process_query_with_fallback returns a tuple: (formatted_response_dict, reasoning_path_string)
            final_response_obj, actual_reasoning_path_val = system_manager.process_query_with_fallback(
                query=query_text_val,
                context=local_context_val,
                query_id=query_id_val,
                forced_path=forced_path_val,
                query_complexity=complexity_score_val,
                supporting_facts=supporting_facts_val,
                dataset_type=dataset_type
            )

            # 'prediction_val' is the actual content of the 'result' field from the SCM's formatted response
            prediction_val = final_response_obj.get('result')
            results_list.append(final_response_obj) # Store the entire SCM output

            print("\n" + "-" * 10 + f" Results for QID: {query_id_val} " + "-" * 10)
            if dataset_type == 'drop' and isinstance(prediction_val, dict):
                print(f"  Prediction (DROP): {prediction_val}")
            else:
                print(f"  Prediction (Text): {str(prediction_val)[:300]}...")
            print(f"  Reasoning Path: {actual_reasoning_path_val}")
            print(f"  Overall Processing Time: {final_response_obj.get('processing_time', 0.0):.3f}s")
            print(f"  Status: {final_response_obj.get('status', 'unknown')}")
            # ... (rest of the printing logic for resource delta, explanation from original)
            res_delta = final_response_obj.get('resource_usage', {})
            if res_delta:
                delta_str_parts = [f"{k.capitalize()}: {v * 100:+.1f}%" for k, v in res_delta.items()]
                if delta_str_parts: print(f"  Resource Delta: {', '.join(delta_str_parts)}")
            explanation_text = final_response_obj.get('explanation')
            if explanation_text: print(f"  Explanation: {explanation_text}")
            print("-" * (30 + len(str(query_id_val))))


            # Evaluation
            if query_type_val.startswith("ground_truth_available") and \
               gt_answer_obj is not None and \
               final_response_obj.get('status') == 'success':
                if query_id_val in ground_truths: # Ensure GT was loaded
                    gt_answer_for_eval = ground_truths[query_id_val]
                    eval_metrics_dict = evaluator.evaluate(
                        predictions={query_id_val: prediction_val}, # prediction_val is the 'result' field
                        ground_truths={query_id_val: gt_answer_for_eval},
                        # supporting_facts and reasoning_chain can be added if relevant for specific metrics
                    )
                    print("  Evaluation Metrics:")
                    print(f"    Exact Match: {eval_metrics_dict.get('average_exact_match', 0.0):.3f}")
                    print(f"    F1 Score: {eval_metrics_dict.get('average_f1', 0.0):.3f}")
                    if dataset_type != 'drop':
                        print(f"    ROUGE-L: {eval_metrics_dict.get('average_rougeL', 0.0):.3f}")
                        # Add other text metrics if needed
                    final_response_obj['evaluation_metrics'] = eval_metrics_dict # Attach to main result log

                    sample_debugger.print_debug_if_selected(
                        query_id=query_id_val, query_text=query_text_val,
                        ground_truth_answer=gt_answer_for_eval, system_prediction_value=prediction_val,
                        actual_reasoning_path=actual_reasoning_path_val, eval_metrics=eval_metrics_dict,
                        dataset_type=dataset_type
                    )
                else:
                    logger.warning(f"Ground truth not found for query_id {query_id_val} during evaluation phase.")
            elif final_response_obj.get('status') != 'success':
                logger.warning(f"Skipping evaluation for QID {query_id_val} due to status: '{final_response_obj.get('status')}'.")

        except Exception as e:
            # Error handling from original
            print(f"\n--- ERROR in main processing loop for query {query_id_val} ---")
            logger.exception(f"Unhandled exception for query_id {query_id_val} in main loop: {e}")
            results_list.append({
                'query_id': query_id_val, 'query': query_text_val, 'status': 'critical_error_in_main_loop',
                'error': str(e), 'reasoning_path': actual_reasoning_path_val
            })
            print(f"Error: {e}")
            print("-" * 30)

    # After processing all queries (from original code)
    print("\n" + "=" * 20 + " System Performance Summary (from SystemControlManager) " + "=" * 20)
    try:
        scm_perf_summary = system_manager.get_performance_metrics()
        if scm_perf_summary.get('total_queries', 0) > 0:
            print(f"- Total Queries Processed by SCM: {scm_perf_summary['total_queries']}")
            print(f"- SCM Successful Queries: {scm_perf_summary['successful_queries']}")
            print(f"- SCM Success Rate: {scm_perf_summary.get('success_rate', 0.0):.1f}%")
            print(f"- SCM Error Count (retries/internal): {scm_perf_summary.get('error_count', 0)}")
            print(f"- SCM Avg Successful Response Time: {scm_perf_summary.get('avg_successful_response_time_sec', 0.0):.3f}s")
        else:
            print("- No queries were processed by SystemControlManager according to its metrics.")

        print("\nSCM Reasoning Path Distribution (Execution Counts):")
        scm_path_stats = system_manager.get_reasoning_path_stats()
        if scm_path_stats:
            for path, path_data in scm_path_stats.items():
                count = path_data.get('execution_count', 0)
                success_count = path_data.get('success_count', 0) # Renamed for clarity
                avg_time = path_data.get('avg_time_sec', 0.0)
                perc_total = path_data.get('percentage_of_total_queries', 0.0)
                print(f"- Path '{path}': Executed={count} ({perc_total:.1f}%), Succeeded={success_count}, AvgTime={avg_time:.3f}s")
        else:
            print("- No path execution statistics available from SystemControlManager.")

        print("\nResource Utilization (Final State via ResourceManager):")
        final_res_usage = resource_manager.check_resources()
        print(f"- CPU Usage: {(final_res_usage.get('cpu', 0.0) or 0.0) * 100:.1f}%")
        print(f"- Memory Usage: {(final_res_usage.get('memory', 0.0) or 0.0) * 100:.1f}%")
        if 'gpu' in final_res_usage and final_res_usage.get('gpu') is not None:
            print(f"- GPU Usage: {(final_res_usage.get('gpu', 0.0) or 0.0) * 100:.1f}%")
    except Exception as report_err:
        logger.error(f"Error generating SystemControlManager performance summary: {report_err}", exc_info=True)

    print("\n" + "=" * 20 + " Comprehensive Academic Analysis (from MetricsCollector) " + "=" * 20)
    # Log directory logic from original
    current_dataset_log_dir = os.path.join(args.log_dir if args and hasattr(args, 'log_dir') else 'logs', dataset_type)
    os.makedirs(current_dataset_log_dir, exist_ok=True)

    try:
        academic_report = metrics_collector.generate_academic_report()
        if not academic_report or academic_report.get("experiment_summary", {}).get("total_queries", 0) == 0:
            print("No data collected by MetricsCollector to generate an academic report.")
            return {"status": "No data for academic report"}

        # Performance Metrics print from original
        perf_metrics = academic_report.get('performance_metrics', {})
        if 'processing_time' in perf_metrics and isinstance(perf_metrics['processing_time'], dict) and perf_metrics['processing_time'].get('count', 0) > 0:
            proc_time_stats = perf_metrics['processing_time']
            print("\nPerformance Metrics (from Metrics Collector):")
            print(f"- Avg Processing Time: {proc_time_stats.get('mean', 0.0):.3f}s (Std: {proc_time_stats.get('std', 0.0):.3f}, Median: {proc_time_stats.get('median', 0.0):.3f})")
            if 'percentile_95' in proc_time_stats: print(f"- 95th Percentile Time: {proc_time_stats.get('percentile_95', 0.0):.3f}s")
        else:
            print("- No processing time metrics collected by MetricsCollector or count is zero.")
        # ... (Rest of the academic report printing from original: Reasoning Analysis, Resource Efficiency, Statistical Analysis) ...
        print("\nReasoning Analysis (from Metrics Collector):")
        ra = academic_report.get('reasoning_analysis', {})
        cc = ra.get('chain_characteristics', {})
        print(f"- Avg Chain Length: {cc.get('avg_length', 0.0):.2f}")
        print(f"- Avg Confidence (Fusion): {cc.get('avg_confidence', 0.0):.3f}")
        print(f"- Path Distribution (MetricsCollector): {ra.get('path_distribution', {})}")

        print("\nResource Efficiency (from Metrics Collector):")
        em_metrics = academic_report.get('efficiency_metrics', {})
        for resource, metrics_val in em_metrics.items():
            if resource != 'trends' and isinstance(metrics_val, dict):
                mean_u = metrics_val.get('mean_usage', 0.0) * 100
                peak_u = metrics_val.get('peak_usage', 0.0) * 100
                score = metrics_val.get('efficiency_score', None)
                print(f"- {resource.capitalize()}: Mean Delta {mean_u:+.1f}%, Peak Delta {peak_u:.1f}%, Score {score:.2f if score is not None else 'N/A'}")
        if 'trends' in em_metrics: print(f"- Efficiency Trends: {em_metrics.get('trends', {})}")

        print("\nStatistical Analysis (from Metrics Collector):")
        sa_metrics = academic_report.get('statistical_analysis', {})
        if 'tests' in sa_metrics:
            for metric, stats_val in sa_metrics.get('tests', {}).items():
                if isinstance(stats_val, dict) and 'p_value' in stats_val:
                    print(f"- {metric}: p-value={stats_val['p_value']:.3g}, effect_size={stats_val.get('effect_size', 0.0):.2f}")
        print(f"- Correlations: {sa_metrics.get('correlations', {})}")
        print(f"- Regression: {sa_metrics.get('regression_analysis', {})}")


        report_file_name = f"academic_report_{dataset_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report_file_path = os.path.join(current_dataset_log_dir, report_file_name)
        try:
            with open(report_file_path, 'w', encoding='utf-8') as rf:
                json.dump(academic_report, rf, indent=2, default=str) # Use default=str for non-serializable
            print(f"\nAcademic report saved to: {report_file_path}")
        except Exception as save_err:
             print(f"Warning: Could not save academic report to {report_file_path}: {save_err}")

        print("\n" + "=" * 20 + " End of Run " + "=" * 20)
        return academic_report

    except Exception as acad_err:
        logger.error(f"Error generating academic report: {acad_err}", exc_info=True)
        return {"error": f"Academic report generation failed: {acad_err}"}

def execute_ablation_study(args: argparse.Namespace):
    """Sets up and runs the ablation study."""
    print(f"\n=== Initializing Ablation Study for Dataset: {args.dataset.upper()} ===")
    dataset_type = args.dataset.lower()

    # 1. Load main configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "src", "config", "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Main configuration file not found at: {config_path}")
        return
    config = ConfigLoader.load_config(config_path)
    model_name = config.get("model_name")
    if not model_name:
        logger.error("model_name not found in main configuration.")
        return

    default_data_dir = os.path.join(script_dir, "data") # Assuming main.py is in project root

    # 2. Load Ablation Configurations
    ablation_config_yaml_path = args.ablation_config
    if not os.path.exists(ablation_config_yaml_path):
        logger.error(f"Ablation configuration file not found at: {ablation_config_yaml_path}")
        return
    with open(ablation_config_yaml_path, 'r') as f:
        loaded_ablation_configs_yaml = yaml.safe_load(f)

    # Resolve rule file paths from keys
    common_paths = {
        "dynamic_rules_path_drop": loaded_ablation_configs_yaml.get("dynamic_rules_path_drop", os.path.join(default_data_dir, "rules_drop_dynamic.json")),
        "static_rules_path_drop": loaded_ablation_configs_yaml.get("static_rules_path_drop", os.path.join(default_data_dir, "rules_drop.json")),
        "no_rules_path": loaded_ablation_configs_yaml.get("no_rules_path", os.path.join(default_data_dir, "empty_rules.json")),
        "rules_path_hotpotqa_baseline": config.get("hotpotqa_rules_file", os.path.join(default_data_dir, "rules_hotpotqa.json"))
    }
    # Create empty_rules.json if it doesn't exist for 'no_rules_path'
    if not os.path.exists(common_paths["no_rules_path"]):
        os.makedirs(os.path.dirname(common_paths["no_rules_path"]), exist_ok=True)
        with open(common_paths["no_rules_path"], "w") as f_empty:
            json.dump([], f_empty)


    ablation_configurations_list = []
    if dataset_type == 'drop':
        for cfg_params in loaded_ablation_configs_yaml.get('drop_ablations', []):
            processed_cfg = cfg_params.copy()
            if 'rules_file_key' in processed_cfg:
                processed_cfg['rules_file'] = common_paths.get(processed_cfg['rules_file_key'], processed_cfg.get('rules_file'))
            ablation_configurations_list.append(processed_cfg)
    elif dataset_type == 'hotpotqa':
        # Similar logic if you have hotpotqa_ablations in your YAML
        for cfg_params in loaded_ablation_configs_yaml.get('hotpotqa_ablations', []): # Assuming a key like 'hotpotqa_ablations'
            processed_cfg = cfg_params.copy()
            if 'rules_file_key' in processed_cfg: # e.g. rules_path_hotpotqa_baseline
                 processed_cfg['rules_file'] = common_paths.get(processed_cfg['rules_file_key'], processed_cfg.get('rules_file'))
            ablation_configurations_list.append(processed_cfg)
    else:
        logger.error(f"Unsupported dataset_type '{dataset_type}' for ablation.")
        return

    if not ablation_configurations_list:
        logger.error(f"No ablation configurations found for dataset_type '{dataset_type}' in {args.ablation_config}.")
        return

    # 3. Load Dataset Samples
    samples_for_ablation: List[Dict[str, Any]] = []
    if dataset_type == 'drop':
        drop_path = config.get("drop_dataset_path", os.path.join(default_data_dir, "drop_dataset_dev.json"))
        samples_for_ablation = load_drop_dataset(drop_path, max_samples=args.samples)
    elif dataset_type == 'hotpotqa':
        hotpotqa_path = config.get("hotpotqa_dataset_path", os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        samples_for_ablation = load_hotpotqa(hotpotqa_path, max_samples=args.samples)

    if not samples_for_ablation:
        logger.error(f"Failed to load samples for {dataset_type} for ablation study.")
        return

    # 4. Initialize Baseline Components (similar to run_hysym_system setup)
    device = DeviceManager.get_device()
    resource_manager = ResourceManager(
        config_path=os.path.join(script_dir, "src", "config", "resource_config.yaml"),
        enable_performance_tracking=True, history_window_size=100
    )
    dimensionality_manager = DimensionalityManager(
        target_dim=config.get('alignment', {}).get('target_dim', 768), device=device
    )
    rules_path_baseline = ""
    if dataset_type == 'drop':
        # For DROP baseline, usually dynamic rules are preferred.
        # Check if dynamic rules were generated in the main config and use that path.
        # This logic assumes dynamic rules are the baseline for DROP.
        rules_path_baseline = common_paths["dynamic_rules_path_drop"]
        if not os.path.exists(rules_path_baseline) or os.path.getsize(rules_path_baseline) <= 2: # Check if empty JSON []
             logger.warning(f"Dynamic DROP rules at {rules_path_baseline} not found or empty. Falling back to static rules for baseline.")
             rules_path_baseline = common_paths["static_rules_path_drop"]
             if not os.path.exists(rules_path_baseline): # Final fallback
                 rules_path_baseline = os.path.join(default_data_dir, "rules_drop.json") # Default static path
                 logger.warning(f"Static DROP rules also not found at {common_paths['static_rules_path_drop']}. Using default {rules_path_baseline}")
    else: # hotpotqa
        rules_path_baseline = common_paths["rules_path_hotpotqa_baseline"]

    logger.info(f"Baseline rules path for {dataset_type} ablation: {rules_path_baseline}")


    symbolic_reasoner_baseline: Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop]
    embedding_model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
    if dataset_type == 'drop':
        symbolic_reasoner_baseline = GraphSymbolicReasonerDrop(
            rules_file=rules_path_baseline,
            match_threshold=config.get('symbolic_match_threshold_drop', 0.1),
            max_hops=config.get('symbolic_max_hops_drop', 3),
            embedding_model=embedding_model_name, device=device, dim_manager=dimensionality_manager
        )
    else:
        symbolic_reasoner_baseline = GraphSymbolicReasoner(
            rules_file=rules_path_baseline,
            match_threshold=config.get('symbolic_match_threshold_hotpot', 0.1),
            max_hops=config.get('symbolic_max_hops_hotpot', 5),
            embedding_model=embedding_model_name, device=device, dim_manager=dimensionality_manager
        )

    nr_few_shot_path_baseline = None
    if dataset_type == 'drop' and bool(config.get("use_drop_few_shots", 0)):
        nr_few_shot_path_baseline = config.get("drop_few_shot_examples_path", os.path.join(default_data_dir, "drop_few_shot_examples.json"))
        if not os.path.exists(nr_few_shot_path_baseline):
            logger.warning(f"Baseline DROP few-shot file not found: {nr_few_shot_path_baseline}. NR will not use them for baseline.")
            nr_few_shot_path_baseline = None

    neural_retriever_baseline = NeuralRetriever(
        model_name,
        use_quantization=config.get('neural_use_quantization', False),
        device=device,
        few_shot_examples_path=nr_few_shot_path_baseline
    )
    complexity_config_yaml_path = os.path.join(script_dir, "src", "config", "complexity_rules.yaml")
    query_expander_baseline = QueryExpander(
        complexity_config=complexity_config_yaml_path if os.path.exists(complexity_config_yaml_path) else None
    )
    response_aggregator_baseline = UnifiedResponseAggregator(include_explanations=True) # Include explanations for debug

    hybrid_integrator_baseline = HybridIntegrator(
        symbolic_reasoner=symbolic_reasoner_baseline,
        neural_retriever=neural_retriever_baseline,
        query_expander=query_expander_baseline,
        dim_manager=dimensionality_manager,
        dataset_type=dataset_type
    )
    # MetricsCollector for baseline SCM is temporary; ablation_study.py creates its own per config.
    # Log directory for this SCM's MC isn't critical as its report isn't the primary output here.
    temp_mc_log_dir = os.path.join(args.log_dir if args and hasattr(args, 'log_dir') else 'logs', dataset_type, "metrics_temp_baseline_scm")
    system_manager_baseline = SystemControlManager(
        hybrid_integrator=hybrid_integrator_baseline,
        resource_manager=resource_manager,
        aggregator=response_aggregator_baseline,
        metrics_collector=MetricsCollector(dataset_type=dataset_type, metrics_dir=temp_mc_log_dir),
        error_retry_limit=config.get('error_retry_limit', 2),
        max_query_time=config.get('max_query_time', 30.0)
    )

    # 5. Run the Ablation Study
    ablation_results_summary = run_ablation_study(
        samples=samples_for_ablation,
        dataset_type=dataset_type,
        rules_path_baseline=rules_path_baseline,
        device=device,
        neural_retriever_baseline=neural_retriever_baseline,
        query_expander_baseline=query_expander_baseline,
        response_aggregator_baseline=response_aggregator_baseline,
        resource_manager_baseline=resource_manager,
        system_manager_baseline=system_manager_baseline,
        dimensionality_manager_baseline=dimensionality_manager,
        ablation_configurations=ablation_configurations_list
    )

    # 6. Save or Print Ablation Summary
    ablation_log_dir = os.path.join(args.log_dir if args and hasattr(args, 'log_dir') else 'logs', dataset_type, "ablation_results")
    os.makedirs(ablation_log_dir, exist_ok=True)
    ablation_summary_file = os.path.join(ablation_log_dir, f"ablation_summary_{dataset_type}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(ablation_summary_file, 'w') as f:
            json.dump(ablation_results_summary, f, indent=2, default=lambda o: '<not serializable>') # Handle non-serializable
        print(f"\nAblation study summary saved to: {ablation_summary_file}")
    except Exception as e:
        print(f"Warning: Could not save ablation study summary: {e}")
        print("\n--- Ablation Results Summary ---")
        # Basic print if JSON fails (could be due to complex objects like torch tensors if not handled)
        for cfg_name, report_data in ablation_results_summary.get('all_ablation_reports', {}).items():
            print(f"\nConfig: {cfg_name}")
            if isinstance(report_data, dict) and 'error' not in report_data:
                print(f"  Mean F1: {report_data.get('response_quality_f1', {}).get('mean', 'N/A')}")
            elif isinstance(report_data, dict) and 'error' in report_data:
                 print(f"  Error: {report_data['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HySym-RAG system with output capture and optional ablation study.')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'drop'],
                        help='Dataset to use for evaluation (hotpotqa or drop)')
    parser.add_argument('--log-dir', default='logs', help='Directory to save log files')
    parser.add_argument('--no-output-capture', action='store_true', help='Disable output capture to file')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to process (for standard run or ablation)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging for main and src components')
    parser.add_argument('--show-progress', action='store_true', help='Show tqdm progress bars')
    # Arguments for ablation study
    parser.add_argument('--run-ablation', action='store_true', help='Run the ablation study instead of a standard evaluation.')
    parser.add_argument('--ablation-config', type=str, default='src/config/ablation_config.yaml',
                        help='Path to the YAML file defining ablation configurations.')


    parsed_args = parser.parse_args()

    # Set logger level based on debug flag
    if parsed_args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG) # Set base 'src' logger to DEBUG
        print("--- DEBUG Logging Enabled for main.py and 'src' package ---")
        # Specific components can also be set to DEBUG if needed, already handled in run_hysym_system
    else:
        logger.setLevel(logging.INFO) # Default main logger to INFO if not debug

    ProgressManager.SHOW_PROGRESS = parsed_args.show_progress
    dataset_log_dir = os.path.join(parsed_args.log_dir, parsed_args.dataset)
    os.makedirs(dataset_log_dir, exist_ok=True)

    if parsed_args.run_ablation:
        print(f"--- Starting Ablation Study for {parsed_args.dataset.upper()} ---")
        # Output capture for ablation study can be handled similarly if desired
        if parsed_args.no_output_capture:
            execute_ablation_study(parsed_args)
        else:
            ablation_output_dir = os.path.join(dataset_log_dir, "ablation_run_logs")
            os.makedirs(ablation_output_dir, exist_ok=True)
            try:
                with capture_output(output_dir=ablation_output_dir) as output_path:
                    print(f"Ablation study output is being captured to: {output_path}")
                    execute_ablation_study(parsed_args)
            except Exception as e:
                print(f"ERROR: Failed to set up output capture or run ablation study: {e}", file=sys.stderr)
                logger.error(f"Failed to set up output capture or run ablation study: {e}", exc_info=True)
    else:
        # Standard system run
        if parsed_args.no_output_capture:
            print("--- Running standard HySym-RAG evaluation without output capture. ---")
            run_hysym_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
        else:
            try:
                with capture_output(output_dir=dataset_log_dir) as output_path:
                    print(f"Output from this run is being captured to: {output_path}")
                    run_hysym_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
            except Exception as e:
                print(f"ERROR: Failed to set up output capture or run system: {e}", file=sys.stderr)
                logger.error(f"Failed to set up output capture or run system: {e}", exc_info=True)