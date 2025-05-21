# main.py

import os
import sys
import json
import time
import warnings
import argparse
import logging
import urllib3 # type: ignore
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
# import torch.nn as nn  # Removed: This import seems unused in main.py based on provided snippets
# import numpy as np  # Removed: This import seems unused in main.py based on provided snippets

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
except ImportError:
    print("Transformers library not found. Please install it: pip install transformers")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer # type: ignore
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
    from src.feedback.feedback_manager import FeedbackManager # Assuming this is used or intended
    from src.config.config_loader import ConfigLoader
    from src.queries.query_expander import QueryExpander
    from src.utils.evaluation import Evaluation
    # Updated import statement below to use the __init__.py in src.system
    from src.system import SystemControlManager, UnifiedResponseAggregator
    from src.utils.metrics_collector import MetricsCollector
    from src.utils.device_manager import DeviceManager
    from src.utils.progress import tqdm, ProgressManager
    from src.utils.output_capture import capture_output
except ImportError as e:
    print(f"Error importing HySym-RAG components: {e}")
    print("Please ensure main.py is run from the project root directory or PYTHONPATH is set correctly.")
    sys.exit(1)

urllib3.disable_warnings() # type: ignore
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Default to DEBUG, can be overridden by args


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


def run_hysym_system(samples: int = 200, dataset_type: str = 'hotpotqa', args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Main execution function for the HySym-RAG system."""
    print(f"\n=== Initializing HySym-RAG System for Dataset: {dataset_type.upper()} ===")

    # Configure library logging levels
    for lib in ['transformers', 'sentence_transformers', 'urllib3.connectionpool', 'h5py', 'numexpr', 'spacy']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Configure project-specific logger levels (can be overridden by --debug)
    logging.getLogger('src.utils.dimension_manager').setLevel(logging.INFO)
    logging.getLogger('src.integrators.hybrid_integrator').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_base').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.networkx_symbolic_reasoner_drop').setLevel(logging.INFO)
    logging.getLogger('src.reasoners.neural_retriever').setLevel(logging.INFO)
    logging.getLogger('src.system.system_control_manager').setLevel(logging.INFO)
    logging.getLogger('src.system.response_aggregator').setLevel(logging.INFO)
    logging.getLogger('src.system.system_logic_helpers').setLevel(logging.INFO)

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

        project_root = os.path.dirname(script_dir)

        default_data_dir = os.path.join(project_root, "data")
        hotpotqa_path = config.get("hotpotqa_dataset_path",
                                   os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        drop_path = config.get("drop_dataset_path", os.path.join(default_data_dir, "drop_dataset_dev.json"))

        rules_path_default_name = "rules.json"
        hotpotqa_rules_path = config.get("hotpotqa_rules_file",
                                         os.path.join(default_data_dir, "rules_hotpotqa.json"))
        drop_rules_path = config.get("drop_rules_file",
                                       os.path.join(default_data_dir, "rules_drop.json"))

        kb_path_default = os.path.join(default_data_dir, "small_knowledge_base.txt")
        kb_path = config.get("knowledge_base", kb_path_default)

    except Exception as e:
        logger.exception(f"Failed to load configuration: {e}")
        return {"error": f"Configuration loading failed: {e}"}

    # 2. DeviceManager
    device = DeviceManager.get_device()
    print(f"Using device: {device}")

    # 3. ResourceManager
    print("Initializing Resource Manager...")
    resource_config_path = os.path.join(script_dir, "src", "config", "resource_config.yaml")
    if not os.path.exists(resource_config_path):
        logger.warning(
            f"Resource configuration file not found at {resource_config_path}. ResourceManager will use defaults.")
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

    # 5. Select and Ensure Rules File
    rules_path = "" # Initialize rules_path
    if dataset_type.lower() == 'drop':
        rules_path = drop_rules_path
        if not os.path.exists(rules_path):
            rules_path = os.path.join(default_data_dir, rules_path_default_name)
    else:  # hotpotqa or other
        rules_path = hotpotqa_rules_path
        if not os.path.exists(rules_path):
            rules_path = os.path.join(default_data_dir, rules_path_default_name)

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
    test_queries: List[Dict[str, Any]] = []
    ground_truths: Dict[str, Any] = {}

    if dataset_type.lower() == 'drop':
        if not os.path.exists(drop_path):
            logger.error(f"DROP dataset file not found at: {drop_path}")
            return {"error": f"DROP dataset not found at {drop_path}"}
        print(f"Loading DROP dataset from {drop_path}...")
        test_queries = load_drop_dataset(drop_path, max_samples=samples)
        ground_truths = {s["query_id"]: s["answer"] for s in test_queries if "query_id" in s and "answer" in s}

        questions = [qa['query'] for qa in test_queries if qa.get('query')]
        passages = [qa['context'] for qa in test_queries if qa.get('context')]

        if questions and passages:
            print("Extracting DROP-specific rules dynamically...")
            try:
                dynamic_rules = rule_extractor.extract_rules_from_drop(
                    drop_json_path=drop_path,
                    questions=questions,
                    passages=passages,
                    min_support=config.get('drop_rule_min_support', 5)
                )
                rules_path = os.path.join(default_data_dir, "rules_drop_dynamic.json")
                os.makedirs(os.path.dirname(rules_path), exist_ok=True)
                with open(rules_path, "w", encoding="utf-8") as f:
                    json.dump(dynamic_rules, f, indent=2)
                print(f"Saved {len(dynamic_rules)} dynamically extracted DROP rules to {rules_path}")
                if not dynamic_rules:
                    logger.warning(
                        "No dynamic rules were extracted for DROP. Symbolic reasoner might be less effective.")
            except Exception as e:
                logger.error(
                    f"Failed to extract dynamic DROP rules: {e}. Continuing with potentially pre-existing rules file: {drop_rules_path}")
                rules_path = drop_rules_path
        else:
            logger.warning("Not enough data from loaded samples to extract dynamic DROP rules.")
            rules_path = drop_rules_path

    elif dataset_type.lower() == 'hotpotqa':
        if not os.path.exists(hotpotqa_path):
            logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
            return {"error": f"HotpotQA dataset not found at {hotpotqa_path}"}
        print(f"Loading HotpotQA dataset from {hotpotqa_path}...")
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=samples)
        for sample_item in test_queries:
            ground_truths[sample_item["query_id"]] = sample_item["answer"]
        rules_path = hotpotqa_rules_path
    else:
        logger.error(f"Unknown dataset_type '{dataset_type}'. Cannot load data.")
        return {"error": f"Unknown dataset_type '{dataset_type}'"}

    if not test_queries:
        logger.error("No test queries loaded. Exiting.")
        return {"error": "No test queries loaded"}
    print(f"Loaded {len(test_queries)} test queries.")

    # 8. GraphSymbolicReasoner
    print(f"Initializing Symbolic Reasoner for {dataset_type.upper()} using rules from: {rules_path}...")
    symbolic: Optional[Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop]] = None
    embedding_model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
    try:
        if dataset_type.lower() == 'drop':
            symbolic = GraphSymbolicReasonerDrop(
                rules_file=rules_path,
                match_threshold=config.get('symbolic_match_threshold_drop', 0.1),
                max_hops=config.get('symbolic_max_hops_drop', 3),
                embedding_model=embedding_model_name,
                device=device,
                dim_manager=dimensionality_manager
            )
        else:
            symbolic = GraphSymbolicReasoner(
                rules_file=rules_path,
                match_threshold=config.get('symbolic_match_threshold_hotpot', 0.1),
                max_hops=config.get('symbolic_max_hops_hotpot', 5),
                embedding_model=embedding_model_name,
                device=device,
                dim_manager=dimensionality_manager
            )
        count_rules = len(getattr(symbolic, 'rules', []))
        print(f"Symbolic Reasoner loaded {count_rules} rules successfully from {rules_path}.")
        if count_rules == 0:
            logger.warning(
                f"Symbolic reasoner initialized, but no rules were loaded from {rules_path}. Symbolic path may be ineffective.")
    except Exception as e:
        logger.exception(f"Fatal error initializing symbolic reasoner: {e}")
        return {"error": f"Symbolic reasoner initialization failed: {e}"}

    # 9. NeuralRetriever
    print("Initializing Neural Retriever...")
    try:
        neural = NeuralRetriever(
            model_name,
            use_quantization=config.get('neural_use_quantization', False),
            max_context_length=config.get('neural_max_context_length', 2048),
            chunk_size=config.get('neural_chunk_size', 512),
            overlap=config.get('neural_overlap', 128),
            device=device
        )
    except Exception as e:
        logger.exception(f"Fatal error initializing neural retriever: {e}")
        return {"error": f"Neural retriever initialization failed: {e}"}

    # 10. Support components
    print("Initializing support components...")
    log_dir_for_query_logger = args.log_dir if args and hasattr(args, 'log_dir') else 'logs'
    query_logger = QueryLogger(log_dir=os.path.join(log_dir_for_query_logger, dataset_type))

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
        logger.exception(f"Fatal error initializing Hybrid Integrator: {e}")
        return {"error": f"Hybrid Integrator initialization failed: {e}"}

    # 13. SystemControlManager
    print("Initializing System Control Manager...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    log_dir_for_metrics_collector = args.log_dir if args and hasattr(args, 'log_dir') else 'logs'
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=os.path.join(log_dir_for_metrics_collector, dataset_type, "metrics_collection")
    )
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
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
            else:
                logger.warning(f"General knowledge base file {kb_path} is empty.")
        except Exception as e:
            logger.warning(f"Could not load general knowledge base from {kb_path}: {e}")
    elif kb_path:
        logger.warning(f"General knowledge base file specified but not found: {kb_path}")

    # Main Query Processing Loop
    print(f"\n=== Testing System with {len(test_queries)} Queries from {dataset_type.upper()} Dataset ===")
    results_list: List[Dict[str, Any]] = []

    query_iterator = tqdm(test_queries, desc="Processing Queries", unit="query",
                          disable=not ProgressManager.SHOW_PROGRESS)

    for q_info in query_iterator:
        query_id = q_info.get("query_id")
        query = q_info.get("query")
        the_answer_obj = q_info.get("answer")
        local_context = q_info.get("context", general_context)
        forced_path = q_info.get("forced_path")
        data_type = q_info.get("type", "")
        supporting_facts_for_query = q_info.get("supporting_facts")

        if not query_id or not query:
            logger.warning(f"Skipping query due to missing ID or text: {q_info}")
            continue

        if ProgressManager.SHOW_PROGRESS:
            query_iterator.set_description(f"Processing QID: {str(query_id)[:8]}")

        logger.info(f"Processing Query ID: {query_id}, Query: '{query[:100]}...'")
        logger.debug(f"Query Type: {data_type}, Forced Path: {forced_path}")

        final_response_from_system: Dict[str, Any] = {}
        actual_reasoning_path: str = 'error_before_processing'

        try:
            complexity = expander.get_query_complexity(query)
            logger.info(f"Query '{query_id}' Complexity Score: {complexity:.4f}")

            final_response_from_system, actual_reasoning_path = system_manager.process_query_with_fallback(
                query=query,
                context=local_context,
                query_id=query_id,
                forced_path=forced_path,
                query_complexity=complexity,
                supporting_facts=supporting_facts_for_query,
                dataset_type=dataset_type
            )

            prediction_val = final_response_from_system.get('result',
                                                         ({'error': 'Result key missing in system response'}
                                                          if dataset_type == 'drop' else "Error: Result key missing"))
            results_list.append(final_response_from_system)

            print("\n" + "-" * 10 + f" Results for QID: {query_id} " + "-" * 10)
            if dataset_type == 'drop' and isinstance(prediction_val, dict):
                print(f"  Prediction (DROP): {prediction_val}")
            else:
                print(f"  Prediction (Text): {str(prediction_val)[:300]}...")

            print(f"  Reasoning Path: {actual_reasoning_path}")
            print(f"  Overall Processing Time: {final_response_from_system.get('processing_time', 0.0):.3f}s")
            print(f"  Status: {final_response_from_system.get('status', 'unknown')}")

            res_delta = final_response_from_system.get('resource_usage', {})
            if res_delta:
                delta_str_parts = [f"{k.capitalize()}: {v * 100:+.1f}%" for k, v in res_delta.items()]
                if delta_str_parts:
                    print(f"  Resource Delta: {', '.join(delta_str_parts)}")

            explanation = final_response_from_system.get('explanation')
            if explanation:
                print(f"  Explanation: {explanation}")
            print("-" * (30 + len(str(query_id))))

            if data_type.startswith("ground_truth_available") and \
                    the_answer_obj is not None and \
                    final_response_from_system.get('status') == 'success':
                if query_id in ground_truths:
                    gt_answer = ground_truths[query_id]
                    sf_for_eval = {query_id: supporting_facts_for_query} if supporting_facts_for_query and dataset_type != 'drop' else None
                    rc_for_eval = None
                    if 'debug_info' in final_response_from_system and isinstance(final_response_from_system['debug_info'], dict):
                        chain_steps = final_response_from_system['debug_info'].get('steps',
                                                                                final_response_from_system['debug_info'].get('chain_info', {}).get('steps'))
                        if chain_steps:
                            rc_for_eval = {query_id: chain_steps}

                    eval_metrics = evaluator.evaluate(
                        predictions={query_id: prediction_val},
                        ground_truths={query_id: gt_answer},
                        supporting_facts=sf_for_eval,
                        reasoning_chain=rc_for_eval
                    )
                    print("  Evaluation Metrics:")
                    print(f"    Exact Match: {eval_metrics.get('average_exact_match', 0.0):.3f}")
                    print(f"    F1 Score: {eval_metrics.get('average_f1', 0.0):.3f}")
                    if dataset_type != 'drop':
                        print(f"    ROUGE-L: {eval_metrics.get('average_rougeL', 0.0):.3f}")
                        print(f"    BLEU: {eval_metrics.get('average_bleu', 0.0):.3f}")
                        print(f"    Semantic Sim: {eval_metrics.get('average_semantic_similarity', 0.0):.3f}")
                    final_response_from_system['evaluation_metrics'] = eval_metrics
                else:
                    logger.warning(f"Ground truth not found for query_id {query_id} during evaluation.")
            elif final_response_from_system.get('status') != 'success':
                logger.warning(f"Skipping evaluation for QID {query_id} due to status: '{final_response_from_system.get('status')}'.")

        except Exception as e:
            print(f"\n--- ERROR in main processing loop for query {query_id} ---")
            logger.exception(f"Unhandled exception for query_id {query_id} in main loop: {e}")
            results_list.append({
                'query_id': query_id, 'query': query, 'status': 'critical_error_in_main_loop',
                'error': str(e), 'reasoning_path': actual_reasoning_path
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
            print(f"- SCM Avg Successful Response Time: {scm_perf_summary.get('avg_successful_response_time_sec', 0.0):.3f}s")
        else:
            print("- No queries were processed by SystemControlManager according to its metrics.")

        print("\nSCM Reasoning Path Distribution (Execution Counts):")
        scm_path_stats = system_manager.get_reasoning_path_stats()
        if scm_path_stats:
            for path, path_data in scm_path_stats.items():
                count = path_data.get('execution_count', 0)
                success = path_data.get('success_count', 0)
                avg_time = path_data.get('avg_time_sec', 0.0)
                perc_total = path_data.get('percentage_of_total_queries', 0.0)
                print(f"- Path '{path}': Executed={count} ({perc_total:.1f}%), Succeeded={success}, AvgTime={avg_time:.3f}s")
        else:
            print("- No path execution statistics available from SystemControlManager.")
            if scm_perf_summary.get('total_queries', 0) > 0:
                logger.warning("SCM reported processing queries but get_reasoning_path_stats() returned empty.")

        print("\nResource Utilization (Final State via ResourceManager):")
        final_res = resource_manager.check_resources()
        cpu_final = final_res.get('cpu', 0.0) or 0.0
        mem_final = final_res.get('memory', 0.0) or 0.0
        print(f"- CPU Usage: {cpu_final * 100:.1f}%")
        print(f"- Memory Usage: {mem_final * 100:.1f}%")
        if 'gpu' in final_res and final_res.get('gpu') is not None:
            gpu_final = final_res.get('gpu', 0.0) or 0.0
            print(f"- GPU Usage: {gpu_final * 100:.1f}%")
    except Exception as report_err:
        logger.error(f"Error generating SystemControlManager performance summary: {report_err}", exc_info=True)

    print("\n" + "=" * 20 + " Comprehensive Academic Analysis (from MetricsCollector) " + "=" * 20)
    current_dataset_log_dir = 'logs'
    if args and hasattr(args, 'log_dir') and hasattr(args, 'dataset'):
        current_dataset_log_dir = os.path.join(args.log_dir, args.dataset)
    else:
        current_dataset_log_dir = os.path.join('logs', dataset_type)
    os.makedirs(current_dataset_log_dir, exist_ok=True)

    try:
        academic_report = metrics_collector.generate_academic_report()
        if not academic_report or academic_report.get("experiment_summary", {}).get("total_queries", 0) == 0:
            print("No data collected by MetricsCollector to generate an academic report.")
            return {"status": "No data for academic report"}

        print("\nPerformance Metrics (from Metrics Collector):")
        perf_metrics = academic_report.get('performance_metrics', {})
        if 'processing_time' in perf_metrics and isinstance(perf_metrics['processing_time'], dict) and \
           perf_metrics['processing_time'].get('count', 0) > 0:
            proc_time_stats = perf_metrics['processing_time']
            print(
                f"- Avg Processing Time: {proc_time_stats.get('mean', 0.0):.3f}s (Std: {proc_time_stats.get('std', 0.0):.3f}, Median: {proc_time_stats.get('median', 0.0):.3f})")
            if 'percentile_95' in proc_time_stats:
                 print(f"- 95th Percentile Time: {proc_time_stats.get('percentile_95', 0.0):.3f}s")
        else:
            print("- No processing time metrics collected by MetricsCollector or count is zero.")

        print("\nReasoning Analysis (from Metrics Collector):")
        ra = academic_report.get('reasoning_analysis', {})
        cc = ra.get('chain_characteristics', {})
        print(f"- Avg Chain Length: {cc.get('avg_length', 0.0):.2f}")
        print(f"- Avg Confidence (Fusion): {cc.get('avg_confidence', 0.0):.3f}")
        path_dist_mc = ra.get('path_distribution', {})
        print(f"- Path Distribution (MetricsCollector): {path_dist_mc}")

        print("\nResource Efficiency (from Metrics Collector):")
        em = academic_report.get('efficiency_metrics', {})
        for resource, metrics_val in em.items():
            if resource != 'trends' and isinstance(metrics_val, dict):
                mean_u = metrics_val.get('mean_usage', 0.0) * 100
                peak_u = metrics_val.get('peak_usage', 0.0) * 100
                score = metrics_val.get('efficiency_score', None)
                score_str = f"{score:.2f}" if score is not None else "N/A"
                print(f"- {resource.capitalize()}: Mean Delta {mean_u:+.1f}%, Peak Delta {peak_u:.1f}%, Score {score_str}")
        if 'trends' in em:
            print(f"- Efficiency Trends: {em.get('trends', {})}")

        print("\nStatistical Analysis (from Metrics Collector):")
        sa = academic_report.get('statistical_analysis', {})
        if 'tests' in sa:
            for metric, stats_val in sa.get('tests', {}).items():
                if isinstance(stats_val, dict) and 'p_value' in stats_val:
                    print(f"- {metric}: p-value={stats_val['p_value']:.3g}, effect_size={stats_val.get('effect_size', 0.0):.2f}")
        print(f"- Correlations: {sa.get('correlations', {})}")
        print(f"- Regression: {sa.get('regression_analysis', {})}")

        report_file_name = f"academic_report_{dataset_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report_file_path = os.path.join(current_dataset_log_dir, report_file_name)
        try:
            with open(report_file_path, 'w', encoding='utf-8') as rf:
                json.dump(academic_report, rf, indent=2, default=str)
            print(f"\nAcademic report saved to: {report_file_path}")
        except Exception as save_err:
            print(f"Warning: Could not save academic report to {report_file_path}: {save_err}")

        print("\n" + "=" * 20 + " End of Run " + "=" * 20)
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
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to process')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG level logging for main and src components')
    parser.add_argument('--show-progress', action='store_true', help='Show tqdm progress bars')

    parsed_args = parser.parse_args() # Renamed to avoid conflict with 'args' in run_hysym_system

    if parsed_args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG)
        print("--- DEBUG Logging Enabled for main.py and 'src' package ---")
        logging.getLogger('src.system.system_control_manager').setLevel(logging.DEBUG)
        logging.getLogger('src.integrators.hybrid_integrator').setLevel(logging.DEBUG)
        logging.getLogger('src.reasoners.neural_retriever').setLevel(logging.DEBUG)
        logging.getLogger('src.reasoners.networkx_symbolic_reasoner_drop').setLevel(logging.DEBUG)

    ProgressManager.SHOW_PROGRESS = parsed_args.show_progress

    dataset_log_dir = os.path.join(parsed_args.log_dir, parsed_args.dataset)
    os.makedirs(dataset_log_dir, exist_ok=True)

    if parsed_args.no_output_capture:
        print("--- Running without output capture. ---")
        run_hysym_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
    else:
        try:
            with capture_output(output_dir=dataset_log_dir) as output_path:
                print(f"Output from this run is being captured to: {output_path}")
                run_hysym_system(samples=parsed_args.samples, dataset_type=parsed_args.dataset, args=parsed_args)
        except Exception as e:
            print(f"ERROR: Failed to set up output capture or run system: {e}", file=sys.stderr)
            logger.error(f"Failed to set up output capture or run system: {e}", exc_info=True)