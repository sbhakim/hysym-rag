# main.py

from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.reasoners.neural_retriever import NeuralRetriever
from src.integrators.hybrid_integrator import HybridIntegrator
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os
import json
import torch
import warnings

# Suppress specific spaCy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")

if __name__ == "__main__":
    print("\n=== Initializing HySym-RAG System ===")

    # 1. Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config("src/config/config.yaml")
    model_name = config["model_name"]

    # 2. Initialize Resource Manager with performance tracking enabled
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager(
        config_path="src/config/resource_config.yaml",
        enable_performance_tracking=True,
        history_window_size=100  # Keep history of last 100 queries
    )

    # 3. Extract symbolic rules into data/rules.json
    print("Extracting rules from deforestation.txt...")
    RuleExtractor.extract_rules("data/deforestation.txt", "data/rules.json")

    # 4. Initialize the graph-based symbolic reasoner (max_hops set to 5)
    print("Initializing Graph-Based Symbolic Reasoner...")
    symbolic = GraphSymbolicReasoner(
        "data/rules.json",
        match_threshold=0.25,  # modest threshold
        max_hops=5  # allow deeper multi-hop chaining
    )

    # 5. Initialize the Neural Retriever (optionally enable quantization)
    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(model_name, use_quantization=False)

    # 6. Initialize support components: logger, feedback manager, query expander, evaluation
    print("Initializing support components...")
    logger = QueryLogger()
    feedback_manager = FeedbackManager()
    print("Initializing QueryExpander...")
    expander = QueryExpander(
        complexity_config="src/config/complexity_rules.yaml"
    )
    print("Loading evaluation dataset...")
    with open("data/ground_truths.json", "r") as gt_file:
        ground_truths = json.load(gt_file)
    evaluator = Evaluation()

    # 7. Create Hybrid Integrator (deciding symbolic vs. neural calls)
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(symbolic, neural, resource_manager, expander)

    # 8. Initialize UnifiedResponseAggregator and SystemControlManager
    print("Initializing System Control Components...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
        error_retry_limit=2,
        max_query_time=10
    )

    # 9. Initialize Application and Feedback Handler
    print("Initializing Application...")
    feedback_handler = FeedbackHandler(feedback_manager)
    app = App(
        symbolic=symbolic,
        neural=neural,
        logger=logger,
        feedback=resource_manager,  # resource-aware scheduling
        evaluator=evaluator,
        expander=expander,
        ground_truths=ground_truths,
        system_manager=system_manager
    )

    # 10. Load knowledge base (for neural context)
    print("Loading knowledge base...")
    with open("data/small_knowledge_base.txt", "r") as kb_file:
        context = kb_file.read()

    # 11. Test queries using SystemControlManager
    print("\n=== Testing System with Various Queries ===")
    test_queries = [
        {"query": "What are the environmental effects of deforestation?", "type": "ground_truth_available"},
        {"query": "What is the social impact of deforestation?", "type": "ground_truth_available"},
        {"query": "What is deforestation?", "type": "exploratory"},
        {"query": "How does deforestation cause climate change?", "type": "ground_truth_available"}
    ]

    for q_info in test_queries:
        query = q_info["query"]
        print(f"\nProcessing Query: {query}")
        print(f"Query Type: {q_info['type']}")
        print("-" * 50)
        try:
            # Process query: get complexity, resource metrics, and final answer via hybrid processing
            complexity = expander.get_query_complexity(query)
            print(f"Query Complexity Score: {complexity:.4f}")

            initial_metrics = resource_manager.check_resources()
            final_answer = system_manager.process_query_with_fallback(query, context)
            final_metrics = resource_manager.check_resources()
            resource_delta = {
                key: final_metrics[key] - initial_metrics[key]
                for key in final_metrics
            }

            # Log the query and result
            logger.log_query(
                query=query,
                result=final_answer,
                source="hybrid",
                complexity=complexity,
                resource_usage=resource_delta
            )

            print("\nProcessing Results:")
            print("-" * 20)
            print(final_answer)
            print("\nResource Usage:")
            print(f"CPU Delta: {resource_delta['cpu'] * 100:.1f}%")
            print(f"Memory Delta: {resource_delta['memory']:.2f} GB")
            print(f"GPU Delta: {resource_delta['gpu'] * 100:.1f}%")
            print("-" * 20)

            if q_info["type"] == "ground_truth_available":
                print("\nEvaluation Metrics:")
                eval_metrics = evaluator.evaluate({query: final_answer}, ground_truths)
                print(f"Similarity Score: {eval_metrics['average_similarity']:.2f}")

        except KeyError as e:
            print(f"Error: Missing ground truth for query evaluation - {str(e)}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

    # 12. Comparison Experiment: Hybrid vs Neural-Only
    print("\n=== Comparison Experiment: Hybrid vs Neural-Only ===")
    comparison_queries = [
        "What are the environmental effects of deforestation?"
    ]

    # Print header for comparison table
    header = f"{'Query':<50} | {'Mode':<15} | {'CPU Δ (%)':<10} | {'Memory Δ (GB)':<15} | {'GPU Δ (%)':<10} | {'Response'}"
    print(header)
    print("-" * len(header))

    for query in comparison_queries:
        # Run Hybrid Reasoning
        initial_metrics_hybrid = resource_manager.check_resources()
        hybrid_answer = system_manager.process_query_with_fallback(query, context)
        final_metrics_hybrid = resource_manager.check_resources()
        hybrid_delta = {k: final_metrics_hybrid[k] - initial_metrics_hybrid[k] for k in final_metrics_hybrid}

        # Run Neural-Only Reasoning (direct call)
        initial_metrics_neural = resource_manager.check_resources()
        neural_answer = neural.retrieve_answer(context, query)
        final_metrics_neural = resource_manager.check_resources()
        neural_delta = {k: final_metrics_neural[k] - initial_metrics_neural[k] for k in final_metrics_neural}

        # Prepare and print table rows
        row_hybrid = f"{query:<50} | {'Hybrid':<15} | {hybrid_delta['cpu'] * 100:>10.1f} | {hybrid_delta['memory']:>15.2f} | {hybrid_delta['gpu'] * 100:>10.1f} | {hybrid_answer}"
        row_neural = f"{query:<50} | {'Neural':<15} | {neural_delta['cpu'] * 100:>10.1f} | {neural_delta['memory']:>15.2f} | {neural_delta['gpu'] * 100:>10.1f} | {neural_answer}"
        print(row_hybrid)
        print(row_neural)
        print("-" * len(header))

    # 13. Final performance summary
    print("\n=== System Performance Summary ===")
    performance_stats = system_manager.get_performance_metrics()
    print("\nOverall Performance:")
    print(f"- Total Queries: {performance_stats['total_queries']}")
    print(f"- Average Response Time: {performance_stats['avg_response_time']:.2f}s")
    print(f"- Success Rate: {performance_stats['success_rate']:.1f}%")

    print("\nResource Utilization:")
    final_resources = resource_manager.check_resources()
    print(f"- CPU Usage: {final_resources['cpu'] * 100:.1f}%")
    print(f"- Memory Usage: {final_resources['memory'] * 100:.1f}%")
    print(f"- GPU Usage: {final_resources['gpu'] * 100:.1f}%")

    print("\nReasoning Path Distribution:")
    path_stats = system_manager.get_reasoning_path_stats()
    for path, percentage in path_stats.items():
        print(f"- {path}: {percentage:.1f}%")

    print("\nSystem Information:")
    print(f"Model Path: {neural.model.config._name_or_path}")
    print(f"Cache Location: {os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")
    print("=== End of Run ===")
