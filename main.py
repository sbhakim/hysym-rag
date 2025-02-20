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
from src.utils.metrics_collector import MetricsCollector

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os
import json
import torch
import warnings
import logging

# Suppress specific spaCy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
            "query": question,
            "answer": answer,
            "context": context_str,
            "type": "ground_truth_available",
            "supporting_facts": supporting_facts  # Store supporting facts
        })
        count += 1
        if max_samples and count >= max_samples:
            break
    return dataset


if __name__ == "__main__":
    print("\n=== Initializing HySym-RAG System ===")

    # 1. Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config("src/config/config.yaml")
    model_name = config["model_name"]

    # 2. Initialize Resource Manager
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager(
        config_path="src/config/resource_config.yaml",
        enable_performance_tracking=True,
        history_window_size=100
    )

    # ----------------------------------------------------------------
    # 3. (Optional) We skip extracting rules from deforestation.txt.
    #    Instead, we ensure 'data/rules.json' exists but is empty or minimal.
    # ----------------------------------------------------------------
    rules_path = "data/rules.json"
    if not os.path.exists(rules_path):
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    print(f"Loading existing rules from {rules_path} (initially empty or minimal).")

    # 4. Initialize the Graph-Based Symbolic Reasoner
    print("Initializing Graph-Based Symbolic Reasoner...")
    symbolic = GraphSymbolicReasoner(
        rules_file=rules_path,
        match_threshold=0.25,
        max_hops=5
    )

    # 5. Initialize the Neural Retriever
    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(model_name, use_quantization=False)

    # 6. Additional components
    print("Initializing support components...")
    logger = logging.getLogger(__name__)  # Standard logger for warnings
    query_logger = QueryLogger()
    feedback_manager = FeedbackManager()
    print("Initializing QueryExpander...")
    expander = QueryExpander(
        complexity_config="src/config/complexity_rules.yaml"
    )
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()  # Instantiate RuleExtractor
    print("Loading evaluation dataset...")

    # Decide whether to load HotpotQA
    use_hotpotqa = True
    hotpotqa_path = "data/hotpot_dev_distractor_v1.json"
    max_hotpot_samples = 20

    if use_hotpotqa and os.path.exists(hotpotqa_path):
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=max_hotpot_samples)
        ground_truths = {}

        # Build rules from HotpotQA contexts and store ground truths
        for i, sample in enumerate(test_queries):
            new_rules = rule_extractor.extract_hotpot_facts(sample["context"], min_confidence=0.7)
            if new_rules:
                # Wrap in try-except to catch missing _track_rule_addition
                try:
                    symbolic.add_dynamic_rules(new_rules)
                except AttributeError as e:
                    logger.warning(f"Could not track new rules automatically (missing method?): {str(e)}")
            ground_truths[sample["query"]] = sample["answer"]
    else:
        print("Warning: HotpotQA not found, and no fallback dataset provided.")
        test_queries = []
        ground_truths = {}

    evaluator = Evaluation()

    # 7. Create Hybrid Integrator
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(symbolic, neural, resource_manager, expander)

    # 8. System Control - Initialize MetricsCollector and pass it to SystemControlManager
    print("Initializing System Control Components...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    metrics_collector = MetricsCollector()
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
        metrics_collector=metrics_collector,
        error_retry_limit=2,
        max_query_time=10
    )

    # 9. Initialize Application
    print("Initializing Application...")
    feedback_handler = FeedbackHandler(feedback_manager)
    app = App(
        symbolic=symbolic,
        neural=neural,
        logger=logger,
        feedback=resource_manager,
        evaluator=evaluator,
        expander=expander,
        ground_truths=ground_truths,
        system_manager=system_manager
    )

    # 10. Possibly load a knowledge base for neural context
    kb_path = "data/small_knowledge_base.txt"
    if os.path.exists(kb_path):
        with open(kb_path, "r") as kb_file:
            context = kb_file.read()
    else:
        context = ""

    print("\n=== Testing System with Queries ===")
    for q_info in test_queries:
        query = q_info["query"]
        the_answer = q_info.get("answer", None)
        forced_path = q_info.get("forced_path", None)
        data_type = q_info.get("type", "ground_truth_available")
        supporting_facts = q_info.get("supporting_facts", None)  # Get supporting facts

        print(f"\nProcessing Query: {query}")
        print(f"Query Type: {data_type}")
        if forced_path:
            print(f"Forced Path: {forced_path}")
        print("-" * 50)
        try:
            complexity = expander.get_query_complexity(query)
            print(f"Query Complexity Score: {complexity:.4f}")

            initial_metrics = resource_manager.check_resources()
            local_context = q_info.get("context", context)

            final_answer = system_manager.process_query_with_fallback(
                query, local_context, forced_path=forced_path, query_complexity=complexity
            )
            final_metrics = resource_manager.check_resources()
            resource_delta = {
                key: final_metrics[key] - initial_metrics[key]
                for key in final_metrics
            }

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
            print(f"Memory Delta: {resource_delta['memory'] * 100:.1f}%")
            print(f"GPU Delta: {resource_delta['gpu'] * 100:.1f}%")
            print("-" * 20)

            if data_type == "ground_truth_available" and the_answer is not None:
                # Pass supporting facts to evaluator.evaluate
                eval_metrics = evaluator.evaluate(
                    predictions={query: final_answer[0]},
                    ground_truths={query: the_answer},
                    supporting_facts={query: supporting_facts}  # Pass supporting facts
                )
                print("\nEvaluation Metrics:")
                print(f"Similarity Score: {eval_metrics['average_semantic_similarity']:.2f}")
                print(f"ROUGE-L Score: {eval_metrics['average_rougeL']:.2f}")
                print(f"BLEU Score: {eval_metrics['average_bleu']:.2f}")  # Print BLEU
                print(f"F1 Score: {eval_metrics['average_f1']:.2f}")

        except KeyError as e:
            print(f"Error: Missing ground truth for query evaluation - {str(e)}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

    # 11. Optional Comparison Experiment
    print("\n=== Comparison Experiment (Sample) ===")
    comparison_queries = ["Compare and contrast the film adaptations of 'Pride and Prejudice'."]
    header = f"{'Query':<50} | {'Mode':<15} | {'CPU Δ (%)':<10} | {'Memory Δ (%)':<15} | {'GPU Δ (%)':<10} | {'Response'}"
    print(header)
    print("-" * len(header))

    for query in comparison_queries:
        initial_metrics_hybrid = resource_manager.check_resources()
        hybrid_answer = system_manager.process_query_with_fallback(query, context)
        final_metrics_hybrid = resource_manager.check_resources()
        hybrid_delta = {k: final_metrics_hybrid[k] - initial_metrics_hybrid[k] for k in final_metrics_hybrid}

        initial_metrics_neural = resource_manager.check_resources()
        neural_answer_raw = system_manager.process_query_with_fallback(query, context, forced_path="neural")
        neural_answer = aggregator.format_response({'result': neural_answer_raw})
        final_metrics_neural = resource_manager.check_resources()
        neural_delta = {k: final_metrics_neural[k] - initial_metrics_neural[k] for k in final_metrics_neural}

        row_hybrid = (
            f"{query:<50} | {'Hyb.':<15} | {hybrid_delta['cpu'] * 100:>10.1f} "
            f"| {hybrid_delta['memory'] * 100:>15.1f} | {hybrid_delta['gpu'] * 100:>10.1f} "
            f"| {hybrid_answer}"
        )
        row_neural = (
            f"{query:<50} | {'Neural':<15} | {neural_delta['cpu'] * 100:>10.1f} "
            f"| {neural_delta['memory'] * 100:>15.1f} | {neural_delta['gpu'] * 100:>10.1f} "
            f"| {str(neural_answer)[:150]}..."
        )
        print(row_hybrid)
        print(row_neural)
        print("-" * len(header))

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
    total_queries = performance_stats['total_queries']
    for path, stats in path_stats.items():
        count = stats.get('count', 0)
        percentage = (count / total_queries) * 100 if total_queries > 0 else 0
        print(f"- {path}: {percentage:.1f}%")

    print("\nSystem Information:")
    print(f"Model Path: {neural.model.config._name_or_path}")
    print(f"Cache Location: {os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")
    print("=== End of Run ===")

    # 12. Generate and print academic evaluation report
    academic_report = metrics_collector.generate_academic_report()
    print("\n=== Academic Evaluation Results ===")
    print(json.dumps(academic_report, indent=2))
