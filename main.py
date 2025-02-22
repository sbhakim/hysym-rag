# main.py

from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os
import json
import torch
import warnings
import logging
import time
from collections import defaultdict

# Set up basic logging for all components
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress specific spaCy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.util")
ProgressManager.SHOW_PROGRESS = False  # Globally disable progress bars

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

    ProgressManager.SHOW_PROGRESS = False  # Globally disable progress bars
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('DimensionalityManager').setLevel(logging.INFO)

    # 1. Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config("src/config/config.yaml")
    config.update({  # This config.update is unnecessary as config.yaml is corrected.  Keep it for now, but good to simplify later
        'alignment': {
            'target_dim': 768,
            'num_heads': 4
        }
    })
    model_name = config["model_name"]

    # 2. Acquire a unified device from DeviceManager
    device = DeviceManager.get_device()

    # 3. Initialize Resource Manager
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager(
        config_path="src/config/resource_config.yaml",
        enable_performance_tracking=True,
        history_window_size=100
    )

    # 4. Initialize Dimensionality Manager (NEW - Initialize before Symbolic Reasoner)
    print("Initializing Dimensionality Manager...")
    dimensionality_manager = DimensionalityManager(target_dim=config['alignment']['target_dim'], device=device)

    # 5. Ensure that 'data/rules.json' exists (empty or minimal)
    rules_path = "data/rules.json"
    if not os.path.exists(rules_path):
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    print(f"Loading existing rules from {rules_path} (initially empty or minimal).")

    # 6. Initialize the Graph-Based Symbolic Reasoner (NEW - Pass DimensionalityManager)
    print("Initializing Graph-Based Symbolic Reasoner...")
    try:
        symbolic = GraphSymbolicReasoner(
            rules_file=rules_path,
            match_threshold=0.25,
            max_hops=5,
            embedding_model='all-MiniLM-L6-v2',
            device=device,
            dim_manager=dimensionality_manager # Pass dimensionality_manager here
        )
        if hasattr(symbolic, 'rules') and symbolic.rules:
            print(f"Loaded {len(symbolic.rules)} rules successfully")
        else:
            print("Warning: No rules loaded in symbolic reasoner")
    except Exception as e:
        print(f"Error initializing symbolic reasoner: {str(e)}")
        print("Continuing with empty rule set...")
        symbolic = GraphSymbolicReasoner( # Keep this for fallback
            rules_file=rules_path,
            match_threshold=0.25,
            max_hops=5,
            embedding_model='all-MiniLM-L6-v2',
            device=device,
            dim_manager=dimensionality_manager
        )

    # 7. Initialize the Neural Retriever
    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(
        model_name,
        use_quantization=False,
        device=device
    )

    # 8. Additional components
    print("Initializing support components...")
    logger = logging.getLogger(__name__)
    query_logger = QueryLogger()
    feedback_manager = FeedbackManager()
    print("Initializing QueryExpander...")
    expander = QueryExpander(
        complexity_config="src/config/complexity_rules.yaml"
    )
    print("Initializing RuleExtractor...")
    rule_extractor = RuleExtractor()
    print("Loading evaluation dataset...")


    use_hotpotqa = True
    hotpotqa_path = "data/hotpot_dev_distractor_v1.json"
    max_hotpot_samples = 8

    if use_hotpotqa and os.path.exists(hotpotqa_path):
        test_queries = load_hotpotqa(hotpotqa_path, max_samples=max_hotpot_samples)
        ground_truths = {}
        for i, sample in enumerate(test_queries):
            new_rules = rule_extractor.extract_hotpot_facts(sample["context"], min_confidence=0.7)
            if new_rules:
                try:
                    symbolic.add_dynamic_rules(new_rules)  #This should now work without error
                except AttributeError as e:
                    logger.warning(f"Could not track new rules automatically (missing method?): {str(e)}")
            ground_truths[sample["query"]] = sample["answer"]
    else:
        print("Warning: HotpotQA not found, and no fallback dataset provided.")
        test_queries = []
        ground_truths = {}

    evaluator = Evaluation()

    # 9. Create Hybrid Integrator
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(
        symbolic,
        neural,
        resource_manager,
        expander
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
        max_query_time=10
    )

    # 11. Initialize Application
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

    # 12. Load knowledge base for neural context (if available)
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
        supporting_facts = q_info.get("supporting_facts", None)

        print(f"\nProcessing Query: {query}")
        print(f"Query Type: {data_type}")
        if forced_path:
            print(f"Forced Path: {forced_path}")
        print("-" * 50)
        try:
            initial_time = time.time()
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

            prediction_val = final_answer.get('result', '')
            if isinstance(prediction_val, tuple):
                prediction_val = prediction_val[0]

            metrics_collector.collect_query_metrics(
                query=query,
                prediction=prediction_val,
                ground_truth=the_answer,
                reasoning_path=(
                    symbolic.extract_reasoning_pattern(query, final_answer.get('reasoning_path', []))
                        .get('pattern_type', 'unknown')
                    if hasattr(symbolic, "extract_reasoning_pattern")
                    else 'unknown'
                ),
                processing_time=time.time() - initial_time,
                resource_usage=resource_delta,
                complexity_score=complexity
            )

            if isinstance(final_answer, dict):
                metrics_collector.component_metrics['symbolic']['execution_time'].append(
                    final_answer.get('symbolic_time', 0.0)
                )
                metrics_collector.component_metrics['neural']['execution_time'].append(
                    final_answer.get('neural_time', 0.0)
                )

            query_logger.log_query(
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
                reasoning_chain = symbolic.extract_reasoning_pattern(
                    query,
                    final_answer.get('reasoning_path', [])
                )

                eval_pred_text = final_answer.get('result', '')
                if isinstance(eval_pred_text, tuple):
                    eval_pred_text = eval_pred_text[0]

                eval_metrics = evaluator.evaluate(
                    predictions={query: eval_pred_text},
                    ground_truths={query: the_answer},
                    supporting_facts={query: supporting_facts},
                    reasoning_chain=reasoning_chain
                )
                print("\nEvaluation Metrics:")
                print(f"Similarity Score: {eval_metrics['average_semantic_similarity']:.2f}")
                print(f"ROUGE-L Score: {eval_metrics['average_rougeL']:.2f}")
                print(f"BLEU Score: {eval_metrics['average_bleu']:.2f}")
                print(f"F1 Score: {eval_metrics['average_f1']:.2f}")
                if 'reasoning_analysis' in eval_metrics:
                    print("\nReasoning Analysis:")
                    print(f"Pattern Type: {eval_metrics['reasoning_analysis'].get('pattern_type', 'unknown')}")
                    print(f"Chain Length: {eval_metrics['reasoning_analysis'].get('chain_length', 0)}")
                    print(f"Pattern Confidence: {eval_metrics['reasoning_analysis'].get('pattern_confidence', 0.0):.2f}")

        except KeyError as e:
            print(f"Error: Missing ground truth for query evaluation - {str(e)}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

    # Pass the dimensionality_manager to the ablation study
    ablation_results = run_ablation_study(
        rules_path=rules_path,
        device=device,
        neural=neural,
        expander=expander,
        aggregator=aggregator,
        resource_manager=resource_manager,
        system_manager=system_manager,
        dimensionality_manager=dimensionality_manager,  # Pass dim_manager
        context=context
    )

    metrics_collector.ablation_results = ablation_results

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

    print("\n=== Comprehensive Academic Analysis ===")
    academic_report = metrics_collector.generate_academic_report()

    print("\nPerformance Analysis:")
    if 'performance_metrics' in academic_report:
        perf = academic_report['performance_metrics']
        if 'processing_time' in perf and 'mean' in perf['processing_time']:
            print(f"- Average Processing Time: {perf['processing_time']['mean']:.2f}s")
        if 'processing_time' in perf and 'percentile_95' in perf['processing_time']:
            print(f"- 95th Percentile Time: {perf['processing_time']['percentile_95']:.2f}s")

    print("\nReasoning Analysis:")
    if 'reasoning_analysis' in academic_report:
        ra = academic_report['reasoning_analysis']
        cc = ra.get('chain_characteristics', {})
        print(f"- Average Chain Length: {cc.get('avg_length', 0.0):.2f}")
        print(f"- Average Confidence: {cc.get('avg_confidence', 0.0):.2f}")
        print(f"- Average Inference Depth: {cc.get('avg_inference_depth', 0.0):.2f}")

    print("\nResource Efficiency:")
    if 'efficiency_metrics' in academic_report:
        em = academic_report['efficiency_metrics']
        for resource, metrics in em.items():
            if resource != 'trends':
                print(f"- {resource.capitalize()}:")
                print(f"  * Mean Usage: {metrics.get('mean_usage', 0.0)*100:.1f}%")
                print(f"  * Peak Usage: {metrics.get('peak_usage', 0.0)*100:.1f}%")
                print(f"  * Efficiency Score: {metrics.get('efficiency_score', 0.0):.2f}")

    print("\nStatistical Analysis:")
    if 'statistical_analysis' in academic_report:
        sa = academic_report['statistical_analysis']
        print("Significance Tests:")
        for metric, stats in sa.items():
            if isinstance(stats, dict) and 'p_value' in stats:
                print(f"- {metric}:")
                print(f"  * p-value: {stats['p_value']:.3f}")
                print(f"  * effect size: {stats.get('effect_size', 0.0):.2f}")

    print("\n=== System Performance Summary (Extended) ===")
    pattern_metrics = symbolic.get_reasoning_metrics()
    print(f"- Average Chain Length: {pattern_metrics['path_analysis']['average_length']:.2f}")
    print(f"- Average Confidence: {pattern_metrics['confidence_analysis']['mean_confidence']:.2f}")
    print("\nPattern Distribution:")
    for length, frequency in pattern_metrics['path_analysis']['path_distribution'].items():
        print(f"- {length}-hop paths: {frequency * 100:.1f}%")

    print("\n=== End of Run ===")
    print("\n=== Academic Evaluation Results ===")
    print(json.dumps(academic_report, indent=2))