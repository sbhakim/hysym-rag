# src/ablation_study.py

from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import time
from scipy import stats
import logging

from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
from src.reasoners.networkx_symbolic_reasoner_drop import GraphSymbolicReasonerDrop
from src.reasoners.neural_retriever import NeuralRetriever
from src.integrators.hybrid_integrator import HybridIntegrator
from src.system.system_control_manager import SystemControlManager
from src.system.response_aggregator import UnifiedResponseAggregator
from src.utils.metrics_collector import MetricsCollector
from src.reasoners.dummy_reasoners import DummySymbolicReasoner, DummyNeuralRetriever
from src.queries.query_expander import QueryExpander
from src.resources.resource_manager import ResourceManager
from src.utils.evaluation import Evaluation

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_ablation_study(
    samples: List[Dict[str, Any]],
    dataset_type: str,
    rules_path_baseline: str,
    device: Any,
    neural_retriever_baseline: NeuralRetriever,
    query_expander_baseline: QueryExpander,
    response_aggregator_baseline: UnifiedResponseAggregator,
    resource_manager_baseline: ResourceManager,
    system_manager_baseline: SystemControlManager,
    dimensionality_manager_baseline: Any,
    ablation_configurations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Runs an ablation study for the HySym-RAG system on the specified dataset.

    Args:
        samples: List of query samples, each containing 'query', 'context', 'answer', 'query_id', and 'type'.
        dataset_type: Dataset type ('drop' or 'hotpotqa').
        rules_path_baseline: Path to the baseline rules file.
        device: Device for computation (e.g., torch.device).
        neural_retriever_baseline: Baseline NeuralRetriever instance.
        query_expander_baseline: Baseline QueryExpander instance.
        response_aggregator_baseline: Baseline UnifiedResponseAggregator instance.
        resource_manager_baseline: Baseline ResourceManager instance.
        system_manager_baseline: Baseline SystemControlManager instance.
        dimensionality_manager_baseline: Baseline DimensionalityManager instance.
        ablation_configurations: List of ablation configurations from ablation_config.yaml.

    Returns:
        Dict containing ablation results and statistical analysis.
    """
    print(f"\n=== Enhanced Ablation Study for Dataset: {dataset_type.upper()} ===")
    ablation_results = defaultdict(dict)

    # Define consistent evaluation metrics
    eval_metrics = ['exact_match', 'f1', 'processing_time', 'resource_usage']

    # Run baseline first to establish reference performance
    print("Running baseline configuration...")
    baseline_metrics = _run_configuration(
        "1. Baseline Hybrid (Dynamic Rules, Few-Shots)",
        samples,
        system_manager_baseline,
        resource_manager_baseline,
        dataset_type
    )

    # Define ablation configurations (already passed as argument)
    ablation_configs = ablation_configurations

    # Run each configuration and compare to baseline
    for config in ablation_configs:
        config_name = config['name']
        print(f"\nTesting Configuration: {config_name}")

        try:
            # Reset paired results for this ablation
            paired_results = defaultdict(lambda: defaultdict(list))

            modified_metrics = _run_configuration_with_params(
                config_name,
                config,
                samples,
                rules_path_baseline,
                device,
                neural_retriever_baseline,
                query_expander_baseline,
                response_aggregator_baseline,
                resource_manager_baseline,
                dimensionality_manager_baseline,
                dataset_type
            )

            # Store results
            ablation_results[config_name] = {
                'baseline': baseline_metrics,
                'modified': modified_metrics
            }

            # Calculate relative changes for each metric
            relative_changes = {}
            for metric in eval_metrics:
                if metric in baseline_metrics and metric in modified_metrics:
                    base_val = baseline_metrics[metric]['mean']
                    mod_val = modified_metrics[metric]['mean']
                    if abs(base_val) > 1e-9:  # Avoid division by zero
                        rel_change = (mod_val - base_val) / base_val
                        relative_changes[metric] = rel_change

                    # Store for paired significance testing (scoped to this ablation)
                    for qid in baseline_metrics.get('query_specific', {}):
                        if qid in modified_metrics.get('query_specific', {}):
                            paired_results[metric]['baseline'].append(baseline_metrics['query_specific'][qid].get(metric, 0))
                            paired_results[metric]['modified'].append(modified_metrics['query_specific'][qid].get(metric, 0))

            ablation_results[config_name]['relative_changes'] = relative_changes

            # Calculate statistical significance for this ablation vs. baseline
            significance_results = {}
            for metric, data in paired_results.items():
                if len(data['baseline']) >= 2 and len(data['modified']) >= 2 and len(data['baseline']) == len(data['modified']):
                    try:
                        t_stat, p_value = stats.ttest_rel(data['baseline'], data['modified'])
                        mean_diff = np.mean([b - m for b, m in zip(data['baseline'], data['modified'])])
                        std_diff = np.std([b - m for b, m in zip(data['baseline'], data['modified'])], ddof=1)
                        effect_size = mean_diff / std_diff if std_diff != 0 else 0.0
                        diffs = [b - m for b, m in zip(data['baseline'], data['modified'])]
                        ci_lower, ci_upper = stats.t.interval(
                            alpha=0.95,
                            df=len(diffs) - 1,
                            loc=np.mean(diffs),
                            scale=stats.sem(diffs)
                        )
                        significance_results[metric] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'effect_size': float(effect_size),
                            'ci_lower': float(ci_lower),
                            'ci_upper': float(ci_upper),
                            'significant': p_value < 0.05
                        }
                    except Exception as t_test_e:
                        significance_results[metric] = {
                            'error': f"Error calculating t-test: {str(t_test_e)}",
                            't_statistic': None,
                            'p_value': None,
                            'effect_size': None,
                            'ci_lower': None,
                            'ci_upper': None,
                            'significant': False
                        }
                else:
                    significance_results[metric] = {
                        'error': "Insufficient or mismatched data points for t-test",
                        't_statistic': None,
                        'p_value': None,
                        'effect_size': None,
                        'ci_lower': None,
                        'ci_upper': None,
                        'significant': False
                    }

            ablation_results[config_name]['statistical_significance_vs_baseline'] = significance_results

            # Print summary
            print(f"\nAblation Results for {config_name}:")
            for metric, change in relative_changes.items():
                print(f"  {metric}: {change * 100:.1f}% change from baseline")
            print("  Statistical Significance vs. Baseline:")
            for metric, stats in significance_results.items():
                if 'error' in stats:
                    print(f"    {metric}: {stats['error']}")
                else:
                    print(f"    {metric}: p-value={stats['p_value']:.3g}, effect_size={stats['effect_size']:.2f}, "
                          f"CI=[{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}], "
                          f"significant={stats['significant']}")

        except Exception as e:
            print(f"Error in ablation study for {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    return {
        'ablation_results': ablation_results,
        'paired_data': {}  # No longer needed globally, but kept for compatibility
    }

def _run_configuration(
    name: str,
    samples: List[Dict[str, Any]],
    system_manager: SystemControlManager,
    resource_manager: ResourceManager,
    dataset_type: str
) -> Dict[str, Any]:
    """
    Runs a single configuration and collects metrics.

    Args:
        name: Configuration name.
        samples: List of query samples.
        system_manager: SystemControlManager instance.
        resource_manager: ResourceManager instance.
        dataset_type: Dataset type ('drop' or 'hotpotqa').

    Returns:
        Dict with aggregated metrics and query-specific results.
    """
    results = {}
    query_specific = {}
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=f"logs/{dataset_type}/metrics_ablation_{name.replace(' ', '_').lower()}"
    )
    evaluator = Evaluation(dataset_type=dataset_type)

    for sample in samples:
        query = sample.get("query")
        context = sample.get("context", "")
        ground_truth = sample.get("answer")
        query_id = sample.get("query_id", "unknown")
        query_type = sample.get("type", "")

        print(f"  Testing query (QID: {query_id}): {query[:50]}...")
        initial_resources = resource_manager.check_resources()
        start_time = time.time()

        response_data, reasoning_path = system_manager.process_query_with_fallback(
            query=query,
            context=context,
            query_id=query_id,
            forced_path=None,
            query_complexity=system_manager.hybrid_integrator.query_expander.get_query_complexity(query),
            supporting_facts=None,
            dataset_type=dataset_type
        )

        processing_time = time.time() - start_time
        final_resources = resource_manager.check_resources()
        resource_delta = {k: final_resources[k] - initial_resources[k] for k in final_resources}

        # Evaluate response
        prediction = response_data.get('result')
        eval_metrics = evaluator.evaluate(
            predictions={query_id: prediction},
            ground_truths={query_id: ground_truth}
        )

        exact_match = eval_metrics.get('average_exact_match', 0.0)
        f1_score = eval_metrics.get('average_f1', 0.0)

        query_specific[query_id] = {
            'exact_match': exact_match,
            'f1': f1_score,
            'processing_time': processing_time,
            'resource_usage': sum(resource_delta.values()),  # Sum of deltas as a simple metric
            'reasoning_path': reasoning_path
        }

        metrics_collector.collect_query_metrics(
            query=query,
            prediction=prediction,
            ground_truth=ground_truth,
            reasoning_path=reasoning_path,
            processing_time=processing_time,
            resource_usage=resource_delta,
            complexity_score=system_manager.hybrid_integrator.query_expander.get_query_complexity(query),
            query_id=query_id
        )

    academic_report = metrics_collector.generate_academic_report()

    # Aggregate metrics
    results['exact_match'] = {
        'mean': np.mean([q['exact_match'] for q in query_specific.values()]),
        'std': np.std([q['exact_match'] for q in query_specific.values()])
    }
    results['f1'] = {
        'mean': np.mean([q['f1'] for q in query_specific.values()]),
        'std': np.std([q['f1'] for q in query_specific.values()])
    }
    results['processing_time'] = {
        'mean': np.mean([q['processing_time'] for q in query_specific.values()]),
        'std': np.std([q['processing_time'] for q in query_specific.values()])
    }
    results['resource_usage'] = {
        'mean': np.mean([q['resource_usage'] for q in query_specific.values()]),
        'std': np.std([q['resource_usage'] for q in query_specific.values()]),
        'trends': academic_report.get('efficiency_metrics', {}).get('trends', {})
    }

    results['query_specific'] = query_specific
    results['academic_report'] = academic_report

    return results

def _run_configuration_with_params(
    config_name: str,
    config: Dict[str, Any],
    samples: List[Dict[str, Any]],
    rules_path_baseline: str,
    device: Any,
    neural_retriever: NeuralRetriever,
    query_expander: QueryExpander,
    response_aggregator: UnifiedResponseAggregator,
    resource_manager: ResourceManager,
    dimensionality_manager: Any,
    dataset_type: str
) -> Dict[str, Any]:
    """
    Runs a configuration with modified parameters.

    Args:
        config_name: Configuration name.
        config: Configuration dictionary with parameters.
        samples: List of query samples.
        rules_path_baseline: Baseline rules path.
        device: Device for computation.
        neural_retriever: Baseline NeuralRetriever instance.
        query_expander: QueryExpander instance.
        response_aggregator: UnifiedResponseAggregator instance.
        resource_manager: ResourceManager instance.
        dimensionality_manager: DimensionalityManager instance.
        dataset_type: Dataset type ('drop' or 'hotpotqa').

    Returns:
        Dict with metrics for the modified configuration.
    """
    print(f"Running modified configuration: {config_name}...")
    modified_metrics = _run_configuration(
        config_name,
        samples,
        _create_modified_system_manager(
            config,
            rules_path_baseline,
            device,
            neural_retriever,
            query_expander,
            response_aggregator,
            resource_manager,
            dimensionality_manager,
            dataset_type
        ),
        resource_manager,
        dataset_type
    )
    return modified_metrics

def _create_modified_system_manager(
    config: Dict[str, Any],
    rules_path_baseline: str,
    device: Any,
    neural_retriever: NeuralRetriever,
    query_expander: QueryExpander,
    response_aggregator: UnifiedResponseAggregator,
    resource_manager: ResourceManager,
    dimensionality_manager: Any,
    dataset_type: str
) -> SystemControlManager:
    """
    Creates a modified SystemControlManager for an ablation configuration.

    Args:
        config: Configuration dictionary with ablation parameters.
        rules_path_baseline: Baseline rules path.
        device: Device for computation.
        neural_retriever: Baseline NeuralRetriever instance.
        query_expander: QueryExpander instance.
        response_aggregator: UnifiedResponseAggregator instance.
        resource_manager: ResourceManager instance.
        dimensionality_manager: DimensionalityManager instance.
        dataset_type: Dataset type ('drop' or 'hotpotqa').

    Returns:
        Modified SystemControlManager instance.
    """
    # Determine rules file to use
    rules_file = config.get('rules_file', rules_path_baseline)

    # Get symbolic reasoner parameters from config
    sym_match_threshold = config.get('match_threshold', 0.1 if dataset_type.lower() == 'drop' else 0.25)
    sym_max_hops = config.get('max_hops', 3 if dataset_type.lower() == 'drop' else 5)
    embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')

    # Initialize symbolic reasoner
    symbolic_reasoner: Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop, DummySymbolicReasoner]
    if dataset_type.lower() == 'drop':
        symbolic_reasoner = GraphSymbolicReasonerDrop(
            rules_file=rules_file,
            match_threshold=sym_match_threshold,
            max_hops=sym_max_hops,
            embedding_model=embedding_model_name,
            device=device,
            dim_manager=dimensionality_manager
        )
    else:
        symbolic_reasoner = GraphSymbolicReasoner(
            rules_file=rules_file,
            match_threshold=sym_match_threshold,
            max_hops=sym_max_hops,
            embedding_model=embedding_model_name,
            device=device,
            dim_manager=dimensionality_manager
        )

    # Initialize neural retriever with few-shot setting
    use_few_shots = config.get('use_few_shots', True)  # Default to True if not specified
    modified_neural = neural_retriever
    if 'use_few_shots' in config:
        modified_neural = NeuralRetriever(
            model_name=neural_retriever.model_name,
            use_quantization=neural_retriever.use_quantization,
            max_context_length=neural_retriever.max_context_length,
            chunk_size=neural_retriever.chunk_size,
            overlap=neural_retriever.overlap,
            device=device,
            few_shot_examples_path=neural_retriever.few_shot_examples_path if use_few_shots else None
        )

    # Create hybrid integrator
    modified_integrator = HybridIntegrator(
        symbolic_reasoner=symbolic_reasoner,
        neural_retriever=modified_neural,
        query_expander=query_expander,
        dim_manager=dimensionality_manager,
        dataset_type=dataset_type
    )

    # Create modified system manager
    modified_system_manager = SystemControlManager(
        hybrid_integrator=modified_integrator,
        resource_manager=resource_manager,
        aggregator=response_aggregator,
        metrics_collector=MetricsCollector(
            dataset_type=dataset_type,
            metrics_dir=f"logs/{dataset_type}/metrics_ablation_{config['name'].replace(' ', '_').lower()}"
        ),
        error_retry_limit=2,
        max_query_time=30.0
    )

    # Apply dummy reasoners for ablation
    if config.get('disable_neural', False):
        modified_system_manager.hybrid_integrator.neural_retriever = DummyNeuralRetriever()
    if config.get('disable_symbolic', False):
        modified_system_manager.hybrid_integrator.symbolic_reasoner = DummySymbolicReasoner()

    return modified_system_manager

if __name__ == "__main__":
    print("This is the ablation study module. It is intended to be called from main.py.")