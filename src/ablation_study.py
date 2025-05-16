# src/ablation_study.py

from collections import defaultdict
from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.system.system_control_manager import SystemControlManager, UnifiedResponseAggregator
from src.utils.metrics_collector import MetricsCollector
from src.reasoners.dummy_reasoners import DummySymbolicReasoner, DummyNeuralRetriever
from src.config.config_loader import ConfigLoader
from src.queries.query_expander import QueryExpander
from src.resources.resource_manager import ResourceManager
import numpy as np
import time
from scipy import stats  # Import scipy.stats for t-tests
from src.utils.evaluation import Evaluation # Import Evaluation

def run_ablation_study(
        rules_path,
        device,
        neural,
        expander,
        aggregator,
        resource_manager,
        system_manager,
        dimensionality_manager,
        context
):
    print("\n=== Enhanced Ablation Study ===")
    ablation_results = defaultdict(dict)

    # Define standard test queries that cover different complexities
    test_queries = [
        "What are the environmental effects of deforestation?",  # Simple query
        "How does deforestation impact both biodiversity and climate change?",  # Multi-aspect query
        "Compare the effects of deforestation in tropical versus temperate forests"  # Complex comparison query
    ]

    # Define consistent evaluation metrics across configurations
    eval_metrics = ['processing_time', 'resource_usage', 'response_quality'] # response_quality is placeholder, define how to measure

    # Track paired results for statistical significance testing
    paired_results = defaultdict(lambda: defaultdict(list))

    # Run baseline first to establish reference performance
    print("Running baseline configuration...")
    baseline_metrics = _run_configuration(
        "Baseline",
        test_queries,
        system_manager,
        resource_manager,
        context,
        neural, # Pass neural retriever for dummy replacement in configurations
        expander,
        aggregator,
        dimensionality_manager,
        rules_path,
        device
    )
    # Fix #1: Define baseline_metrics_full_hybrid here
    baseline_metrics_full_hybrid = baseline_metrics.get('academic_report', {}).get('performance_metrics', {})


    # Define ablation configurations with more meaningful parameters
    ablation_configs = [
        {'name': 'No Pattern Analysis', 'disable_patterns': True},
        {'name': 'Limited Hops (2)', 'max_hops': 2},
        {'name': 'High Threshold (0.5)', 'match_threshold': 0.5},
        {'name': 'Symbolic Only', 'disable_neural': True, 'match_threshold': 0.25},
        {'name': 'Neural Only', 'disable_symbolic': True},
        {'name': 'Hybrid', 'disable_symbolic': False, 'disable_neural': False} # Explicit Hybrid config
    ]

    # Run each configuration and compare to baseline
    for config in ablation_configs:
        config_name = config['name']
        print(f"\nTesting Configuration: {config_name}")

        try:
            # Fix #2: Use _run_configuration_with_params to run modified configurations
            modified_metrics = _run_configuration_with_params(
                config_name,
                config,
                test_queries,
                rules_path,
                device,
                neural,
                expander,
                aggregator,
                resource_manager,
                dimensionality_manager,
                context
            )

            # Extract needed metrics from the results (for clarity)
            modified_perf = modified_metrics.get('academic_report', {}).get('performance_metrics', {})
            processing_time = modified_metrics.get('processing_time', {}).get('mean', 0.0)
            modified_resource_delta = modified_metrics.get('resource_usage', {})


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

                    # Store for paired significance testing
                    for query, values in baseline_metrics.get('query_specific', {}).items():
                        paired_results[metric]['baseline'].append(values.get(metric, 0))
                    for query, values in modified_metrics.get('query_specific', {}).items():
                        paired_results[metric]['modified'].append(values.get(metric, 0))

            ablation_results[config_name]['relative_changes'] = relative_changes

            # Print summary
            print(f"\nAblation Results for {config['name']}:")
            for metric, change in relative_changes.items():
                print(f"  {metric}: {change * 100:.1f}% change from baseline")

        except Exception as e:
            print(f"Error in ablation study for {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Calculate statistical significance where possible
    significance_results = {}
    for metric, data in paired_results.items():
        # ADD THIS CHECK: Ensure both lists have enough data points
        if len(data['baseline']) >= 2 and len(data['modified']) >= 2:  # Changed from 3 to 2 for minimal data points
            from scipy import stats
            try:  # Add try-except block for robustness
                t_stat, p_value = stats.ttest_rel(data['baseline'], data['modified'])
                significance_results[metric] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as t_test_e:  # Catch potential t-test errors
                significance_results[metric] = {
                    'error': f"Error calculating t-test: {str(t_test_e)}",
                    't_statistic': None,
                    'p_value': None,
                    'significant': False
                }
        else:  # Handle cases with insufficient data points
            significance_results[metric] = {
                'error': "Insufficient data points for t-test",
                't_statistic': None,
                'p_value': None,
                'significant': False
            }

    ablation_results['statistical_significance'] = significance_results

    return {
        'ablation_results': ablation_results,
        'paired_data': paired_results
    }

# Helper function to run a single configuration
def _run_configuration(name, queries, system_manager, resource_manager, context, neural, expander, aggregator, dimensionality_manager, rules_path, device): # Added neural, expander etc.
    results = {}
    query_specific = {}
    metrics_collector = MetricsCollector()  # Use a MetricsCollector for each configuration

    for query in queries:
        print(f"  Testing query: {query[:50]}...") # Increased query[:50] for better readability
        initial_resources = resource_manager.check_resources()
        start_time = time.time()

        response_data = system_manager.process_query_with_fallback(query, context) # Get full response data
        response_text = response_data.get('result', '') # Extract result text
        if isinstance(response_text, tuple): # Handle tuple results
            response_text = response_text[0]

        processing_time = time.time() - start_time
        final_resources = resource_manager.check_resources()
        resource_delta = {k: final_resources[k] - initial_resources[k] for k in final_resources}

        # Evaluate response quality (using F1 score as example - refine as needed)
        eval_metrics = Evaluation().evaluate(
            predictions={query: response_text},
            ground_truths={} # No ground truths in ablation study, modify if needed
        )
        response_quality = eval_metrics.get('average_f1', 0.0) # Use F1 as response quality, refine if needed

        query_specific[query] = {
            'processing_time': processing_time,
            'resource_usage': resource_delta,
            'response_quality': response_quality # Capture response quality metric
        }

        metrics_collector.collect_query_metrics( # Collect metrics for academic report
            query=query,
            prediction=response_text,
            ground_truth=None,
            reasoning_path="ablation",
            processing_time=processing_time,
            resource_usage=resource_delta,
            complexity_score=0.0 # Complexity not relevant here, or calculate if needed
        )

    academic_report = metrics_collector.generate_academic_report() # Generate report *after* processing all queries

    # Calculate aggregate metrics
    results['processing_time'] = {
        'mean': np.mean([q['processing_time'] for q in query_specific.values()]),
        'std': np.std([q['processing_time'] for q in query_specific.values()])
    }
    results['resource_usage'] = { # Aggregate resource usage metrics
        'mean': np.mean([sum(q['resource_usage'].values()) for q in query_specific.values()]), # Example: mean of sum of resource deltas
        'std': np.std([sum(q['resource_usage'].values()) for q in query_specific.values()]),
        'trends': academic_report.get('efficiency_metrics', {}).get('trends', {}) # Capture resource trends from report
    }
    results['response_quality'] = { # Aggregate response quality metrics
        'mean': np.mean([q['response_quality'] for q in query_specific.values()]), # Mean F1 score
        'std': np.std([q['response_quality'] for q in query_specific.values()])
    }


    # Add query-specific data
    results['query_specific'] = query_specific
    results['academic_report'] = academic_report # Attach the full academic report for more detailed analysis

    return results


# Helper function to run a configuration with modified parameters
def _run_configuration_with_params(config_name, config, queries, rules_path, device, neural, expander, aggregator, resource_manager, dimensionality_manager, context):
    print(f"Running modified configuration: {config_name}...")
    modified_metrics = _run_configuration(
        config_name,
        queries,
        _create_modified_system_manager(
            config,
            rules_path,
            device,
            neural,
            expander,
            aggregator,
            resource_manager,
            dimensionality_manager
        ),
        resource_manager,
        context,
        neural, # Pass neural retriever
        expander,
        aggregator,
        dimensionality_manager,
        rules_path,
        device
    )
    return modified_metrics


# Helper function to create modified system manager for each configuration
def _create_modified_system_manager(config, rules_path, device, neural, expander, aggregator, resource_manager, dimensionality_manager):
    modified_symbolic = GraphSymbolicReasoner(
        rules_file=rules_path,
        match_threshold=config.get('match_threshold', 0.25),
        max_hops=config.get('max_hops', 5),
        embedding_model='all-MiniLM-L6-v2',
        device=device,
        dim_manager=dimensionality_manager
    )
    modified_integrator = HybridIntegrator(
        modified_symbolic,
        neural, # Use the original neural retriever
        resource_manager,
        expander,
        dim_manager=dimensionality_manager
    )
    modified_aggregator = UnifiedResponseAggregator(include_explanations=True)

    modified_system_manager = SystemControlManager(
        hybrid_integrator=modified_integrator,
        resource_manager=resource_manager,
        aggregator=modified_aggregator,
        metrics_collector=MetricsCollector(), # Use a fresh MetricsCollector for each config
        error_retry_limit=2,
        max_query_time=10
    )

    # Apply Dummy Reasoners for ablation
    if config.get('disable_neural', False):
        modified_system_manager.hybrid_integrator.neural_retriever = DummyNeuralRetriever()  # Fixed property name
    if config.get('disable_symbolic', False):
        modified_system_manager.hybrid_integrator.symbolic_reasoner = DummySymbolicReasoner()

    return modified_system_manager


if __name__ == "__main__":
    print("This is the ablation study module. It is intended to be called from main.py.")