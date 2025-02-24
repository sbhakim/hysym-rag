# src/ablation_study.py

from collections import defaultdict
from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.system.system_control_manager import SystemControlManager, UnifiedResponseAggregator # Import UnifiedResponseAggregator
from src.utils.metrics_collector import MetricsCollector
from src.reasoners.dummy_reasoners import DummySymbolicReasoner, DummyNeuralRetriever  # Import dummy classes
from src.config.config_loader import ConfigLoader # Import ConfigLoader
from src.queries.query_expander import QueryExpander # Import QueryExpander
from src.resources.resource_manager import ResourceManager # Import ResourceManager

def run_ablation_study(
        rules_path,
        device,
        neural,
        expander,
        aggregator,
        resource_manager,
        system_manager,  # Original system manager for baseline run
        dimensionality_manager,  # Accept DimensionalityManager as argument
        context  # Pass context
):
    """
    Runs the ablation study experiment and returns ablation results.
    Enhanced to include additional configurations and detailed metrics.
    """
    print("\n=== Ablation Study ===")
    ablation_results = defaultdict(dict)
    performance_metrics = defaultdict(lambda: defaultdict(list))
    query = "What are the environmental effects of deforestation?"

    ablation_configs = [
        {'name': 'No Pattern Analysis', 'disable_patterns': True},
        {'name': 'Limited Hops', 'max_hops': 2},
        {'name': 'High Threshold', 'match_threshold': 0.5},
        {'name': 'Symbolic Only', 'disable_neural': True, 'match_threshold': 0.25},
        {'name': 'Neural Only', 'disable_symbolic': True},
        {'name': 'Hybrid', 'disable_symbolic': False, 'disable_neural': False} # Explicit Hybrid config
    ]

    baseline_metrics_full_hybrid = None # Store baseline metrics outside the loop

    for config in ablation_configs:
        config_name = config['name']
        print(f"\nTesting Configuration: {config_name}")
        try:
            # 1. Baseline Run (Full Hybrid - run only once before the loop)
            if baseline_metrics_full_hybrid is None: # Check if baseline is already computed
                print("  Running Baseline (Full Hybrid)...")
                baseline_metrics_collector = MetricsCollector() # Use separate collector for baseline
                initial_metrics_baseline = resource_manager.check_resources()

                # Use the original system_manager for the baseline run
                baseline_answer = system_manager.process_query_with_fallback(query, context)
                final_metrics_baseline = resource_manager.check_resources()
                baseline_resource_delta = {k: final_metrics_baseline[k] - initial_metrics_baseline[k]
                                           for k in final_metrics_baseline}
                baseline_academic_report = baseline_metrics_collector.generate_academic_report() # Use baseline collector
                baseline_metrics_full_hybrid = baseline_academic_report.get('performance_metrics', {}) # Store baseline metrics


            modified_metrics_collector = MetricsCollector()
            initial_metrics_modified = resource_manager.check_resources()

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
            modified_aggregator = UnifiedResponseAggregator(include_explanations=True) # Use separate aggregator

            modified_system_manager = SystemControlManager(
                hybrid_integrator=modified_integrator,
                resource_manager=resource_manager,
                aggregator=modified_aggregator, # Use modified aggregator
                metrics_collector=modified_metrics_collector, # Use modified metrics collector
                error_retry_limit=2,
                max_query_time=10
            )

            # Disable modalities using dummy classes if configured
            if config.get('disable_neural', False):
                modified_system_manager.hybrid_integrator.neural = DummyNeuralRetriever() # Use .neural, not .neural_retriever
            if config.get('disable_symbolic', False):
                modified_system_manager.hybrid_integrator.symbolic_reasoner = DummySymbolicReasoner()

            modified_answer = modified_system_manager.process_query_with_fallback(query, context)
            final_metrics_modified = resource_manager.check_resources()
            modified_resource_delta = {k: final_metrics_modified[k] - initial_metrics_modified[k]
                                       for k in final_metrics_modified}
            modified_academic_report = modified_system_manager.metrics_collector.generate_academic_report() # Use modified metrics collector
            modified_perf = modified_academic_report.get('performance_metrics', {})

            # Track detailed metrics
            performance_metrics[config_name].update({
                'processing_time': modified_perf.get('processing_time', {}),
                'accuracy': modified_perf.get('average_f1', 0.0), # Capture F1 score
                'resource_usage': modified_resource_delta,
                'reasoning_quality': modified_academic_report.get('reasoning_analysis', {})
            })

            ablation_results[config_name] = {
                'baseline_report': baseline_metrics_full_hybrid, # Use pre-computed baseline metrics
                'modified_report': modified_perf
            }

            print("\nPerformance Comparison:")
            baseline_time = baseline_metrics_full_hybrid.get('processing_time', {}).get('mean', 0.0) # Get baseline time from stored metrics
            modified_time = modified_perf.get('processing_time', {}).get('mean', 0.0)
            print(f"Baseline processing time (Hybrid): {baseline_time:.2f}s") # Indicate baseline is Hybrid
            print(f"Modified processing time ({config_name}): {modified_time:.2f}s")
            print(f"Ablation Results for {config_name}: {ablation_results[config_name]}")

        except Exception as e:
            print(f"Error in ablation study for {config_name}: {str(e)}")

    return {
        'ablation_results': ablation_results,
        'performance_metrics': dict(performance_metrics)
    }


if __name__ == "__main__":
    print("This is the ablation study module. It is intended to be called from main.py.")