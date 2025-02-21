# src/ablation_study.py

from collections import defaultdict
from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.system.system_control_manager import SystemControlManager
from src.utils.metrics_collector import MetricsCollector


def run_ablation_study(
    rules_path,
    device,
    neural,
    expander,
    aggregator,
    resource_manager,
    system_manager,  # Pass the original system manager for baseline
    context  # Pass context
):
    """
    Runs the ablation study experiment and returns ablation results.
    """
    print("\n=== Ablation Study ===")
    ablation_results = defaultdict(dict)
    ablation_configs = [
        {'name': 'No Pattern Analysis', 'disable_patterns': True},  # Example - not used in code right now
        {'name': 'Limited Hops', 'max_hops': 2},
        {'name': 'High Threshold', 'match_threshold': 0.5}
    ]
    for config in ablation_configs:
        config_name = config['name']
        print(f"\nTesting Configuration: {config_name}")
        try:
            # 1. Baseline Run (Original System)
            print(f"  Running Baseline...")
            metrics_collector = MetricsCollector()  # Instantiate metrics_collector here for each run
            initial_metrics_baseline = resource_manager.check_resources()
            baseline_answer = system_manager.process_query_with_fallback(
                "What are the environmental effects of deforestation?",
                context
            )
            final_metrics_baseline = resource_manager.check_resources()
            baseline_resource_delta = {k: final_metrics_baseline[k] - initial_metrics_baseline[k] for k in final_metrics_baseline}
            baseline_academic_report = metrics_collector.generate_academic_report()

            # 2. Modified Run (Ablated System)
            print(f"  Running Modified Configuration: {config_name}...")
            metrics_collector_modified = MetricsCollector()  # New MetricsCollector for modified run
            initial_metrics_modified = resource_manager.check_resources()
            # **Important: Create a NEW SystemControlManager with modified symbolic reasoner**
            modified_symbolic = GraphSymbolicReasoner(
                rules_file=rules_path,
                match_threshold=config.get('match_threshold', 0.25),
                max_hops=config.get('max_hops', 5),
                embedding_model='all-MiniLM-L6-v2',
                device=device
            )
            modified_integrator = HybridIntegrator(
                modified_symbolic,
                neural,
                resource_manager,
                expander
            )
            modified_system_manager = SystemControlManager(
                hybrid_integrator=modified_integrator,
                resource_manager=resource_manager,
                aggregator=aggregator,
                metrics_collector=metrics_collector_modified,  # Pass new MetricsCollector
                error_retry_limit=2,
                max_query_time=10
            )

            modified_answer = modified_system_manager.process_query_with_fallback(
                "What are the environmental effects of deforestation?",
                context
            )
            final_metrics_modified = resource_manager.check_resources()
            modified_resource_delta = {k: final_metrics_modified[k] - initial_metrics_modified[k] for k in final_metrics_modified}
            modified_academic_report = modified_system_manager.metrics_collector.generate_academic_report()  # Access metrics from the modified_system_manager

            ablation_results[config_name] = {
                'baseline_report': baseline_academic_report['performance_metrics'],
                'modified_report': modified_academic_report['performance_metrics']
            }

            print("\nPerformance Comparison:")
            print(f"Baseline processing time: {baseline_academic_report['performance_metrics']['processing_time']['mean']:.2f}s")
            print(f"Modified processing time: {modified_academic_report['performance_metrics']['processing_time']['mean']:.2f}s")
            print(f"Ablation Results for {config_name}: {ablation_results[config_name]}")

        except Exception as e:
            print(f"Error in ablation study for {config_name}: {str(e)}")

        return ablation_results  # Return ablation_results


if __name__ == "__main__":
    # Example usage (for testing ablation study in isolation if needed)
    # You'd need to mock or initialize the dependencies for testing here.
    print("This is the ablation study module. It is intended to be called from main.py.")
    pass  # Add example test call if necessary for isolated testing