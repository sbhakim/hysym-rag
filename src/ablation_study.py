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
logger.setLevel(logging.INFO)  # Changed to INFO, set to DEBUG if more verbose logs are needed for this file


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
    # Note: The name "1. Baseline Hybrid (Dynamic Rules, Few-Shots)" is hardcoded here.
    # This assumes the first configuration in ablation_configurations or a specific one is always the baseline.
    # If the baseline is not explicitly run as part of ablation_configurations,
    # ensure this baseline_metrics is truly representative of the intended baseline setup.
    baseline_metrics = _run_configuration(
        name="Baseline_Reference_Run",  # Giving a distinct name for clarity
        samples=samples,
        system_manager=system_manager_baseline,  # Use the passed baseline SCM
        resource_manager=resource_manager_baseline,
        dataset_type=dataset_type
    )
    logger.info(f"Baseline metrics established: {baseline_metrics.keys()}")

    # Run each configuration and compare to baseline
    for config_idx, config in enumerate(ablation_configurations):
        config_name = config['name']
        print(f"\n--- Testing Ablation Configuration {config_idx + 1}/{len(ablation_configurations)}: {config_name} ---")

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
            logger.info(f"Metrics for modified config '{config_name}' obtained: {modified_metrics.keys()}")

            # Store results
            ablation_results[config_name] = {
                'baseline': baseline_metrics,
                'modified': modified_metrics
            }

            # Calculate relative changes for each metric
            relative_changes = {}
            for metric in eval_metrics:
                if metric in baseline_metrics and metric in modified_metrics and \
                        isinstance(baseline_metrics.get(metric), dict) and \
                        isinstance(modified_metrics.get(metric), dict) and \
                        'mean' in baseline_metrics[metric] and 'mean' in modified_metrics[metric]:

                    base_val = baseline_metrics[metric]['mean']
                    mod_val = modified_metrics[metric]['mean']

                    if isinstance(base_val, (int, float)) and isinstance(mod_val, (int, float)):
                        if abs(base_val) > 1e-9:  # Avoid division by zero
                            rel_change = (mod_val - base_val) / base_val
                            relative_changes[metric] = rel_change
                        elif mod_val == 0 and base_val == 0:  # Both zero, no change
                            relative_changes[metric] = 0.0
                        else:  # Base is near zero, mod is not
                            relative_changes[metric] = float('inf') if mod_val > 0 else float(
                                '-inf') if mod_val < 0 else 0.0
                    else:
                        logger.warning(
                            f"Metric '{metric}' mean values are not numeric for relative change. Base: {base_val}, Mod: {mod_val}")

                    # Store for paired significance testing (scoped to this ablation)
                    baseline_query_specific = baseline_metrics.get('query_specific')
                    modified_query_specific = modified_metrics.get('query_specific')

                    if isinstance(baseline_query_specific, dict) and isinstance(modified_query_specific, dict):
                        valid_pairs_for_metric = 0
                        for qid in baseline_query_specific:
                            if qid in modified_query_specific:
                                baseline_q_metric = baseline_query_specific[qid].get(metric)
                                modified_q_metric = modified_query_specific[qid].get(metric)

                                if isinstance(baseline_q_metric, (int, float)) and \
                                        isinstance(modified_q_metric, (int, float)) and \
                                        np.isfinite(baseline_q_metric) and np.isfinite(
                                    modified_q_metric):  # Ensure finite
                                    paired_results[metric]['baseline'].append(baseline_q_metric)
                                    paired_results[metric]['modified'].append(modified_q_metric)
                                    valid_pairs_for_metric += 1
                                else:
                                    logger.debug(
                                        f"Metric '{metric}' for QID '{qid}' missing, non-numeric, or non-finite. Baseline: {baseline_q_metric}, Mod: {modified_q_metric}. Skipping for paired test.")
                        logger.info(
                            f"Collected {valid_pairs_for_metric} valid paired results for metric '{metric}' for t-test.")
                    else:
                        logger.warning(
                            f"Query-specific results missing or not dicts for metric '{metric}'. Baseline type: {type(baseline_query_specific)}, Modified type: {type(modified_query_specific)}")

            ablation_results[config_name]['relative_changes'] = relative_changes

            # Calculate statistical significance for this ablation vs. baseline
            significance_results = {}
            for metric, data in paired_results.items():
                logger.info(f"Calculating significance for metric: {metric}")
                logger.debug(
                    f"Paired Baseline Data ({len(data['baseline'])} points) for {metric}: {str(data['baseline'])[:200]}...")
                logger.debug(
                    f"Paired Modified Data ({len(data['modified'])} points) for {metric}: {str(data['modified'])[:200]}...")

                if len(data['baseline']) >= 2 and \
                        len(data['modified']) >= 2 and \
                        len(data['baseline']) == len(data['modified']):
                    try:
                        # Data should already be cleaned of non-finites when populating paired_results
                        clean_baseline = data['baseline']
                        clean_modified = data['modified']

                        # Additional check just in case
                        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in clean_baseline) or \
                                not all(isinstance(x, (int, float)) and np.isfinite(x) for x in clean_modified):
                            logger.error(
                                f"Data for metric '{metric}' still contains non-finite values before t-test. This should not happen.")
                            raise ValueError(f"Data for metric '{metric}' contains non-finite values.")

                        logger.debug(
                            f"Cleaned Baseline Data for t-test ({len(clean_baseline)} points) for {metric}: {clean_baseline[:10]}...")
                        logger.debug(
                            f"Cleaned Modified Data for t-test ({len(clean_modified)} points) for {metric}: {clean_modified[:10]}...")

                        t_stat, p_value = stats.ttest_rel(clean_baseline, clean_modified)
                        logger.debug(f"Metric: {metric}, t_stat={t_stat}, p_value={p_value}")

                        diffs = [b - m for b, m in zip(clean_baseline, clean_modified)]
                        if not diffs:
                            logger.error(f"Differences list is empty for metric {metric} despite data length checks.")
                            raise ValueError("Differences list is empty after zipping cleaned data.")

                        logger.debug(f"Differences ({len(diffs)} points) for {metric}: {diffs[:10]}...")

                        mean_diff = np.mean(diffs)
                        std_diff = np.std(diffs, ddof=1)
                        logger.debug(f"Metric: {metric}, mean_diff={mean_diff}, std_diff={std_diff}")

                        if std_diff == 0 or not np.isfinite(std_diff):  # Check for NaN/Inf std_diff as well
                            effect_size = 0.0 if mean_diff == 0 else (
                                np.inf * np.sign(mean_diff) if np.isfinite(mean_diff) else 0.0)
                            logger.warning(
                                f"Std_diff is zero or non-finite for metric {metric}. Effect size set to {effect_size}.")
                        else:
                            effect_size = mean_diff / std_diff
                        logger.debug(f"Metric: {metric}, effect_size={effect_size}")

                        current_sem = 0.0
                        df_for_ci = len(diffs) - 1

                        if df_for_ci > 0:
                            sem_val = stats.sem(diffs, nan_policy='propagate')
                            current_sem = 0.0 if np.isnan(sem_val) else sem_val
                            logger.debug(f"Metric: {metric}, sem_val={sem_val}, current_sem={current_sem}")
                        else:
                            logger.warning(
                                f"Metric: {metric}, Not enough data points in diffs for SEM and CI calculation (len: {len(diffs)}). df_for_ci={df_for_ci}")

                        if df_for_ci > 0 and current_sem >= 0 and np.isfinite(current_sem) and np.isfinite(mean_diff):
                            ci_lower, ci_upper = stats.t.interval(
                                confidence=0.95,  # Changed from alpha to confidence
                                df=df_for_ci,
                                loc=mean_diff,
                                scale=current_sem
                            )
                            logger.debug(f"Metric: {metric}, ci_lower={ci_lower}, ci_upper={ci_upper}")
                        else:
                            ci_lower, ci_upper = (mean_diff, mean_diff) if np.isfinite(mean_diff) else (None, None)
                            logger.warning(
                                f"Metric: {metric}, Using mean_diff for CI due to invalid df/scale. df={df_for_ci}, scale={current_sem}, mean_diff={mean_diff}")

                        significance_results[metric] = {
                            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
                            'p_value': float(p_value) if not np.isnan(p_value) else None,
                            'effect_size': float(effect_size) if np.isfinite(effect_size) else None,
                            'ci_lower': float(ci_lower) if ci_lower is not None and np.isfinite(ci_lower) else None,
                            'ci_upper': float(ci_upper) if ci_upper is not None and np.isfinite(ci_upper) else None,
                            'significant': (p_value < 0.05) if p_value is not None and not np.isnan(p_value) else False
                        }
                    except Exception as t_test_e:
                        logger.error(f"Exception during t-test for metric '{metric}': {str(t_test_e)}", exc_info=True)
                        significance_results[metric] = {
                            'error': f"Error calculating t-test for {metric}: {str(t_test_e)}",
                            't_statistic': None, 'p_value': None, 'effect_size': None,
                            'ci_lower': None, 'ci_upper': None, 'significant': False
                        }
                else:
                    logger.warning(
                        f"Insufficient or mismatched data for t-test on metric '{metric}'. Baseline: {len(data['baseline'])}, Modified: {len(data['modified'])}")
                    significance_results[metric] = {
                        'error': "Insufficient or mismatched data points for t-test",
                        't_statistic': None, 'p_value': None, 'effect_size': None,
                        'ci_lower': None, 'ci_upper': None, 'significant': False
                    }

            ablation_results[config_name]['statistical_significance_vs_baseline'] = significance_results

            # Print summary
            print(f"\nAblation Results for {config_name}:")
            for metric_key, change_val in relative_changes.items():
                if isinstance(change_val, (int, float)) and np.isfinite(change_val):
                    print(f"  {metric_key}: {change_val * 100:.1f}% change from baseline")
                else:
                    print(
                        f"  {metric_key}: Change N/A (Base value was near zero or issue with calculation: {change_val})")

            print("  Statistical Significance vs. Baseline:")
            for metric_key, stats_dict in significance_results.items():  # Renamed stats to stats_dict
                if 'error' in stats_dict and stats_dict['error']:
                    print(f"    {metric_key}: {stats_dict['error']}")
                elif all(k in stats_dict for k in
                         ['p_value', 'effect_size', 'ci_lower', 'ci_upper', 'significant']):  # Check all keys
                    p_val_str = f"{stats_dict['p_value']:.3g}" if stats_dict['p_value'] is not None and np.isfinite(
                        stats_dict['p_value']) else "N/A"
                    eff_size_str = f"{stats_dict['effect_size']:.2f}" if stats_dict[
                                                                             'effect_size'] is not None and np.isfinite(
                        stats_dict['effect_size']) else "N/A"
                    ci_low_str = f"{stats_dict['ci_lower']:.2f}" if stats_dict['ci_lower'] is not None and np.isfinite(
                        stats_dict['ci_lower']) else "N/A"
                    ci_up_str = f"{stats_dict['ci_upper']:.2f}" if stats_dict['ci_upper'] is not None and np.isfinite(
                        stats_dict['ci_upper']) else "N/A"
                    sig_str = str(stats_dict['significant'])

                    print(f"    {metric_key}: p-value={p_val_str}, effect_size={eff_size_str}, "
                          f"CI=[{ci_low_str}, {ci_up_str}], significant={sig_str}")
                else:
                    print(
                        f"    {metric_key}: Statistical results incomplete or contains None/NaN values. Data: {stats_dict}")


        except Exception as e:
            print(f"Error in ablation study for {config_name}: {str(e)}")
            logger.error(f"Outer exception in ablation study for {config_name}: {str(e)}", exc_info=True)
            # Ensure the key exists even if an error occurs before significance_results is populated
            ablation_results[config_name].setdefault('statistical_significance_vs_baseline',
                                                     {m: {'error': f'Outer loop error: {str(e)}'} for m in
                                                      eval_metrics})

    return {
        'ablation_results': dict(ablation_results),
        'paired_data': {}
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
    This function is called for both the baseline and each modified ablation configuration.
    """
    logger.info(f"--- Running Configuration: {name} ---")
    results = {}  # To store aggregated mean/std for this run
    query_specific = {}  # Stores per-query raw scores for this run, used for paired t-tests

    # Create a new MetricsCollector for each configuration run to keep reports separate
    run_specific_metrics_dir = f"logs/{dataset_type}/metrics_ablation_{name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace(',', '').lower()}"
    logger.info(f"Metrics for config '{name}' will be saved to: {run_specific_metrics_dir}")
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=run_specific_metrics_dir
    )
    evaluator = Evaluation(dataset_type=dataset_type)

    # Lists to store individual metric scores for this run's aggregation
    all_exact_matches = []
    all_f1_scores = []
    all_processing_times = []
    all_resource_usages = []  # Sum of resource deltas

    for sample_idx, sample in enumerate(samples):
        query = sample.get("query")
        context = sample.get("context", "")
        ground_truth = sample.get("answer")  # This is the GT object (str for text, dict for DROP)
        query_id = sample.get("query_id", f"unknown_qid_{sample_idx}")
        # query_type = sample.get("type", "") # 'type' from dataset sample, not used directly here

        logger.info(
            f"  Processing sample {sample_idx + 1}/{len(samples)} for config '{name}' (QID: {query_id}): '{query[:50]}...'")
        initial_resources = resource_manager.check_resources()
        start_time = time.time()

        # System processes the query
        # response_data is the final formatted response from UnifiedResponseAggregator
        response_data, reasoning_path = system_manager.process_query_with_fallback(
            query=query,
            context=context,
            query_id=query_id,
            forced_path=None,  # This might be controlled by the 'config' dict for specific ablations
            query_complexity=system_manager.hybrid_integrator.query_expander.get_query_complexity(query),
            supporting_facts=sample.get("supporting_facts"),  # Pass supporting facts if available
            dataset_type=dataset_type
        )

        processing_time = time.time() - start_time
        final_resources = resource_manager.check_resources()
        resource_delta = {}
        for k in final_resources:
            if k != 'timestamp':
                resource_delta[k] = final_resources.get(k, 0.0) - initial_resources.get(k, 0.0)

        # Evaluate response
        prediction_obj = response_data.get('result')  # This is the actual answer payload
        # The 'status' in response_data indicates if the SCM processing succeeded or failed overall.

        eval_metrics_dict = {}
        if ground_truth is not None and prediction_obj is not None and response_data.get('status') == 'success':
            gt_for_eval = {query_id: ground_truth}
            pred_for_eval = {query_id: prediction_obj}
            try:
                eval_metrics_dict = evaluator.evaluate(
                    predictions=pred_for_eval,
                    ground_truths=gt_for_eval
                )
            except Exception as eval_e:
                logger.error(f"QID {query_id}: Evaluation call failed: {eval_e}", exc_info=True)
                # Assign default zero metrics if evaluation itself fails
                eval_metrics_dict = {'average_exact_match': 0.0, 'average_f1': 0.0}
        else:
            logger.warning(
                f"QID {query_id}: Skipping evaluation. GT: {ground_truth is not None}, Pred: {prediction_obj is not None}, Status: {response_data.get('status')}")
            # Still record 0 for metrics if not evaluated successfully for consistency in paired tests
            eval_metrics_dict = {'average_exact_match': 0.0, 'average_f1': 0.0}

        exact_match = eval_metrics_dict.get('average_exact_match', 0.0)
        f1_score = eval_metrics_dict.get('average_f1', 0.0)
        current_resource_usage_sum = sum(v for v in resource_delta.values() if isinstance(v, (int, float)))

        # Store per-query raw scores for this configuration for paired t-tests later
        query_specific[query_id] = {
            'exact_match': exact_match,
            'f1': f1_score,
            'processing_time': processing_time,
            'resource_usage': current_resource_usage_sum,
            'reasoning_path': reasoning_path,
            'response_status': response_data.get('status', 'unknown')
        }

        # Append to lists for aggregation for this configuration's summary
        all_exact_matches.append(exact_match)
        all_f1_scores.append(f1_score)
        all_processing_times.append(processing_time)
        all_resource_usages.append(current_resource_usage_sum)

        # Collect with MetricsCollector for this configuration's detailed academic report
        metrics_collector.collect_query_metrics(
            query=query,
            prediction=prediction_obj,
            ground_truth=ground_truth,
            reasoning_path=reasoning_path,
            processing_time=processing_time,
            resource_usage=resource_delta,
            complexity_score=system_manager.hybrid_integrator.query_expander.get_query_complexity(query),
            query_id=query_id,
            confidence=response_data.get('confidence', prediction_obj.get('confidence') if isinstance(prediction_obj,
                                                                                                      dict) else None),
            operation_type=response_data.get('type',
                                             prediction_obj.get('type') if isinstance(prediction_obj, dict) else None)
        )

    logger.info(f"--- Finished processing samples for config: {name} ---")
    academic_report = metrics_collector.generate_academic_report()
    if not academic_report or academic_report.get("experiment_summary", {}).get("total_queries", 0) == 0:
        logger.warning(
            f"MetricsCollector generated an empty or invalid academic report for config: {name}. Report: {academic_report}")

    # Aggregate metrics for this configuration's run summary
    results['exact_match'] = {
        'mean': float(np.mean(all_exact_matches)) if all_exact_matches else 0.0,
        'std': float(np.std(all_exact_matches)) if all_exact_matches else 0.0,
        'values': all_exact_matches  # Keep raw values if needed for other analyses
    }
    results['f1'] = {
        'mean': float(np.mean(all_f1_scores)) if all_f1_scores else 0.0,
        'std': float(np.std(all_f1_scores)) if all_f1_scores else 0.0,
        'values': all_f1_scores
    }
    results['processing_time'] = {
        'mean': float(np.mean(all_processing_times)) if all_processing_times else 0.0,
        'std': float(np.std(all_processing_times)) if all_processing_times else 0.0,
        'values': all_processing_times
    }
    results['resource_usage'] = {
        'mean': float(np.mean(all_resource_usages)) if all_resource_usages else 0.0,
        'std': float(np.std(all_resource_usages)) if all_resource_usages else 0.0,
        'values': all_resource_usages,
        'trends': academic_report.get('efficiency_metrics', {}).get('trends', {}) if academic_report else {}
    }

    results['query_specific'] = query_specific
    results['academic_report'] = academic_report

    logger.info(
        f"Aggregated metrics for config '{name}': EM Mean={results['exact_match']['mean']:.3f}, F1 Mean={results['f1']['mean']:.3f}")
    return results


def _run_configuration_with_params(
        config_name: str,
        config: Dict[str, Any],  # This is the specific ablation configuration dict
        samples: List[Dict[str, Any]],
        rules_path_baseline: str,
        device: Any,
        neural_retriever_baseline: NeuralRetriever,
        query_expander_baseline: QueryExpander,
        response_aggregator_baseline: UnifiedResponseAggregator,
        resource_manager_baseline: ResourceManager,
        dimensionality_manager_baseline: Any,
        dataset_type: str
) -> Dict[str, Any]:
    """
    Runs a configuration with modified parameters by creating a new SystemControlManager.
    """
    logger.info(f"--- Preparing to run modified configuration: {config_name} ---")
    logger.debug(f"Ablation config params for '{config_name}': {config}")

    modified_system_manager = _create_modified_system_manager(
        config=config,
        rules_path_baseline=rules_path_baseline,
        device=device,
        neural_retriever_baseline=neural_retriever_baseline,
        query_expander_baseline=query_expander_baseline,
        response_aggregator_baseline=response_aggregator_baseline,
        resource_manager_baseline=resource_manager_baseline,
        dimensionality_manager_baseline=dimensionality_manager_baseline,
        dataset_type=dataset_type
    )

    # Run this modified configuration using the standard _run_configuration function
    modified_metrics = _run_configuration(
        name=config_name,  # Use the ablation config name for logging and metrics dir
        samples=samples,
        system_manager=modified_system_manager,  # Pass the newly created SCM
        resource_manager=resource_manager_baseline,  # Use the baseline RM for this run
        dataset_type=dataset_type
    )
    return modified_metrics


def _create_modified_system_manager(
        config: Dict[str, Any],
        rules_path_baseline: str,
        device: Any,
        neural_retriever_baseline: NeuralRetriever,
        query_expander_baseline: QueryExpander,
        response_aggregator_baseline: UnifiedResponseAggregator,
        resource_manager_baseline: ResourceManager,
        dimensionality_manager_baseline: Any,
        dataset_type: str
) -> SystemControlManager:
    """
    Creates a modified SystemControlManager for an ablation configuration.
    """
    logger.info(f"Creating SystemControlManager for ablation: {config.get('name', 'Unknown Ablation')}")
    # Determine rules file to use
    current_rules_file = config.get('rules_file', rules_path_baseline)  # 'rules_file' should be an actual path here
    logger.info(f"Ablation '{config.get('name')}': Using rules file: {current_rules_file}")

    # Get symbolic reasoner parameters from config
    sym_match_threshold = config.get('match_threshold', 0.1 if dataset_type.lower() == 'drop' else 0.25)
    sym_max_hops = config.get('max_hops', 3 if dataset_type.lower() == 'drop' else 5)
    sym_embedding_model = config.get('embedding_model', 'all-MiniLM-L6-v2')

    # Initialize symbolic reasoner for this ablation
    local_symbolic_reasoner: Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop, DummySymbolicReasoner]
    if config.get('disable_symbolic', False):
        logger.info(f"Ablation '{config.get('name')}': Symbolic reasoner is DISABLED.")
        local_symbolic_reasoner = DummySymbolicReasoner()
    else:
        logger.info(f"Ablation '{config.get('name')}': Initializing Symbolic Reasoner (type: {dataset_type}).")
        if dataset_type.lower() == 'drop':
            local_symbolic_reasoner = GraphSymbolicReasonerDrop(
                rules_file=current_rules_file,
                match_threshold=sym_match_threshold,
                max_hops=sym_max_hops,
                embedding_model=sym_embedding_model,
                device=device,
                dim_manager=dimensionality_manager_baseline
            )
        else:
            local_symbolic_reasoner = GraphSymbolicReasoner(
                rules_file=current_rules_file,
                match_threshold=sym_match_threshold,
                max_hops=sym_max_hops,
                embedding_model=sym_embedding_model,
                device=device,
                dim_manager=dimensionality_manager_baseline
            )

    # Initialize neural retriever with few-shot setting for this ablation
    local_neural_retriever: Union[NeuralRetriever, DummyNeuralRetriever]
    if config.get('disable_neural', False):
        logger.info(f"Ablation '{config.get('name')}': Neural retriever is DISABLED.")
        local_neural_retriever = DummyNeuralRetriever()
    else:
        logger.info(f"Ablation '{config.get('name')}': Initializing Neural Retriever.")
        use_few_shots_this_ablation = config.get('use_few_shots', False)  # Default to False for safety if not specified

        # Get the few_shot_examples_path from the specific ablation configuration
        # This path is resolved in main.py's execute_ablation_study
        few_shot_path_for_this_ablation = config.get('few_shot_examples_path') if use_few_shots_this_ablation else None

        if use_few_shots_this_ablation and few_shot_path_for_this_ablation:
            logger.info(
                f"Ablation '{config.get('name')}': Neural Retriever will use few-shot examples from '{few_shot_path_for_this_ablation}'.")
        elif use_few_shots_this_ablation and not few_shot_path_for_this_ablation:
            logger.warning(
                f"Ablation '{config.get('name')}': 'use_few_shots' is True, but 'few_shot_examples_path' is not provided or is None. NR will not use few-shots.")
        else:
            logger.info(
                f"Ablation '{config.get('name')}': Neural Retriever will NOT use few-shot examples (use_few_shots: {use_few_shots_this_ablation}).")

        # Ensure model_name is in the config for this ablation
        ablation_model_name = config.get('model_name')
        if not ablation_model_name:
            # This should have been added by main.py's execute_ablation_study, but as a fallback:
            ablation_model_name = neural_retriever_baseline.model.name_or_path if hasattr(
                neural_retriever_baseline.model, 'name_or_path') else "meta-llama/Llama-3.2-3B"  # Absolute fallback
            logger.warning(f"model_name not found in ablation configuration for '{config.get('name')}'."
                           f" Defaulting to '{ablation_model_name}'.")

        local_neural_retriever = NeuralRetriever(
            model_name=ablation_model_name,  # Should be from config
            use_quantization=config.get('neural_use_quantization', False),  # Default if not in ablation config
            max_context_length=getattr(neural_retriever_baseline, 'max_context_length', 2048),
            chunk_size=getattr(neural_retriever_baseline, 'chunk_size', 512),
            overlap=getattr(neural_retriever_baseline, 'overlap', 128),
            device=device,
            few_shot_examples_path=few_shot_path_for_this_ablation  # Use the path determined for this ablation
        )

    # Create hybrid integrator
    logger.info(f"Ablation '{config.get('name')}': Initializing Hybrid Integrator.")
    modified_integrator = HybridIntegrator(
        symbolic_reasoner=local_symbolic_reasoner,
        neural_retriever=local_neural_retriever,
        query_expander=query_expander_baseline,  # Using baseline query expander
        dim_manager=dimensionality_manager_baseline,  # Using baseline dim manager
        dataset_type=dataset_type
    )

    # MetricsCollector specific to this SCM instance for this ablation config run
    ablation_run_metrics_dir = f"logs/{dataset_type}/metrics_ablation_SCM_{config.get('name', 'unknown_ablation').replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace(',', '').lower()}"
    scm_metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=ablation_run_metrics_dir
    )

    # Create modified system manager
    logger.info(f"Ablation '{config.get('name')}': Initializing SystemControlManager.")
    modified_system_manager = SystemControlManager(
        hybrid_integrator=modified_integrator,
        resource_manager=resource_manager_baseline,  # Using baseline resource manager
        aggregator=response_aggregator_baseline,  # Using baseline aggregator
        metrics_collector=scm_metrics_collector,  # SCM-specific collector
        error_retry_limit=config.get('error_retry_limit', 2),  # Allow override from ablation config
        max_query_time=config.get('max_query_time', 30.0)  # Allow override
    )

    return modified_system_manager


if __name__ == "__main__":
    print("This is the ablation study module. It is intended to be called from main.py.")