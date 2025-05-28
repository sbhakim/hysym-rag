# src/ablation_study.py

import os
import sys
import json
import time
import yaml
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
import logging
import argparse  # Added for type hinting args

# HySym-RAG component imports
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
from src.config.config_loader import ConfigLoader  # For loading main config
from src.utils.device_manager import DeviceManager  # For getting device
from src.utils.dimension_manager import DimensionalityManager  # For DM init
from src.utils.progress import tqdm, ProgressManager  # For progress bars
# Import data loaders from their new location
from src.utils.data_loaders import load_hotpotqa, load_drop_dataset

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _run_ablation_iterations(  # Renamed from run_ablation_study to avoid confusion
        samples: List[Dict[str, Any]],
        dataset_type: str,
        rules_path_baseline: str,  # Path to rules used for baseline Symbolic Reasoner
        device: Any,
        # Baseline components:
        neural_retriever_baseline: NeuralRetriever,
        query_expander_baseline: QueryExpander,
        response_aggregator_baseline: UnifiedResponseAggregator,
        resource_manager_baseline: ResourceManager,  # This RM instance will be shared across ablation runs
        system_manager_baseline: SystemControlManager,  # The SCM configured for the baseline run
        dimensionality_manager_baseline: DimensionalityManager,
        ablation_configurations: List[Dict[str, Any]]  # Iterator/List of config dicts
) -> Dict[str, Any]:
    """
    Core loop that runs each ablation configuration and compares it to a pre-computed baseline.
    """
    logger.info(f"--- Starting Ablation Iterations for Dataset: {dataset_type.upper()} ---")
    ablation_results = defaultdict(dict)
    eval_metrics = ['exact_match', 'f1', 'processing_time', 'resource_usage']

    # Run baseline configuration first to establish reference performance
    # This uses the system_manager_baseline passed in, which is already configured.
    print("Running baseline configuration for comparison points...")
    actual_baseline_metrics = _run_configuration(
        name="Baseline_Reference_Run_for_Ablation_Comparison",  # Specific name for this internal baseline run
        samples=samples,
        system_manager=system_manager_baseline,
        resource_manager=resource_manager_baseline,  # Use the shared RM
        dataset_type=dataset_type
    )
    logger.info(
        f"Baseline metrics for comparison established: EM Mean={actual_baseline_metrics.get('exact_match', {}).get('mean', 'N/A')}")

    for config_idx, config_params in enumerate(ablation_configurations):  # Iterate through the iterator/list
        config_name = config_params['name']
        print(f"\n--- Testing Ablation Configuration {config_idx + 1}: {config_name} ---")

        try:
            paired_results = defaultdict(lambda: defaultdict(list))
            modified_metrics = _run_configuration_with_params(  # This creates a new SCM for the ablation
                config_name=config_name,
                config=config_params,
                samples=samples,
                rules_path_baseline=rules_path_baseline,  # Fallback for rules if not in config_params
                device=device,
                # Pass baseline components for _create_modified_system_manager to potentially reuse/get defaults from
                neural_retriever_baseline=neural_retriever_baseline,
                query_expander_baseline=query_expander_baseline,
                response_aggregator_baseline=response_aggregator_baseline,
                resource_manager_baseline=resource_manager_baseline,  # Pass the shared RM
                dimensionality_manager_baseline=dimensionality_manager_baseline,
                dataset_type=dataset_type
            )
            logger.info(f"Metrics for modified config '{config_name}' obtained.")

            ablation_results[config_name]['modified'] = modified_metrics
            ablation_results[config_name]['baseline'] = actual_baseline_metrics  # Store the common baseline

            relative_changes = {}
            for metric in eval_metrics:
                if metric in actual_baseline_metrics and metric in modified_metrics and \
                        isinstance(actual_baseline_metrics.get(metric), dict) and \
                        isinstance(modified_metrics.get(metric), dict) and \
                        'mean' in actual_baseline_metrics[metric] and 'mean' in modified_metrics[metric]:

                    base_val = actual_baseline_metrics[metric]['mean']
                    mod_val = modified_metrics[metric]['mean']

                    if isinstance(base_val, (int, float)) and isinstance(mod_val, (int, float)):
                        if abs(base_val) > 1e-9:
                            rel_change = (mod_val - base_val) / base_val
                            relative_changes[metric] = rel_change
                        elif mod_val == 0 and base_val == 0:
                            relative_changes[metric] = 0.0
                        else:
                            relative_changes[metric] = float('inf') if mod_val > 0 else float(
                                '-inf') if mod_val < 0 else 0.0
                    else:
                        logger.warning(f"Metric '{metric}' mean values non-numeric. Base: {base_val}, Mod: {mod_val}")

                    baseline_qs = actual_baseline_metrics.get('query_specific', {})
                    modified_qs = modified_metrics.get('query_specific', {})
                    if isinstance(baseline_qs, dict) and isinstance(modified_qs, dict):
                        valid_pairs_count = 0
                        for qid in baseline_qs:
                            if qid in modified_qs:
                                b_q_metric = baseline_qs[qid].get(metric)
                                m_q_metric = modified_qs[qid].get(metric)
                                if isinstance(b_q_metric, (int, float)) and isinstance(m_q_metric, (int, float)) and \
                                        np.isfinite(b_q_metric) and np.isfinite(m_q_metric):
                                    paired_results[metric]['baseline'].append(b_q_metric)
                                    paired_results[metric]['modified'].append(m_q_metric)
                                    valid_pairs_count += 1
                        logger.info(
                            f"For metric '{metric}', collected {valid_pairs_count} valid paired samples for t-test.")
            ablation_results[config_name]['relative_changes'] = relative_changes

            significance_results = {}
            for metric, data in paired_results.items():
                logger.info(f"Calculating significance for metric: {metric}")
                logger.debug(
                    f"Paired Baseline Data ({len(data['baseline'])} pts) for {metric}: {str(data['baseline'])[:100]}...")
                logger.debug(
                    f"Paired Modified Data ({len(data['modified'])} pts) for {metric}: {str(data['modified'])[:100]}...")
                if len(data['baseline']) >= 2 and len(data['modified']) >= 2 and len(data['baseline']) == len(
                        data['modified']):
                    try:
                        clean_baseline = data['baseline']  # Already cleaned
                        clean_modified = data['modified']

                        t_stat, p_value = stats.ttest_rel(clean_baseline, clean_modified)
                        diffs = [b - m for b, m in zip(clean_baseline, clean_modified)]
                        mean_diff = np.mean(diffs)
                        std_diff = np.std(diffs, ddof=1)

                        effect_size = (mean_diff / std_diff) if (std_diff != 0 and np.isfinite(std_diff)) else (
                            0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff) if np.isfinite(mean_diff) else 0.0)

                        current_sem = 0.0
                        df_for_ci = len(diffs) - 1
                        ci_lower, ci_upper = None, None

                        if df_for_ci > 0:
                            sem_val = stats.sem(diffs, nan_policy='propagate')
                            current_sem = 0.0 if np.isnan(sem_val) or not np.isfinite(sem_val) else sem_val

                        if df_for_ci > 0 and current_sem >= 0 and np.isfinite(current_sem) and np.isfinite(mean_diff):
                            ci_lower, ci_upper = stats.t.interval(
                                confidence=0.95, df=df_for_ci, loc=mean_diff, scale=current_sem
                            )
                        elif np.isfinite(mean_diff):  # Fallback if CI cannot be calculated properly
                            ci_lower, ci_upper = mean_diff, mean_diff
                            logger.warning(f"Metric {metric}: Using mean_diff for CI due to invalid df/scale.")

                        significance_results[metric] = {
                            't_statistic': float(t_stat) if np.isfinite(t_stat) else None,
                            'p_value': float(p_value) if np.isfinite(p_value) else None,
                            'effect_size': float(effect_size) if np.isfinite(effect_size) else None,
                            'ci_lower': float(ci_lower) if ci_lower is not None and np.isfinite(ci_lower) else None,
                            'ci_upper': float(ci_upper) if ci_upper is not None and np.isfinite(ci_upper) else None,
                            'significant': (p_value < 0.05) if p_value is not None and np.isfinite(p_value) else False
                        }
                    except Exception as t_test_e:
                        logger.error(f"Exception during t-test for metric '{metric}': {str(t_test_e)}", exc_info=True)
                        significance_results[metric] = {
                            'error': f"Error calculating t-test for {metric}: {str(t_test_e)}"}
                else:
                    logger.warning(
                        f"Insufficient/mismatched data for t-test on '{metric}'. Baseline: {len(data['baseline'])}, Mod: {len(data['modified'])}")
                    significance_results[metric] = {'error': "Insufficient or mismatched data points for t-test"}
            ablation_results[config_name]['statistical_significance_vs_baseline'] = significance_results
        except Exception as e:
            logger.error(f"Outer exception processing ablation config '{config_name}': {str(e)}", exc_info=True)
            ablation_results[config_name]['error'] = str(e)
            ablation_results[config_name].setdefault('statistical_significance_vs_baseline',
                                                     {m: {'error': 'Outer loop error'} for m in eval_metrics})
    return {
        'ablation_results': dict(ablation_results),
        'paired_data': {}  # Kept for potential future use or compatibility
    }


def setup_and_orchestrate_ablation(args: argparse.Namespace):
    """
    Sets up all components and orchestrates the ablation study,
    including loading configurations, data, initializing baseline components,
    running ablation iterations, and saving/printing the summary.
    """
    print(f"\n=== Setting up Ablation Study for Dataset: {args.dataset.upper()} ===")
    dataset_type = args.dataset.lower()

    # 1. Load main configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Correct for src/ablation_study.py
    project_root_dir = os.path.dirname(script_dir)  # Assuming src is one level down from project root

    main_config_path = os.path.join(project_root_dir, "src", "config", "config.yaml")
    if not os.path.exists(main_config_path):
        logger.error(f"Main configuration file not found at: {main_config_path}")
        return
    main_config = ConfigLoader.load_config(main_config_path)
    model_name = main_config.get("model_name")
    if not model_name:
        logger.error("model_name not found in main configuration.")
        return

    default_data_dir = os.path.join(project_root_dir, "data")

    # 2. Load Ablation Configurations from YAML
    ablation_config_yaml_path = os.path.join(project_root_dir, args.ablation_config) if not os.path.isabs(
        args.ablation_config) else args.ablation_config

    if not os.path.exists(ablation_config_yaml_path):
        logger.error(f"Ablation configuration file not found at: {ablation_config_yaml_path}")
        return
    with open(ablation_config_yaml_path, 'r') as f:
        loaded_ablation_configs_yaml = yaml.safe_load(f)

    # Extract common parameters for symbolic reasoner and few-shot examples from YAML
    common_embedding_model = loaded_ablation_configs_yaml.get('common_embedding_model', 'all-MiniLM-L6-v2')
    common_max_hops_drop = loaded_ablation_configs_yaml.get('common_max_hops_drop', 3)
    common_match_threshold_drop = loaded_ablation_configs_yaml.get('common_match_threshold_drop', 0.1)
    common_max_hops_hotpotqa = loaded_ablation_configs_yaml.get('common_max_hops_hotpotqa', 5)
    common_match_threshold_hotpotqa = loaded_ablation_configs_yaml.get('common_match_threshold_hotpotqa', 0.25)
    common_few_shot_examples_path_drop_key = 'drop_few_shot_examples_path'  # Key in ablation_config.yaml
    default_few_shot_path = os.path.join(default_data_dir, "drop_few_shot_examples.json")
    common_few_shot_examples_path_drop = loaded_ablation_configs_yaml.get(common_few_shot_examples_path_drop_key,
                                                                          default_few_shot_path)

    # Resolve rule file paths from keys in ablation_config.yaml
    common_paths = {
        "dynamic_rules_path_drop": loaded_ablation_configs_yaml.get("dynamic_rules_path_drop",
                                                                    os.path.join(default_data_dir,
                                                                                 "rules_drop_dynamic.json")),
        "static_rules_path_drop": loaded_ablation_configs_yaml.get("static_rules_path_drop",
                                                                   os.path.join(default_data_dir, "rules_drop.json")),
        "no_rules_path": loaded_ablation_configs_yaml.get("no_rules_path",
                                                          os.path.join(default_data_dir, "empty_rules.json")),
        "rules_path_hotpotqa_baseline": main_config.get("hotpotqa_rules_file",
                                                        os.path.join(default_data_dir, "rules_hotpotqa.json"))
    }
    if not os.path.exists(common_paths["no_rules_path"]):
        os.makedirs(os.path.dirname(common_paths["no_rules_path"]), exist_ok=True)
        with open(common_paths["no_rules_path"], "w") as f_empty:
            json.dump([], f_empty)

    # Prepare list of ablation configurations
    ablation_configurations_list = []
    if dataset_type == 'drop':
        for cfg_params in loaded_ablation_configs_yaml.get('drop_ablations', []):
            processed_cfg = cfg_params.copy()
            if 'rules_file_key' in processed_cfg:
                processed_cfg['rules_file'] = common_paths.get(processed_cfg['rules_file_key'],
                                                               processed_cfg.get('rules_file'))
            processed_cfg['embedding_model'] = processed_cfg.get('embedding_model', common_embedding_model)
            processed_cfg['match_threshold'] = processed_cfg.get('match_threshold', common_match_threshold_drop)
            processed_cfg['max_hops'] = processed_cfg.get('max_hops', common_max_hops_drop)
            # Set the actual path for few-shot examples for this config
            processed_cfg['few_shot_examples_path'] = processed_cfg.get('few_shot_examples_path',
                                                                        common_few_shot_examples_path_drop)
            processed_cfg['model_name'] = model_name
            ablation_configurations_list.append(processed_cfg)
    elif dataset_type == 'hotpotqa':
        for cfg_params in loaded_ablation_configs_yaml.get('hotpotqa_ablations', []):
            processed_cfg = cfg_params.copy()
            if 'rules_file_key' in processed_cfg:
                processed_cfg['rules_file'] = common_paths.get(processed_cfg['rules_file_key'],
                                                               processed_cfg.get('rules_file'))
            processed_cfg['embedding_model'] = processed_cfg.get('embedding_model', common_embedding_model)
            processed_cfg['match_threshold'] = processed_cfg.get('match_threshold', common_match_threshold_hotpotqa)
            processed_cfg['max_hops'] = processed_cfg.get('max_hops', common_max_hops_hotpotqa)
            processed_cfg['model_name'] = model_name
            ablation_configurations_list.append(processed_cfg)
    else:
        logger.error(f"Unsupported dataset_type '{dataset_type}' for ablation setup.")
        return

    if args.ablation_name:
        ablation_configurations_list = [cfg for cfg in ablation_configurations_list if
                                        cfg.get('name') == args.ablation_name]
        if not ablation_configurations_list:
            logger.error(
                f"No ablation configuration found with name '{args.ablation_name}'. Check ablation_config.yaml.")
            return
    if not ablation_configurations_list:
        logger.error(f"No ablation configurations to run for dataset_type '{dataset_type}'.")
        return

    # 3. Load Dataset Samples for Ablation
    samples_for_ablation: List[Dict[str, Any]] = []
    if dataset_type == 'drop':
        drop_dataset_path = main_config.get("drop_dataset_path",
                                            os.path.join(default_data_dir, "drop_dataset_dev.json"))
        if not os.path.exists(drop_dataset_path):
            logger.error(f"DROP dataset for ablation not found at: {drop_dataset_path}")
            return
        samples_for_ablation = load_drop_dataset(drop_dataset_path, max_samples=args.samples)
    elif dataset_type == 'hotpotqa':
        hotpotqa_dataset_path = main_config.get("hotpotqa_dataset_path",
                                                os.path.join(default_data_dir, "hotpot_dev_distractor_v1.json"))
        if not os.path.exists(hotpotqa_dataset_path):
            logger.error(f"HotpotQA dataset for ablation not found at: {hotpotqa_dataset_path}")
            return
        samples_for_ablation = load_hotpotqa(hotpotqa_dataset_path, max_samples=args.samples)

    if not samples_for_ablation:
        logger.error(f"Failed to load samples for {dataset_type} for ablation study.")
        return
    logger.info(f"Loaded {len(samples_for_ablation)} samples for {dataset_type} ablation.")

    # 4. Initialize Baseline Components (to be passed to _run_ablation_iterations)
    device = DeviceManager.get_device()
    resource_manager_for_ablation = ResourceManager(
        config_path=os.path.join(project_root_dir, "src", "config", "resource_config.yaml"),
        enable_performance_tracking=True,
        history_window_size=100
    )
    dimensionality_manager_for_ablation = DimensionalityManager(
        target_dim=main_config.get('alignment', {}).get('target_dim', 768),
        device=device
    )

    rules_path_for_sym_reasoner_baseline = ""  # This is the rules file for the reference Symbolic Reasoner
    if dataset_type == 'drop':
        # For DROP baseline in ablation, prefer dynamic rules if they exist, else static
        dynamic_baseline_rules = common_paths["dynamic_rules_path_drop"]
        if os.path.exists(dynamic_baseline_rules) and os.path.getsize(dynamic_baseline_rules) > 2:
            rules_path_for_sym_reasoner_baseline = dynamic_baseline_rules
        else:
            rules_path_for_sym_reasoner_baseline = common_paths["static_rules_path_drop"]
            if not os.path.exists(rules_path_for_sym_reasoner_baseline):
                rules_path_for_sym_reasoner_baseline = os.path.join(default_data_dir,
                                                                    "rules_drop.json")  # Ultimate fallback
        logger.info(
            f"Baseline Symbolic Reasoner for DROP ablation will use rules from: {rules_path_for_sym_reasoner_baseline}")
    else:  # hotpotqa
        rules_path_for_sym_reasoner_baseline = common_paths["rules_path_hotpotqa_baseline"]
        logger.info(
            f"Baseline Symbolic Reasoner for HotpotQA ablation will use rules from: {rules_path_for_sym_reasoner_baseline}")

    sym_reasoner_baseline_instance: Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop]
    baseline_embedding_model_name = main_config.get('embeddings', {}).get('model_name', common_embedding_model)

    if dataset_type == 'drop':
        sym_reasoner_baseline_instance = GraphSymbolicReasonerDrop(
            rules_file=rules_path_for_sym_reasoner_baseline,
            match_threshold=common_match_threshold_drop,
            max_hops=common_max_hops_drop,
            embedding_model=baseline_embedding_model_name,
            device=device,
            dim_manager=dimensionality_manager_for_ablation
        )
    else:
        sym_reasoner_baseline_instance = GraphSymbolicReasoner(
            rules_file=rules_path_for_sym_reasoner_baseline,
            match_threshold=common_match_threshold_hotpotqa,
            max_hops=common_max_hops_hotpotqa,
            embedding_model=baseline_embedding_model_name,
            device=device,
            dim_manager=dimensionality_manager_for_ablation
        )

    # Baseline Neural Retriever: its few-shot usage is determined by main_config's 'use_drop_few_shots'
    use_few_shots_for_baseline_nr_in_ablation = bool(main_config.get("use_drop_few_shots", 0))
    nr_few_shot_path_for_baseline_in_ablation = None
    if dataset_type == 'drop' and use_few_shots_for_baseline_nr_in_ablation:
        nr_few_shot_path_for_baseline_in_ablation = common_few_shot_examples_path_drop  # Resolved common path
        if not os.path.exists(nr_few_shot_path_for_baseline_in_ablation):
            logger.warning(
                f"Baseline DROP few-shot file '{nr_few_shot_path_for_baseline_in_ablation}' not found. Baseline NR in ablation will not use them.")
            nr_few_shot_path_for_baseline_in_ablation = None

    neural_retriever_baseline_instance = NeuralRetriever(
        model_name=model_name,  # Main model
        use_quantization=main_config.get('neural_use_quantization', False),
        max_context_length=main_config.get('neural_max_context_length', 2048),
        chunk_size=main_config.get('neural_chunk_size', 512),
        overlap=main_config.get('neural_overlap', 128),
        device=device,
        few_shot_examples_path=nr_few_shot_path_for_baseline_in_ablation
    )

    complexity_rules_yaml_path = os.path.join(project_root_dir, "src", "config", "complexity_rules.yaml")
    query_expander_baseline_instance = QueryExpander(
        complexity_config=complexity_rules_yaml_path if os.path.exists(complexity_rules_yaml_path) else None
    )
    response_aggregator_baseline_instance = UnifiedResponseAggregator(include_explanations=True)

    hybrid_integrator_baseline_instance = HybridIntegrator(
        symbolic_reasoner=sym_reasoner_baseline_instance,
        neural_retriever=neural_retriever_baseline_instance,
        query_expander=query_expander_baseline_instance,
        dim_manager=dimensionality_manager_for_ablation,
        dataset_type=dataset_type
    )

    baseline_scm_metrics_dir = os.path.join(project_root_dir, args.log_dir, dataset_type,
                                            "metrics_ablation_SCM_BaselineRef")  # Ensure path relative to project
    baseline_metrics_collector_for_scm = MetricsCollector(dataset_type=dataset_type,
                                                          metrics_dir=baseline_scm_metrics_dir)

    system_manager_baseline_instance = SystemControlManager(
        hybrid_integrator=hybrid_integrator_baseline_instance,
        resource_manager=resource_manager_for_ablation,
        aggregator=response_aggregator_baseline_instance,
        metrics_collector=baseline_metrics_collector_for_scm,
        error_retry_limit=main_config.get('error_retry_limit', 2),
        max_query_time=main_config.get('max_query_time', 30.0)
    )

    # 5. Run the Ablation Iterations
    print(f"\n--- Starting core ablation iterations with {len(ablation_configurations_list)} configurations... ---")
    config_iterator = tqdm(ablation_configurations_list, desc="Ablation Configurations Iteration", unit="config",
                           disable=not ProgressManager.SHOW_PROGRESS)

    ablation_results_summary = None
    try:
        ablation_results_summary = _run_ablation_iterations(
            samples=samples_for_ablation,
            dataset_type=dataset_type,
            rules_path_baseline=rules_path_for_sym_reasoner_baseline,
            device=device,
            neural_retriever_baseline=neural_retriever_baseline_instance,
            query_expander_baseline=query_expander_baseline_instance,
            response_aggregator_baseline=response_aggregator_baseline_instance,
            resource_manager_baseline=resource_manager_for_ablation,
            system_manager_baseline=system_manager_baseline_instance,  # This is the SCM used for the baseline run
            dimensionality_manager_baseline=dimensionality_manager_for_ablation,
            ablation_configurations=config_iterator
        )
    except Exception as e:
        logger.error(f"Core ablation iterations failed: {e}", exc_info=True)
        # Ensure a return dictionary even on failure
        return {"error": f"Core ablation iterations failed: {e}"}

    # 6. Save and Print Ablation Summary
    ablation_log_dir = os.path.join(project_root_dir, args.log_dir, dataset_type,
                                    "ablation_results")  # Ensure path relative to project
    os.makedirs(ablation_log_dir, exist_ok=True)
    ablation_summary_file = os.path.join(ablation_log_dir,
                                         f"ablation_summary_{dataset_type}_{args.ablation_name or 'all'}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(ablation_summary_file, 'w') as f:
            json.dump(ablation_results_summary, f, indent=2,
                      default=lambda o: str(o) if isinstance(o, np.generic) else o.__dict__ if hasattr(o,
                                                                                                       '__dict__') else str(
                          o))
        print(f"\nAblation study summary saved to: {ablation_summary_file}")
    except Exception as e:
        print(f"Warning: Could not save ablation study summary to {ablation_summary_file}: {e}")

    # Print detailed statistical significance results from the summary
    print("\n=== Ablation Study Final Results Summary ===")
    if ablation_results_summary and 'ablation_results' in ablation_results_summary:
        processed_ablation_results = ablation_results_summary.get('ablation_results', {})
        for config_name_key, report_data in processed_ablation_results.items():
            print(f"\nConfiguration: {config_name_key}")
            if isinstance(report_data, dict) and 'error' in report_data and report_data['error']:
                print(f"  Error during this configuration run: {report_data['error']}")
                continue
            if not isinstance(report_data, dict):
                print(f"  Unexpected report_data format for {config_name_key}: {type(report_data)}")
                continue

            relative_changes = report_data.get('relative_changes', {})
            print("  Relative Changes from Baseline:")
            if isinstance(relative_changes, dict):
                for metric, change in relative_changes.items():
                    if isinstance(change, (int, float)) and np.isfinite(change):
                        print(f"    {metric}: {change * 100:.1f}%")
                    else:
                        print(f"    {metric}: {change} (Change N/A or non-finite)")
            else:
                print("    Relative changes data is not in the expected format.")

            stats_results = report_data.get('statistical_significance_vs_baseline', {})
            print("  Statistical Significance vs. Baseline:")
            if isinstance(stats_results, dict):
                for metric, stats_dict in stats_results.items():
                    if isinstance(stats_dict, dict) and 'error' in stats_dict and stats_dict['error']:
                        print(f"    {metric}: {stats_dict['error']}")
                    elif isinstance(stats_dict, dict) and all(
                            k in stats_dict for k in ['p_value', 'effect_size', 'ci_lower', 'ci_upper', 'significant']):
                        p_val_str = f"{stats_dict['p_value']:.3g}" if stats_dict['p_value'] is not None and np.isfinite(
                            stats_dict['p_value']) else "N/A"
                        eff_size_str = f"{stats_dict['effect_size']:.2f}" if stats_dict[
                                                                                 'effect_size'] is not None and np.isfinite(
                            stats_dict['effect_size']) else "N/A"
                        ci_low_str = f"{stats_dict['ci_lower']:.2f}" if stats_dict[
                                                                            'ci_lower'] is not None and np.isfinite(
                            stats_dict['ci_lower']) else "N/A"
                        ci_up_str = f"{stats_dict['ci_upper']:.2f}" if stats_dict[
                                                                           'ci_upper'] is not None and np.isfinite(
                            stats_dict['ci_upper']) else "N/A"
                        sig_str = str(stats_dict.get('significant', 'N/A'))

                        print(f"    {metric}: p-value={p_val_str}, effect_size={eff_size_str}, "
                              f"CI=[{ci_low_str}, {ci_up_str}], significant={sig_str}")
                    else:
                        print(f"    {metric}: Statistical results incomplete or contains None/NaN. Data: {stats_dict}")
            else:
                print("    Statistical significance data is not in the expected format or is None.")
    else:
        logger.error("Ablation results summary is empty or not in the expected dict format after run.")

    return ablation_results_summary  # Return the summary for potential further use


# The functions _run_configuration, _run_configuration_with_params, _create_modified_system_manager
# remain in this file as they are integral parts of the ablation study logic.

def _run_configuration(
        name: str,
        samples: List[Dict[str, Any]],
        system_manager: SystemControlManager,  # This SCM is specific to this config run
        resource_manager: ResourceManager,  # Shared RM instance
        dataset_type: str
) -> Dict[str, Any]:
    """
    Runs a single configuration (either baseline or a modified ablation) and collects metrics.
    """
    logger.info(f"--- Running Configuration: {name} ---")
    results = {}
    query_specific = {}

    # MetricsCollector for this specific configuration run
    # Sanitize name for directory creation
    safe_name = name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace(':',
                                                                                                                   '').lower()
    run_specific_metrics_dir = f"logs/{dataset_type}/metrics_collection_ablation_{safe_name}"
    logger.info(f"Metrics for config '{name}' will be saved to: {run_specific_metrics_dir}")
    metrics_collector = MetricsCollector(
        dataset_type=dataset_type,
        metrics_dir=run_specific_metrics_dir
    )
    evaluator = Evaluation(dataset_type=dataset_type)

    all_exact_matches = []
    all_f1_scores = []
    all_processing_times = []
    all_resource_usages = []

    for sample_idx, sample in enumerate(samples):
        query = sample.get("query")
        context = sample.get("context", "")
        ground_truth = sample.get("answer")
        query_id = sample.get("query_id", f"unknown_qid_{name}_{sample_idx}")

        logger.info(
            f"  Processing sample {sample_idx + 1}/{len(samples)} for config '{name}' (QID: {query_id}): '{str(query)[:50]}...'")

        # It's crucial that resource_manager tracks deltas correctly for *this specific query*
        # This means baseline should be established once, and deltas measured against it.
        # Or, if RM is reset/reused, it should be done carefully.
        # Assuming resource_manager.check_resources() gives current state, and deltas are calculated per query.
        initial_resources = resource_manager.check_resources()
        start_time = time.time()

        response_data, reasoning_path = system_manager.process_query_with_fallback(
            query=query,
            context=context,
            query_id=query_id,
            # forced_path could be part of the ablation config, SystemControlManager needs to handle it
            # For now, assuming it's not used unless specifically set by the SCM internal logic
            forced_path=None,
            query_complexity=system_manager.hybrid_integrator.query_expander.get_query_complexity(query),
            supporting_facts=sample.get("supporting_facts"),
            dataset_type=dataset_type
        )

        processing_time = time.time() - start_time
        final_resources = resource_manager.check_resources()
        resource_delta = {}
        for k in final_resources:
            if k != 'timestamp':
                resource_delta[k] = final_resources.get(k, 0.0) - initial_resources.get(k, 0.0)

        prediction_obj = response_data.get('result')
        eval_metrics_dict = {}
        if ground_truth is not None and prediction_obj is not None and response_data.get('status') == 'success':
            gt_for_eval = {query_id: ground_truth}
            pred_for_eval = {query_id: prediction_obj}
            try:
                eval_metrics_dict = evaluator.evaluate(predictions=pred_for_eval, ground_truths=gt_for_eval)
            except Exception as eval_e:
                logger.error(f"QID {query_id} in config '{name}': Evaluation call failed: {eval_e}", exc_info=True)
                eval_metrics_dict = {'average_exact_match': 0.0, 'average_f1': 0.0}
        else:
            logger.warning(
                f"QID {query_id} in config '{name}': Skipping evaluation. GT: {ground_truth is not None}, Pred: {prediction_obj is not None}, Status: {response_data.get('status')}")
            eval_metrics_dict = {'average_exact_match': 0.0, 'average_f1': 0.0}

        exact_match = float(eval_metrics_dict.get('average_exact_match', 0.0))
        f1_score = float(eval_metrics_dict.get('average_f1', 0.0))
        current_resource_usage_sum = sum(v for v in resource_delta.values() if isinstance(v, (int, float)))

        query_specific[query_id] = {
            'exact_match': exact_match, 'f1': f1_score,
            'processing_time': processing_time, 'resource_usage': current_resource_usage_sum,
            'reasoning_path': reasoning_path, 'response_status': response_data.get('status', 'unknown')
        }
        all_exact_matches.append(exact_match)
        all_f1_scores.append(f1_score)
        all_processing_times.append(processing_time)
        all_resource_usages.append(current_resource_usage_sum)

        metrics_collector.collect_query_metrics(
            query=query, prediction=prediction_obj, ground_truth=ground_truth,
            reasoning_path=reasoning_path, processing_time=processing_time,
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
        logger.warning(f"MetricsCollector generated an empty or invalid report for config: {name}.")

    results['exact_match'] = {'mean': np.mean(all_exact_matches) if all_exact_matches else 0.0,
                              'std': np.std(all_exact_matches) if all_exact_matches else 0.0,
                              'values': all_exact_matches}
    results['f1'] = {'mean': np.mean(all_f1_scores) if all_f1_scores else 0.0,
                     'std': np.std(all_f1_scores) if all_f1_scores else 0.0, 'values': all_f1_scores}
    results['processing_time'] = {'mean': np.mean(all_processing_times) if all_processing_times else 0.0,
                                  'std': np.std(all_processing_times) if all_processing_times else 0.0,
                                  'values': all_processing_times}
    results['resource_usage'] = {'mean': np.mean(all_resource_usages) if all_resource_usages else 0.0,
                                 'std': np.std(all_resource_usages) if all_resource_usages else 0.0,
                                 'values': all_resource_usages,
                                 'trends': academic_report.get('efficiency_metrics', {}).get('trends',
                                                                                             {}) if academic_report else {}}
    results['query_specific'] = query_specific
    results['academic_report'] = academic_report
    logger.info(
        f"Aggregated metrics for config '{name}': EM Mean={results['exact_match']['mean']:.3f}, F1 Mean={results['f1']['mean']:.3f}")
    return results


def _run_configuration_with_params(  # This remains largely the same as it calls the other helpers
        config_name: str,
        config: Dict[str, Any],
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
    logger.info(f"--- Preparing to run modified configuration: {config_name} ---")
    logger.debug(f"Ablation config params for '{config_name}': {config}")
    modified_system_manager = _create_modified_system_manager(
        config=config, rules_path_baseline=rules_path_baseline, device=device,
        neural_retriever_baseline=neural_retriever_baseline,
        query_expander_baseline=query_expander_baseline,
        response_aggregator_baseline=response_aggregator_baseline,
        resource_manager_baseline=resource_manager_baseline,
        dimensionality_manager_baseline=dimensionality_manager_baseline,
        dataset_type=dataset_type
    )
    modified_metrics = _run_configuration(
        name=config_name, samples=samples, system_manager=modified_system_manager,
        resource_manager=resource_manager_baseline, dataset_type=dataset_type
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
    logger.info(f"Creating SystemControlManager for ablation: {config.get('name', 'Unknown Ablation')}")
    current_rules_file = config.get('rules_file', rules_path_baseline)
    logger.info(f"Ablation '{config.get('name')}': Using rules file: {current_rules_file}")

    sym_match_threshold = config.get('match_threshold', 0.1 if dataset_type.lower() == 'drop' else 0.25)
    sym_max_hops = config.get('max_hops', 3 if dataset_type.lower() == 'drop' else 5)
    sym_embedding_model = config.get('embedding_model', 'all-MiniLM-L6-v2')

    local_symbolic_reasoner: Union[GraphSymbolicReasoner, GraphSymbolicReasonerDrop, DummySymbolicReasoner]
    if config.get('disable_symbolic', False):
        logger.info(f"Ablation '{config.get('name')}': Symbolic reasoner is DISABLED.")
        local_symbolic_reasoner = DummySymbolicReasoner()
    else:
        logger.info(f"Ablation '{config.get('name')}': Initializing Symbolic Reasoner (type: {dataset_type}).")
        if dataset_type.lower() == 'drop':
            local_symbolic_reasoner = GraphSymbolicReasonerDrop(
                rules_file=current_rules_file, match_threshold=sym_match_threshold, max_hops=sym_max_hops,
                embedding_model=sym_embedding_model, device=device, dim_manager=dimensionality_manager_baseline
            ) # [cite: 470]
        else:
            local_symbolic_reasoner = GraphSymbolicReasoner(
                rules_file=current_rules_file, match_threshold=sym_match_threshold, max_hops=sym_max_hops,
                embedding_model=sym_embedding_model, device=device, dim_manager=dimensionality_manager_baseline
            ) # [cite: 471]

    local_neural_retriever: Union[NeuralRetriever, DummyNeuralRetriever]
    if config.get('disable_neural', False):
        logger.info(f"Ablation '{config.get('name')}': Neural retriever is DISABLED.")
        local_neural_retriever = DummyNeuralRetriever()
    else:
        logger.info(f"Ablation '{config.get('name')}': Initializing Neural Retriever.")
        use_few_shots_this_ablation = config.get('use_few_shots', False)
        few_shot_path_for_this_ablation = config.get('few_shot_examples_path') if use_few_shots_this_ablation else None

        if use_few_shots_this_ablation and few_shot_path_for_this_ablation: # [cite: 472]
            logger.info(
                f"Ablation '{config.get('name')}': NR will use few-shot examples from '{few_shot_path_for_this_ablation}'.") # [cite: 472]
        # ... (other logging for few-shot path based on conditions)

        ablation_model_name = config.get('model_name')
        if not ablation_model_name:
            ablation_model_name = neural_retriever_baseline.model.name_or_path if hasattr(
                neural_retriever_baseline.model, 'name_or_path') else "meta-llama/Llama-3.2-3B" # [cite: 473]
            logger.warning(
                f"model_name not found in ablation config for '{config.get('name')}'. Defaulting to '{ablation_model_name}'.") #

        local_neural_retriever = NeuralRetriever(
            model_name=ablation_model_name,
            use_quantization=config.get('neural_use_quantization', False),
            max_context_length=getattr(neural_retriever_baseline, 'max_context_length', 2048),
            chunk_size=getattr(neural_retriever_baseline, 'chunk_size', 512),
            overlap=getattr(neural_retriever_baseline, 'overlap', 128),
            device=device,
            few_shot_examples_path=few_shot_path_for_this_ablation
        ) # [cite: 475]

    logger.info(f"Ablation '{config.get('name')}': Initializing Hybrid Integrator.")
    modified_integrator = HybridIntegrator(
        symbolic_reasoner=local_symbolic_reasoner, neural_retriever=local_neural_retriever,
        query_expander=query_expander_baseline, dim_manager=dimensionality_manager_baseline,
        dataset_type=dataset_type
    )

    safe_config_name_for_dir = config.get('name', 'unknown_ablation').replace(' ', '_').replace('.', '').replace('(',
                                                                                                              '').replace(
        ')', '').replace(',', '').replace(':', '').lower() #
    ablation_scm_metrics_dir = f"logs/{dataset_type}/metrics_ablation_SCM_{safe_config_name_for_dir}"
    scm_metrics_collector = MetricsCollector(dataset_type=dataset_type, metrics_dir=ablation_scm_metrics_dir) #

    # << NEW: Read the use_adaptive_scm_logic flag from the ablation config >>
    use_adaptive_logic_for_scm = config.get('use_adaptive_scm_logic', True)  # Default to True if not specified in ablation_config.yaml

    logger.info(f"Ablation '{config.get('name')}': Initializing SystemControlManager with use_adaptive_logic={use_adaptive_logic_for_scm}.")
    modified_system_manager = SystemControlManager(
        hybrid_integrator=modified_integrator,
        resource_manager=resource_manager_baseline,
        aggregator=response_aggregator_baseline,
        metrics_collector=scm_metrics_collector,
        error_retry_limit=config.get('error_retry_limit', 2),
        max_query_time=config.get('max_query_time', 30.0),
        use_adaptive_logic=use_adaptive_logic_for_scm  # << NEW: Pass the flag here >>
    )
    return modified_system_manager


if __name__ == "__main__":
    # This is for testing the module if run directly, not part of the main flow
    # For example, to test if imports work or a small utility function.
    # The main execution is handled by main.py.
    print("This is the ablation study module. It is intended to be called from main.py.")
    logger.info("Ablation study module executed directly (e.g., for basic checks).")