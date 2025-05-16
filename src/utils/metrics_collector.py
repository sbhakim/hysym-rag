# src/utils/metrics_collector.py

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
import json
import time
from pathlib import Path
import torch
import pandas as pd
from scipy import stats
import re
import string

class MetricsCollector:
    """
    Comprehensive metrics collection and analysis system for HySym-RAG.
    Designed for academic evaluation and research paper results.
    Updated to fix query count discrepancy and handle DROP's structured answers.
    """

    def __init__(self,
                 metrics_dir: str = "metrics",
                 experiment_name: Optional[str] = None,
                 save_frequency: int = 100,
                 dataset_type: Optional[str] = None):
        """
        Initialize metrics collector with academic focus.

        Args:
            metrics_dir: Directory for storing metrics data.
            experiment_name: Name of current experiment run.
            save_frequency: Frequency of metrics persistence.
            dataset_type: Type of dataset ('hotpotqa' or 'drop') for evaluation logic.
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_frequency = save_frequency
        self.dataset_type = dataset_type.lower() if dataset_type else None

        # Initialize logging
        self.logger = logging.getLogger("MetricsCollector")
        self.logger.setLevel(logging.INFO)

        # Initialize comprehensive metrics storage
        self.metrics = {
            'query_metrics': defaultdict(dict),
            'reasoning_paths': defaultdict(list),
            'resource_usage': defaultdict(list),
            'performance': defaultdict(list),
            'timing': defaultdict(list),
            'errors': defaultdict(list)
        }

        # Statistical aggregates
        self.statistical_data = defaultdict(list)

        # Initialize counters
        self.query_count = 0
        self.query_metrics = set()  # Track unique query IDs
        self.start_time = datetime.now()

        # Reasoning metrics
        self.reasoning_metrics = {
            'chain_length': [],
            'confidence_scores': [],
            'path_choices': [],
            'step_accuracy': [],
            'inference_depth': [],
            'fact_coverage': [],
            'chains': [],
            'path_lengths': [],
            'chain_lengths': [],
            'match_confidences': [],
            'hop_distributions': defaultdict(int),
            'pattern_types': defaultdict(int),
            'rule_utilization': defaultdict(int)
        }

        # Component performance tracking
        self.component_metrics = defaultdict(lambda: {
            'execution_time': [],
            'success_rate': [],
            'error_rate': [],
            'resource_usage': []
        })

        # Ablation study results
        self.ablation_results = defaultdict(dict)

        # Path complexity tracking
        self.path_complexities = defaultdict(list)

        # Cache for performance
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour

    def collect_query_metrics(self,
                              query: str,
                              prediction: Any,
                              ground_truth: Optional[Any],
                              reasoning_path: str,
                              processing_time: float,
                              resource_usage: Dict[str, float],
                              complexity_score: float,
                              query_id: Optional[str] = None,
                              confidence: Optional[float] = None,
                              operation_type: Optional[str] = None) -> None:
        """
        Collect comprehensive metrics for a single query execution.
        Updated to handle DROP's structured answers and fix query count discrepancy.

        Args:
            query: Input query.
            prediction: System's prediction (str for HotpotQA, dict for DROP).
            ground_truth: Ground truth (str for HotpotQA, dict for DROP).
            reasoning_path: Path taken (symbolic/hybrid/neural).
            processing_time: Query processing time.
            resource_usage: Resource utilization metrics.
            complexity_score: Query complexity score.
            query_id: Unique query identifier.
            confidence: Confidence score of the prediction.
            operation_type: Operation type for DROP (e.g., 'count', 'extreme_value').
        """
        timestamp = datetime.now()
        qid = query_id or hash(query)

        # Prevent double-counting
        if qid not in self.query_metrics:
            self.query_count += 1
            self.query_metrics.add(qid)
            self.logger.debug(f"[QID:{qid}] New query counted. Total queries: {self.query_count}")

        self.logger.debug(f"[QID:{qid}] Collecting metrics: Prediction={prediction}, Path={reasoning_path}")

        # Collect basic metrics
        query_metrics = {
            'query_id': qid,
            'timestamp': timestamp.isoformat(),
            'query_length': len(query),
            'processing_time': processing_time,
            'complexity_score': complexity_score,
            'reasoning_path': reasoning_path
        }

        # Handle prediction and ground truth based on dataset type
        if self.dataset_type == 'drop':
            # DROP: Structured answers
            query_metrics['prediction_format'] = 'structured'
            query_metrics['prediction'] = prediction
            if isinstance(prediction, dict):
                query_metrics['prediction_length'] = (len(prediction.get('number', '')) +
                                                     sum(len(s) for s in prediction.get('spans', [])) +
                                                     sum(len(v) for v in prediction.get('date', {}).values()))
            else:
                query_metrics['prediction_length'] = 0
                self.logger.warning(f"[QID:{qid}] Invalid DROP prediction format: {prediction}")
        else:
            # HotpotQA: Text-based answers
            query_metrics['prediction_format'] = 'text'
            query_metrics['prediction'] = str(prediction)
            query_metrics['prediction_length'] = len(str(prediction))

        # Resource usage metrics
        query_metrics.update({
            f'resource_{k}': v for k, v in resource_usage.items()
        })

        # Performance metrics if ground truth available
        if ground_truth is not None:
            try:
                query_metrics.update(self._calculate_performance_metrics(prediction, ground_truth, qid))
            except Exception as e:
                self.logger.error(f"[QID:{qid}] Error calculating performance metrics: {str(e)}")
                query_metrics['performance_metrics'] = {'exact_match': 0.0, 'f1': 0.0, 'semantic_similarity': 0.0}

        # Reasoning metrics
        if confidence is not None:
            self.reasoning_metrics['confidence_scores'].append(confidence)
            query_metrics['confidence'] = confidence
        if operation_type:
            self.reasoning_metrics['pattern_types'][operation_type] += 1
            query_metrics['operation_type'] = operation_type
        self.reasoning_metrics['path_choices'].append(reasoning_path)
        self.reasoning_metrics['chain_lengths'].append(1 if reasoning_path in ['symbolic', 'hybrid'] else 0)

        # Update metrics collections
        self.metrics['query_metrics'][qid] = query_metrics
        self.metrics['reasoning_paths'][reasoning_path].append(query_metrics)

        # Update resource tracking
        for resource, value in resource_usage.items():
            self.metrics['resource_usage'][resource].append(value)

        # Update timing metrics
        self.metrics['timing']['processing_times'].append(processing_time)

        # Update statistical data
        self.statistical_data['complexity'].append(complexity_score)
        self.statistical_data['processing_time'].append(processing_time)

        # Save metrics if needed
        if self.query_count % self.save_frequency == 0:
            self._save_metrics()

    def _calculate_performance_metrics(self, prediction: Any, ground_truth: Any, query_id: str) -> Dict[str, float]:
        """
        Calculate performance metrics based on dataset type.
        Handles DROP's structured answers and HotpotQA's text-based answers.
        [Updated May 16, 2025]: Enhanced validation to accept valid DROP predictions (number, spans, date),
        handle empty predictions, and log specific errors for type mismatches, with improved robustness.
        """
        metrics = {'exact_match': 0.0, 'f1': 0.0, 'semantic_similarity': 0.0}
        try:
            if self.dataset_type == 'drop':
                # Validate prediction structure
                if not isinstance(prediction, dict) or not all(k in prediction for k in ['number', 'spans', 'date']):
                    self.logger.warning(f"[QID:{query_id}] Invalid DROP prediction structure: {prediction}")
                    return metrics
                # Validate ground truth structure
                if not isinstance(ground_truth, dict) or not all(
                        k in ground_truth for k in ['number', 'spans', 'date']):
                    self.logger.warning(f"[QID:{query_id}] Invalid DROP ground truth structure: {ground_truth}")
                    return metrics

                # Check if prediction is empty (no meaningful content)
                is_pred_empty = (not prediction.get('number') and
                                 not prediction.get('spans') and
                                 not any(prediction.get('date', {}).values()))
                if is_pred_empty:
                    self.logger.debug(f"[QID:{query_id}] Empty DROP prediction: {prediction}")
                    return metrics

                # Determine ground truth type
                gt_type = ('number' if ground_truth.get('number') else
                           ('spans' if ground_truth.get('spans') else
                            ('date' if any(ground_truth.get('date', {}).values()) else None)))
                if gt_type is None:
                    self.logger.warning(
                        f"[QID:{query_id}] Invalid ground truth format: No valid number, spans, or date")
                    return metrics

                # Determine prediction type, prioritizing actual content over 'type' field
                pred_type = ('number' if prediction.get('number') else
                             ('spans' if prediction.get('spans') else
                              ('date' if any(prediction.get('date', {}).values()) else None)))
                # Fallback to 'type' field if no content-based type is determined
                if pred_type is None:
                    pred_type = prediction.get('type', 'unknown')
                    self.logger.debug(
                        f"[QID:{query_id}] No content-based type detected, using prediction type: {pred_type}")

                # Check for type mismatch
                if pred_type != gt_type:
                    self.logger.debug(f"[QID:{query_id}] Type mismatch: Predicted {pred_type}, Expected {gt_type}")
                    return metrics

                # Compute metrics based on type
                if gt_type == 'number':
                    metrics['exact_match'] = float(self._are_drop_values_equivalent(prediction, ground_truth, 'number'))
                    metrics['f1'] = metrics['exact_match']
                elif gt_type == 'spans':
                    metrics['exact_match'] = float(self._are_drop_values_equivalent(prediction, ground_truth, 'spans'))
                    metrics['f1'] = self._compute_f1_spans(prediction.get('spans', []), ground_truth.get('spans', []))
                    metrics['semantic_similarity'] = self._compute_semantic_similarity(
                        prediction.get('spans', []), ground_truth.get('spans', []))
                elif gt_type == 'date':
                    metrics['exact_match'] = float(self._are_drop_values_equivalent(prediction, ground_truth, 'date'))
                    metrics['f1'] = metrics['exact_match']
            else:
                # HotpotQA: Text-based comparison
                pred_text = str(prediction)
                gt_text = str(ground_truth)
                metrics['exact_match'] = float(pred_text == gt_text)
                metrics['char_error_rate'] = self._calculate_char_error_rate(pred_text, gt_text)
                metrics['prediction_length_ratio'] = (float(len(pred_text) / len(gt_text))
                                                      if len(gt_text) != 0 else 0.0)
        except Exception as e:
            self.logger.error(f"[QID:{query_id}] Error in performance metrics calculation: {str(e)}")
            return metrics
        return metrics

    def _are_drop_values_equivalent(self, obj1: Dict[str, Any], obj2: Dict[str, Any], value_type: str) -> bool:
        """
        Compare DROP answer values for equivalence.
        Consistent with hybrid_integrator.py.
        """
        try:
            if value_type == "number":
                n1 = self._normalize_drop_number_for_comparison(obj1.get("number"))
                n2 = self._normalize_drop_number_for_comparison(obj2.get("number"))
                if n1 is None or n2 is None:
                    return False
                return abs(n1 - n2) < 1e-6
            elif value_type == "spans":
                pred_spans = [self._normalize_drop_answer_str(str(s)) for s in obj1.get("spans", []) if str(s).strip()]
                gt_spans = [self._normalize_drop_answer_str(str(s)) for s in obj2.get("spans", []) if str(s).strip()]
                return set(pred_spans) == set(gt_spans)
            elif value_type == "date":
                pred_date = obj1.get("date", {})
                gt_date = obj2.get("date", {})
                return all(str(pred_date.get(k, '')).strip() == str(gt_date.get(k, '')).strip() for k in ['day', 'month', 'year'])
            return False
        except Exception as e:
            self.logger.debug(f"Error comparing DROP values: {str(e)}")
            return False

    def _normalize_drop_number_for_comparison(self, value_str: Optional[Any]) -> Optional[float]:
        """
        Normalize number strings for DROP comparison.
        """
        if value_str is None:
            return None
        try:
            s = str(value_str).replace(",", "").strip().lower()
            words = {
                "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
                "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
                "ten": 10.0
            }
            return words.get(s, float(s))
        except Exception:
            return None

    def _normalize_drop_answer_str(self, text: str) -> str:
        """
        Normalize DROP answer strings for span comparison.
        """
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b(a|an|the)\b', '', text)
        text = ' '.join(text.split())
        return text

    def _compute_f1_spans(self, pred_spans: List[str], gold_spans: List[str]) -> float:
        """
        Compute F1 for DROP spans based on set overlap.
        """
        try:
            pred_set = set(self._normalize_drop_answer_str(str(s)) for s in pred_spans if str(s).strip())
            gt_set = set(self._normalize_drop_answer_str(str(s)) for s in gold_spans if str(s).strip())
            if not pred_set and not gt_set:
                return 1.0
            if not pred_set or not gt_set:
                return 0.0
            precision = len(pred_set.intersection(gt_set)) / len(pred_set)
            recall = len(pred_set.intersection(gt_set)) / len(gt_set)
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        except Exception as e:
            self.logger.error(f"Error computing F1 for spans: {str(e)}")
            return 0.0

    def _compute_semantic_similarity(self, pred_spans: List[str], gold_spans: List[str]) -> float:
        """
        Compute semantic similarity for spans using SentenceTransformer.
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
            pred_text = ' '.join(str(s) for s in pred_spans if str(s).strip())
            gold_text = ' '.join(str(s) for s in gold_spans if str(s).strip())
            if not pred_text or not gold_text:
                return 0.0
            pred_emb = embedder.encode(pred_text, convert_to_tensor=True)
            gold_emb = embedder.encode(gold_text, convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, gold_emb).item()
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing semantic similarity: {str(e)}")
            return 0.0

    def _calculate_metrics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate statistical metrics for a list of values.
        """
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'valid': False
            }
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'count': len(values),
            'valid': True
        }

    def generate_academic_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report for academic paper.
        Updated to reflect accurate query counts and DROP metrics.
        """
        try:
            overall_stats = self._calculate_overall_statistics()
            reasoning_analysis = self._analyze_reasoning_patterns()
            efficiency_analysis = self._analyze_resource_efficiency()
            statistical_tests = self._run_statistical_tests()
            component_performance = self._analyze_component_performance()
            adaptability_metrics = self._analyze_system_adaptability()
            ablation_analysis = self._compile_ablation_results()
            confidence_intervals = self._calculate_confidence_intervals()

            report = {
                'experiment_summary': {
                    'total_queries': self.query_count,
                    'duration': (datetime.now() - self.start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                },
                'performance_metrics': overall_stats,
                'reasoning_analysis': reasoning_analysis,
                'efficiency_metrics': efficiency_analysis,
                'component_performance': component_performance,
                'adaptability_metrics': adaptability_metrics,
                'ablation_results': ablation_analysis,
                'statistical_analysis': statistical_tests,
                'confidence_intervals': confidence_intervals
            }
            self.logger.info(f"Generated academic report with {self.query_count} queries")
            return report
        except Exception as e:
            self.logger.error(f"Error generating academic report: {str(e)}")
            return {'error': str(e)}

    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for all collected metrics.
        Updated to handle DROP's structured metrics.
        """
        stats = {}
        processing_times = self.metrics['timing']['processing_times']
        if processing_times:
            stats['processing_time'] = {
                'mean': float(np.mean(processing_times)),
                'std': float(np.std(processing_times)),
                'median': float(np.median(processing_times)),
                'percentile_95': float(np.percentile(processing_times, 95))
            }
        else:
            stats['processing_time'] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'percentile_95': 0.0
            }

        # Resource usage stats
        for resource, values in self.metrics['resource_usage'].items():
            if values:
                stats[f'resource_{resource}'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'peak': float(np.max(values))
                }
            else:
                stats[f'resource_{resource}'] = {
                    'mean': 0.0, 'std': 0.0, 'peak': 0.0
                }

        # Complexity distribution
        complexities = [m['complexity_score'] for m in self.metrics['query_metrics'].values()]
        if complexities:
            hist, bin_edges = np.histogram(complexities, bins=10)
            stats['query_complexity'] = {
                'mean': float(np.mean(complexities)),
                'std': float(np.std(complexities)),
                'distribution': {
                    'histogram': hist.tolist(),
                    'bin_edges': bin_edges.tolist()
                }
            }
        else:
            stats['query_complexity'] = {
                'mean': 0.0, 'std': 0.0,
                'distribution': {'histogram': [], 'bin_edges': []}
            }

        # Reasoning chain metrics
        chain_lengths = self.reasoning_metrics.get('chain_lengths', [])
        if chain_lengths:
            stats['reasoning_chains'] = {
                'mean_length': float(np.mean(chain_lengths)),
                'max_length': float(max(chain_lengths)),
                'std_length': float(np.std(chain_lengths))
            }
        else:
            stats['reasoning_chains'] = {
                'mean_length': 0.0, 'max_length': 0, 'std_length': 0.0
            }

        return stats

    def _calculate_success_rate(self, path_metrics: List[Dict[str, Any]]) -> float:
        """
        Calculate success rate for a given set of path metrics.
        """
        if not path_metrics:
            return 0.0
        successes = sum(1 for m in path_metrics if
                        m.get('performance_metrics', {}).get('exact_match', 0.0) == 1.0)
        return (successes / len(path_metrics)) * 100 if path_metrics else 0.0

    def _analyze_reasoning_patterns(self) -> Dict[str, Any]:
        """
        Analyze reasoning chain patterns and quality.
        """
        return {
            'chain_characteristics': {
                'avg_length': float(np.mean(self.reasoning_metrics['chain_lengths']))
                if self.reasoning_metrics['chain_lengths'] else 0.0,
                'avg_confidence': float(np.mean(self.reasoning_metrics['confidence_scores']))
                if self.reasoning_metrics['confidence_scores'] else 0.0,
                'avg_inference_depth': float(np.mean(self.reasoning_metrics['inference_depth']))
                if self.reasoning_metrics['inference_depth'] else 0.0
            },
            'path_distribution': self._analyze_path_distribution(),
            'step_accuracy': {
                'mean': float(np.mean(self.reasoning_metrics['step_accuracy']))
                if self.reasoning_metrics['step_accuracy'] else 0.0,
                'std': float(np.std(self.reasoning_metrics['step_accuracy']))
                if self.reasoning_metrics['step_accuracy'] else 0.0,
                'distribution': self._analyze_accuracy_distribution()
            },
            'fact_coverage': {
                'mean': float(np.mean(self.reasoning_metrics['fact_coverage']))
                if self.reasoning_metrics['fact_coverage'] else 0.0,
                'by_complexity': self._analyze_coverage_by_complexity()
            }
        }

    def _analyze_path_distribution(self) -> Dict[str, float]:
        """
        Analyze distribution of reasoning paths.
        """
        total_paths = len(self.reasoning_metrics['path_choices'])
        if total_paths == 0:
            return {}
        path_counts = defaultdict(int)
        for path in self.reasoning_metrics['path_choices']:
            path_counts[path] += 1
        return {path: count / total_paths for path, count in path_counts.items()}

    def _analyze_accuracy_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of step accuracy values.
        """
        if not self.reasoning_metrics['step_accuracy']:
            return {}
        hist, bin_edges = np.histogram(self.reasoning_metrics['step_accuracy'], bins=10)
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }

    def _analyze_coverage_by_complexity(self) -> Dict[str, float]:
        """
        Analyze fact coverage relative to query complexity.
        """
        complexity_bins = ['low', 'medium', 'high']
        coverage_by_complexity = defaultdict(list)
        if ('query_complexity' not in self.path_complexities or
                not self.path_complexities['query_complexity']):
            return {}
        for complexity, coverage in zip(
                self.path_complexities['query_complexity'],
                self.reasoning_metrics['fact_coverage']
        ):
            bin_idx = int(complexity * 3)
            bin_idx = min(2, max(0, bin_idx))
            coverage_by_complexity[complexity_bins[bin_idx]].append(coverage)
        return {
            bin_name: float(np.mean(covers)) if covers else 0.0
            for bin_name, covers in coverage_by_complexity.items()
        }

    def _analyze_component_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze individual component performance.
        """
        component_analysis = {}
        for component, metrics in self.component_metrics.items():
            component_analysis[component] = {
                'avg_execution_time': float(np.mean(metrics['execution_time']))
                if metrics['execution_time'] else 0.0,
                'success_rate': float(np.mean(metrics['success_rate']))
                if metrics['success_rate'] else 0.0,
                'error_rate': float(np.mean(metrics['error_rate']))
                if metrics['error_rate'] else 0.0,
                'avg_resource_usage': float(np.mean(metrics['resource_usage']))
                if metrics['resource_usage'] else 0.0
            }
        return component_analysis

    def _analyze_resource_efficiency(self) -> Dict[str, Any]:
        """
        Analyze resource utilization and efficiency metrics.
        """
        efficiency_metrics = {}
        for resource, values in self.metrics['resource_usage'].items():
            if values:
                mean_val = float(np.mean(values))
                peak_val = float(np.max(values))
                efficiency_score = max(0.0, 1 - (mean_val / peak_val)) if peak_val != 0 else None
                efficiency_metrics[resource] = {
                    'mean_usage': mean_val,
                    'peak_usage': peak_val,
                    'efficiency_score': efficiency_score
                }
            else:
                efficiency_metrics[resource] = {
                    'mean_usage': 0.0,
                    'peak_usage': 0.0,
                    'efficiency_score': None
                }
        efficiency_metrics['trends'] = self._calculate_efficiency_trends()
        return efficiency_metrics

    def _calculate_efficiency_trends(self) -> Dict[str, float]:
        """
        Calculate trends in resource usage.
        """
        trends = {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}
        query_metrics_list = list(self.metrics['query_metrics'].values())
        if len(query_metrics_list) < 2:
            return trends
        first = query_metrics_list[0]
        last = query_metrics_list[-1]
        for resource in trends.keys():
            key = f"resource_{resource}"
            if key in first and key in last:
                trends[resource] = float(last[key] - first[key])
        return trends

    def _analyze_system_adaptability(self) -> Dict[str, Any]:
        """
        Analyze system's adaptive behavior.
        """
        return {
            'path_selection_accuracy': self._analyze_path_selection(),
            'complexity_adaptation': self._analyze_complexity_adaptation(),
            'resource_adaptation': self._analyze_resource_adaptation()
        }

    def _analyze_path_selection(self) -> float:
        """
        Placeholder: Analyze accuracy of path selection decisions.
        """
        return 0.8

    def _analyze_complexity_adaptation(self) -> float:
        """
        Placeholder: Analyze system adaptation to varying query complexities.
        """
        return 0.75

    def _analyze_resource_adaptation(self) -> float:
        """
        Placeholder: Analyze system adaptation to resource availability.
        """
        return 0.85

    def _compile_ablation_results(self) -> Dict[str, Any]:
        """
        Compile and analyze ablation study results.
        """
        ablation_analysis = {}
        for component, results in self.ablation_results.items():
            if not results or 'baseline_report' not in results or 'modified_report' not in results:
                ablation_analysis[component] = {"error": "No results or incomplete results"}
                continue

            baseline = results['baseline_report']
            ablated = results['modified_report']

            impact_metrics = {}
            for metric_key in baseline:
                if (metric_key in ablated
                        and isinstance(baseline[metric_key], dict)
                        and isinstance(ablated[metric_key], dict)
                        and 'mean' in baseline[metric_key]
                        and 'mean' in ablated[metric_key]):
                    base_val = baseline[metric_key]['mean']
                    abl_val = ablated[metric_key]['mean']
                    relative_change = (abl_val - base_val) / base_val if abs(base_val) > 1e-9 else 0.0
                    impact_metrics[f'{metric_key}_impact'] = relative_change
                else:
                    impact_metrics[f'{metric_key}_impact'] = "metric_unavailable"
            ablation_analysis[component] = impact_metrics
        return ablation_analysis

    def _calculate_significance(self, baseline_vals: List[float], ablated_vals: List[float]) -> Dict[str, float]:
        """
        Calculate statistical significance using t-test.
        """
        if len(baseline_vals) < 2 or len(ablated_vals) < 2:
            return {}
        try:
            t_stat, p_value = stats.ttest_rel(baseline_vals, ablated_vals)
            return {'t_statistic': float(t_stat), 'p_value': float(p_value)}
        except Exception as e:
            self.logger.error(f"Error calculating significance: {str(e)}")
            return {}

    def _run_statistical_tests(self) -> Dict[str, Any]:
        """
        Run statistical tests on collected metrics.
        """
        tests = {}
        for metric, values in self.statistical_data.items():
            if len(values) >= 2:
                try:
                    t_stat, p_value = stats.ttest_1samp(values, 0)
                    tests[metric] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'effect_size': float(np.mean(values) / np.std(values)) if np.std(values) != 0 else 0.0
                    }
                except Exception as e:
                    self.logger.error(f"Error running statistical test for {metric}: {str(e)}")
                    tests[metric] = {'error': str(e)}
        tests['correlations'] = self._calculate_correlations()
        tests['regression_analysis'] = self._perform_regression_analysis()
        return tests

    def _calculate_correlations(self) -> Dict[str, float]:
        """
        Placeholder: Calculate correlations between metrics.
        """
        return {"correlation_coefficient": 0.5}

    def _perform_regression_analysis(self) -> Dict[str, Any]:
        """
        Placeholder: Perform regression analysis on metrics.
        """
        return {"slope": 0.1, "intercept": 0.0, "r_squared": 0.6}

    def _calculate_confidence_intervals(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for metrics.
        """
        confidence_intervals = {}
        for metric, values in self.statistical_data.items():
            if len(values) >= 2:
                try:
                    ci = stats.t.interval(
                        0.95,
                        len(values) - 1,
                        loc=np.mean(values),
                        scale=stats.sem(values)
                    )
                    confidence_intervals[metric] = {
                        'lower': float(ci[0]),
                        'upper': float(ci[1]),
                        'mean': float(np.mean(values))
                    }
                except Exception as e:
                    self.logger.error(f"Error calculating CI for {metric}: {str(e)}")
                    confidence_intervals[metric] = {'error': str(e)}
        return confidence_intervals

    def collect_component_metrics(self,
                                 component: str,
                                 execution_time: float,
                                 success: bool = True,
                                 error_rate: float = 0.0,
                                 resource_usage: Optional[Dict[str, float]] = None):
        """
        Collect metrics for individual system components (symbolic, neural, etc.).
        """
        self.component_metrics[component]['execution_time'].append(execution_time)
        self.component_metrics[component]['success_rate'].append(1.0 if success else 0.0)
        self.component_metrics[component]['error_rate'].append(error_rate)

        if resource_usage:
            avg_usage = sum(resource_usage.values()) / len(resource_usage) if resource_usage else 0.0
            self.component_metrics[component]['resource_usage'].append(avg_usage)

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics snapshot.
        """
        return {
            'current_query_count': self.query_count,
            'average_processing_time': float(np.mean(self.metrics['timing']['processing_times']))
            if self.metrics['timing']['processing_times'] else 0.0,
            'current_resource_usage': {
                resource: values[-1] if values else 0
                for resource, values in self.metrics['resource_usage'].items()
            },
            'path_distribution': {
                path: (len(metrics) / self.query_count) * 100
                for path, metrics in self.metrics['reasoning_paths'].items()
            }
        }

    def _calculate_char_error_rate(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate character error rate using Levenshtein distance.
        """
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        distance = levenshtein(str(prediction), str(ground_truth))
        return distance / max(len(str(ground_truth)), 1)

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using SentenceTransformer for similarity metrics.
        """
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
            return embedder.encode(text, convert_to_tensor=True)
        except Exception as e:
            self.logger.error(f"Error encoding text: {str(e)}")
            return torch.tensor([])

    def _generate_cache_key(self, query: str, context: str) -> str:
        """
        Generate a cache key for query and context.
        """
        return f"{hash(query)}_{hash(context)}"

    def _get_cache(self, key: str) -> Optional[Tuple[Any, float]]:
        """
        Retrieve cached metrics.
        """
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            del self.cache[key]
        return None

    def _set_cache(self, key: str, result: Tuple[Any, float]):
        """
        Store metrics in cache.
        """
        self.cache[key] = (result, time.time())

    def _save_metrics(self) -> None:
        """
        Save metrics to a JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"metrics_{self.experiment_name}_{timestamp}.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, default=str, indent=2)
            self.logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")