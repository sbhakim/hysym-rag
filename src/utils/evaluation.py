# src/utils/evaluation.py

import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import logging
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from scipy import stats


class Evaluation:
    """
    Enhanced evaluation system for HySym-RAG with comprehensive academic metrics.
    Includes support for multi-hop reasoning evaluation, resource efficiency,
    advanced reasoning-quality metrics, ablation tracking, and significance testing.
    """

    def __init__(self,
                 use_semantic_scoring: bool = True,
                 semantic_threshold: float = 0.7,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the evaluation system with configurable parameters.

        Args:
            use_semantic_scoring: Whether to use semantic similarity scoring.
            semantic_threshold: Threshold for semantic similarity matches.
            embedding_model: Model to use for semantic embeddings.
        """
        # Initialize semantic scoring components
        self.use_semantic_scoring = use_semantic_scoring
        self.semantic_threshold = semantic_threshold
        self.embedder = SentenceTransformer(embedding_model)

        # Set up logging
        self.logger = logging.getLogger("Evaluation")
        self.logger.setLevel(logging.INFO)

        # Initialize metric tracking
        self.metric_history = defaultdict(list)

        # Define comprehensive metric weights for academic evaluation
        self.metric_weights = {
            'answer_accuracy': 0.3,       # Base answer correctness
            'reasoning_fidelity': 0.2,    # Quality of reasoning path
            'factual_consistency': 0.2,   # Consistency with known facts
            'multi_hop_coherence': 0.2,   # Multi-hop reasoning quality
            'resource_efficiency': 0.1    # Computational efficiency
        }

        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_time': 0.0,
            'path_performance': defaultdict(list)
        }

        # Initialize ROUGE scorer and BLEU smoothing
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bleu_smoothing = SmoothingFunction().method1

        # --- New reasoning metrics ---
        # Store step-level or path-level metrics for advanced evaluation
        self.reasoning_metrics = {
            'path_coherence': [],
            'fact_coverage': [],
            'inference_depth': [],
            'step_accuracy': []
        }

        # Ablation study tracking
        self.ablation_results = defaultdict(list)

        # Statistical data for significance tests
        self.statistical_data = defaultdict(list)

    def evaluate(self,
                 predictions: Dict[str, str],
                 ground_truths: Dict[str, Union[str, Dict[str, Any]]],
                 supporting_facts: Optional[Dict[str, List[Tuple[str, int]]]] = None,
                 reasoning_chain: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, float]:
        """
        Comprehensive evaluation of system predictions. Includes ROUGE, BLEU,
        multi-hop checks, and optionally advanced reasoning metrics if a
        reasoning_chain is provided.

        Args:
            predictions: Dictionary of {query: predicted_answer} or {query: (answer, ...)}.
            ground_truths: Dictionary of {query: ground_truth} (string or dict with 'answer' key).
            supporting_facts: Optional dict of {query: list of (fact_text, index)} for multi-hop.
            reasoning_chain: Optional dict with details about the chain (pattern_type, steps, etc.).

        Returns:
            A dictionary containing aggregated metrics:
            - average_semantic_similarity
            - average_rougeL
            - average_bleu
            - average_f1
            - reasoning_analysis (if reasoning_chain is provided)
            ...
        """
        # Basic container for all metrics we accumulate
        metrics = {
            'exact_match': [],
            'semantic_similarity': [],
            'rougeL': [],
            'bleu': []
        }

        # We also keep track of the final F1 or partial scoring if needed
        # (Optional placeholder: you can implement a real F1 if you want.)
        f1_scores = []

        for query, pred_value in predictions.items():
            # Extract predicted text
            if isinstance(pred_value, (tuple, list)):
                # If your system returns (answer, debug_info)
                pred_text = pred_value[0]
            else:
                pred_text = pred_value

            # Retrieve ground truth
            if query not in ground_truths:
                continue

            gt = ground_truths[query]
            if isinstance(gt, dict) and 'answer' in gt:
                truth_text = gt['answer']
            else:
                truth_text = gt

            # 1) Exact match
            em = float(self._normalize_text(pred_text) == self._normalize_text(truth_text))
            metrics['exact_match'].append(em)

            # 2) Semantic similarity
            sem_sim = self._calculate_semantic_similarity(pred_text, truth_text)
            metrics['semantic_similarity'].append(sem_sim)

            # 3) ROUGE-L
            rouge_l_val = self._calculate_rouge_l(pred_text, truth_text)
            metrics['rougeL'].append(rouge_l_val)

            # 4) BLEU
            bleu_val = self._calculate_bleu(pred_text, truth_text)
            metrics['bleu'].append(bleu_val)

            # 5) (Optional) F1 placeholder
            # You could implement a real token-level F1. For demonstration:
            f1_scores.append( (2.0 * em * sem_sim) / (em + sem_sim + 1e-9) )

        # Summarize core text metrics
        aggregated = {}
        aggregated['average_semantic_similarity'] = float(np.mean(metrics['semantic_similarity'])) if metrics['semantic_similarity'] else 0.0
        aggregated['average_rougeL'] = float(np.mean(metrics['rougeL'])) if metrics['rougeL'] else 0.0
        aggregated['average_bleu'] = float(np.mean(metrics['bleu'])) if metrics['bleu'] else 0.0
        aggregated['average_f1'] = float(np.mean(f1_scores)) if f1_scores else 0.0

        # If a reasoning_chain was passed, do advanced reasoning analysis
        if reasoning_chain:
            # Example usage: analyzing pattern_type, chain_length, etc.
            reasoning_analysis = {
                'pattern_type': reasoning_chain.get('pattern_type', 'unknown'),
                'chain_length': reasoning_chain.get('hop_count', 0),
                'pattern_confidence': reasoning_chain.get('pattern_confidence', 0.0)
            }
            aggregated['reasoning_analysis'] = reasoning_analysis

        return aggregated

    # ---------------------------------------------------------------------
    # The following methods remain the same as your original code
    # but we incorporate them for completeness.
    # ---------------------------------------------------------------------

    def _calculate_semantic_similarity(self,
                                       prediction: str,
                                       ground_truth: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        """
        try:
            pred_emb = self.embedder.encode(prediction, convert_to_tensor=True)
            truth_emb = self.embedder.encode(ground_truth, convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, truth_emb).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def _calculate_rouge_l(self, prediction: str, ground_truth: str) -> float:
        """Calculate ROUGE-L score."""
        scores = self.rouge_scorer.score(ground_truth, prediction)
        return scores['rougeL'].fmeasure

    def _calculate_bleu(self, prediction: str, ground_truth: str) -> float:
        """Calculate BLEU score."""
        reference = [ground_truth.split()]
        candidate = prediction.split()
        return sentence_bleu(reference, candidate, smoothing_function=self.bleu_smoothing)

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent comparison.
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text.strip()

    # ---------------------------------------------------------------------
    # Optionally, advanced reasoning path evaluation (if you want more detail)
    # and ablation + statistical significance. These are placeholders you can
    # call from your code if you store the data properly.
    # ---------------------------------------------------------------------

    def evaluate_reasoning_quality(self,
                                   prediction: str,
                                   reasoning_path: List[Dict],
                                   supporting_facts: Optional[List[Tuple[str, int]]] = None
                                   ) -> Dict[str, float]:
        """
        Evaluate multi-hop reasoning quality (fact coverage, path coherence, etc.)
        """
        metrics = {}

        # Step-level accuracy
        step_acc = self._evaluate_step_accuracy(reasoning_path, supporting_facts)
        metrics['step_accuracy'] = step_acc

        # Fact coverage
        fact_cov = self._calculate_fact_coverage(reasoning_path, supporting_facts)
        metrics['fact_coverage'] = fact_cov

        # Path coherence
        coherence = self._evaluate_path_coherence(reasoning_path)
        metrics['path_coherence'] = coherence

        # Depth
        metrics['inference_depth'] = len(reasoning_path)

        return metrics

    def _evaluate_step_accuracy(self,
                                reasoning_path: List[Dict],
                                supporting_facts: Optional[List[Tuple[str, int]]]
                                ) -> float:
        """
        Evaluate accuracy of each step vs. supporting_facts or partial checks.
        """
        if not reasoning_path or not supporting_facts:
            return 0.0

        # Simple approach: for each step, see if it semantically matches any supporting fact
        step_scores = []
        for step_dict in reasoning_path:
            step_text = step_dict.get('content', '')
            best_sim = 0.0
            for fact_text, _ in supporting_facts:
                sim = self._calculate_semantic_similarity(step_text, fact_text)
                if sim > best_sim:
                    best_sim = sim
            step_scores.append(best_sim)

        return float(np.mean(step_scores)) if step_scores else 0.0

    def _calculate_fact_coverage(self,
                                 reasoning_path: List[Dict],
                                 supporting_facts: Optional[List[Tuple[str, int]]]
                                 ) -> float:
        """
        How many supporting facts are matched by at least one step in the path?
        """
        if not supporting_facts or not reasoning_path:
            return 0.0

        covered = 0
        total = len(supporting_facts)
        for fact_text, _ in supporting_facts:
            matched = False
            for step_dict in reasoning_path:
                step_text = step_dict.get('content', '')
                sim = self._calculate_semantic_similarity(step_text, fact_text)
                if sim >= self.semantic_threshold:
                    matched = True
                    break
            if matched:
                covered += 1

        return covered / total if total > 0 else 0.0

    def _evaluate_path_coherence(self, reasoning_path: List[Dict]) -> float:
        """
        Evaluate semantic coherence among consecutive steps.
        """
        if len(reasoning_path) < 2:
            return 1.0
        sims = []
        for i in range(len(reasoning_path) - 1):
            step_text_1 = reasoning_path[i].get('content', '')
            step_text_2 = reasoning_path[i+1].get('content', '')
            sim = self._calculate_semantic_similarity(step_text_1, step_text_2)
            sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0

    # Ablation tracking

    def calculate_ablation_metrics(self,
                                   component: str,
                                   base_performance: Dict[str, float],
                                   ablated_performance: Dict[str, float]
                                   ) -> Dict[str, float]:
        """
        Compare baseline vs. ablated performance for a given component.
        """
        impact_metrics = {}
        for metric in base_performance:
            if metric in ablated_performance:
                base_val = base_performance[metric]
                abl_val = ablated_performance[metric]
                if abs(base_val) > 1e-9:
                    relative_change = (abl_val - base_val) / base_val
                else:
                    relative_change = 0.0
                impact_metrics[f'{metric}_impact'] = relative_change

        self.ablation_results[component].append(impact_metrics)
        return impact_metrics

    # Statistical significance

    def record_metric(self, metric_name: str, value: float):
        """
        Store a single metric value for significance testing.
        """
        self.statistical_data[metric_name].append(value)

    def calculate_statistical_significance(self) -> Dict[str, Dict[str, float]]:
        """
        Perform a simple t-test across collected metric arrays.
        """
        significance_results = {}
        for metric, values in self.statistical_data.items():
            if len(values) >= 2:
                t_stat, p_value = stats.ttest_1samp(values, 0.0)
                significance_results[metric] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        return significance_results
