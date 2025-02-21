# src/utils/evaluation.py

import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import torch
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class Evaluation:
    """
    Enhanced evaluation system for HySym-RAG with comprehensive academic metrics.
    Includes support for multi-hop reasoning evaluation, resource efficiency,
    and detailed performance analysis. **Added ROUGE and BLEU metrics.**
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

    def evaluate(self,
                 predictions: Dict[str, str],
                 ground_truths: Dict[str, Union[str, Dict[str, Any]]],
                 supporting_facts: Optional[Dict[str, List[Tuple[str, int]]]] = None,
                 resource_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of system predictions. **Added ROUGE and BLEU metrics to output.**

        Args:
            predictions: Dictionary of query-prediction pairs.
            ground_truths: Dictionary of query-ground truth pairs. The value can be a string or a dict
                           (in which case, the 'answer' key will be used).
            supporting_facts: Optional supporting facts for multi-hop evaluation.
            resource_metrics: Optional resource usage metrics.

        Returns:
            Dictionary containing comprehensive evaluation metrics.
        """
        metrics = {
            'exact_match': [],
            'semantic_similarity': [],
            'factual_accuracy': [],
            'reasoning_quality': [],
            'multi_hop_accuracy': [],
            'rougeL': [],  # Added ROUGE-L
            'bleu': []     # Added BLEU
        }

        # Process each prediction
        for query, prediction in predictions.items():
            if query not in ground_truths:
                continue

            # Extract prediction text
            pred_text = prediction['result'] if isinstance(prediction, dict) else prediction
            ground_truth = ground_truths[query]

            # If ground_truth is a dict (e.g., HotpotQA style), use its 'answer' field
            if isinstance(ground_truth, dict) and 'answer' in ground_truth:
                truth_text = ground_truth['answer']
            else:
                truth_text = ground_truth

            # Calculate base metrics
            metrics['exact_match'].append(
                self._calculate_exact_match(pred_text, truth_text)
            )

            metrics['semantic_similarity'].append(
                self._calculate_semantic_similarity(pred_text, truth_text)
            )

            # Calculate ROUGE-L
            rouge_l_score = self._calculate_rouge_l(pred_text, truth_text)
            metrics['rougeL'].append(rouge_l_score)

            # Calculate BLEU
            bleu_score_val = self._calculate_bleu(pred_text, truth_text)
            metrics['bleu'].append(bleu_score_val)

            # Calculate reasoning quality if multi-hop supporting facts are provided
            if supporting_facts and query in supporting_facts:
                metrics['reasoning_quality'].append(
                    self._evaluate_reasoning_quality(pred_text, supporting_facts[query])
                )
                metrics['multi_hop_accuracy'].append(
                    self._evaluate_multi_hop_accuracy(pred_text, supporting_facts[query])
                )

        # Calculate resource efficiency if resource metrics provided
        if resource_metrics:
            efficiency_score = self._calculate_efficiency_score(resource_metrics)
            metrics['resource_efficiency'] = [efficiency_score]

        return self._aggregate_metrics(metrics)

    def _calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate normalized exact match score with robust text normalization.
        """
        pred_norm = self._normalize_text(prediction)
        truth_norm = self._normalize_text(ground_truth)
        return float(pred_norm == truth_norm)

    def _calculate_semantic_similarity(self,
                                       prediction: str,
                                       ground_truth: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        Handles errors gracefully and includes confidence scoring.
        """
        try:
            # Generate embeddings
            pred_emb = self.embedder.encode(prediction, convert_to_tensor=True)
            truth_emb = self.embedder.encode(ground_truth, convert_to_tensor=True)

            # Calculate similarity
            similarity = util.cos_sim(pred_emb, truth_emb).item()

            # Apply confidence threshold
            confidence = max(0.0, min(1.0, similarity))
            return confidence

        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def _calculate_rouge_l(self, prediction: str, ground_truth: str) -> float:
        """Calculate ROUGE-L score."""
        scores = self.rouge_scorer.score(ground_truth, prediction)
        return scores['rougeL'].fmeasure

    def _calculate_bleu(self, prediction: str, ground_truth: str) -> float:
        """Calculate BLEU score."""
        reference = [ground_truth.split()]  # BLEU expects list of lists for reference
        candidate = prediction.split()
        return sentence_bleu(reference, candidate, smoothing_function=self.bleu_smoothing)

    def _evaluate_reasoning_quality(self,
                                    prediction: str,
                                    supporting_facts: List[Tuple[str, int]]) -> float:
        """
        Evaluate the quality of the reasoning path.
        Considers fact coverage, logical coherence, and step validity.
        """
        steps = self._extract_reasoning_steps(prediction)
        fact_coverage = self._calculate_fact_coverage(steps, supporting_facts)
        coherence_score = self._evaluate_logical_coherence(steps)
        quality_score = (0.6 * fact_coverage + 0.4 * coherence_score)
        return quality_score

    def _evaluate_multi_hop_accuracy(self,
                                     prediction: str,
                                     supporting_facts: List[Tuple[str, int]]) -> float:
        """
        Evaluate accuracy of multi-hop reasoning steps.
        Checks both intermediate and final conclusions.
        """
        steps = self._extract_reasoning_steps(prediction)
        if not steps:
            return 0.0
        step_scores = []
        for step in steps:
            step_score = self._evaluate_step_accuracy(step, supporting_facts)
            step_scores.append(step_score)
        weights = np.linspace(0.5, 1.0, len(step_scores))
        weighted_score = np.average(step_scores, weights=weights)
        return weighted_score

    def _calculate_efficiency_score(self, resource_metrics: Dict[str, float]) -> float:
        """
        Calculate efficiency score based on resource usage metrics.
        Lower resource usage yields higher efficiency score.
        """
        weights = {
            'cpu': 0.3,
            'memory': 0.3,
            'gpu': 0.4
        }
        efficiency_score = 0.0
        for resource, usage in resource_metrics.items():
            if resource in weights:
                efficiency = 1.0 - min(1.0, usage)
                efficiency_score += weights[resource] * efficiency
        return efficiency_score

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent comparison.
        Handles various text normalization cases.
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text.strip()

    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """
        Extract individual reasoning steps from prediction text.
        Handles various reasoning step formats.
        """
        step_matches = re.findall(r'Step \d+:.*?(?=Step \d+:|$)', text, re.DOTALL)
        if step_matches:
            return [step.strip() for step in step_matches]
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences

    def _calculate_fact_coverage(self,
                                 steps: List[str],
                                 supporting_facts: List[Tuple[str, int]]) -> float:
        """
        Calculate how well the reasoning steps cover the supporting facts.
        """
        if not supporting_facts:
            return 0.0
        covered_facts = 0
        for fact in supporting_facts:
            fact_text = fact[0]
            for step in steps:
                if self._calculate_semantic_similarity(step, fact_text) > self.semantic_threshold:
                    covered_facts += 1
                    break
        return covered_facts / len(supporting_facts)

    def _evaluate_logical_coherence(self, steps: List[str]) -> float:
        """
        Evaluate logical coherence between reasoning steps.
        Higher score indicates better logical flow.
        """
        if len(steps) < 2:
            return 1.0
        coherence_scores = []
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            step_coherence = self._calculate_semantic_similarity(current_step, next_step)
            coherence_scores.append(step_coherence)
        return np.mean(coherence_scores)

    def _evaluate_step_accuracy(self,
                                step: str,
                                supporting_facts: List[Tuple[str, int]]) -> float:
        """
        Evaluate accuracy of individual reasoning step.
        """
        step_scores = []
        for fact in supporting_facts:
            fact_text = fact[0]
            similarity = self._calculate_semantic_similarity(step, fact_text)
            step_scores.append(similarity)
        return max(step_scores) if step_scores else 0.0

    def _aggregate_metrics(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Aggregate all metrics into final scores.
        Includes confidence intervals where appropriate.
        """
        aggregated = {}
        for metric, values in metrics.items():
            if values:
                aggregated[f"average_{metric}"] = float(np.mean(values))
                aggregated[f"max_{metric}"] = float(np.max(values))
                aggregated[f"min_{metric}"] = float(np.min(values))
                if len(values) >= 5:
                    ci = np.percentile(values, [2.5, 97.5])
                    aggregated[f"{metric}_ci_lower"] = float(ci[0])
                    aggregated[f"{metric}_ci_upper"] = float(ci[1])
        if all(f"average_{metric}" in aggregated for metric in self.metric_weights):
            weighted_score = 0.0
            for metric, weight in self.metric_weights.items():
                weighted_score += aggregated[f"average_{metric}"] * weight
            aggregated['overall_score'] = weighted_score
        return aggregated

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary for academic reporting.
        """
        return {
            'metrics': self.metric_history,
            'performance_stats': self.performance_stats,
            'efficiency_analysis': self._analyze_efficiency(),
            'reasoning_analysis': self._analyze_reasoning_paths()
        }
