# src/utils/evaluation.py

import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import logging
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from scipy import stats
import string  # For punctuation removal in DROP tokenization

class Evaluation:
    """
    Enhanced evaluation system for HySym-RAG with comprehensive academic metrics.
    Includes support for multi-hop reasoning evaluation (HotpotQA) and discrete reasoning (DROP),
    resource efficiency, advanced reasoning-quality metrics, ablation tracking, and significance testing.
    Updated to robustly handle DROP's structured answer format while preserving HotpotQA functionality.
    """

    def __init__(self,
                 dataset_type: Optional[str] = None,
                 use_semantic_scoring: bool = True,
                 semantic_threshold: float = 0.7,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the evaluation system with configurable parameters.

        Args:
            dataset_type: Type of the dataset being evaluated (e.g., 'hotpotqa', 'drop').
            use_semantic_scoring: Whether to use semantic similarity scoring (for text).
            semantic_threshold: Threshold for semantic similarity matches.
            embedding_model: Model to use for semantic embeddings.
        """
        self.dataset_type = dataset_type.lower() if dataset_type else None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Initialize semantic scoring components (primarily for text-based QA)
        self.use_semantic_scoring = use_semantic_scoring
        self.semantic_threshold = semantic_threshold
        if self.use_semantic_scoring or self.dataset_type != 'drop':
            try:
                self.embedder = SentenceTransformer(embedding_model)
                self.logger.info(f"Successfully loaded SentenceTransformer model '{embedding_model}'.")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load SentenceTransformer model '{embedding_model}': {e}. Semantic scoring will be disabled.")
                self.embedder = None
                self.use_semantic_scoring = False
        else:
            self.embedder = None

        # Initialize metric tracking
        self.metric_history = defaultdict(list)

        # Define comprehensive metric weights for academic evaluation (less relevant for DROP)
        self.metric_weights = {
            'answer_accuracy': 0.3,
            'reasoning_fidelity': 0.2,
            'factual_consistency': 0.2,
            'multi_hop_coherence': 0.2,
            'resource_efficiency': 0.1
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

        # Reasoning metrics (primarily for multi-hop text QA)
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

        # Cache for semantic similarity to optimize performance
        self.similarity_cache = {}

    def evaluate(self,
                 predictions: Dict[str, Any],
                 ground_truths: Dict[str, Union[str, Dict[str, Any]]],
                 supporting_facts: Optional[Dict[str, List[Tuple[str, int]]]] = None,
                 reasoning_chain: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, float]:
        """
        Comprehensive evaluation of system predictions for HotpotQA and DROP.
        Uses query_id as the key for predictions and ground_truths.
        """
        all_query_metrics = []
        self.performance_stats['total_queries'] += len(predictions)

        for query_id, pred_value in predictions.items():
            if query_id not in ground_truths:
                self.logger.warning(f"[QID:{query_id}] No ground truth for query")
                continue

            gt_value = ground_truths[query_id]
            current_query_metrics = {"query_id": query_id}

            try:
                if self.dataset_type == 'drop':
                    # DROP evaluation
                    drop_eval = self._evaluate_drop_answer(pred_value, gt_value, query_id)
                    current_query_metrics.update(drop_eval)
                    # ROUGE-L and BLEU for DROP (stringified answers)
                    pred_str = self._stringify_drop_answer(pred_value, query_id)
                    gt_str = self._stringify_drop_answer(gt_value, query_id)
                    current_query_metrics['rougeL'] = self._calculate_rouge_l(pred_str, gt_str, query_id)
                    current_query_metrics['bleu'] = self._calculate_bleu(pred_str, gt_str, query_id)
                else:  # HotpotQA / text-based evaluation
                    # Extract predicted text
                    pred_text = pred_value[0] if isinstance(pred_value, tuple) else str(pred_value)
                    # Extract ground truth text
                    if isinstance(gt_value, dict) and 'answer' in gt_value:
                        truth_text = str(gt_value['answer'])
                    else:
                        truth_text = str(gt_value)

                    # Exact match
                    em = float(self._normalize_text(pred_text) == self._normalize_text(truth_text))
                    current_query_metrics['exact_match_text'] = em

                    # Semantic similarity
                    if self.use_semantic_scoring and self.embedder:
                        sem_sim = self._calculate_semantic_similarity(pred_text, truth_text, query_id)
                        current_query_metrics['semantic_similarity'] = sem_sim
                    else:
                        current_query_metrics['semantic_similarity'] = 0.0

                    # ROUGE-L
                    current_query_metrics['rougeL'] = self._calculate_rouge_l(pred_text, truth_text, query_id)

                    # BLEU
                    current_query_metrics['bleu'] = self._calculate_bleu(pred_text, truth_text, query_id)

                    # F1 (text-based)
                    sem_sim_for_f1 = current_query_metrics.get('semantic_similarity', 0.0)
                    current_query_metrics['f1_text'] = (2.0 * em * sem_sim_for_f1) / (em + sem_sim_for_f1 + 1e-9) if (
                        em + sem_sim_for_f1) > 0 else 0.0

                    # Reasoning analysis
                    if reasoning_chain and reasoning_chain.get(query_id):
                        rc = reasoning_chain[query_id]
                        current_query_metrics['reasoning_analysis'] = {
                            'pattern_type': rc.get('pattern_type', 'unknown'),
                            'chain_length': rc.get('hop_count', rc.get('chain_length', 0)),
                            'pattern_confidence': rc.get('pattern_confidence', 0.0)
                        }

                all_query_metrics.append(current_query_metrics)
                self.performance_stats['successful_queries'] += 1

            except Exception as e:
                self.logger.error(f"[QID:{query_id}] Error evaluating query: {str(e)}")
                current_query_metrics.update({
                    'exact_match': 0.0,
                    'f1': 0.0,
                    'rougeL': 0.0,
                    'bleu': 0.0,
                    'semantic_similarity': 0.0
                })
                all_query_metrics.append(current_query_metrics)

        # Aggregate metrics
        aggregated_metrics = {
            'average_exact_match': 0.0,
            'average_f1': 0.0,
            'average_semantic_similarity': 0.0,
            'average_rougeL': 0.0,
            'average_bleu': 0.0
        }
        if not all_query_metrics:
            self.logger.info("No queries processed for evaluation")
            return aggregated_metrics

        metric_values = defaultdict(list)
        for q_metrics in all_query_metrics:
            for key, value in q_metrics.items():
                if key in ['exact_match', 'f1', 'semantic_similarity', 'rougeL', 'bleu'] and isinstance(value, (int, float)):
                    metric_values[key].append(value)
                    self.logger.debug(f"[QID:{q_metrics['query_id']}] Recorded {key}: {value:.2f}")

        for metric_name, values_list in metric_values.items():
            if values_list:
                avg_value = float(np.mean(values_list))
                aggregated_metrics[f'average_{metric_name}'] = avg_value
                self.metric_history[metric_name].append(avg_value)
                self.logger.debug(f"Aggregated {metric_name}: {avg_value:.2f} (from {len(values_list)} queries)")
            else:
                aggregated_metrics[f'average_{metric_name}'] = 0.0
                self.logger.debug(f"No values for {metric_name} to aggregate")

        self.logger.info(f"Evaluation metrics: {aggregated_metrics}")
        return aggregated_metrics

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent comparison (for HotpotQA).
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        self.logger.debug(f"Normalized text: '{text[:50]}...'")
        return text.strip()

    def _compute_f1_spans(self, pred_spans: List[str], gold_spans: List[str], query_id: str) -> float:
        """
        Compute F1 for DROP spans with semantic similarity for partial matches.
        """
        try:
            pred_set = set(self._normalize_drop_answer_str(str(s), query_id) for s in pred_spans if str(s).strip())
            gt_set = set(self._normalize_drop_answer_str(str(s), query_id) for s in gold_spans if str(s).strip())
            self.logger.debug(f"[QID:{query_id}] Pred spans: {pred_set}, Gold spans: {gt_set}")

            if not pred_set and not gt_set:
                self.logger.debug(f"[QID:{query_id}] Both pred and gold spans empty: F1=1.0")
                return 1.0
            if not pred_set or not gt_set:
                self.logger.debug(f"[QID:{query_id}] One set empty: F1=0.0")
                return 0.0

            # Exact match component
            exact_matches = len(pred_set.intersection(gt_set))
            exact_precision = exact_matches / len(pred_set)
            exact_recall = exact_matches / len(gt_set)

            # Semantic similarity component
            semantic_score = 0.0
            if self.embedder and self.use_semantic_scoring:
                for pred_span in pred_set:
                    max_sim = 0.0
                    for gt_span in gt_set:
                        sim = self._calculate_semantic_similarity(pred_span, gt_span, query_id)
                        max_sim = max(max_sim, sim)
                    semantic_score += max_sim / len(pred_set)
                self.logger.debug(f"[QID:{query_id}] Semantic similarity score for spans: {semantic_score:.2f}")
            else:
                semantic_score = exact_precision  # Fallback to exact precision

            # Combine exact and semantic scores
            precision = 0.7 * exact_precision + 0.3 * semantic_score
            recall = exact_recall  # Recall based on exact matches
            if precision + recall == 0:
                self.logger.debug(f"[QID:{query_id}] Precision+Recall=0: F1=0.0")
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            self.logger.debug(f"[QID:{query_id}] F1 for spans: {f1:.2f} (Precision: {precision:.2f}, Recall: {recall:.2f})")
            return f1

        except Exception as e:
            self.logger.error(f"[QID:{query_id}] Error computing F1 for spans: {str(e)}")
            return 0.0

    # --- DROP Specific Evaluation Methods ---
    def _evaluate_drop_answer(self,
                             pred: Any,
                             gt: Dict[str, Any],
                             qid: str
                             ) -> Dict[str, float]:
        """
        Evaluate a single DROP answer, comparing number, spans, or date fields.
        Enhanced to handle structured answers and edge cases robustly.
        """
        try:
            # Handle invalid or error predictions
            if not isinstance(pred, dict) or pred.get('error'):
                self.logger.debug(f"[QID:{qid}] Invalid prediction: {pred}")
                return {'exact_match': 0.0, 'f1': 0.0, 'answer_type_drop': 'invalid'}

            if not isinstance(gt, dict):
                self.logger.debug(f"[QID:{qid}] Invalid ground truth format: {gt}")
                return {'exact_match': 0.0, 'f1': 0.0, 'answer_type_drop': 'invalid'}

            # Determine answer type
            if gt.get('number'):
                answer_type = 'number'
            elif gt.get('spans'):
                answer_type = 'spans'
            elif gt.get('date') and any(gt['date'].values()):
                answer_type = 'date'
            else:
                self.logger.debug(f"[QID:{qid}] Invalid ground truth format: {gt}")
                return {'exact_match': 0.0, 'f1': 0.0, 'answer_type_drop': 'invalid'}

            # Exact Match and F1
            if answer_type == 'number':
                em = float(self._are_drop_values_equivalent(pred, gt, 'number', qid))
                f1 = em  # F1 same as EM for numbers
            elif answer_type == 'spans':
                em = float(self._are_drop_values_equivalent(pred, gt, 'spans', qid))
                f1 = self._compute_f1_spans(pred.get('spans', []), gt.get('spans', []), qid)
            elif answer_type == 'date':
                em = float(self._are_drop_values_equivalent(pred, gt, 'date', qid))
                f1 = em  # F1 same as EM for dates
            else:
                self.logger.debug(f"[QID:{qid}] Unsupported answer type: {answer_type}")
                em, f1 = 0.0, 0.0
                answer_type = 'unknown'

            self.logger.debug(f"[QID:{qid}] DROP evaluation: Pred={pred}, GT={gt}, EM={em:.2f}, F1={f1:.2f}, Type={answer_type}")
            return {'exact_match': em, 'f1': f1, 'answer_type_drop': answer_type}

        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error evaluating DROP answer: {str(e)}")
            return {'exact_match': 0.0, 'f1': 0.0, 'answer_type_drop': 'error'}

    def _are_drop_values_equivalent(self, obj1: Dict[str, Any], obj2: Dict[str, Any], value_type: str, qid: str) -> bool:
        """
        Compare DROP answer values for equivalence.
        Consistent with hybrid_integrator.py for number, spans, and date comparison.
        """
        try:
            if value_type == "number":
                n1 = self._normalize_drop_number_for_comparison(obj1.get("number"), qid)
                n2 = self._normalize_drop_number_for_comparison(obj2.get("number"), qid)
                if n1 is None or n2 is None:
                    self.logger.debug(f"[QID:{qid}] Number comparison failed: One or both values are None (Pred: {obj1.get('number')}, GT: {obj2.get('number')})")
                    return False
                result = abs(n1 - n2) < 1e-6
                self.logger.debug(f"[QID:{qid}] Number comparison: {n1} == {n2} -> {result}")
                return result
            elif value_type == "spans":
                pred_spans = [self._normalize_drop_answer_str(str(s), qid) for s in obj1.get("spans", []) if str(s).strip()]
                gt_spans = [self._normalize_drop_answer_str(str(s), qid) for s in obj2.get("spans", []) if str(s).strip()]
                result = set(pred_spans) == set(gt_spans)
                self.logger.debug(f"[QID:{qid}] Span comparison: {pred_spans} == {gt_spans} -> {result}")
                return result
            elif value_type == "date":
                pred_date = obj1.get("date", {})
                gt_date = obj2.get("date", {})
                result = all(str(pred_date.get(k, '')).strip() == str(gt_date.get(k, '')).strip() for k in ['day', 'month', 'year'])
                self.logger.debug(f"[QID:{qid}] Date comparison: {pred_date} == {gt_date} -> {result}")
                return result
            self.logger.debug(f"[QID:{qid}] Unsupported value type for comparison: {value_type}")
            return False
        except Exception as e:
            self.logger.debug(f"[QID:{qid}] Error comparing DROP values: {str(e)}")
            return False

    def _normalize_drop_number_for_comparison(self, value_str: Optional[Any], qid: str) -> Optional[float]:
        """
        Normalize number strings for comparison.
        Consistent with hybrid_integrator.py.
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
            result = words.get(s, float(s))
            self.logger.debug(f"[QID:{qid}] Normalized number: '{value_str}' -> {result}")
            return result
        except Exception as e:
            self.logger.debug(f"[QID:{qid}] Error normalizing number '{value_str}': {str(e)}")
            return None

    def _normalize_drop_answer_str(self, text: str, qid: str) -> str:
        """
        Normalize DROP answer strings for span comparison.
        Preserve essential content while removing irrelevant punctuation.
        """
        text = str(text).lower()
        # Remove specific punctuation but preserve meaningful characters
        text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        text = re.sub(r'\b(a|an|the)\b', '', text)
        text = ' '.join(text.split())
        self.logger.debug(f"[QID:{qid}] Normalized span text: '{text[:50]}...'")
        return text

    def _tokenize_drop(self, text: str, qid: str) -> List[str]:
        """
        Tokenize for DROP F1 calculation.
        """
        tokens = self._normalize_drop_answer_str(text, qid).split()
        self.logger.debug(f"[QID:{qid}] Tokenized text: {tokens}")
        return tokens

    def _stringify_drop_answer(self, answer: Dict[str, Any], qid: str) -> str:
        """
        Convert a DROP answer to a string for ROUGE-L and BLEU metrics.
        """
        try:
            if answer.get('number'):
                return str(answer['number']).strip()
            if answer.get('spans'):
                return ' '.join(str(s).strip() for s in answer['spans'])
            if answer.get('date'):
                date = answer['date']
                return f"{date.get('month', '')}/{date.get('day', '')}/{date.get('year', '')}".strip('/')
            self.logger.debug(f"[QID:{qid}] Empty DROP answer stringified to ''")
            return ''
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error stringifying DROP answer: {str(e)}")
            return ''

    # --- HotpotQA / Text-based QA Methods ---
    def _calculate_semantic_similarity(self, prediction: str, ground_truth: str, qid: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        """
        if not self.embedder:
            self.logger.debug(f"[QID:{qid}] Embedder not available for semantic similarity")
            return 0.0
        try:
            prediction_str = str(prediction)
            ground_truth_str = str(ground_truth)
            if not prediction_str.strip() or not ground_truth_str.strip():
                self.logger.debug(f"[QID:{qid}] Empty prediction or ground truth for semantic similarity")
                return 0.0

            cache_key = hash(prediction_str + ground_truth_str)
            if cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]

            pred_emb = self.embedder.encode(prediction_str, convert_to_tensor=True)
            truth_emb = self.embedder.encode(ground_truth_str, convert_to_tensor=True)
            similarity = util.cos_sim(pred_emb, truth_emb).item()
            similarity = max(0.0, min(1.0, similarity))
            self.similarity_cache[cache_key] = similarity
            self.logger.debug(f"[QID:{qid}] Semantic similarity: {similarity:.2f}")
            return similarity
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error calculating semantic similarity: {str(e)}")
            return 0.0

    def _calculate_rouge_l(self, prediction: str, ground_truth: str, qid: str) -> float:
        """
        Calculate ROUGE-L score.
        """
        try:
            prediction_str = str(prediction)
            ground_truth_str = str(ground_truth)
            if not prediction_str.strip() or not ground_truth_str.strip():
                self.logger.debug(f"[QID:{qid}] Empty prediction or ground truth for ROUGE-L")
                return 0.0
            scores = self.rouge_scorer.score(ground_truth_str, prediction_str)
            rouge_l = scores['rougeL'].fmeasure
            self.logger.debug(f"[QID:{qid}] ROUGE-L score: {rouge_l:.2f}")
            return rouge_l
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error calculating ROUGE-L: {str(e)}")
            return 0.0

    def _calculate_bleu(self, prediction: str, ground_truth: str, qid: str) -> float:
        """
        Calculate BLEU score.
        """
        try:
            prediction_str = str(prediction)
            ground_truth_str = str(ground_truth)
            if not prediction_str.strip() or not ground_truth_str.strip():
                self.logger.debug(f"[QID:{qid}] Empty prediction or ground truth for BLEU")
                return 0.0
            reference = [self._tokenize_drop(ground_truth_str, qid)]
            candidate = self._tokenize_drop(prediction_str, qid)
            if not candidate:
                self.logger.debug(f"[QID:{qid}] Empty candidate tokens for BLEU")
                return 0.0
            bleu = sentence_bleu(reference, candidate, smoothing_function=self.bleu_smoothing)
            self.logger.debug(f"[QID:{qid}] BLEU score: {bleu:.2f}")
            return bleu
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error calculating BLEU: {str(e)}")
            return 0.0

    def _compute_f1(self, predicted: str, ground_truth: str, qid: str) -> float:
        """
        Compute F1 Score for text-based answers (HotpotQA).
        """
        try:
            pred_tokens = set(predicted.lower().split())
            gt_tokens = set(ground_truth.lower().split())
            if not pred_tokens and not gt_tokens:
                self.logger.debug(f"[QID:{qid}] Both pred and gt tokens empty: F1=1.0")
                return 1.0
            if not pred_tokens or not gt_tokens:
                self.logger.debug(f"[QID:{qid}] One set empty: F1=0.0")
                return 0.0
            precision = len(pred_tokens.intersection(gt_tokens)) / len(pred_tokens)
            recall = len(pred_tokens.intersection(gt_tokens)) / len(gt_tokens)
            if precision + recall == 0:
                self.logger.debug(f"[QID:{qid}] Precision+Recall=0: F1=0.0")
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            self.logger.debug(f"[QID:{qid}] F1 score: {f1:.2f} (Precision: {precision:.2f}, Recall: {recall:.2f})")
            return f1
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error computing F1: {str(e)}")
            return 0.0

    def evaluate_reasoning_quality(self,
                                  prediction: str,
                                  reasoning_path: List[Dict],
                                  supporting_facts: Optional[List[Tuple[str, int]]] = None,
                                  qid: str = "unknown"
                                  ) -> Dict[str, float]:
        """
        Evaluate multi-hop reasoning quality (HotpotQA).
        """
        metrics = {}
        if not self.use_semantic_scoring or not self.embedder:
            self.logger.debug(f"[QID:{qid}] Semantic scoring disabled for reasoning quality")
            return {
                'step_accuracy': 0.0, 'fact_coverage': 0.0,
                'path_coherence': 0.0, 'inference_depth': float(len(reasoning_path))
            }

        # Step-level accuracy
        step_acc = self._evaluate_step_accuracy(reasoning_path, supporting_facts, qid)
        metrics['step_accuracy'] = step_acc
        self.logger.debug(f"[QID:{qid}] Step accuracy: {step_acc:.2f}")

        # Fact coverage
        fact_cov = self._calculate_fact_coverage(reasoning_path, supporting_facts, qid)
        metrics['fact_coverage'] = fact_cov
        self.logger.debug(f"[QID:{qid}] Fact coverage: {fact_cov:.2f}")

        # Path coherence
        coherence = self._evaluate_path_coherence(reasoning_path, qid)
        metrics['path_coherence'] = coherence
        self.logger.debug(f"[QID:{qid}] Path coherence: {coherence:.2f}")

        # Depth
        metrics['inference_depth'] = float(len(reasoning_path))
        self.logger.debug(f"[QID:{qid}] Inference depth: {metrics['inference_depth']}")

        return metrics

    def _evaluate_step_accuracy(self,
                               reasoning_path: List[Dict],
                               supporting_facts: Optional[List[Tuple[str, int]]],
                               qid: str
                               ) -> float:
        """
        Evaluate accuracy of each step vs. supporting facts (HotpotQA).
        """
        if not reasoning_path or not supporting_facts or not self.embedder:
            self.logger.debug(f"[QID:{qid}] No reasoning path, supporting facts, or embedder for step accuracy")
            return 0.0

        step_scores = []
        for step_dict in reasoning_path:
            step_text = str(step_dict.get('content', ''))
            if not step_text.strip():
                self.logger.debug(f"[QID:{qid}] Empty step text, skipping")
                continue
            best_sim = 0.0
            for fact_text_tuple in supporting_facts:
                fact_text = str(fact_text_tuple[0])
                if not fact_text.strip():
                    continue
                sim = self._calculate_semantic_similarity(step_text, fact_text, qid)
                if sim > best_sim:
                    best_sim = sim
            step_scores.append(best_sim)
            self.logger.debug(f"[QID:{qid}] Step similarity: {best_sim:.2f}")

        avg_score = float(np.mean(step_scores)) if step_scores else 0.0
        self.logger.debug(f"[QID:{qid}] Average step accuracy: {avg_score:.2f}")
        return avg_score

    def _calculate_fact_coverage(self,
                                reasoning_path: List[Dict],
                                supporting_facts: Optional[List[Tuple[str, int]]],
                                qid: str
                                ) -> float:
        """
        Calculate fact coverage for HotpotQA reasoning paths.
        """
        if not supporting_facts or not reasoning_path or not self.embedder:
            self.logger.debug(f"[QID:{qid}] No supporting facts, reasoning path, or embedder for fact coverage")
            return 0.0

        covered = 0
        total_valid_facts = 0
        for fact_text_tuple in supporting_facts:
            fact_text = str(fact_text_tuple[0])
            if not fact_text.strip():
                continue
            total_valid_facts += 1
            matched = False
            for step_dict in reasoning_path:
                step_text = str(step_dict.get('content', ''))
                if not step_text.strip():
                    continue
                sim = self._calculate_semantic_similarity(step_text, fact_text, qid)
                if sim >= self.semantic_threshold:
                    matched = True
                    break
            if matched:
                covered += 1
            self.logger.debug(f"[QID:{qid}] Fact '{fact_text[:50]}...': {'Covered' if matched else 'Not covered'}")

        coverage = covered / total_valid_facts if total_valid_facts > 0 else 0.0
        self.logger.debug(f"[QID:{qid}] Fact coverage: {coverage:.2f} ({covered}/{total_valid_facts})")
        return coverage

    def _evaluate_path_coherence(self, reasoning_path: List[Dict], qid: str) -> float:
        """
        Evaluate semantic coherence among consecutive steps (HotpotQA).
        """
        if len(reasoning_path) < 2 or not self.embedder:
            self.logger.debug(f"[QID:{qid}] Insufficient steps or no embedder for path coherence")
            return 1.0

        sims = []
        for i in range(len(reasoning_path) - 1):
            step_text_1 = str(reasoning_path[i].get('content', ''))
            step_text_2 = str(reasoning_path[i + 1].get('content', ''))
            if not step_text_1.strip() or not step_text_2.strip():
                self.logger.debug(f"[QID:{qid}] Empty step text pair, skipping")
                sims.append(0.0)
                continue
            sim = self._calculate_semantic_similarity(step_text_1, step_text_2, qid)
            sims.append(sim)
            self.logger.debug(f"[QID:{qid}] Coherence between steps {i} and {i+1}: {sim:.2f}")

        avg_coherence = float(np.mean(sims)) if sims else 0.0
        self.logger.debug(f"[QID:{qid}] Average path coherence: {avg_coherence:.2f}")
        return avg_coherence

    def calculate_ablation_metrics(self,
                                  component: str,
                                  base_performance: Dict[str, float],
                                  ablated_performance: Dict[str, float],
                                  qid: str = "unknown"
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
                elif abs(abl_val) > 1e-9:
                    relative_change = float('inf') if abl_val > 0 else float('-inf')
                else:
                    relative_change = 0.0
                impact_metrics[f'{metric}_impact'] = relative_change
                self.logger.debug(f"[QID:{qid}] Ablation impact for {metric}: {relative_change:.2f}")

        self.ablation_results[component].append(impact_metrics)
        return impact_metrics

    def record_metric(self, metric_name: str, value: float, qid: str = "unknown"):
        """
        Store a single metric value for significance testing.
        """
        if isinstance(value, (int, float)):
            self.statistical_data[metric_name].append(value)
            self.logger.debug(f"[QID:{qid}] Recorded metric {metric_name}: {value}")
        else:
            self.logger.warning(f"[QID:{qid}] Non-numerical value '{value}' for metric '{metric_name}'. Skipping.")

    def calculate_statistical_significance(self, qid: str = "unknown") -> Dict[str, Dict[str, float]]:
        """
        Perform a t-test across collected metric arrays.
        """
        significance_results = {}
        for metric, values in self.statistical_data.items():
            if len(values) >= 2:
                try:
                    t_stat, p_value = stats.ttest_1samp(values, 0.0)
                    significance_results[metric] = {
                        't_statistic': float(t_stat) if not np.isnan(t_stat) else 0.0,
                        'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
                    self.logger.debug(f"[QID:{qid}] Statistical significance for {metric}: t={t_stat:.2f}, p={p_value:.4f}")
                except Exception as e:
                    self.logger.error(f"[QID:{qid}] Error calculating t-test for metric {metric}: {e}")
                    significance_results[metric] = {'error': str(e)}
            else:
                significance_results[metric] = {'error': 'Insufficient data for t-test'}
                self.logger.debug(f"[QID:{qid}] Insufficient data for t-test on {metric}")
        return significance_results
