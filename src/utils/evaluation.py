# src/utils/evaluation.py

import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import logging
from collections import defaultdict, Counter, OrderedDict
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
        if self.use_semantic_scoring:
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
            'skipped_queries': 0,
            'average_time': 0.0,
            'path_performance': defaultdict(list)
        }

        # Initialize ROUGE scorer and BLEU smoothing (used only for HotpotQA)
        if self.dataset_type != 'drop':
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.bleu_smoothing = SmoothingFunction().method1
        else:
            self.rouge_scorer = None
            self.bleu_smoothing = None

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

        # Cache for semantic similarity to optimize performance with size limit
        self.similarity_cache = OrderedDict()
        self.max_cache_size = 10000

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
                self.performance_stats['skipped_queries'] += 1
                self.logger.warning(f"[QID:{query_id}] No ground truth for query")
                continue

            gt_value = ground_truths[query_id]
            current_query_metrics = {"query_id": query_id}

            try:
                if self.dataset_type == 'drop':
                    # DROP evaluation
                    drop_eval = self._evaluate_drop_answer(pred_value, gt_value, query_id)
                    current_query_metrics.update(drop_eval)
                    # Semantic similarity for spans if applicable
                    if drop_eval['answer_type_drop'] == 'spans' and self.use_semantic_scoring and self.embedder:
                        pred_spans = pred_value.get('spans', []) if isinstance(pred_value, dict) else []
                        gt_spans = gt_value.get('spans', [])
                        pred_str = ' '.join(str(s).strip() for s in pred_spans) if pred_spans else ''
                        gt_str = ' '.join(str(s).strip() for s in gt_spans) if gt_spans else ''
                        if pred_str and gt_str:
                            sem_sim = self._calculate_semantic_similarity(pred_str, gt_str, query_id)
                            current_query_metrics['semantic_similarity'] = sem_sim
                        else:
                            current_query_metrics['semantic_similarity'] = 0.0
                    else:
                        current_query_metrics['semantic_similarity'] = 0.0
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
                    'semantic_similarity': 0.0
                })
                if self.dataset_type != 'drop':
                    current_query_metrics.update({
                        'rougeL': 0.0,
                        'bleu': 0.0
                    })
                all_query_metrics.append(current_query_metrics)

        # Aggregate metrics
        aggregated_metrics = {
            'average_exact_match': 0.0,
            'average_f1': 0.0,
            'average_semantic_similarity': 0.0
        }
        if self.dataset_type != 'drop':
            aggregated_metrics.update({
                'average_rougeL': 0.0,
                'average_bleu': 0.0
            })

        if not all_query_metrics:
            self.logger.info("No queries processed for evaluation")
            return aggregated_metrics

        metric_values = defaultdict(list)
        for q_metrics in all_query_metrics:
            for key, value in q_metrics.items():
                if key in ['exact_match', 'f1', 'semantic_similarity', 'rougeL', 'bleu'] and isinstance(value,
                                                                                                        (int, float)):
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

    def _compute_f1_spans(self, pred_spans_raw: List[str], gold_spans_raw: List[str], query_id: str) -> float:
        """
        Compute F1 for DROP spans with semantic similarity for partial matches.
        Uses normalized span text for robust comparison.
        """
        try:
            # Normalize and deduplicate spans while preserving order for semantic scoring if needed (not for exact set for F1)
            # For F1 based on token overlap, the set of normalized stringsInherited
            pred_set_normalized = {self._normalize_drop_answer_str(str(s), query_id) for s in pred_spans_raw if
                                   str(s).strip()}
            gt_set_normalized = {self._normalize_drop_answer_str(str(s), query_id) for s in gold_spans_raw if
                                 str(s).strip()}

            self.logger.debug(
                f"[QID:{query_id}] F1 Spans: Pred normalized set: {pred_set_normalized}, Gold normalized set: {gt_set_normalized}")

            if not pred_set_normalized and not gt_set_normalized:  # Both empty
                self.logger.debug(f"[QID:{query_id}] Both pred and gold spans empty after normalization: F1=1.0")
                return 1.0
            if not pred_set_normalized or not gt_set_normalized:  # One is empty, the other is not
                self.logger.debug(f"[QID:{query_id}] One span set empty after normalization, other not: F1=0.0")
                return 0.0

            # --- Exact Match Component for F1 (Official DROP F1 is token-based, but this is span-set F1) ---
            # This method as written calculates F1 based on exact matches of *normalized spans*.
            # The official DROP F1 is more complex (token-level overlap).
            # The current implementation calculates a span-level F1.

            common_spans = pred_set_normalized.intersection(gt_set_normalized)
            num_common = len(common_spans)

            precision = num_common / len(pred_set_normalized) if pred_set_normalized else 0.0
            recall = num_common / len(gt_set_normalized) if gt_set_normalized else 0.0

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            self.logger.debug(
                f"[QID:{query_id}] Span-level F1: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f}, Common: {num_common})")

            # --- Semantic Similarity Component (Optional, if you want to augment F1) ---
            # The prompt for this method in Evaluation.py [source 1804] mentions "semantic similarity for partial matches".
            # This part makes it a custom F1, not the standard token-based DROP F1.
            semantic_score_precision = 0.0
            if self.embedder and self.use_semantic_scoring and pred_set_normalized and gt_set_normalized:
                # For precision: for each pred_span, find max similarity to any gt_span
                # This is a simplified semantic precision. True partial match F1 is more involved.
                # Convert sets to lists for iteration if semantic scoring needs original case or order.
                pred_list_for_sem = [s for s in pred_spans_raw if
                                     self._normalize_drop_answer_str(s, query_id) in pred_set_normalized]
                gt_list_for_sem = [s for s in gold_spans_raw if
                                   self._normalize_drop_answer_str(s, query_id) in gt_set_normalized]

                total_sim_for_precision = 0.0
                for pred_span in pred_list_for_sem:
                    max_sim_to_gt = 0.0
                    for gt_span in gt_list_for_sem:
                        sim = self._calculate_semantic_similarity(pred_span, gt_span,
                                                                  query_id)  # Uses original case for embedding
                        if sim > max_sim_to_gt:
                            max_sim_to_gt = sim
                    total_sim_for_precision += max_sim_to_gt
                semantic_score_precision = total_sim_for_precision / len(
                    pred_list_for_sem) if pred_list_for_sem else 0.0
                self.logger.debug(
                    f"[QID:{query_id}] Avg Max Semantic Sim for Precision (Pred to GT): {semantic_score_precision:.3f}")

                # Hybrid F1 (example: weighted average of exact precision and semantic precision)
                hybrid_precision = 0.7 * precision + 0.3 * semantic_score_precision
                if hybrid_precision + recall == 0:
                    f1_hybrid = 0.0
                else:
                    f1_hybrid = 2 * (hybrid_precision * recall) / (hybrid_precision + recall)
                self.logger.debug(
                    f"[QID:{query_id}] Hybrid Span F1 (with semantic precision): {f1_hybrid:.3f} (Hybrid Precision: {hybrid_precision:.3f})")
                return f1_hybrid  # Return this hybrid F1 if desired

            return f1  # Return the exact span-set F1

        except Exception as e:
            self.logger.error(f"[QID:{query_id}] Error computing F1 for spans: {str(e)}", exc_info=True)
            return 0.0

    def _evaluate_drop_answer(self,
                              pred: Any,  # Prediction from the system
                              gt: Dict[str, Any],  # Ground truth
                              qid: str
                              ) -> Dict[str, Any]:  # Changed return to Any from float for metrics dict
        """
        Evaluate a single DROP answer, comparing number, spans, or date fields.
        Prediction 'pred' is expected to be a dictionary for valid DROP answers.
        """
        # Initialize metrics for this query
        query_eval_metrics = {'exact_match': 0.0, 'f1': 0.0, 'answer_type_drop': 'invalid_pred_format'}

        try:
            # --- Validate Prediction Format ---
            if not isinstance(pred, dict):
                self.logger.warning(
                    f"[QID:{qid}] Invalid prediction format for DROP: Expected dict, got {type(pred)}. Value: '{str(pred)[:100]}'")
                # If pred is a string indicating error from upstream, capture that
                if isinstance(pred, str) and pred.lower().startswith("error:"):
                    query_eval_metrics['rationale'] = f"Upstream error: {pred}"
                return query_eval_metrics  # Return default 0 scores

            # Ensure basic DROP keys exist in pred, even if empty, for consistent processing
            # This should ideally be guaranteed by the output of HybridIntegrator/_create_drop_answer_obj
            pred.setdefault('number', "")
            pred.setdefault('spans', [])
            pred.setdefault('date', {'day': '', 'month': '', 'year': ''})
            pred.setdefault('status', 'unknown')  # Status from the prediction object itself

            # --- Validate Ground Truth Format ---
            if not isinstance(gt, dict):
                self.logger.error(
                    f"[QID:{qid}] Invalid ground truth format for DROP: Expected dict, got {type(gt)}. Value: '{str(gt)[:100]}'")
                query_eval_metrics['answer_type_drop'] = 'invalid_gt_format'
                return query_eval_metrics

            # --- Determine Ground Truth Answer Type ---
            gt_type: Optional[str] = None
            if gt.get('number') is not None and str(gt.get('number')).strip() != "":  # Check if number has a value
                gt_type = 'number'
            elif gt.get('spans') is not None and (isinstance(gt.get('spans'), list) and any(
                    str(s).strip() for s in gt.get('spans'))):  # Check if spans list is not empty and has content
                gt_type = 'spans'
            elif gt.get('date') and isinstance(gt.get('date'), dict) and any(
                    str(v).strip() for v in gt['date'].values()):  # Check if any date field has a value
                gt_type = 'date'

            if gt_type is None:
                self.logger.warning(
                    f"[QID:{qid}] Could not determine a valid answer type from ground truth: {gt}. Possibly an empty GT.")
                # If GT expects nothing, and prediction also provides nothing, it's an EM.
                is_pred_empty_semantically = (not pred.get('number') and \
                                              not any(s for s in pred.get('spans', []) if s) and \
                                              not any(v for v in pred.get('date', {}).values() if v))
                if is_pred_empty_semantically:
                    query_eval_metrics.update({'exact_match': 1.0, 'f1': 1.0, 'answer_type_drop': 'empty_match'})
                else:  # GT is empty, but prediction is not
                    query_eval_metrics['answer_type_drop'] = 'gt_empty_pred_not'
                return query_eval_metrics

            query_eval_metrics['answer_type_drop'] = gt_type  # Store determined GT type

            # --- Perform Evaluation based on GT Type ---
            # pred['number'] should be native int/float by now due to upstream fixes.
            # pred['spans'] should be List[str].
            # pred['date'] should be Dict[str,str].

            if gt_type == 'number':
                em = float(self._are_drop_values_equivalent(pred, gt, 'number', qid))
                f1 = em  # F1 is same as EM for numbers in DROP
            elif gt_type == 'spans':
                em = float(
                    self._are_drop_values_equivalent(pred, gt, 'spans', qid))  # Exact set match of normalized spans
                # For F1 of spans, use the more sophisticated _compute_f1_spans
                # which might incorporate semantic similarity or just be exact span-set F1
                f1 = self._compute_f1_spans(pred.get('spans', []), gt.get('spans', []), qid)
            elif gt_type == 'date':
                em = float(self._are_drop_values_equivalent(pred, gt, 'date', qid))
                f1 = em  # F1 is same as EM for dates
            else:  # Should not be reached if gt_type is determined above
                self.logger.error(
                    f"[QID:{qid}] Internal error: Unhandled gt_type '{gt_type}' in _evaluate_drop_answer.")
                em, f1 = 0.0, 0.0
                query_eval_metrics['answer_type_drop'] = 'internal_error_unknown_gt_type'

            query_eval_metrics['exact_match'] = em
            query_eval_metrics['f1'] = f1

            self.logger.debug(
                f"[QID:{qid}] DROP evaluation: Pred='{str(pred)[:100]}...', GT='{str(gt)[:100]}...', "
                f"EM={em:.3f}, F1={f1:.3f}, AnswerType(GT)='{gt_type}'"
            )
            return query_eval_metrics

        except Exception as e:
            self.logger.error(f"[QID:{qid}] Critical error evaluating DROP answer: {str(e)}. Pred: '{str(pred)[:100]}'",
                              exc_info=True)
            # Return default 0 scores with error type
            query_eval_metrics['answer_type_drop'] = 'evaluation_exception'
            return query_eval_metrics

    def _are_drop_values_equivalent(self, obj1: Dict[str, Any], obj2: Dict[str, Any], value_type: str,
                                    qid: str) -> bool:
        """
        Compare DROP answer values for equivalence.
        obj1 is prediction, obj2 is ground truth.
        """
        try:
            if value_type == "number":
                # Values from the prediction (obj1) and ground truth (obj2) dicts
                val1_raw = obj1.get("number")
                val2_raw = obj2.get("number")

                n1 = self._normalize_drop_number_for_comparison(val1_raw, qid)
                n2 = self._normalize_drop_number_for_comparison(val2_raw, qid)

                if n1 is None and n2 is None:
                    # If both normalize to None, consider them equivalent only if their raw forms were also similar
                    # (e.g. both empty string, or both explicitly "N/A" if that was a convention)
                    # For now, if both fail normalization, they aren't equivalent valid numbers for scoring.
                    # But if raw were both '', it is an EM if GT is also '' expecting nothing.
                    # This logic is tricky: if GT expects "no number" and pred gives "no number" = True.
                    # If GT expects a number, and pred gives "no number" = False.
                    # The current logic in _evaluate_drop_answer implies that if gt_type is 'number', a number is expected.
                    raw1_is_empty_like = (val1_raw == '' or val1_raw is None)
                    raw2_is_empty_like = (val2_raw == '' or val2_raw is None)
                    if raw1_is_empty_like and raw2_is_empty_like:
                        self.logger.debug(
                            f"[QID:{qid}] Number comparison: Both raw values indicate empty. Result: True")
                        return True
                    self.logger.debug(
                        f"[QID:{qid}] Number comparison: Both normalized to None, but raw values differ or not both empty. Pred raw: '{val1_raw}', GT raw: '{val2_raw}'. Result: False")
                    return False
                if n1 is None or n2 is None:  # Only one normalized successfully
                    self.logger.debug(
                        f"[QID:{qid}] Number comparison: One normalized to None. Pred norm: {n1}, GT norm: {n2}. Pred raw: '{val1_raw}', GT raw: '{val2_raw}'. Result: False")
                    return False

                # Both n1 and n2 are valid floats here
                result = abs(n1 - n2) < 1e-6  # Compare floats
                self.logger.debug(
                    f"[QID:{qid}] Number comparison: {n1} (from '{val1_raw}') == {n2} (from '{val2_raw}') -> {result}")
                return result

            elif value_type == "spans":
                # Spans are lists of strings. Normalize each string for comparison.
                pred_spans_raw = obj1.get("spans", [])
                gt_spans_raw = obj2.get("spans", [])

                # Use the _normalize_drop_answer_str for each span string
                pred_spans_normalized = {self._normalize_drop_answer_str(str(s), qid) for s in pred_spans_raw if
                                         str(s).strip()}
                gt_spans_normalized = {self._normalize_drop_answer_str(str(s), qid) for s in gt_spans_raw if
                                       str(s).strip()}

                result = pred_spans_normalized == gt_spans_normalized
                self.logger.debug(
                    f"[QID:{qid}] Span comparison: Normalized Pred {pred_spans_normalized} == Normalized GT {gt_spans_normalized} -> {result}")
                return result

            elif value_type == "date":
                pred_date = obj1.get("date", {})
                gt_date = obj2.get("date", {})

                # Ensure comparison on stripped string versions of day, month, year
                result = all(
                    str(pred_date.get(k, '')).strip() == str(gt_date.get(k, '')).strip()
                    for k in ['day', 'month', 'year']
                )
                self.logger.debug(f"[QID:{qid}] Date comparison: Pred {pred_date} == GT {gt_date} -> {result}")
                return result

            self.logger.debug(f"[QID:{qid}] Unsupported value type for DROP comparison: {value_type}")
            return False
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error comparing DROP values (type '{value_type}'): {str(e)}", exc_info=True)
            return False

    def _normalize_drop_number_for_comparison(self, value_str: Optional[Any], qid: str) -> Optional[float]:
        """
        Normalize number strings/values for comparison, returning float or None.
        Consistent with hybrid_integrator.py.
        """
        if value_str is None:
            return None
        try:
            if isinstance(value_str, (int, float)):  # If already a number
                return float(value_str)

            s = str(value_str).replace(",", "").strip().lower()
            if not s:  # If empty after cleaning
                self.logger.debug(f"[QID:{qid}] Number value '{value_str}' became empty after cleaning.")
                return None

            words = {
                "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
                "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
                "ten": 10.0
            }
            if s in words:
                result = words[s]
            elif re.fullmatch(r'-?\d+(\.\d+)?', s):  # Regex to match float/int strings
                result = float(s)
            else:
                self.logger.debug(
                    f"[QID:{qid}] Could not normalize '{value_str}' (cleaned: '{s}') to a recognized number format or word.")
                return None

            # No need to convert to int if is_integer, as comparison will be float vs float.
            # The main goal is a canonical float representation for comparison.
            # Example: "4" -> 4.0, "4.0" -> 4.0.
            self.logger.debug(f"[QID:{qid}] Normalized number: '{value_str}' -> {result}")
            return result
        except (ValueError, TypeError) as e:
            self.logger.debug(f"[QID:{qid}] Error normalizing number '{value_str}': {str(e)}")
            return None

    def _normalize_drop_answer_str(self, text: str, qid: str) -> str:
        """
        Normalize DROP answer strings for span comparison.
        Preserves hyphens to maintain meaningful compound words (e.g., 'well-known').
        """
        text = str(text).lower()
        # Remove specific punctuation but preserve meaningful characters like hyphens
        text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        text = re.sub(r'\b(a|an|the)\b', '', text)
        text = ' '.join(text.split())
        self.logger.debug(f"[QID:{qid}] Normalized span text: '{text[:50]}...'")
        return text

    def _tokenize_drop(self, text: str, qid: str) -> List[str]:
        """
        Tokenize for DROP F1 twarzy. Used only for HotpotQA BLEU computation.
        """
        tokens = self._normalize_drop_answer_str(text, qid).split()
        self.logger.debug(f"[QID:{qid}] Tokenized text: {tokens}")
        return tokens

    def _stringify_drop_answer(self, answer: Dict[str, Any], qid: str) -> str:
        """
        Convert a DROP answer to a string for ROUGE-L and BLEU metrics (used only for HotpotQA).
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
            if len(self.similarity_cache) >= self.max_cache_size:
                self.similarity_cache.popitem(last=False)
            self.logger.debug(f"[QID:{qid}] Semantic similarity: {similarity:.2f}")
            return similarity
        except Exception as e:
            self.logger.error(f"[QID:{qid}] Error calculating semantic similarity: {str(e)}")
            return 0.0

    def _calculate_rouge_l(self, prediction: str, ground_truth: str, qid: str) -> float:
        """
        Calculate ROUGE-L score. Used only for HotpotQA evaluation.
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
        Calculate BLEU score. Used only for HotpotQA evaluation.
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
        if self.dataset_type == 'drop':
            self.logger.debug(f"[QID:{qid}] Reasoning quality evaluation not applicable for DROP dataset")
            return {'step_accuracy': 0.0, 'fact_coverage': 0.0, 'path_coherence': 0.0, 'inference_depth': 0.0}

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
            self.logger.debug(f"[QID:{qid}] Coherence between steps {i} and {i + 1}: {sim:.2f}")

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