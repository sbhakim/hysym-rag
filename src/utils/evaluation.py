# ===== src/utils/evaluation.py =====

import numpy as np
import re
from typing import Dict
import nltk
try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    sentence_bleu = None
from difflib import SequenceMatcher

class Evaluation:
    def __init__(self):
        pass

    def _normalize_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def evaluate(self, predictions: Dict[str, Dict], ground_truths: Dict[str, str]):
        """
        Evaluate a dictionary of predictions against ground truths.

        predictions: {query: { 'result': <answer string> }}
        or         : {query: <answer string>}
        ground_truths: {query: <ground truth string>}
        """
        similarities = []
        rougeL_scores = []
        f1_scores = []

        for query, pred_obj in predictions.items():
            # If pred_obj is a dictionary with 'result', extract it:
            if isinstance(pred_obj, dict) and 'result' in pred_obj:
                pred_text = pred_obj['result']
            else:
                pred_text = pred_obj

            gt_text = ground_truths.get(query, "")
            sim = self.similarity_score(pred_text, gt_text)
            rougeL = self.rougeL_score(pred_text, gt_text)
            f1 = self.f1_score(pred_text, gt_text)

            similarities.append(sim)
            rougeL_scores.append(rougeL)
            f1_scores.append(f1)

        return {
            "average_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "average_rougeL": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
            "average_f1": float(np.mean(f1_scores)) if f1_scores else 0.0
        }

    def similarity_score(self, pred, gt):
        pred_norm = self._normalize_text(pred)
        gt_norm = self._normalize_text(gt)
        if not pred_norm or not gt_norm:
            return 0.0
        return SequenceMatcher(None, pred_norm, gt_norm).ratio()

    def rougeL_score(self, pred, gt):
        """
        Compute a simple character-based ROUGE-L.
        """
        pred_norm = self._normalize_text(pred)
        gt_norm = self._normalize_text(gt)
        if not pred_norm or not gt_norm:
            return 0.0

        # Longest common subsequence approach:
        lcs_length = self._lcs_length(pred_norm, gt_norm)
        recall = lcs_length / len(gt_norm) if gt_norm else 0
        precision = lcs_length / len(pred_norm) if pred_norm else 0
        if (precision + recall) == 0:
            return 0.0
        f_score = 2 * (precision * recall) / (precision + recall)
        return f_score

    def _lcs_length(self, s1, s2):
        # Dynamic programming approach
        dp = [[0] * (len(s2)+1) for _ in range(len(s1)+1)]
        for i in range(1, len(s1)+1):
            for j in range(1, len(s2)+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[len(s1)][len(s2)]

    def f1_score(self, pred, gt):
        pred_tokens = self._normalize_text(pred).split()
        gt_tokens = self._normalize_text(gt).split()
        common = set(pred_tokens).intersection(set(gt_tokens))
        if (len(pred_tokens) + len(gt_tokens)) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # NOTE: For a thorough HotpotQA multi-hop evaluation,
    # you'd typically check supporting-facts retrieval correctness, etc.
    # That requires partial-labelling of which sentences were used in the final answer.
    # HySym-RAG, by default, doesn't track these sentence-level retrievals.
    # If you want that, you need to store and compare them here.
