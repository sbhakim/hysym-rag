# src/evaluation.py
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import numpy as np

class Evaluation:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate(self, predictions, ground_truths):
        valid_queries = [q for q in predictions if q in ground_truths]
        if not valid_queries:
            return {
                "average_similarity": 0.0,
                "average_rougeL": 0.0,
                "evaluated_queries": 0,
                "queries_with_truth": []
            }
        sim_scores, rouge_scores = [], []
        for q in valid_queries:
            gt = ground_truths[q]
            pred = self.simplify_prediction(predictions[q])
            emb_gt = self.model.encode(gt, convert_to_tensor=True)
            emb_pred = self.model.encode(pred, convert_to_tensor=True)
            sim = util.cos_sim(emb_gt, emb_pred).item()
            sim_scores.append(sim)
            rouge_score = self.rouge.score(gt, pred)['rougeL'].fmeasure
            rouge_scores.append(rouge_score)
        avg_similarity = np.mean(sim_scores)
        avg_rouge = np.mean(rouge_scores)
        return {
            "average_similarity": avg_similarity,
            "average_rougeL": avg_rouge,
            "evaluated_queries": len(valid_queries),
            "queries_with_truth": valid_queries
        }

    def evaluate_advanced(self, predictions, ground_truths, extra_metrics):
        """
        Advanced evaluation merging semantic similarity with extra metrics (e.g. chain depth, energy usage).
        """
        base_eval = self.evaluate(predictions, ground_truths)
        chain_depths = extra_metrics.get("chain_depths", [])
        energies = extra_metrics.get("energy", [])
        base_eval["average_chain_depth"] = np.mean(chain_depths) if chain_depths else 0.0
        base_eval["average_energy"] = np.mean(energies) if energies else 0.0
        return base_eval

    def simplify_prediction(self, prediction):
        """
        For a list prediction, take the first item and strip whitespace.
        """
        if isinstance(prediction, list):
            prediction = prediction[0]
        return prediction.strip()
