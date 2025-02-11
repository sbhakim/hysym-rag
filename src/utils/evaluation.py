# src/utils/evaluation.py

from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import numpy as np
import time


class Evaluation:
    def __init__(self, cpu_wattage=65, gpu_wattage=250):
        # Load a semantic model for similarity evaluation.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize ROUGE scorer.
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # Estimated wattages (in Watts) for energy calculation.
        self.cpu_wattage = cpu_wattage
        self.gpu_wattage = gpu_wattage

    def simplify_prediction(self, prediction):
        """
        Normalizes the prediction output.
        """
        if isinstance(prediction, list):
            prediction = " ".join(str(item) for item in prediction)
        elif isinstance(prediction, dict):
            prediction = prediction.get("result", str(prediction))
        elif not isinstance(prediction, str):
            prediction = str(prediction)
        return prediction.strip()

    def evaluate(self, predictions, ground_truths):
        """
        Evaluate basic metrics: average cosine similarity and ROUGE-L.
        """
        valid_queries = [q for q in predictions if q in ground_truths]
        if not valid_queries:
            return {
                "average_similarity": 0.0,
                "average_rougeL": 0.0, # Initialize ROUGE-L
                "average_f1": 0.0,     # Initialize F1 score
                "evaluated_queries": 0,
                "queries_with_truth": []
            }
        sim_scores, rouge_scores, f1_scores = [], [], [] # Initialize f1_scores list
        for q in valid_queries:
            gt = ground_truths[q]
            pred = self.simplify_prediction(predictions[q]) # Reverted to using simplify_prediction
            emb_gt = self.model.encode(gt, convert_to_tensor=True)
            emb_pred = self.model.encode(pred, convert_to_tensor=True)
            sim = util.cos_sim(emb_gt, emb_pred).item()
            sim_scores.append(sim)
            rouge_score = self.rouge.score(gt, pred)['rougeL'].fmeasure
            rouge_scores.append(rouge_score)
            f1 = self.f1_score(pred, gt) # Calculate F1 score
            f1_scores.append(f1)        # Append F1 score to the list

        avg_similarity = np.mean(sim_scores)
        avg_rouge = np.mean(rouge_scores)
        avg_f1 = np.mean(f1_scores)      # Calculate average F1 score

        return {
            "average_similarity": avg_similarity,
            "average_rougeL": avg_rouge,     # Return average ROUGE-L
            "average_f1": avg_f1,         # Return average F1 score
            "evaluated_queries": len(valid_queries),
            "queries_with_truth": valid_queries
        }

    def f1_score(self, prediction, ground_truth):
        """
        Computes a simple token-level F1 score between prediction and ground truth.
        """
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        if not pred_tokens or not truth_tokens:
            return 0.0
        common = pred_tokens.intersection(truth_tokens)
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def calculate_energy_metrics(self, resource_logs):
        """
        Calculates energy consumption (in Joules) from resource usage logs.
        Each log should contain 'cpu', 'gpu', and 'time' (in seconds).
        """
        cpu_joules = sum(log['cpu'] * self.cpu_wattage * log['time'] for log in resource_logs)
        gpu_joules = sum(log['gpu'] * self.gpu_wattage * log['time'] for log in resource_logs)
        total_joules = cpu_joules + gpu_joules
        return {
            'cpu_joules': cpu_joules,
            'gpu_joules': gpu_joules,
            'total_joules': total_joules
        }

    def energy_per_query(self, resource_logs):
        """
        Computes average energy consumption per query.
        """
        if not resource_logs:
            return 0.0
        total_energy = sum(log.get('total_joules', 0.0) for log in resource_logs)
        return total_energy / len(resource_logs)

    def evaluate_advanced(self, predictions, ground_truths, extra_metrics):
        """
        Advanced evaluation merging semantic similarity, ROUGE, F1 score, energy metrics, and latency.
        extra_metrics should include:
          - "chain_depths": list of chain depth values (floats)
          - "energy_logs": list of resource log dicts (each with 'cpu', 'gpu', 'time')
          - "latency": overall average latency in seconds
        """
        base_eval = self.evaluate(predictions, ground_truths)
        # Compute average F1 score for queries with ground truth.
        f1_scores = []
        for q in base_eval["queries_with_truth"]:
            gt = ground_truths[q]
            pred = self.simplify_prediction(predictions[q])
            f1_scores.append(self.f1_score(pred, gt))
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

        chain_depths = extra_metrics.get("chain_depths", [])
        energy_logs = extra_metrics.get("energy_logs", [])
        latency = extra_metrics.get("latency", 0)

        energy_metrics = self.calculate_energy_metrics(energy_logs) if energy_logs else {'total_joules': 0}
        avg_energy = energy_metrics.get('total_joules', 0) / max(1, len(energy_logs))
        avg_chain_depth = np.mean(chain_depths) if chain_depths else 0.0

        # Normalize latency and energy to a scale [0,1] based on expected upper bounds
        norm_latency = min(latency / 10.0, 1.0)  # assuming 10s as a high-latency threshold
        norm_energy = min(avg_energy / 1000.0, 1.0)  # assuming 1000J as an upper bound

        # Composite score combining similarity, F1, latency, and energy consumption
        composite_score = (
                0.5 * base_eval["average_similarity"] +
                0.2 * avg_f1 +
                0.15 * (1 - norm_latency) +
                0.15 * (1 - norm_energy)
        )

        base_eval["average_f1"] = avg_f1
        base_eval["energy_metrics"] = energy_metrics
        base_eval["average_chain_depth"] = avg_chain_depth
        base_eval["composite_score"] = composite_score
        base_eval["latency"] = latency
        base_eval["avg_energy_per_query"] = avg_energy
        return base_eval
