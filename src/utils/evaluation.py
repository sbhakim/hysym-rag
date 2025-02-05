# src/utils/evaluation.py
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import numpy as np

class Evaluation:
    def __init__(self, cpu_wattage=65, gpu_wattage=250):
        """
        Initialize the evaluator.
        cpu_wattage and gpu_wattage are estimated power draws (in Watts).
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # Estimated wattages for energy calculations
        self.cpu_wattage = cpu_wattage
        self.gpu_wattage = gpu_wattage

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

    def calculate_energy_metrics(self, resource_logs):
        """
        Calculate energy consumption (in Joules) based on resource usage logs.
        Each log in resource_logs is expected to have:
          - 'cpu': CPU usage fraction (0-1)
          - 'gpu': GPU usage fraction (0-1)
          - 'time': duration in seconds
        """
        cpu_joules = sum(log['cpu'] * self.cpu_wattage * log['time'] for log in resource_logs)
        gpu_joules = sum(log['gpu'] * self.gpu_wattage * log['time'] for log in resource_logs)
        total_joules = cpu_joules + gpu_joules
        return {
            'cpu_joules': cpu_joules,
            'gpu_joules': gpu_joules,
            'total_joules': total_joules
        }

    def evaluate_advanced(self, predictions, ground_truths, extra_metrics):
        """
        Advanced evaluation merging semantic similarity with extra metrics,
        such as chain depth and energy consumption.
        extra_metrics should be a dict with keys:
          - "chain_depths": list of chain depth values (floats)
          - "energy_logs": list of resource log dicts (see calculate_energy_metrics)
          - "latency": overall average latency in seconds
        """
        base_eval = self.evaluate(predictions, ground_truths)
        chain_depths = extra_metrics.get("chain_depths", [])
        energy_logs = extra_metrics.get("energy_logs", [])
        latency = extra_metrics.get("latency", 0)

        energy_metrics = self.calculate_energy_metrics(energy_logs) if energy_logs else {'total_joules': 0}
        avg_chain_depth = np.mean(chain_depths) if chain_depths else 0.0

        # Composite score: higher accuracy, lower energy, and lower latency are preferred.
        # We assume similarity score in [0,1] and normalize energy and latency inversely.
        composite_score = (
            0.6 * base_eval["average_similarity"] +
            0.25 * (1 - (energy_metrics['total_joules'] / 1000)) +  # Normalize energy (example scaling)
            0.15 * (1 - (latency / 10))                           # Normalize latency (example scaling)
        )
        base_eval["energy_metrics"] = energy_metrics
        base_eval["average_chain_depth"] = avg_chain_depth
        base_eval["composite_score"] = composite_score
        return base_eval

    def simplify_prediction(self, prediction):
        """
        Normalize the prediction output:
          - If the prediction is a list, join its elements into a single string.
          - If the prediction is a dict and contains a 'result' key, extract it.
          - Otherwise, convert non-string predictions to string.
          - Finally, strip leading/trailing whitespace.
        """
        if isinstance(prediction, list):
            # Join list elements with a space
            prediction = " ".join(str(item) for item in prediction)
        elif isinstance(prediction, dict):
            # Extract the 'result' key if available, else convert the dict to string
            prediction = prediction.get("result", str(prediction))
        elif not isinstance(prediction, str):
            prediction = str(prediction)
        return prediction.strip()
