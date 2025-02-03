# src/evaluation.py
from sklearn.metrics import classification_report
from difflib import SequenceMatcher

class Evaluation:
    @staticmethod
    def evaluate(predictions, ground_truths):
        """
        Evaluate predictions against ground truths using similarity and counts.
        """
        valid_queries = [q for q in predictions if q in ground_truths]
        if not valid_queries:
            print("Warning: No matching ground truths found for evaluation")
            return {
                "average_similarity": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "evaluated_queries": 0
            }

        y_true = [ground_truths[q] for q in valid_queries]
        y_pred = [Evaluation.simplify_prediction(predictions[q]) for q in valid_queries]

        similarities = [
            SequenceMatcher(None, gt, pred).ratio()
            for gt, pred in zip(y_true, y_pred)
        ]
        avg_similarity = sum(similarities)/len(similarities) if similarities else 0.0

        return {
            "average_similarity": avg_similarity,
            "evaluated_queries": len(valid_queries),
            "queries_with_truth": valid_queries
        }

    @staticmethod
    def simplify_prediction(prediction):
        if isinstance(prediction, list):
            prediction = prediction[0]
        return prediction.split('.')[0]  # only first sentence

    @staticmethod
    def generate_report(predictions, ground_truths, labels=None):
        y_true = [ground_truths[q] for q in predictions if q in ground_truths]
        y_pred = [
            Evaluation.simplify_prediction(predictions[q])
            for q in predictions if q in ground_truths
        ]
        return classification_report(y_true, y_pred, target_names=labels, zero_division=0)
