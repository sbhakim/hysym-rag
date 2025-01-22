from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


class Evaluation:
    @staticmethod
    def evaluate(predictions, ground_truths):
        precision = precision_score(ground_truths, predictions, average="weighted")
        recall = recall_score(ground_truths, predictions, average="weighted")
        f1 = f1_score(ground_truths, predictions, average="weighted")
        return {"precision": precision, "recall": recall, "f1_score": f1}

    @staticmethod
    def generate_report(predictions, ground_truths, labels):
        return classification_report(ground_truths, predictions, target_names=labels)
