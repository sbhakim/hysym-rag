# src/feedback/feedback_manager.py
import json
import os

class FeedbackManager:
    def __init__(self, feedback_file="logs/feedback.json"):
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        self.feedback_file = feedback_file

    def submit_feedback(self, query, result, rating, comments=""):
        feedback_entry = {
            "query": query,
            "result": result,
            "rating": rating,
            "comments": comments,
        }
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r") as file:
                feedbacks = json.load(file)
        else:
            feedbacks = []

        feedbacks.append(feedback_entry)
        with open(self.feedback_file, "w") as file:
            json.dump(feedbacks, file, indent=4)
