class FeedbackHandler:
    def __init__(self, feedback_manager):
        self.feedback_manager = feedback_manager

    def collect_feedback(self, query, result):
        skip_feedback = input("Do you want to skip feedback? (yes/no): ").strip().lower()
        if skip_feedback != "yes":
            rating = int(input("Rate the result (1-5): "))
            comments = input("Comments (optional): ")
            self.feedback_manager.submit_feedback(query, result, rating, comments)
            print("Thank you for your feedback!")
        else:
            print("Feedback skipped.")