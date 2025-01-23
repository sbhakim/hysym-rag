class FeedbackHandler:
    """
    Handles feedback collection and management.
    """
    def __init__(self, feedback_manager):
        self.feedback_manager = feedback_manager

    def collect_feedback(self, query, result):
        """
        Collect feedback from the user for a given query result.
        """
        skip_feedback = input("Do you want to skip feedback? (yes/no): ").strip().lower()
        if skip_feedback != "yes":
            try:
                rating = int(input("Rate the result (1-5): "))
                if rating < 1 or rating > 5:
                    raise ValueError("Invalid rating. Must be between 1 and 5.")
                comments = input("Comments (optional): ")
                self.feedback_manager.submit_feedback(query, result, rating, comments)
                print("Thank you for your feedback!")
            except ValueError as e:
                print(f"Invalid input: {e}. Feedback collection aborted.")
        else:
            print("Feedback skipped.")
