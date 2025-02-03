# src/app.py
from src.hybrid_integrator import HybridIntegrator
from src.query_expander import QueryExpander


class App:
    """
    Main application class to integrate symbolic and neural reasoning.
    """

    def __init__(self, symbolic, neural, logger, feedback, evaluator, expander, ground_truths):
        # Pass the query expander to the HybridIntegrator as well.
        self.integrator = HybridIntegrator(symbolic, neural, feedback, expander)
        self.logger = logger
        self.feedback = feedback
        self.evaluator = evaluator
        self.expander = expander or QueryExpander()
        self.ground_truths = ground_truths

    def run(self, query, context):
        """
        Process the query with expanded terms and pass it to the hybrid integrator.
        Returns a tuple: (result, source)
        """
        expanded_query = self.expander.expand_query(query)
        print(f"Expanded Query: {expanded_query}")

        # Unpack the result and source from the integrator
        result, source = self.integrator.process_query(expanded_query, context)

        # Log the source for clarity
        print(f"Answer retrieved using: {source} reasoning.")

        # Evaluate only if ground truth exists for the query
        if self.evaluator and query in self.ground_truths:
            evaluation = self.evaluator.evaluate({query: result}, self.ground_truths)
            print(f"Evaluation Metrics: {evaluation}")
        else:
            print("Note: No ground truth available for evaluation")

        return result, source
