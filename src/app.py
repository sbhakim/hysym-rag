# src/app.py
from src.hybrid_integrator import HybridIntegrator

class App:
    def __init__(self, symbolic, neural, logger, feedback, evaluator, expander):
        self.integrator = HybridIntegrator(symbolic, neural)
        self.logger = logger
        self.feedback = feedback
        self.evaluator = evaluator
        self.expander = expander

    def run(self, query, context):
        # Query expansion and processing logic
        if self.expander:
            expanded_query = self.expander.expand_query(query)
        else:
            expanded_query = query  # Use the original query if no expander is provided
        result = self.integrator.process_query(expanded_query, context)
        return result
