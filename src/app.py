# src/app.py

from src.integrators.hybrid_integrator import HybridIntegrator
from src.queries.query_expander import QueryExpander
from src.reasoners.neural_retriever import NeuralRetriever # Import NeuralRetriever

class App:
    """
    Main application class to integrate symbolic and neural reasoning.
    """
    def __init__(self, symbolic, neural, logger, feedback, evaluator, expander, ground_truths, system_manager=None):
        self.integrator = HybridIntegrator(symbolic, neural, feedback, expander)
        self.logger = logger
        self.feedback = feedback
        self.evaluator = evaluator
        self.expander = expander or QueryExpander()
        self.ground_truths = ground_truths
        self.system_manager = system_manager

    def run(self, query, context):
        """
        Process the query with expanded terms and pass it to the hybrid integrator.
        Returns (result, source).
        """
        if self.system_manager:
            # Use SystemControlManager for enhanced processing
            complexity = self.expander.get_query_complexity(query) # Calculate complexity here in app.run
            return self.system_manager.process_query_with_fallback(query, context, query_complexity=complexity) # Pass complexity
        else:
            # Default fallback to HybridIntegrator
            expanded_query = self.expander.expand_query(query)
            print(f"Expanded Query: {expanded_query}")

            result, source = self.integrator.process_query(expanded_query, context)
            print(f"Answer retrieved using: {source} reasoning.")

            # Evaluate if we have a ground truth
            if self.evaluator and query in self.ground_truths:
                evaluation = self.evaluator.evaluate({query: result}, self.ground_truths)
                print(f"Evaluation Metrics: {evaluation}")
            else:
                print("Note: No ground truth available for evaluation")

            return result, source



if __name__ == '__main__':
    # Example Usage (Conceptual - needs actual component initialization)
    # Assume symbolic, neural, logger, feedback, evaluator, expander, ground_truths are initialized

    # Example instantiation of NeuralRetriever (you'd need to configure model_name correctly)
    neural_retriever = NeuralRetriever(model_name="meta-llama/Llama-3.2-3B") # Replace with your actual model name

    # Example usage of NeuralRetriever's retrieve_answer method (you'd need to provide context and query)
    sample_context = "Forests are vital for the environment and economy..." # Replace with actual context
    sample_query = "What are the environmental effects of deforestation?" # Replace with actual query
    symbolic_rules = [{"response": "Deforestation causes biodiversity loss."}, {"response": "Deforestation leads to soil erosion."}] # Example rules - replace with actual rules

    # Example call to retrieve_answer with RG-Retriever enabled and symbolic guidance
    neural_answer = neural_retriever.retrieve_answer(sample_context, sample_query, symbolic_guidance=symbolic_rules, rule_guided_retrieval=True)
    print("\nNeural Retriever Answer with RG-Retriever:\n", neural_answer)

    # Example call to retrieve_answer with RG-Retriever disabled
    neural_answer_no_rg = neural_retriever.retrieve_answer(sample_context, sample_query, rule_guided_retrieval=False)
    print("\nNeural Retriever Answer without RG-Retriever:\n", neural_answer_no_rg)
