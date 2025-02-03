# src/hybrid_integrator.py

class HybridIntegrator:
    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager, query_expander=None):
        """
        Initializes the HybridIntegrator with symbolic and neural components,
        a resource manager, and an optional query expander for computing query complexity.
        """
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander  # Optional: used to compute query complexity
        self.cache = {}  # Cache to store processed queries for faster subsequent retrievals

    def process_query(self, query, context):
        """
        Processes a query and returns a tuple: (result, source) where source is either 'symbolic' or 'neural'.
        """
        if query in self.cache:
            print("Retrieving from cache...")
            return self.cache[query]

        # Compute query complexity if a query expander is provided; default to 0.5 otherwise.
        query_complexity = self.query_expander.get_query_complexity(query) if self.query_expander else 0.5
        print(f"HybridIntegrator: Computed query complexity score: {query_complexity:.4f}")

        # Decide based on resource usage and complexity whether to use symbolic reasoning.
        if self.resource_manager.should_use_symbolic(query_complexity):
            print("HybridIntegrator: Using symbolic reasoning based on resource and complexity checks")
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                result, source = symbolic_results, "symbolic"
                self.cache[query] = (result, source)
                return result, source
            else:
                print("HybridIntegrator: Symbolic reasoning yielded no valid result; falling back to neural processing.")

        # Fall back to neural processing.
        print("HybridIntegrator: Defaulting to neural reasoning based on resource and complexity checks.")
        neural_result = [self.neural_retriever.retrieve_answer(context, query)]
        result, source = neural_result, "neural"
        self.cache[query] = (result, source)
        return result, source
