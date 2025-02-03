# src/hybrid_integrator.py

class HybridIntegrator:
    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager, query_expander=None):
        """
        Manages whether queries go to symbolic or neural. Caches results.
        """
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander
        self.cache = {}

    def process_query(self, query, context):
        if query in self.cache:
            print("Retrieving from cache...")
            return self.cache[query]

        query_complexity = 0.5
        if self.query_expander:
            query_complexity = self.query_expander.get_query_complexity(query)
        print(f"HybridIntegrator: Computed query complexity score: {query_complexity:.4f}")

        # Resource-based decision
        if self.resource_manager.should_use_symbolic(query_complexity):
            print("HybridIntegrator: Using symbolic reasoning")
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                result, source = symbolic_results, "symbolic"
                self.cache[query] = (result, source)
                return result, source
            else:
                print("Symbolic gave no valid result; fallback to neural.")
        else:
            print("HybridIntegrator: Defaulting to neural reasoning.")

        neural_answer = [self.neural_retriever.retrieve_answer(context, query)]
        result, source = neural_answer, "neural"
        self.cache[query] = (result, source)
        return result, source
