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
        # Check cache
        if query in self.cache:
            print("Retrieving from cache...")
            return self.cache[query]

        query_complexity = 0.5
        if self.query_expander:
            query_complexity = self.query_expander.get_query_complexity(query)
        print(f"HybridIntegrator: Computed query complexity score: {query_complexity:.4f}")

        # Resource-based decision
        use_symbolic = self.resource_manager.should_use_symbolic(query_complexity)
        if use_symbolic:
            print("HybridIntegrator: Using symbolic reasoning")
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                result, source = symbolic_results, "symbolic"
                self.cache[query] = (result, source)
                return result, source
            else:
                # Fallback to neural if symbolic is empty
                print("Symbolic gave no valid result; fallback to neural.")
        else:
            print("HybridIntegrator: Defaulting to neural reasoning.")

        # Neural path:
        import time
        start_t = time.time()
        neural_answer = [self.neural_retriever.retrieve_answer(context, query)]
        end_t = time.time()

        # Record the neural inference time
        self.resource_manager.neural_perf_times.append(end_t - start_t)

        result, source = neural_answer, "neural"
        self.cache[query] = (result, source)
        return result, source
