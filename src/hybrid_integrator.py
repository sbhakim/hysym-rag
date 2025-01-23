class HybridIntegrator:
    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.resource_manager = resource_manager
        # Simple cache for demonstration
        self.cache = {}

    def process_query(self, query, context):
        # Check cache first
        if query in self.cache:
            print("Retrieving from cache...")
            return self.cache[query]

        # Check resource status
        resources = self.resource_manager.check_resources()

        # Use symbolic reasoning if memory usage is high or for simple queries
        if resources['memory_utilization'] > 0.8:
            print("High memory usage detected - prioritizing symbolic reasoning")
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                self.cache[query] = symbolic_results
                return symbolic_results

        # Fall back to neural processing
        print("Using neural processing")
        result = [self.neural_retriever.retrieve_answer(context, query)]
        self.cache[query] = result
        return result