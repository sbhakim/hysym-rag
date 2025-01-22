class HybridIntegrator:
    def __init__(self, symbolic_reasoner, neural_retriever):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever

    def process_query(self, query, context):
        symbolic_results = self.symbolic_reasoner.process_query(query)
        if symbolic_results and "No symbolic match found." not in symbolic_results:
            return symbolic_results
        return [self.neural_retriever.retrieve_answer(context, query)]
