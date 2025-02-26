# src/reasoners/dummy_reasoners.py

class DummySymbolicReasoner:
    def process_query(self, query: str) -> list:
        # Return a safe fallback; for example, a message indicating no symbolic reasoning.
        return ["[DummySymbolicReasoner] No symbolic reasoning available."]


class DummyNeuralRetriever:
    def retrieve(self, query: str):
        # Return a safe fallback for retrieval.
        return []

    def retrieve_answer(self, context: str, question: str, symbolic_guidance=None,
                        supporting_facts=None, query_complexity=None):
        """
        Match the signature of the real NeuralRetriever.retrieve_answer method
        """
        return "This is a dummy response. Neural retriever was disabled for this ablation test."
