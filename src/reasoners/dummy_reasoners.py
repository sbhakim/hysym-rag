# src/reasoners/dummy_reasoners.py

class DummySymbolicReasoner:
    def process_query(self, query: str) -> list:
        # Return a safe fallback; for example, a message indicating no symbolic reasoning.
        return ["[DummySymbolicReasoner] No symbolic reasoning available."]

class DummyNeuralRetriever:
    def retrieve(self, query: str):
        # Return a safe fallback for retrieval.
        return []
