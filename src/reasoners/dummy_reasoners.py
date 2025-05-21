# src/reasoners/dummy_reasoners.py

from typing import List, Dict, Any, Optional, Union, Tuple


class DummySymbolicReasoner:
    def process_query(self,
                      query: str,
                      context: Optional[str] = None,
                      dataset_type: str = 'text',
                      query_id: Optional[str] = None) -> Union[List[str], Dict[str, Any]]:
        """
        Return a safe fallback indicating no symbolic reasoning.
        For DROP, returns a structured error dictionary.
        """
        rationale = "[DummySymbolicReasoner] Symbolic reasoning disabled for this ablation."
        if dataset_type == 'drop':
            return {
                "number": "",
                "spans": [],
                "date": {"day": "", "month": "", "year": ""},
                "status": "error", # Critical for HybridIntegrator's DROP fusion logic
                "confidence": 0.0,
                "rationale": rationale,
                "type": "error_symbolic_disabled" # Specific type for this dummy
            }
        return [rationale]


class DummyNeuralRetriever:
    def retrieve(self, query: str) -> list: # This method seems unused in main flows but kept
        """Return a safe fallback for retrieval (e.g., of chunks)."""
        return []

    def retrieve_answer(self,
                        context: str,
                        question: str,
                        symbolic_guidance: Optional[List[Dict]] = None,
                        supporting_facts: Optional[List[Tuple[str, int]]] = None,
                        query_complexity: Optional[float] = None,
                        dataset_type: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Match the signature of the real NeuralRetriever.retrieve_answer method.
        Returns a dummy response appropriate for the dataset_type.
        """
        rationale = "[DummyNeuralRetriever] Neural retriever disabled for this ablation test."
        if dataset_type == 'drop':
            # For DROP, the HybridIntegrator's fusion logic might expect a dictionary,
            # even if it's an error or a very low-confidence empty parse.
            # Returning a structured error response is safer.
            return {
                "number": "",
                "spans": [],
                "date": {"day": "", "month": "", "year": ""},
                "status": "error", # Indicate failure
                "confidence": 0.0,
                "rationale": rationale,
                "type": "error_neural_disabled" # Specific type
            }
        # For text-based QA, a simple string is often expected.
        return f"Error: {rationale}"