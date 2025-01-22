from src.symbolic_reasoner import SymbolicReasoner
from src.neural_retriever import NeuralRetriever
from src.hybrid_integrator import HybridIntegrator

if __name__ == "__main__":
    # Initialize components
    symbolic = SymbolicReasoner("data/rules.json")
    neural = NeuralRetriever()
    integrator = HybridIntegrator(symbolic, neural)

    # Sample inputs
    context = open("data/small_knowledge_base.txt").read()
    query = "What are the effects of tropical deforestation?"

    # Process query
    result = integrator.process_query(query, context)
    print(f"Result: {result}")
