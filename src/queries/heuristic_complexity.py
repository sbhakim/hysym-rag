# In src/queries/heuristic_complexity.py

class HeuristicQueryComplexityEstimator:
    def __init__(self):
        # Set thresholds and weights as desired
        self.word_count_weight = 0.05
        self.analytical_bonus = 0.5
        self.domain_bonus = 0.3

    def estimate_complexity(self, query):
        word_count = len(query.split())
        complexity = word_count * self.word_count_weight
        if any(term in query.lower() for term in ["why", "how", "explain", "analyze"]):
            complexity += self.analytical_bonus
        if any(term in query.lower() for term in ["biodiversity", "climate", "deforestation"]):
            complexity += self.domain_bonus
        return complexity
