# src/query_expander.py

class QueryExpander:
    def __init__(self, expansion_rules=None):
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }

    def get_query_complexity(self, query):
        """
        Compute a basic complexity score based on:
        1. Length-based complexity (number of words).
        2. Keyword-based complexity for terms like 'why', 'how', 'explain', etc.
        """
        score = 0
        words = query.split()

        # Length-based complexity
        score += len(words) * 0.1

        # Keyword-based complexity
        complexity_keywords = ["why", "how", "explain", "compare", "relationship"]
        score += sum(word in query.lower() for word in complexity_keywords) * 0.3

        return score

    def expand_query(self, query):
        """
        Expand specific terms (e.g., 'deforestation' → '(deforestation OR forest loss OR ...)')
        unless the complexity is already above a threshold.
        """
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity}")

        # Skip expansions if complexity is already high
        if complexity > 1.0:
            print("Skipping expansions due to high complexity.")
            return query

        # Otherwise, proceed with existing expansion logic
        expanded_query = query
        for term, related_terms in self.expansion_rules.items():
            if term in query:
                expansion = f"({term} OR {' OR '.join(related_terms)})"
                expanded_query = expanded_query.replace(term, expansion)

        return expanded_query
