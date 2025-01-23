# src/query_expander.py

class QueryExpander:
    def __init__(self, expansion_rules=None):
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }

    def get_query_complexity(self, query):
        # Simple complexity scoring based on query characteristics
        score = 0
        words = query.split()

        # Length-based complexity
        score += len(words) * 0.1

        # Keyword-based complexity
        complexity_keywords = ["why", "how", "explain", "compare", "relationship"]
        score += sum(word in query.lower() for word in complexity_keywords) * 0.3

        return score

    def expand_query(self, query):
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity}")

        # Existing expansion logic
        expanded_query = query
        for term, related_terms in self.expansion_rules.items():
            if term in query:
                expansion = f"({term} OR {' OR '.join(related_terms)})"
                expanded_query = expanded_query.replace(term, expansion)
        return expanded_query

