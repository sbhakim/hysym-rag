# src/query_expander.py
import yaml
import spacy
import os

class QueryExpander:
    def __init__(self, complexity_config=None, expansion_rules=None):
        """
        Args:
            complexity_config: Path to a YAML file with advanced complexity rules (optional).
            expansion_rules: Optional dict if you'd prefer to pass expansions programmatically.
        """
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }

        self.nlp = spacy.load("en_core_web_sm")

        self.config = {}
        if complexity_config and os.path.exists(complexity_config):
            with open(complexity_config, "r") as f:
                self.config = yaml.safe_load(f)

    def get_query_complexity(self, query):
        """
        If no advanced config is found, fallback to a simpler scoring.
        Otherwise, incorporate advanced rules from complexity_rules.yaml.
        """
        if not self.config:
            # Fallback
            score = 0
            words = query.lower().split()
            score += len(words) * 0.1
            complexity_keywords = ["why", "how", "explain", "compare", "relationship"]
            score += sum(word in query.lower() for word in complexity_keywords) * 0.3
            return score

        # Advanced approach with spaCy + config-based scoring
        doc = self.nlp(query.lower())
        base_score = 0.0

        # (A) Sentence count
        base_score += len(list(doc.sents)) * 0.3

        # (B) Content words
        content_tokens = [token for token in doc if not token.is_stop]
        base_score += len(content_tokens) * 0.1

        # (C) Check question root
        # We look for root in question_types
        root = next((token for token in doc if token.dep_ == "ROOT"), None)
        if root:
            root_text = root.lemma_.lower()
            # If it matches any question type in advanced config
            for category, group in self.config.get("question_types", {}).items():
                for entry in group:
                    for key, w in entry.items():
                        if root_text == key:
                            base_score += w

        # (D) Domain terms
        domain_score = 0.0
        for domain, term_list in self.config.get("domain_terms", {}).items():
            for term_data in term_list:
                main_term = term_data["term"].lower()
                weight = term_data["weight"]
                synonyms = term_data.get("synonyms", [])
                # Check main term
                if main_term in query.lower():
                    domain_score += weight
                # Check synonyms
                for syn in synonyms:
                    if syn in query.lower():
                        domain_score += weight * 0.8

        # (E) Structural patterns
        struct_score = self._analyze_structural_patterns(doc)

        # Weighted combination
        total_score = (base_score * 0.3) + (domain_score * 0.2) + (struct_score * 0.2) + (base_score * 0.3)
        # We do a second weighting of base_score for emphasis; you can tune as needed

        # Clamp
        return min(3.0, max(0.0, total_score))

    def _analyze_structural_patterns(self, doc):
        """
        For multi-part or dependency patterns, using config.
        """
        if not self.config.get("structural_patterns"):
            return 0.0

        score = 0.0
        text = [t.text for t in doc]

        # Multi-part
        multi_part = self.config["structural_patterns"].get("multi_part", [])
        for item in multi_part:
            pattern = item["pattern"]
            weight = item["weight"]
            if pattern in text:
                score += weight

        # Dependency cues
        dependency = self.config["structural_patterns"].get("dependency", [])
        for item in dependency:
            pattern = item["pattern"]
            weight = item["weight"]
            if pattern in " ".join(text):
                score += weight

        return score

    def expand_query(self, query):
        """
        Expand domain terms with synonyms unless the complexity is already high.
        """
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity}")

        if complexity > 1.0:
            print("Skipping expansions due to high complexity.")
            return query

        expanded = query
        for term, synonyms in self.expansion_rules.items():
            if term in query.lower():
                expansion_group = f"({term} OR {' OR '.join(synonyms)})"
                expanded = expanded.replace(term, expansion_group)

        return expanded
