# src/query_expander.py
import yaml
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import os

class QueryExpander:
    def __init__(self, complexity_config=None, expansion_rules=None):
        # Original expansion rules
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }
        # Use a medium-size spacy model for vector support (ensure you have it installed)
        self.nlp = spacy.load("en_core_web_md")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny", output_attentions=True)
        self.config = {}
        if complexity_config and os.path.exists(complexity_config):
            with open(complexity_config, "r") as f:
                self.config = yaml.safe_load(f)
        # Set a similarity threshold for expansion
        self.sim_threshold = 0.6

    def get_query_complexity(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        attn = outputs.attentions[-1].mean().item()
        complexity = attn * len(query.split())
        return complexity

    def _analyze_structural_patterns(self, doc):
        if not self.config.get("structural_patterns"):
            return 0.0
        score = 0.0
        text = [t.text for t in doc]
        multi_part = self.config["structural_patterns"].get("multi_part", [])
        for item in multi_part:
            pattern = item["pattern"]
            weight = item["weight"]
            if pattern in text:
                score += weight
        dependency = self.config["structural_patterns"].get("dependency", [])
        for item in dependency:
            pattern = item["pattern"]
            weight = item["weight"]
            if pattern in " ".join(text):
                score += weight
        return score

    def _semantic_expansion(self, query):
        """
        Perform semantic query expansion using spacy vectors.
        For each term in our expansion rules that appears in the query,
        find similar words (from the provided candidate list) that exceed a similarity threshold.
        """
        doc = self.nlp(query)
        expanded_terms = []
        for term, candidates in self.expansion_rules.items():
            # Process the domain term using spacy
            term_doc = self.nlp(term)
            if term.lower() in query.lower():
                # For each candidate, compute similarity with the domain term
                similar = []
                for cand in candidates:
                    cand_doc = self.nlp(cand)
                    sim = term_doc.similarity(cand_doc)
                    if sim >= self.sim_threshold:
                        similar.append(cand)
                if similar:
                    # Append the domain term along with selected synonyms
                    expanded_terms.append(f"({term} OR {' OR '.join(similar)})")
        return " ".join(expanded_terms)

    def expand_query(self, query):
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity}")
        # If complexity is high, skip expansion
        if complexity > 1.0:
            print("Skipping expansion due to high complexity.")
            return query
        # Perform semantic expansion
        semantic_expansion = self._semantic_expansion(query)
        # Also perform original string-based expansion (if desired)
        basic_expansion = query
        for term, synonyms in self.expansion_rules.items():
            if term.lower() in query.lower():
                expansion_group = f"({term} OR {' OR '.join(synonyms)})"
                basic_expansion = basic_expansion.replace(term, expansion_group)
        # Combine original query with semantic expansion (if any)
        if semantic_expansion:
            expanded = basic_expansion + " " + semantic_expansion
        else:
            expanded = basic_expansion
        return expanded.strip()

# For testing purposes:
if __name__ == "__main__":
    expander = QueryExpander(complexity_config="src/config/complexity_rules.yaml")
    test_query = "What are the environmental effects of deforestation?"
    print("Original Query:", test_query)
    print("Expanded Query:", expander.expand_query(test_query))
