# src/networkx_symbolic_reasoner.py
import json
import re
import spacy
from sentence_transformers import SentenceTransformer, util
import logging

class GraphSymbolicReasoner:
    """
    A graph-based symbolic reasoner that uses sentence embeddings for matching.
    Rules (from rules.json) are assumed to have a "keywords" list and a "response".
    """
    def __init__(self, rules_file, match_threshold=0.25, max_hops=5):
        self.logger = logging.getLogger("GraphSymbolicReasoner")
        self.match_threshold = match_threshold
        self.max_hops = max_hops  # (chaining not further implemented here)
        self.rules = self.load_rules(rules_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.rule_index = {}
        self.build_rule_index()
        self.logger.info("GraphSymbolicReasoner initialized successfully.")

    def load_rules(self, rules_file):
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_rule_index(self):
        for i, rule in enumerate(self.rules):
            rule_id = f"rule_{i}"
            # Concatenate keywords to form a cause representation
            cause_text = " ".join(rule.get("keywords", []))
            embedding = self.embedder.encode(cause_text, convert_to_tensor=True)
            self.rule_index[rule_id] = {
                "embedding": embedding,
                "response": rule.get("response", ""),
                "original_rule": rule
            }

    def process_query(self, query):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        matches = []
        for rule_id, data in self.rule_index.items():
            sim = util.cos_sim(query_embedding, data["embedding"]).item()
            if sim >= self.match_threshold:
                matches.append((rule_id, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        if not matches:
            return ["No symbolic match found."]
        # For simplicity, return the response from the best matching rule.
        best_rule_id = matches[0][0]
        response = self.rule_index[best_rule_id]["response"]
        self.logger.info(f"Symbolic match found (score={matches[0][1]:.2f}).")
        return [response]
