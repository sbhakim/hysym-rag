# src/reasoners/networkx_symbolic_reasoner.py
import json
import spacy
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer, util
import torch

class GraphSymbolicReasoner:
    """
    A graph-based symbolic reasoner that uses sentence embeddings for matching and
    supports multi-hop chaining by traversing a rule graph.
    Rules (from rules.json) are assumed to have a "keywords" list and a "response".
    """
    def __init__(self, rules_file, match_threshold=0.25, max_hops=5):
        self.logger = logging.getLogger("GraphSymbolicReasoner")
        self.match_threshold = match_threshold
        self.max_hops = max_hops  # Maximum hops for multi-hop chaining
        self.rules = self.load_rules(rules_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.rule_index = {}  # Mapping from rule ID to rule dict
        self.build_rule_index()
        self.graph = nx.DiGraph()
        self.build_graph()
        self.logger.info("GraphSymbolicReasoner initialized successfully.")

    def load_rules(self, rules_file):
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_rule_index(self):
        for i, rule in enumerate(self.rules):
            rule_id = f"rule_{i}"
            # Concatenate keywords to form a representation for embedding
            cause_text = " ".join(rule.get("keywords", []))
            embedding = self.embedder.encode(cause_text, convert_to_tensor=True)
            rule["embedding"] = embedding
            rule["id"] = rule_id
            self.rule_index[rule_id] = rule

    def build_graph(self):
        """Build a directed graph linking rules that share at least one keyword."""
        self.graph.clear()
        # Add nodes for each rule
        for rule_id, rule in self.rule_index.items():
            self.graph.add_node(rule_id, rule=rule)
        # Add edges: if two rules share at least one keyword, add a directed edge.
        for id1, rule1 in self.rule_index.items():
            for id2, rule2 in self.rule_index.items():
                if id1 == id2:
                    continue
                if set(rule1.get("keywords", [])) & set(rule2.get("keywords", [])):
                    self.graph.add_edge(id1, id2)

    def encode(self, query):
        """
        Encode the query into an embedding tensor using the SentenceTransformer.
        This method enables alignment with neural embeddings.
        """
        return self.embedder.encode(query, convert_to_tensor=True)

    def traverse_graph(self, query_embedding, max_hops=None):
        """
        Traverse the rule graph starting from initial matches based on the query embedding.
        Returns a list of responses collected over multiple hops.
        """
        if max_hops is None:
            max_hops = self.max_hops
        visited = set()
        results = []

        # Find initial matching nodes based on embedding similarity.
        initial_matches = []
        for rule_id, rule in self.rule_index.items():
            sim = util.cos_sim(query_embedding, rule["embedding"]).item()
            if sim >= self.match_threshold:
                initial_matches.append(rule_id)
        if not initial_matches:
            return []

        current_nodes = initial_matches
        hops = 0
        while current_nodes and hops < max_hops:
            new_nodes = []
            for node_id in current_nodes:
                if node_id in visited:
                    continue
                visited.add(node_id)
                rule = self.rule_index[node_id]
                results.append(rule["response"])
                # Add neighbors for next hop
                new_nodes.extend(list(self.graph.neighbors(node_id)))
            current_nodes = new_nodes
            hops += 1
        return results

    def process_query(self, query):
        """
        Process the query:
          1. Encode the query.
          2. Traverse the rule graph for multi-hop responses.
          3. Return aggregated responses or a default message.
        """
        query_embedding = self.encode(query)
        responses = self.traverse_graph(query_embedding)
        if not responses:
            self.logger.info("No symbolic match found.")
            return ["No symbolic match found."]
        self.logger.info(f"Multi-hop symbolic responses found: {responses}")
        return responses
