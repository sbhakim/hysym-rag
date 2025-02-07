# src/reasoners/networkx_symbolic_reasoner.py
import json
import spacy
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger("GraphSymbolicReasoner")
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model in GraphSymbolicReasoner: {str(e)}")
    raise

class GraphSymbolicReasoner:
    """
    A graph-based symbolic reasoner that uses sentence embeddings for matching and supports
    multi-hop chaining by traversing a rule graph. It also provides an extract_keywords method.
    """
    def __init__(self, rules_file, match_threshold=0.25, max_hops=5):
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.rules = self.load_rules(rules_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.rule_index = {}
        self.build_rule_index()
        self.graph = nx.DiGraph()
        self.build_graph()
        self.build_causal_graph()
        logger.info("GraphSymbolicReasoner initialized successfully.")

    def load_rules(self, rules_file):
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_rule_index(self):
        for i, rule in enumerate(self.rules):
            rule_id = f"rule_{i}"
            cause_text = " ".join(rule.get("keywords", []))
            embedding = self.embedder.encode(cause_text, convert_to_tensor=True)
            rule["embedding"] = embedding
            rule["id"] = rule_id
            self.rule_index[rule_id] = rule

    def build_graph(self):
        self.graph.clear()
        for rule_id, rule in self.rule_index.items():
            self.graph.add_node(rule_id, rule=rule)
        for id1, rule1 in self.rule_index.items():
            for id2, rule2 in self.rule_index.items():
                if id1 == id2:
                    continue
                if set(rule1.get("keywords", [])) & set(rule2.get("keywords", [])):
                    self.graph.add_edge(id1, id2)

    def build_causal_graph(self):
        self.causal_graph = nx.DiGraph()
        for rule_id, rule in self.rule_index.items():
            causes = rule.get("causes", [])
            effects = rule.get("effects", [])
            for cause in causes:
                for effect in effects:
                    self.causal_graph.add_edge(cause.lower(), effect.lower(), rule_id=rule_id)
        logger.info("Causal graph built successfully.")

    def extract_keywords(self, text):
        """
        Extract meaningful keywords from the input text using spaCy.
        """
        doc = self.nlp(text.lower())
        keywords = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ") and not token.is_stop and len(token.text) > 1]
        return list(set(keywords))

    def encode(self, query):
        return self.embedder.encode(query, convert_to_tensor=True)

    def traverse_graph(self, query_embedding, max_hops=None):
        if max_hops is None:
            max_hops = self.max_hops
        visited = set()
        results = []
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
                new_nodes.extend(list(self.graph.neighbors(node_id)))
            current_nodes = new_nodes
            hops += 1
        return results

    def traverse_causal_graph(self, query):
        tokens = set(query.lower().split())
        responses = []
        if not hasattr(self, 'causal_graph'):
            self.build_causal_graph()
        for token in tokens:
            if token in self.causal_graph:
                for target in self.causal_graph.nodes():
                    if target != token:
                        try:
                            paths = nx.all_simple_paths(self.causal_graph, source=token, target=target, cutoff=self.max_hops)
                            for path in paths:
                                responses.append(" -> ".join(path))
                        except nx.NetworkXNoPath:
                            continue
        return responses

    def process_query(self, query):
        query_embedding = self.encode(query)
        responses = self.traverse_graph(query_embedding)
        causal_responses = self.traverse_causal_graph(query)
        responses.extend(causal_responses)
        if not responses:
            logger.info("No symbolic match found.")
            return ["No symbolic match found."]
        logger.info(f"Multi-hop symbolic responses found: {responses}")
        return responses
