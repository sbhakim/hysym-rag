# src/reasoners/networkx_symbolic_reasoner.py

import json
import spacy
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

logger = logging.getLogger("GraphSymbolicReasoner")
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model in GraphSymbolicReasoner: {str(e)}")
    raise

class GraphSymbolicReasoner:
    """
    A graph-based symbolic reasoner that uses sentence embeddings for matching
    and supports multi-hop chaining by traversing a rule graph.
    """
    def __init__(self, rules_file, match_threshold=0.25, max_hops=5):
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.rules = self.load_rules(rules_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.rule_index = {}
        self.traversal_cache = {}
        self.build_rule_index()
        self.build_graph()
        self.build_causal_graph()
        logger.info("GraphSymbolicReasoner initialized successfully.")

    def load_rules(self, rules_file):
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_rule_index(self):
        self.rule_ids = []
        embeddings = []
        for i, rule in enumerate(self.rules):
            rule_id = f"rule_{i}"
            cause_text = " ".join(rule.get("keywords", []))
            embedding = self.embedder.encode(cause_text, convert_to_tensor=True)
            rule["embedding"] = embedding
            rule["id"] = rule_id
            self.rule_index[rule_id] = rule
            self.rule_ids.append(rule_id)
            embeddings.append(embedding)
        if embeddings:
            self.rule_embeddings = torch.stack(embeddings)
        else:
            self.rule_embeddings = torch.empty((0,))

    def build_graph(self):
        self.graph = nx.DiGraph()
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
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc
                  if token.pos_ in ("NOUN", "VERB", "ADJ") and not token.is_stop and len(token.text) > 1]
        return list(set(tokens))

    def encode(self, query):
        if not query:
            return torch.zeros((1,))
        return self.embedder.encode(query, convert_to_tensor=True)

    def traverse_graph(self, query_embedding, max_hops=None):
        if self.rule_embeddings.shape[0] == 0:
            return []
        if max_hops is None:
            max_hops = self.max_hops
        fingerprint = hash(query_embedding.detach().cpu().numpy().tobytes())
        if fingerprint in self.traversal_cache:
            return self.traversal_cache[fingerprint]

        sims = util.cos_sim(query_embedding, self.rule_embeddings).squeeze(0)
        initial_indices = (sims >= self.match_threshold).nonzero(as_tuple=False).flatten().tolist()
        if not initial_indices:
            self.traversal_cache[fingerprint] = []
            return []
        initial_matches = [self.rule_ids[i] for i in initial_indices]

        visited = set()
        results = []
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
                neighbors = list(self.graph.neighbors(node_id))
                new_nodes.extend(neighbors)
            current_nodes = new_nodes
            hops += 1

        self.traversal_cache[fingerprint] = results
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
                            paths = nx.all_simple_paths(self.causal_graph, source=token, target=target,
                                                        cutoff=self.max_hops)
                            for path in paths:
                                responses.append(" -> ".join(path))
                        except nx.NetworkXNoPath:
                            continue
        return responses

    def process_query(self, query):
        query_embedding = self.encode(query)
        if query_embedding.dim() == 0:
            return ["No symbolic match found."]
        responses = self.traverse_graph(query_embedding)
        causal_responses = self.traverse_causal_graph(query)
        responses.extend(causal_responses)
        if not responses:
            logger.info("No symbolic match found.")
            return ["No symbolic match found."]
        logger.info(f"Multi-hop symbolic responses found: {responses}")
        return responses

    def add_dynamic_rules(self, new_rules):
        """
        Add newly extracted rules to both self.rules and the internal graph structures in-memory.
        """
        if not new_rules:
            return

        start_index = len(self.rules)
        new_embeddings = []
        for i, rule in enumerate(new_rules):
            rule_id = f"rule_{start_index + i}"
            cause_text = " ".join(rule.get("keywords", []))
            rule_embedding = self.embedder.encode(cause_text, convert_to_tensor=True)
            rule["embedding"] = rule_embedding
            rule["id"] = rule_id
            self.rules.append(rule)
            self.rule_index[rule_id] = rule

            # Add node to the main graph
            self.graph.add_node(rule_id, rule=rule)
            for existing_id, existing_rule in self.rule_index.items():
                if existing_id == rule_id:
                    continue
                # If overlap in keywords, link them
                if set(rule.get("keywords", [])) & set(existing_rule.get("keywords", [])):
                    self.graph.add_edge(rule_id, existing_id)
                    self.graph.add_edge(existing_id, rule_id)

            new_embeddings.append(rule_embedding)
            logger.debug(f"Added new rule: {rule} with ID {rule_id}")

        # Update rule_ids and rule_embeddings
        self.rule_ids.extend([f"rule_{start_index + i}" for i in range(len(new_rules))])
        if len(self.rule_embeddings.shape) == 0 or self.rule_embeddings.shape[0] == 0:
            # If empty, replace entirely
            self.rule_embeddings = torch.stack(new_embeddings)
        else:
            self.rule_embeddings = torch.cat([self.rule_embeddings, torch.stack(new_embeddings)], dim=0)

        logger.info(f"Added {len(new_rules)} dynamic rules to the symbolic reasoner.")
