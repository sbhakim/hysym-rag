# src/networkx_symbolic_reasoner.py

import json
import re
from collections import defaultdict

import networkx as nx

class GraphSymbolicReasoner:
    """
    A graph-based symbolic reasoner that maintains a directed, weighted rule graph.
    Each node corresponds to a rule, edges indicate semantic or causal relationships
    between rules, and edge weights represent the strength of these relationships.

    Unlike a simple 'keyword intersection' approach, this method can traverse multiple
    rule 'hops' in a controlled manner, factoring in user feedback to adjust confidences.
    """

    def __init__(self, rules_file, match_threshold=0.2, max_hops=2):
        """
        Args:
            rules_file (str): Path to a JSON file containing rules, each with 'keywords' and 'response'.
            match_threshold (float): Minimum chaining score required to consider a rule relevant.
            max_hops (int): Maximum depth for multi-hop chaining.

        Additional attributes:
            chain_decay (float): Factor by which the path score decays for each additional hop.
            rule_performance: Tracks usage stats and feedback-based confidence for each rule node.
        """
        with open(rules_file, 'r', encoding='utf-8') as file:
            self.rules = json.load(file)

        # Rule graph and index
        self.rule_graph = nx.DiGraph()
        self.rule_index = {}  # Maps rule_id -> { "keywords": set(...), "response_tokens": set(...), ... }

        # Core parameters
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.chain_decay = 0.8

        # Feedback-based stats per rule_id
        # For example: rule_performance["rule_0"] = { uses=..., positive_feedback=..., confidence=... }
        self.rule_performance = defaultdict(lambda: {
            'uses': 0,
            'positive_feedback': 0,
            'confidence': 0.5  # start each rule at neutral confidence
        })

        # Build the rule network (nodes + edges + weights)
        self.build_rule_network()

    def build_rule_network(self):
        """
        Constructs a weighted directed graph of rule relationships. Each node
        corresponds to one rule, and edges capture how strongly that rule
        connects to another (semantic overlap, cause-effect, etc.).
        """
        # Add nodes first
        for i, rule in enumerate(self.rules):
            rule_id = f"rule_{i}"
            keywords = set(rule.get("keywords", []))
            response_tokens = set(self._tokenize(rule.get("response", "")))

            self.rule_index[rule_id] = {
                "keywords": keywords,
                "response_tokens": response_tokens,
                "original_rule": rule
            }
            self.rule_graph.add_node(rule_id)

        # Add weighted edges
        for rule_id1 in self.rule_graph.nodes():
            node1 = self.rule_index[rule_id1]
            for rule_id2 in self.rule_graph.nodes():
                if rule_id1 == rule_id2:
                    continue
                node2 = self.rule_index[rule_id2]

                weight = self._calculate_rule_relationship(
                    node1["keywords"], node1["response_tokens"],
                    node2["keywords"], node2["response_tokens"]
                )
                # Only add an edge if the relationship meets a minimal threshold
                if weight > 0.2:
                    self.rule_graph.add_edge(rule_id1, rule_id2, weight=weight)

    def _calculate_rule_relationship(self, keywords1, response1, keywords2, response2):
        """
        Computes a relationship strength (0..1) between two rules using:
        (1) keyword overlap,
        (2) response-to-keyword overlap (e.g. cause->effect),
        (3) concept similarity in the responses themselves.

        This can be extended with embeddings or more sophisticated approaches.
        """
        # 1) Keyword overlap
        if keywords1 or keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / max(len(keywords1.union(keywords2)), 1)
        else:
            keyword_similarity = 0.0

        # 2) Response -> other's keywords overlap
        if response1:
            response_to_keyword = len(response1.intersection(keywords2)) / max(len(response1), 1)
        else:
            response_to_keyword = 0.0

        # 3) Response-to-response overlap
        if response1 or response2:
            concept_similarity = len(response1.intersection(response2)) / max(len(response1.union(response2)), 1)
        else:
            concept_similarity = 0.0

        # Weighted sum
        return (
            0.4 * keyword_similarity
            + 0.4 * response_to_keyword
            + 0.2 * concept_similarity
        )

    def process_query(self, query):
        """
        Public method for processing a natural language query with multi-hop chaining.
        1. Find all rules matching the query tokens (above match_threshold).
        2. For each match, run a DFS to discover possible multi-hop chains.
        3. Convert each chain to a final response string.
        4. Aggregate all chain responses into final output.
        """
        query_tokens = set(self._tokenize(query))
        initial_matches = self._find_matching_rules(query_tokens)
        if not initial_matches:
            return ["No symbolic match found."]

        responses = []
        visited = set()

        # Explore each initial matched rule
        for (rule_id, match_score) in initial_matches:
            if rule_id in visited:
                continue

            chains = self._find_rule_chains(rule_id, match_score, visited, self.max_hops)
            for chain_path, chain_score in chains:
                response_str = self._process_rule_chain(chain_path, chain_score)
                if response_str:
                    responses.append(response_str)

        final_output = self._aggregate_responses(responses)
        return final_output

    def _find_matching_rules(self, query_tokens):
        """
        Finds rules that match the query tokens above self.match_threshold.
        We'll use a simple ratio: intersection(query_tokens, rule_keywords) / len(rule_keywords).
        Returns a list of (rule_id, match_score).
        """
        matches = []
        for rule_id, node_data in self.rule_index.items():
            keywords = node_data["keywords"]
            if not keywords:
                continue

            overlap = len(query_tokens.intersection(keywords)) / float(len(keywords))
            # We also factor in the rule's current confidence
            confidence = self.rule_performance[rule_id]['confidence']
            match_score = overlap * confidence

            if match_score >= self.match_threshold:
                matches.append((rule_id, match_score))

        # Sort descending so we handle the strongest matches first
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _find_rule_chains(self, start_rule_id, initial_score, visited, max_depth):
        """
        Depth-limited DFS that looks for neighbor rules, factoring in:
        - Edge weight
        - Next node's confidence
        - chain_decay per hop
        - Current path_score
        Returns a list of (chain_path, chain_score).
        """
        chains = []
        stack = [(start_rule_id, [start_rule_id], initial_score)]

        while stack:
            current_id, path, path_score = stack.pop()

            # Add the path if it's more than a single rule
            if len(path) > 1:
                chains.append((path, path_score))

            # Depth check
            if len(path) >= max_depth:
                continue

            # Explore neighbors
            for neighbor_id in self.rule_graph.neighbors(current_id):
                if neighbor_id not in visited:
                    edge_weight = self.rule_graph[current_id][neighbor_id]['weight']
                    neighbor_conf = self.rule_performance[neighbor_id]['confidence']

                    new_score = path_score * self.chain_decay * edge_weight * neighbor_conf
                    if new_score >= self.match_threshold:
                        stack.append((neighbor_id, path + [neighbor_id], new_score))

            # Mark current node visited
            visited.add(current_id)

        return chains

    def _process_rule_chain(self, chain_path, chain_score):
        """
        Takes a chain of rule_ids (path) and merges their 'response' fields
        into a coherent single string. We also factor chain_score for reference.
        """
        responses = []
        for i, rule_id in enumerate(chain_path):
            rule_data = self.rule_index[rule_id]["original_rule"]
            text = rule_data.get("response", "")
            if i > 0:
                # Connect this rule’s response to the previous one
                text = self._connect_responses(responses[-1], text)
            responses.append(text)

        return self._combine_responses(responses, chain_score)

    def _connect_responses(self, prev_response, current_response):
        """
        A placeholder that might do a more advanced 'transition' between consecutive rule responses.
        For now, we simply put a newline or a linking phrase between them.
        """
        # Example simplistic approach:
        joined = prev_response.rstrip(". ") + ". " + current_response.lstrip()
        return joined

    def _combine_responses(self, responses, chain_score):
        """
        Combine all partial responses in a chain into one final string. You can get creative here.
        For demonstration, we just join them with newlines, plus a mention of chain_score.
        """
        # Optionally factor chain_score into text, or skip if you prefer.
        combined = "\n".join(responses)
        final_text = f"{combined}\n(Chain Score: {chain_score:.2f})"
        return final_text

    def _aggregate_responses(self, all_responses):
        """
        After generating multiple chain outputs, unify or filter them. E.g.:
        - removing duplicates
        - sorting by chain_score
        - returning the top N
        Here, we'll just return them all as a list.
        """
        unique = list(set(all_responses))
        # Sort them in descending order of chain_score if you want. We put
        # chain_score in text, so we can't trivially parse it here.
        # We'll just return unique as-is:
        return unique

    def update_from_feedback(self, rule_id, feedback_score):
        """
        Updates the confidence of a specific rule based on user feedback.

        Args:
            rule_id: e.g. "rule_0"
            feedback_score (float): a normalized 0..1 rating from the user.
        """
        stats = self.rule_performance[rule_id]
        stats['uses'] += 1
        stats['positive_feedback'] += feedback_score

        # Exponential moving average for confidence
        alpha = 0.1
        new_conf = feedback_score
        stats['confidence'] = (1 - alpha) * stats['confidence'] + alpha * new_conf

        # Optionally adjust match_threshold globally
        if stats['uses'] > 5:
            avg_confidence = sum(
                s['confidence'] for s in self.rule_performance.values()
            ) / len(self.rule_performance)
            # Example approach: tie match_threshold to average confidence
            self.match_threshold = max(0.1, min(0.5, avg_confidence))

    def _tokenize(self, text):
        """
        Basic tokenization: split into alphanumeric words, lowercased.
        """
        return re.findall(r'\b\w+\b', text.lower())
