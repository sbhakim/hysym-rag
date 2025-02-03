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

    This version performs multi-hop chaining, but if no chain yields a valid result,
    it falls back to returning the best direct match.
    """

    def __init__(self, rules_file, match_threshold=0.2, max_hops=2):
        with open(rules_file, 'r', encoding='utf-8') as file:
            self.rules = json.load(file)

        self.rule_graph = nx.DiGraph()
        self.rule_index = {}  # Maps rule_id -> { "keywords": set(...), "response_tokens": set(...), ... }
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.chain_decay = 0.8
        self.rule_performance = defaultdict(lambda: {
            'uses': 0,
            'positive_feedback': 0,
            'confidence': 0.5
        })
        self.build_rule_network()

    def build_rule_network(self):
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
                if weight > 0.2:
                    self.rule_graph.add_edge(rule_id1, rule_id2, weight=weight)

    def _calculate_rule_relationship(self, keywords1, response1, keywords2, response2):
        if keywords1 or keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / max(len(keywords1.union(keywords2)), 1)
        else:
            keyword_similarity = 0.0
        if response1:
            response_to_keyword = len(response1.intersection(keywords2)) / max(len(response1), 1)
        else:
            response_to_keyword = 0.0
        if response1 or response2:
            concept_similarity = len(response1.intersection(response2)) / max(len(response1.union(response2)), 1)
        else:
            concept_similarity = 0.0
        return (0.4 * keyword_similarity + 0.4 * response_to_keyword + 0.2 * concept_similarity)

    def process_query(self, query):
        query_tokens = set(self._tokenize(query))
        cache_key = frozenset(query_tokens)
        if cache_key in getattr(self, "response_cache", {}):
            return self.response_cache[cache_key]
        direct_matches = self.match_rules(query_tokens)
        if not direct_matches:
            return ["No symbolic match found."]
        responses = []
        visited = set()
        context = {}
        for ratio, rule in direct_matches:
            rule_id = id(rule)
            if rule_id in visited:
                continue
            visited.add(rule_id)
            response_list = self._process_rule(rule, context)
            if response_list:
                responses.extend(response_list)
            if self.max_hops > 0:
                chain_responses = self._chain_rules_with_context(rule, context, visited, depth=1)
                responses.extend(chain_responses)
        final_responses = self._filter_responses(responses)
        # Fallback: if no chain produced a valid response, return the best direct match.
        if not final_responses and direct_matches:
            best_rule = direct_matches[0][1]
            final_responses = self._process_rule(best_rule, context)
        if not hasattr(self, "response_cache"):
            self.response_cache = {}
        self.response_cache[cache_key] = final_responses
        return final_responses

    def _process_rule(self, rule, context):
        response = rule["response"]
        if "subject" in context:
            response = response.replace("it", context["subject"]).replace("this", context["subject"])
        tokens = self._tokenize(response)
        if tokens:
            # Update context with the first token as subject (simplistic)
            context["subject"] = tokens[0]
        return [response]

    def _chain_rules_with_context(self, current_rule, context, visited, depth):
        if depth >= self.max_hops:
            return []
        responses = []
        next_rules = self._find_related_rules(current_rule, context)
        for rule in next_rules:
            rule_id = id(rule)
            if rule_id not in visited:
                visited.add(rule_id)
                response_list = self._process_rule(rule, context)
                if response_list:
                    responses.extend(response_list)
                chain_responses = self._chain_rules_with_context(rule, context, visited, depth + 1)
                responses.extend(chain_responses)
        return responses

    def _find_related_rules(self, current_rule, context):
        response_tokens = set(current_rule["response_tokens"])
        if "subject" in context and context["subject"]:
            subject_tokens = set(self._tokenize(context["subject"]))
            response_tokens.update(subject_tokens)
        related = []
        for rule in self.rules:
            if id(rule) == id(current_rule):
                continue
            if response_tokens.intersection(set(rule.get("keywords", []))):
                related.append(rule)
        return related

    def _filter_responses(self, responses):
        seen = set()
        unique = []
        for resp in responses:
            if resp not in seen:
                seen.add(resp)
                unique.append(resp)
        unique.sort(key=len, reverse=True)
        return unique

    def match_rules(self, tokens):
        matching = []
        for rule in self.rules:
            rule_tokens = set(rule.get("keywords", []))
            if rule_tokens:
                ratio = len(tokens.intersection(rule_tokens)) / len(rule_tokens)
            else:
                ratio = 0.0
            if ratio >= self.match_threshold:
                matching.append((ratio, rule))
        matching.sort(key=lambda x: x[0], reverse=True)
        return matching

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
