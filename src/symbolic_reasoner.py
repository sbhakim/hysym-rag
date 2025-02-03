# src/symbolic_reasoner.py
import json
import re


class SymbolicReasoner:
    def __init__(self, rules_file, match_threshold=0.2):
        """
        Initialize SymbolicReasoner with rules from a file.
        match_threshold: minimum ratio of matching keywords required to consider a rule.
        """
        with open(rules_file, 'r') as file:
            self.rules = json.load(file)
        self.validate_rules()
        self.build_rule_index()
        self.match_threshold = match_threshold

    def validate_rules(self):
        """
        Validate rules to ensure they have the required structure.
        """
        for rule in self.rules:
            if not isinstance(rule, dict) or 'keywords' not in rule or 'response' not in rule:
                raise ValueError("Invalid rule structure. Each rule must contain 'keywords' and 'response'.")
        print("Rules validated successfully.")

    def build_rule_index(self):
        """
        Build an inverted index and precompute keyword sets for each rule.
        """
        self.index = {}
        for rule in self.rules:
            # Precompute a set of keywords for each rule
            rule["keywords_set"] = set(rule["keywords"])
            for keyword in rule["keywords_set"]:
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(rule)
        print("Rule index built successfully.")

    def process_query(self, query):
        """
        Process a query by computing the intersection between the query tokens
        and each rule's keywords. Returns responses for rules that exceed the match threshold.
        """
        # Tokenize and normalize the query
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        matching_rules = []
        for rule in self.rules:
            intersection = query_tokens.intersection(rule["keywords_set"])
            if rule["keywords_set"]:
                ratio = len(intersection) / len(rule["keywords_set"])
            else:
                ratio = 0
            if ratio >= self.match_threshold:
                matching_rules.append((ratio, rule))

        if matching_rules:
            # Sort rules by best match ratio descending
            matching_rules.sort(key=lambda x: x[0], reverse=True)
            responses = [rule["response"] for _, rule in matching_rules]
            print(f"Matching rules found: {len(responses)} (best match ratio: {matching_rules[0][0]:.4f})")
            return responses
        else:
            return ["No symbolic match found."]
