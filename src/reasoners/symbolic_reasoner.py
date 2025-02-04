# src/reasoners/symbolic_reasoner.py
import json
import re
import spacy
from sentence_transformers import SentenceTransformer, util

class SymbolicReasoner:
    """
    An enhanced symbolic reasoner with multi-hop chaining, context-awareness, and caching.
    """

    def __init__(self, rules_file, match_threshold=0.2, max_hops=2):
        """
        Initialize SymbolicReasoner with rules from a file.

        Args:
            rules_file: Path to the JSON file containing symbolic rules.
            match_threshold: Minimum ratio of matching keywords required to consider a rule.
            max_hops: Maximum depth for multi-hop chaining of rule consequences.
        """
        with open(rules_file, 'r') as file:
            self.rules = json.load(file)

        self.validate_rules()
        self.build_rule_index()

        self.match_threshold = match_threshold
        self.max_hops = max_hops

        # Additional enhancements
        self.response_cache = {}       # Cache final aggregated responses
        self.similarity_threshold = 0.6  # For future expansions if you want to filter out low-similarity responses

        # Load spaCy for context-based processing in rules
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize SentenceTransformer for encoding queries
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
        Also, compute and store an embedding for each rule's keywords.
        """
        self.index = {}
        for rule in self.rules:
            # Precompute a set of keywords for each rule
            rule["keywords_set"] = set(rule["keywords"])
            # Store a tokenized version of the response for potential chaining
            rule["response_tokens"] = set(re.findall(r'\b\w+\b', rule["response"].lower()))
            # Compute an embedding for the concatenated keywords (for matching)
            keywords_text = " ".join(rule.get("keywords", []))
            rule["embedding"] = self.embedder.encode(keywords_text, convert_to_tensor=True)

            # Build an inverted index for the rule's keywords
            for keyword in rule["keywords_set"]:
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(rule)
        print("Rule index built successfully.")

    def encode(self, query):
        """
        Encode the query into an embedding tensor using the SentenceTransformer.
        """
        return self.embedder.encode(query, convert_to_tensor=True)

    def process_query(self, query):
        """
        Enhanced query processing:
          1. Encodes the query.
          2. Finds direct matches using cosine similarity with rule embeddings.
          3. Returns the best matching rule's response.
        """
        query_embedding = self.encode(query)
        matches = []
        # Compare query embedding with each rule's embedding
        for rule_id, rule in enumerate(self.rules):
            sim = util.cos_sim(query_embedding, rule["embedding"]).item()
            if sim >= self.match_threshold:
                matches.append((rule_id, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        if not matches:
            return ["No symbolic match found."]
        # Return the response from the best matching rule
        best_rule = self.rules[matches[0][0]]
        print(f"Symbolic match found (score={matches[0][1]:.2f}).")
        return [best_rule["response"]]

    def _process_rule(self, rule, context):
        """
        Process a single rule in a context-aware manner.
        - Replaces references to 'it'/'this' if we already have a subject in context.
        - Updates context with new subject(s) from the response.
        """
        response = rule["response"]

        # Replace pronouns if a subject is known in context
        if "subject" in context:
            response = response.replace("it", context["subject"]).replace("this", context["subject"])

        # Parse the response for a subject to update the context
        doc = self.nlp(response)
        subjects = [tok.text for tok in doc if tok.dep_ == "nsubj"]
        if subjects:
            context["subject"] = subjects[0]
        return [response]

    def _chain_rules_with_context(self, current_rule, context, visited, depth):
        """
        Recursively chain to other rules using context and the current rule's response tokens.
        Depth-limited by self.max_hops.
        """
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
        """
        Find rules that overlap with the current rule's response tokens or context tokens.
        """
        response_tokens = set(current_rule["response_tokens"])
        if "subject" in context and context["subject"]:
            subject_tokens = set(re.findall(r'\b\w+\b', context["subject"].lower()))
            response_tokens.update(subject_tokens)

        related = []
        for rule in self.rules:
            if id(rule) == id(current_rule):
                continue
            if response_tokens.intersection(rule["keywords_set"]):
                related.append(rule)
        return related

    def _filter_responses(self, responses):
        """
        Remove duplicate responses and sort by length (as a proxy for detail).
        """
        seen = set()
        unique_responses = []
        for resp in responses:
            if resp not in seen:
                seen.add(resp)
                unique_responses.append(resp)
        unique_responses.sort(key=len, reverse=True)
        return unique_responses

    def match_rules(self, tokens):
        """
        Return all rules that match the given token set above the match threshold.
        """
        matching_rules = []
        for rule in self.rules:
            intersection = tokens.intersection(rule["keywords_set"])
            ratio = len(intersection) / len(rule["keywords_set"]) if rule["keywords_set"] else 0.0
            if ratio >= self.match_threshold:
                matching_rules.append((ratio, rule))
        matching_rules.sort(key=lambda x: x[0], reverse=True)
        return matching_rules
