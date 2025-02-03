# src/symbolic_reasoner.py
import json
import re
import spacy

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
            # Also store a tokenized version of the response to potentially chain further
            rule["response_tokens"] = set(re.findall(r'\b\w+\b', rule["response"].lower()))

            # Build an inverted index for the rule's keywords
            for keyword in rule["keywords_set"]:
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(rule)
        print("Rule index built successfully.")

    def process_query(self, query):
        """
        Enhanced query processing:
          1. Tokenizes the query and checks a response cache.
          2. Finds direct matches using self.match_rules().
          3. For each matching rule, calls _process_rule() and chains further
             via _chain_rules_with_context().
          4. Filters and deduplicates final responses, caches them, and returns.
        """
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        cache_key = frozenset(query_tokens)

        # Cache check
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        # Match relevant rules
        direct_matches = self.match_rules(query_tokens)
        if not direct_matches:
            return ["No symbolic match found."]

        responses = []
        visited = set()   # keep track of rule object IDs we've used to avoid loops
        context = {}      # track evolving context across multi-hop chaining

        # Process each matched rule
        for ratio, rule in direct_matches:
            rule_id = id(rule)
            if rule_id in visited:
                continue
            visited.add(rule_id)

            # Process this rule's response
            response_list = self._process_rule(rule, context)
            if response_list:
                responses.extend(response_list)

            # Chain further, if allowed
            if self.max_hops > 0:
                chain_responses = self._chain_rules_with_context(rule, context, visited, depth=1)
                responses.extend(chain_responses)

        # Remove duplicates, sort by relevance
        final_responses = self._filter_responses(responses)

        # Cache the final result
        self.response_cache[cache_key] = final_responses
        return final_responses

    def _process_rule(self, rule, context):
        """
        Process a single rule in a context-aware manner.
        - Replaces references to 'it'/'this' if we already have a subject in context.
        - Updates context with new subject(s) from the response.
        """
        response = rule["response"]

        # If we already have a known subject, replace 'it'/'this' with that subject
        if "subject" in context:
            response = response.replace("it", context["subject"]).replace("this", context["subject"])

        # Now parse the response to see if there's a new subject to track
        doc = self.nlp(response)
        subjects = [tok.text for tok in doc if tok.dep_ == "nsubj"]
        if subjects:
            # The first subject can become context for subsequent rules
            context["subject"] = subjects[0]

        return [response]

    def _chain_rules_with_context(self, current_rule, context, visited, depth):
        """
        Recursively chain to other rules using 'context' and the current rule's response tokens.
        Depth-limited by self.max_hops.
        """
        if depth >= self.max_hops:
            return []

        responses = []
        # Identify possible next rules by overlapping keywords or context tokens
        next_rules = self._find_related_rules(current_rule, context)

        for rule in next_rules:
            rule_id = id(rule)
            if rule_id not in visited:
                visited.add(rule_id)
                # Process the rule in a context-aware way
                response_list = self._process_rule(rule, context)
                if response_list:
                    responses.extend(response_list)

                # Continue chaining further
                chain_responses = self._chain_rules_with_context(rule, context, visited, depth + 1)
                responses.extend(chain_responses)

        return responses

    def _find_related_rules(self, current_rule, context):
        """
        Find rules that overlap with the current rule's response tokens or context tokens.
        This version merges context with the rule's tokens to find additional rules.
        """
        response_tokens = set(current_rule["response_tokens"])

        # If there's a subject in context, incorporate that into the tokens
        if "subject" in context and context["subject"]:
            subject_tokens = set(re.findall(r'\b\w+\b', context["subject"].lower()))
            response_tokens.update(subject_tokens)

        # Return rules whose keywords intersect with our current token set
        related = []
        for rule in self.rules:
            if id(rule) == id(current_rule):
                continue  # skip the same rule
            if response_tokens.intersection(rule["keywords_set"]):
                related.append(rule)

        return related

    def _filter_responses(self, responses):
        """
        Filter out duplicates, then sort by length (longer or more detailed first).
        You could also add more advanced filtering or ranking here if needed.
        """
        seen = set()
        unique_responses = []
        for resp in responses:
            if resp not in seen:
                seen.add(resp)
                unique_responses.append(resp)

        # Sort by length descending as a basic 'detail' proxy
        unique_responses.sort(key=len, reverse=True)
        return unique_responses

    def match_rules(self, tokens):
        """
        Return all rules that match the given token set above self.match_threshold.
        """
        matching_rules = []
        for rule in self.rules:
            intersection = tokens.intersection(rule["keywords_set"])
            if rule["keywords_set"]:
                ratio = len(intersection) / len(rule["keywords_set"])
            else:
                ratio = 0.0
            if ratio >= self.match_threshold:
                matching_rules.append((ratio, rule))

        # Sort by ratio descending so the most relevant come first
        matching_rules.sort(key=lambda x: x[0], reverse=True)
        return matching_rules
