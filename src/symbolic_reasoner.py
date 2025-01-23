import json


class SymbolicReasoner:
    def __init__(self, rules_file):
        """
        Initialize SymbolicReasoner with rules from a file.
        """
        with open(rules_file, 'r') as file:
            self.rules = json.load(file)
        self.validate_rules()
        self.build_rule_index()

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
        Build an inverted index for efficient keyword-based rule matching.
        """
        self.index = {}
        for rule in self.rules:
            for keyword in rule['keywords']:
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(rule)
        print("Rule index built successfully.")

    def process_query(self, query):
        """
        Process a query by matching keywords to rules.
        """
        matching_rules = set()
        for keyword in self.index:
            if keyword in query:
                matching_rules.update(self.index[keyword])

        if matching_rules:
            responses = [rule['response'] for rule in matching_rules]
            print(f"Matching rules found: {len(responses)}")
            return responses
        else:
            return ["No symbolic match found."]
