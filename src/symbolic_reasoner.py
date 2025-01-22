import json

class SymbolicReasoner:
    def __init__(self, rules_file):
        with open(rules_file, 'r') as file:
            self.rules = json.load(file)

    def process_query(self, query):
        results = []
        for rule in self.rules:
            if all(word in query for word in rule['keywords']):
                results.append(rule['response'])
        return results if results else ["No symbolic match found."]
