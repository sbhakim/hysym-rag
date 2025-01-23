import json
import re

class RuleExtractor:
    """
    RuleExtractor class to extract symbolic reasoning rules from text data.
    """
    @staticmethod
    def extract_rules(input_file, output_file):
        with open(input_file, 'r') as file:
            content = file.read()

        # Basic rule extraction: Identify "if...then" patterns
        pattern = r"If\s+(.*?)\s+then\s+(.*?)[.?!]"
        matches = re.findall(pattern, content, re.IGNORECASE)

        rules = []
        for match in matches:
            condition, response = match
            keywords = re.findall(r'\b\w+\b', condition)
            rules.append({"keywords": keywords, "response": response.strip()})

        with open(output_file, 'w') as out_file:
            json.dump(rules, out_file, indent=4)
