import json

class RuleExtractor:
    @staticmethod
    def extract_rules(input_file, output_file):
        with open(input_file, 'r') as file:
            content = file.read()

        # Example rule extraction logic
        rules = [
            {
                "keywords": ["deforestation", "soil erosion"],
                "response": "Deforestation leads to soil erosion."
            },
            {
                "keywords": ["deforestation", "biodiversity"],
                "response": "Deforestation causes a loss of biodiversity."
            },
            {
                "keywords": ["deforestation", "climate change"],
                "response": "Deforestation contributes to climate change by increasing greenhouse gases."
            },
            {
                "keywords": ["deforestation", "water cycle"],
                "response": "Deforestation disrupts the water cycle, leading to droughts and floods."
            },
            {
                "keywords": ["deforestation", "indigenous communities"],
                "response": "Deforestation displaces indigenous communities and threatens their cultural heritage."
            }
        ]

        with open(output_file, 'w') as out_file:
            json.dump(rules, out_file, indent=4)
