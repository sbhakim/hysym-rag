# src/rule_extractor.py
import json
import re
import spacy

class RuleExtractor:
    """
    RuleExtractor class to extract symbolic reasoning rules from text data.
    You can choose to use a simple regex or spaCy for more robust extraction.
    """
    @staticmethod
    def extract_rules(input_file, output_file, use_spacy=True):
        with open(input_file, 'r') as file:
            content = file.read()

        rules = []
        if use_spacy:
            # Use spaCy for extraction
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(content)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                # Look for sentences with at least one cue word
                if any(cue in sent_text.lower() for cue in ["if", "when", "whenever", "then", "leads to"]):
                    # Try to split using "if" and "then"
                    parts = re.split(r'\bif\b|\bthen\b', sent_text, flags=re.IGNORECASE)
                    if len(parts) >= 3:
                        condition = parts[1].strip()
                        consequence = parts[2].strip().rstrip(".?!")
                        # Use spaCy to extract candidate keywords from the condition
                        cond_doc = nlp(condition)
                        keywords = [token.lemma_.lower() for token in cond_doc if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")]
                        keywords = list(set(keywords))
                        if keywords and consequence:
                            rules.append({"keywords": keywords, "response": consequence})
            print(f"Extracted {len(rules)} rules using spaCy.")
        else:
            # Simple regex-based extraction
            pattern = r"If\s+(.*?)\s+then\s+(.*?)[.?!]"
            matches = re.findall(pattern, content, re.IGNORECASE)
            for condition, response in matches:
                keywords = re.findall(r'\b\w+\b', condition.lower())
                keywords = list(set(keywords))
                if keywords and response.strip():
                    rules.append({"keywords": keywords, "response": response.strip()})
            print(f"Extracted {len(rules)} rules using regex.")

        if not rules:
            print("Warning: No rules were extracted. Check the input file or try using spaCy extraction.")
        with open(output_file, 'w') as out_file:
            json.dump(rules, out_file, indent=4)
