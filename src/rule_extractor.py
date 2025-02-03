# src/rule_extractor.py
import json
import re
import spacy

class RuleExtractor:
    """
    Advanced RuleExtractor that uses both spaCy for sentence segmentation
    and a simplified OpenIE-like approach to detect cause-effect relations.

    The pipeline:
    1. Split text into sentences with spaCy.
    2. For each sentence, attempt a basic subject-relation-object parse.
    3. If the relation or sentence contains causal cues, treat it as a potential rule.
    4. Store cause/effect tokens as 'keywords' and the effect as 'response'.
    """

    CAUSAL_CUES = {"cause", "causes", "lead", "leads", "led", "results", "due to", "because"}

    @staticmethod
    def extract_rules(input_file, output_file, use_spacy=True):
        """
        Extract rules from the input file and save them to JSON.
        Incorporates a basic OpenIE-like approach to identify cause/effect pairs.
        """
        with open(input_file, 'r') as file:
            content = file.read()

        if use_spacy:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(content)
            sentences = list(doc.sents)
        else:
            # Fallback: naive sentence splitting (not recommended for real usage).
            sentences = re.split(r'[.?!]\s*', content)

        rules = []
        for sent in sentences:
            sent_text = sent.text.strip() if use_spacy else sent.strip()
            if not sent_text:
                continue

            # 1. Look for any known cue words in the sentence
            lower_sent = sent_text.lower()
            if any(cue in lower_sent for cue in RuleExtractor.CAUSAL_CUES):
                # 2. Perform a simplified triple extraction
                sro_triple = RuleExtractor.extract_sro_triplet(sent_text)

                if sro_triple:
                    subject_tokens, relation_tokens, object_tokens = sro_triple

                    # If the relation includes any causal cue, treat this as a cause->effect rule
                    if any(cue in " ".join(relation_tokens).lower() for cue in RuleExtractor.CAUSAL_CUES):
                        # The 'keywords' here are from the subject (condition),
                        # and the 'response' can be the object (consequence).
                        rule_keywords = list(set([tok.lower() for tok in subject_tokens if tok]))
                        rule_response = " ".join(object_tokens).strip()

                        # If we have a valid rule, store it
                        if rule_keywords and rule_response:
                            rules.append({
                                "keywords": rule_keywords,
                                "response": rule_response
                            })

        # If no advanced cause-effect found, fallback to spaCy's if-then approach as well
        additional_rules = RuleExtractor.extract_if_then(content, use_spacy)
        rules.extend(additional_rules)

        if rules:
            print(f"Extracted {len(rules)} advanced rules.")
        else:
            print("Warning: No rules were extracted with the advanced pipeline.")

        # Write out to JSON
        with open(output_file, 'w') as out_file:
            json.dump(rules, out_file, indent=4)

    @staticmethod
    def extract_sro_triplet(sentence):
        """
        Extremely simplified subject-relation-object 'extraction'.
        For real usage, consider using AllenNLP OpenIE or another library.
        """
        # Basic approach: split on ' cause ' or ' leads to ', etc., as a naive sign of relation
        # In practice, you'd do dependency parsing or a real OpenIE pipeline.
        lower_sent = sentence.lower()
        # We'll scan for a small set of patterns
        patterns = [
            r"(.*)\s+(cause[s]?|lead[s]? to|result[s]? in)\s+(.*)",
            r"(.*)\s+because\s+(.*)"
        ]
        for pattern in patterns:
            match = re.search(pattern, lower_sent)
            if match:
                subj = match.group(1).split()
                rel = match.group(2).split()
                obj = match.group(3).split()
                return (subj, rel, obj)

        # If no match, return None
        return None

    @staticmethod
    def extract_if_then(content, use_spacy=True):
        """
        Fallback extraction using the if-then approach from the existing code.
        """
        rules = []
        if use_spacy:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(content)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                # Look for sentences with at least one cue word
                if any(cue in sent_text.lower() for cue in ["if", "when", "whenever", "then", "leads to"]):
                    parts = re.split(r'\bif\b|\bthen\b', sent_text, flags=re.IGNORECASE)
                    if len(parts) >= 3:
                        condition = parts[1].strip()
                        consequence = parts[2].strip().rstrip(".?!")
                        cond_doc = nlp(condition)
                        keywords = [token.lemma_.lower() for token in cond_doc if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")]
                        keywords = list(set(keywords))
                        if keywords and consequence:
                            rules.append({"keywords": keywords, "response": consequence})
        else:
            pattern = r"If\s+(.*?)\s+then\s+(.*?)[.?!]"
            matches = re.findall(pattern, content, re.IGNORECASE)
            for condition, response in matches:
                keywords = re.findall(r'\b\w+\b', condition.lower())
                keywords = list(set(keywords))
                if keywords and response.strip():
                    rules.append({"keywords": keywords, "response": response.strip()})

        if rules:
            print(f"Extracted {len(rules)} fallback if-then rules.")
        return rules
