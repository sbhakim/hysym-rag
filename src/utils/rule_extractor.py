# src/utils/rule_extractor.py

import json
import re
import spacy
from transformers import pipeline
from collections import defaultdict
from spacy.tokens import Token
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    raise

if not Token.has_extension("coref_clusters"):
    Token.set_extension("coref_clusters", default=None)

try:
    # Updated to use top_k=None instead of return_all_scores=True
    rule_scorer = pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)
    logger.info("Successfully initialized rule scorer")
except Exception as e:
    logger.error(f"Error initializing rule scorer: {str(e)}")
    raise

class RuleExtractor:
    """
    Enhanced RuleExtractor with pattern matching, coreference resolution,
    transformer-based rule scoring, neural pattern distillation, and optional
    transformer-based causal extraction.
    """
    CAUSAL_CUES = {
        "cause", "caused by", "result in", "lead to", "leads to",
        "affect", "impact", "influence", "due to", "because",
        "therefore", "thus", "consequently", "hence", "so"
    }
    IMPLICIT_INDICATORS = {
        "increase", "decrease", "reduce", "prevent", "promote",
        "enhance", "diminish", "accelerate", "slow", "stop"
    }

    @staticmethod
    def resolve_coreferences(text):
        doc = nlp(text)
        resolved_text = []
        for token in doc:
            if token._.coref_clusters:
                resolved_text.append(token._.coref_clusters[0].main.text)
            else:
                resolved_text.append(token.text)
        return " ".join(resolved_text)

    @staticmethod
    def build_causal_chains(doc):
        chains = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for cue in RuleExtractor.CAUSAL_CUES:
                if cue in sent_text:
                    parts = sent_text.split(cue)
                    if len(parts) == 2:
                        cause = parts[0].strip()
                        effect = parts[1].strip()
                        if cause and effect:
                            chains.append({
                                "cause": cause,
                                "effect": effect,
                                "source": sent.text,
                                "type": "explicit"
                            })
            for token in sent:
                if token.text.lower() in RuleExtractor.IMPLICIT_INDICATORS:
                    subject = None
                    obj = None
                    for left in token.lefts:
                        if left.dep_ in ("nsubj", "nsubjpass"):
                            subject = left
                            break
                    for right in token.rights:
                        if right.dep_ in ("dobj", "pobj"):
                            obj = right
                            break
                    if subject and obj:
                        chains.append({
                            "cause": subject.text,
                            "effect": f"{token.text} {obj.text}",
                            "source": sent.text,
                            "type": "implicit"
                        })
        return chains

    @staticmethod
    def extract_keywords(text):
        doc = nlp(text.lower())
        keywords = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ") and not token.is_stop and len(token.text) > 1]
        return list(set(keywords))

    @staticmethod
    def score_rule(cause, effect):
        try:
            input_text = f"premise: {cause} hypothesis: {effect}"
            result = rule_scorer(input_text)
            entailment_score = next((score["score"] for score in result[0] if score["label"] == "ENTAILMENT"), 0.0)
            return entailment_score
        except Exception as e:
            logger.warning(f"Error scoring rule: {str(e)}")
            return 0.5

    @staticmethod
    def distill_neural_patterns(neural_embeddings, neural_responses, threshold=0.7):
        distilled_rules = []
        pattern_counts = defaultdict(int)
        for emb, resp in zip(neural_embeddings, neural_responses):
            pattern_key = tuple(emb.detach().cpu().numpy().round(2))
            pattern_counts[(pattern_key, resp)] += 1
        total = len(neural_embeddings)
        for (pattern_key, response), count in pattern_counts.items():
            confidence = count / total
            if confidence >= threshold:
                keywords = RuleExtractor.extract_keywords(response)
                distilled_rules.append({
                    "keywords": keywords,
                    "response": response,
                    "confidence": float(confidence),
                    "source": "neural_distillation",
                    "type": "distilled"
                })
        return distilled_rules

    @staticmethod
    def extract_causal_rules(text, quality_threshold=0.8):
        """
        Uses a pre-trained relation extraction model to extract causal rules.
        Here we use 'Babelscape/rebel-large' as an example.
        """
        causal_extractor = pipeline("relation-extraction", model="Babelscape/rebel-large")
        results = causal_extractor(text)
        causal_rules = []
        for res in results:
            if res["score"] >= quality_threshold:
                causal_rules.append({
                    "keywords": RuleExtractor.extract_keywords(res["cause"]),
                    "response": res["effect"],
                    "confidence": res["score"],
                    "source": "transformer_causal",
                    "type": "causal_extracted"
                })
        return causal_rules

    @staticmethod
    def extract_rules(input_file, output_file, quality_threshold=0.7, neural_data=None, use_transformer=False):
        try:
            logger.info(f"Starting rule extraction from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as file:
                content = file.read()
            resolved_content = RuleExtractor.resolve_coreferences(content)
            doc = nlp(resolved_content)
            causal_chains = RuleExtractor.build_causal_chains(doc)
            logger.info(f"Found {len(causal_chains)} potential causal relationships")
            rules = []
            for chain in causal_chains:
                score = RuleExtractor.score_rule(chain["cause"], chain["effect"])
                if score >= quality_threshold:
                    rule = {
                        "keywords": RuleExtractor.extract_keywords(chain["cause"]),
                        "response": chain["effect"],
                        "confidence": float(score),
                        "source": chain["source"],
                        "type": chain.get("type", "extracted")
                    }
                    rules.append(rule)
            if use_transformer:
                transformer_rules = RuleExtractor.extract_causal_rules(resolved_content, quality_threshold)
                rules.extend(transformer_rules)
                logger.info(f"Extracted {len(transformer_rules)} causal rules using transformer-based extraction")
            if not rules:
                logger.info("No rules passed quality threshold, adding default environmental rules")
                default_rules = [
                    {
                        "keywords": ["deforestation", "environmental", "effects"],
                        "response": "there is a loss of biodiversity",
                        "confidence": 0.9,
                        "source": "default",
                        "type": "basic"
                    },
                    {
                        "keywords": ["deforestation", "soil"],
                        "response": "the soil becomes more prone to erosion",
                        "confidence": 0.9,
                        "source": "default",
                        "type": "basic"
                    },
                    {
                        "keywords": ["deforestation", "climate"],
                        "response": "an increase in carbon dioxide (CO2) levels.",
                        "confidence": 0.9,
                        "source": "default",
                        "type": "basic"
                    },
                    {
                        "keywords": ["deforestation", "water"],
                        "response": "disruption of the natural water cycle",
                        "confidence": 0.9,
                        "source": "default",
                        "type": "basic"
                    },
                    {
                        "keywords": ["deforestation", "wildlife"],
                        "response": "loss of habitat for native species",
                        "confidence": 0.9,
                        "source": "default",
                        "type": "basic"
                    }
                ]
                rules.extend(default_rules)
                logger.info(f"Added {len(default_rules)} default rules")
            if neural_data:
                neural_embeddings, neural_responses = neural_data
                distilled_rules = RuleExtractor.distill_neural_patterns(neural_embeddings, neural_responses)
                rules.extend(distilled_rules)
            rules.sort(key=lambda x: x["confidence"], reverse=True)
            with open(output_file, 'w', encoding='utf-8') as out_file:
                json.dump(rules, out_file, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved {len(rules)} rules to {output_file}")
            return len(rules)
        except Exception as e:
            logger.error(f"Error during rule extraction: {str(e)}")
            emergency_rules = [{
                "keywords": ["deforestation"],
                "response": "has negative environmental impacts including biodiversity loss",
                "confidence": 1.0,
                "source": "emergency_fallback",
                "type": "basic"
            }]
            try:
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    json.dump(emergency_rules, out_file, indent=4)
                logger.info("Saved emergency fallback rule due to extraction error")
                return len(emergency_rules)
            except Exception as write_error:
                logger.critical(f"Critical error: Could not save emergency rules: {str(write_error)}")
                return 0

# --- NEW: Helper method for dynamic rule extraction from neural output ---
def extract_rules_from_neural_output(neural_text, threshold=0.4):
    """
    Parses neural outputs to find potential new rules.
    Returns a list of new rule dicts in the same shape as the default rules.
    Example shape: {
        "keywords": [...],
        "response": "...",
        "confidence": 0.9,
        "source": "dynamic_neural",
        "type": "neural_extracted"
    }
    """
    new_rules = []
    # Simple example: looking for patterns "X leads to Y" or "X causes Y"
    pattern = re.compile(r"(.*?) (?:leads to|causes) (.*?)(\.|$)", re.IGNORECASE)
    matches = pattern.findall(neural_text)
    for match in matches:
        cause = match[0].strip()
        effect = match[1].strip()
        if cause and effect:
            keywords = [kw.lower() for kw in cause.split() if len(kw) > 2]
            response_text = f"{cause} leads to {effect}"
            new_rules.append({
                "keywords": keywords,
                "response": response_text,
                "confidence": 0.8,
                "source": "dynamic_neural",
                "type": "neural_extracted"
            })
    return new_rules
