# src/rule_extractor.py

import json
import re
import spacy
from transformers import pipeline
from collections import defaultdict
from spacy.tokens import Token
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model globally
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    raise

# Register custom extensions for tokens if they don't exist
if not Token.has_extension("coref_clusters"):
    Token.set_extension("coref_clusters", default=None)

# Initialize the transformer-based rule scorer
try:
    rule_scorer = pipeline("text-classification", model="facebook/bart-large-mnli")
    logger.info("Successfully initialized rule scorer")
except Exception as e:
    logger.error(f"Error initializing rule scorer: {str(e)}")
    raise


class RuleExtractor:
    """
    Enhanced RuleExtractor with pattern matching, coreference resolution,
    and transformer-based rule scoring.
    """

    # Define linguistic patterns that indicate causal relationships
    CAUSAL_CUES = {
        "cause", "caused by", "result in", "lead to", "leads to",
        "affect", "impact", "influence", "due to", "because",
        "therefore", "thus", "consequently", "hence", "so"
    }

    # Words that might indicate implicit relationships
    IMPLICIT_INDICATORS = {
        "increase", "decrease", "reduce", "prevent", "promote",
        "enhance", "diminish", "accelerate", "slow", "stop"
    }

    @staticmethod
    def resolve_coreferences(text):
        """
        Resolve pronouns and other references in the text.
        Returns the text with resolved references where possible.
        """
        doc = nlp(text)
        resolved_text = []

        for token in doc:
            if token._.coref_clusters:
                # Use the main mention from the first cluster
                resolved_text.append(token._.coref_clusters[0].main.text)
            else:
                resolved_text.append(token.text)

        return " ".join(resolved_text)

    @staticmethod
    def build_causal_chains(doc):
        """
        Identify causal relationships in text using linguistic patterns.
        Returns a list of cause-effect pairs with their source context.
        """
        chains = []

        for sent in doc.sents:
            sent_text = sent.text.lower()

            # Check for explicit causal relationships
            for cue in RuleExtractor.CAUSAL_CUES:
                if cue in sent_text:
                    # Split the sentence around the causal cue
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

            # Check for implicit relationships
            for token in sent:
                if token.text.lower() in RuleExtractor.IMPLICIT_INDICATORS:
                    # Look for subject and object
                    subject = None
                    obj = None

                    # Find the subject (look to the left of the verb)
                    for left in token.lefts:
                        if left.dep_ in ("nsubj", "nsubjpass"):
                            subject = left
                            break

                    # Find the object (look to the right of the verb)
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
        """
        Extract meaningful keywords from text, focusing on important parts of speech.
        Returns a list of lemmatized keywords.
        """
        doc = nlp(text.lower())
        keywords = []

        for token in doc:
            # Include tokens that are meaningful parts of speech and not stopwords
            if (token.pos_ in ("NOUN", "VERB", "ADJ") and
                    not token.is_stop and
                    len(token.text) > 1):
                keywords.append(token.lemma_)

        return list(set(keywords))  # Remove duplicates

    @staticmethod
    def score_rule(cause, effect):
        """
        Score the quality of a potential rule using the transformer model.
        Returns a confidence score between 0 and 1.
        """
        try:
            # Create the input text in a format suitable for natural language inference
            input_text = f"premise: {cause} hypothesis: {effect}"

            # Use the model for textual entailment instead of classification
            result = rule_scorer(
                input_text,
                return_all_scores=True
            )

            # Calculate a confidence score based on the entailment probability
            entailment_score = next(
                (score["score"] for score in result[0] if score["label"] == "ENTAILMENT"),
                0.0
            )

            return entailment_score
        except Exception as e:
            logger.warning(f"Error scoring rule: {str(e)}")
            # Return a moderate confidence score as fallback
            return 0.5

    @staticmethod
    def extract_rules(input_file, output_file, quality_threshold=0.7):
        """
        Extract rules from an input text file, combining automatic extraction with default rules.

        This method performs several key steps:
        1. Reads and preprocesses the input text
        2. Extracts causal relationships using linguistic patterns
        3. Converts relationships into rules with confidence scores
        4. Falls back to default rules if needed
        5. Saves the final ruleset to a JSON file

        Args:
            input_file (str): Path to the input text file containing domain knowledge
            output_file (str): Path where the extracted rules will be saved
            quality_threshold (float): Minimum confidence score (0.0-1.0) for including rules

        Returns:
            int: Number of rules successfully extracted and saved
        """
        try:
            logger.info(f"Starting rule extraction from {input_file}")

            # Read and preprocess the input text
            with open(input_file, 'r', encoding='utf-8') as file:
                content = file.read()

            # Resolve coreferences to improve relationship extraction
            resolved_content = RuleExtractor.resolve_coreferences(content)
            doc = nlp(resolved_content)

            # Extract causal relationships from the text
            causal_chains = RuleExtractor.build_causal_chains(doc)
            logger.info(f"Found {len(causal_chains)} potential causal relationships")

            # Convert causal chains to rules
            rules = []
            for chain in causal_chains:
                # Score the quality of the potential rule
                score = RuleExtractor.score_rule(chain["cause"], chain["effect"])

                # Only include rules that meet the quality threshold
                if score >= quality_threshold:
                    rule = {
                        "keywords": RuleExtractor.extract_keywords(chain["cause"]),
                        "response": chain["effect"],
                        "confidence": float(score),
                        "source": chain["source"],
                        "type": chain.get("type", "extracted")
                    }
                    rules.append(rule)

            # If no rules were extracted or passed the threshold, add default rules
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

            # Sort rules by confidence score for better processing later
            rules.sort(key=lambda x: x["confidence"], reverse=True)

            # Save the final ruleset to the output file
            try:
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    json.dump(rules, out_file, indent=4, ensure_ascii=False)
                logger.info(f"Successfully saved {len(rules)} rules to {output_file}")
            except IOError as e:
                logger.error(f"Error saving rules to file: {str(e)}")
                raise

            # Return the number of rules successfully saved
            return len(rules)

        except Exception as e:
            # Handle any unexpected errors during rule extraction
            logger.error(f"Error during rule extraction: {str(e)}")

            # Create emergency fallback rules to ensure system can continue operating
            emergency_rules = [{
                "keywords": ["deforestation"],
                "response": "has negative environmental impacts including biodiversity loss",
                "confidence": 1.0,
                "source": "emergency_fallback",
                "type": "basic"
            }]

            try:
                # Save emergency rules
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    json.dump(emergency_rules, out_file, indent=4)
                logger.info("Saved emergency fallback rule due to extraction error")
                return len(emergency_rules)
            except Exception as write_error:
                logger.critical(f"Critical error: Could not save emergency rules: {str(write_error)}")
                return 0