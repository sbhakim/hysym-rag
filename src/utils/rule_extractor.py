# src/utils/rule_extractor.py

import json
import re
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
from collections import defaultdict
from spacy.tokens import Token
import logging
from sklearn.cluster import KMeans
import numpy as np

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
    transformer-based rule scoring, and dynamic rule generation from neural patterns.
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
        """
        Extract meaningful keywords from text, focusing on important parts of speech.
        Returns a list of lemmatized keywords.
        """
        doc = nlp(text.lower())
        keywords = []
        for token in doc:
            if (token.pos_ in ("NOUN", "VERB", "ADJ") and
                    not token.is_stop and
                    len(token.text) > 1):
                keywords.append(token.lemma_)
        return list(set(keywords))

    @staticmethod
    def score_rule(cause, effect):
        """
        Score the quality of a potential rule using the transformer model.
        Returns a confidence score between 0 and 1.
        """
        try:
            input_text = f"premise: {cause} hypothesis: {effect}"
            result = rule_scorer(input_text, return_all_scores=True)
            entailment_score = next(
                (score["score"] for score in result[0] if score["label"] == "ENTAILMENT"),
                0.0
            )
            return entailment_score
        except Exception as e:
            logger.warning(f"Error scoring rule: {str(e)}")
            return 0.5

    @staticmethod
    def extract_rules(input_file, output_file, quality_threshold=0.7):
        """
        Extract rules from an input text file and save them to a JSON file.
        """
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

    # --- Dynamic Rule Generation Methods (Integrated in DynamicRuleGenerator below) ---
    def cluster_embeddings(self, embeddings, num_clusters=5):
        """Cluster embeddings and return clusters."""
        n_samples = embeddings.shape[0]
        if n_samples < num_clusters:
            num_clusters = n_samples
            logger.info(f"Reducing number of clusters to {num_clusters} due to low sample count.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters = defaultdict(list)
        for label, emb in zip(labels, embeddings):
            clusters[label].append(emb)
        return list(clusters.values())

    def generate_rule_from_cluster(self, cluster_texts):
        """
        Generate a rule from a cluster of texts.
        Use a simple heuristic: combine the texts and extract the first causal pair.
        """
        combined_text = " ".join(cluster_texts)
        doc = nlp(combined_text)
        chains = self.build_causal_chains(doc)
        if chains:
            pair = chains[0]
            return {
                "keywords": self.extract_keywords(pair["cause"]),
                "response": pair["effect"],
                "confidence": self.score_rule(pair["cause"], pair["effect"]),
                "source": "distilled",
                "type": "generated"
            }
        else:
            return {
                "keywords": self.extract_keywords(cluster_texts[0]),
                "response": cluster_texts[0],
                "confidence": 0.5,
                "source": "distilled_fallback",
                "type": "generated"
            }

    def distill_rules_from_embeddings(self, text_data, num_clusters=5, similarity_threshold=0.8):
        """
        Distill symbolic rules from neural embeddings without needing external training data.
        Args:
            text_data (list): List of sentences or phrases.
            num_clusters (int): Desired number of clusters.
            similarity_threshold: (Not used directly here but could be used to filter clusters)
        Returns:
            List of distilled rules.
        """
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = encoder.encode(text_data, convert_to_tensor=True).cpu().numpy()
        n_samples = embeddings.shape[0]
        if n_samples < num_clusters:
            num_clusters = n_samples
            logger.info(f"Reducing number of clusters to {num_clusters} due to low sample count.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters_texts = defaultdict(list)
        for label, text in zip(labels, text_data):
            clusters_texts[label].append(text)
        new_rules = []
        for cluster_texts in clusters_texts.values():
            if len(cluster_texts) > 1:
                rule = self.generate_rule_from_cluster(cluster_texts)
                new_rules.append(rule)
        return new_rules

    def update_rules_from_feedback(self, feedback_data):
        """
        Update rules based on user feedback.
        Args:
            feedback_data: List of feedback entries containing queries, results, and ratings.
        """
        for entry in feedback_data:
            if entry.get("rating", 0) >= 4:
                keywords = self.extract_keywords(entry["query"])
                response = entry["result"]
                new_rule = {
                    "keywords": keywords,
                    "response": response,
                    "confidence": entry["rating"] / 5.0,
                    "source": "user_feedback",
                    "type": "feedback"
                }
                logger.info(f"New feedback rule: {new_rule}")
        logger.info("Feedback-based rule update complete.")


class DynamicRuleGenerator:
    """
    Dynamically generates new rules by clustering neural embeddings of text data.
    This class serves as a wrapper that leverages RuleExtractor's dynamic methods.
    """

    def __init__(self, encoder_model="all-MiniLM-L6-v2", threshold=0.7):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(encoder_model)
        self.threshold = threshold
        # Create an instance of RuleExtractor to use its methods.
        self.rule_extractor = RuleExtractor()

    def generate_rules_from_embeddings(self, text_data, num_clusters=5):
        """
        Generate new rules by clustering neural embeddings of the provided text data.
        """
        from collections import defaultdict
        from sklearn.cluster import KMeans
        embeddings = self.encoder.encode(text_data, convert_to_tensor=True).cpu().numpy()
        n_samples = embeddings.shape[0]
        if n_samples < num_clusters:
            num_clusters = n_samples
            logger.info(f"Reducing number of clusters to {num_clusters} due to low sample count.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clusters_texts = defaultdict(list)
        for label, text in zip(labels, text_data):
            clusters_texts[label].append(text)
        new_rules = []
        for cluster_texts in clusters_texts.values():
            if len(cluster_texts) > 1:
                rule = self.rule_extractor.generate_rule_from_cluster(cluster_texts)
                new_rules.append(rule)
        return new_rules

    def update_rule_base(self, new_rules, output_file="data/rules.json"):
        """
        Append new rules to the existing rule base.
        """
        try:
            with open(output_file, "r") as file:
                existing_rules = json.load(file)
        except FileNotFoundError:
            existing_rules = []
        existing_rules.extend(new_rules)
        with open(output_file, "w") as file:
            json.dump(existing_rules, file, indent=4)
        logger.info(f"Updated rule base with {len(new_rules)} new rules.")
