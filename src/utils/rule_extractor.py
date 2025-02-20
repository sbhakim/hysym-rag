# src/utils/rule_extractor.py

import json
import re
import spacy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging
from transformers import pipeline
from spacy.tokens import Token
import torch
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    raise

# Ensure spaCy has custom extension for coreference
if not Token.has_extension("coref_clusters"):
    Token.set_extension("coref_clusters", default=None)

# Initialize BART for confidence scoring
try:
    rule_scorer = pipeline("text-classification",
                           model="facebook/bart-large-mnli",
                           top_k=None)
    logger.info("Successfully initialized rule scorer")
except Exception as e:
    logger.error(f"Error initializing rule scorer: {str(e)}")
    raise


class RuleExtractor:
    """
    Enhanced RuleExtractor for HotpotQA-style multi-hop reasoning.
    Supports sophisticated pattern matching and relation extraction.
    """

    def __init__(self):
        # Define comprehensive patterns for various types of relations
        self.relation_patterns = {
            'biographical': [
                r"([A-Z][a-zA-Z\s]+)\s*\(born\s+([^)]+)\)\s+is\s+an?\s+([A-Za-z\s]+)",
                r"([A-Z][a-zA-Z\s]+)\s+was\s+born\s+in\s+([A-Za-z\s,]+)\s+on\s+([A-Za-z0-9\s,]+)"
            ],
            'professional': [
                r"([A-Z][a-zA-Z\s]+)\s+(?:is|was)\s+(?:a|an)\s+([A-Za-z\s]+)",
                r"([A-Z][a-zA-Z\s]+)\s+worked\s+as\s+(?:a|an)\s+([A-Za-z\s]+)"
            ],
            'creation': [
                r"([A-Z][a-zA-Z0-9\s]+)\s+(?:directed|produced|wrote)\s+([A-Z][a-zA-Z0-9\s\"]+)",
                r"([A-Z][a-zA-Z0-9\s\"]+)\s+was\s+(?:directed|produced|written)\s+by\s+([A-Z][a-zA-Z\s]+)"
            ],
            'temporal': [
                r"(?:In|During|Around)\s+(\d{4}),\s+([A-Z][a-zA-Z\s]+)\s+([A-Za-z\s]+)",
                r"([A-Z][a-zA-Z\s]+)\s+(?:began|started|commenced)\s+([A-Za-z\s]+)\s+in\s+(\d{4})"
            ],
            'location': [
                r"([A-Z][a-zA-Z\s]+)\s+(?:is|was)\s+located\s+in\s+([A-Z][a-zA-Z\s,]+)",
                r"([A-Z][a-zA-Z\s]+)\s+(?:moved|relocated)\s+to\s+([A-Z][a-zA-Z\s,]+)"
            ]
        }

        # Initialize SentenceTransformer for semantic similarity
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Confidence thresholds for different relation types
        self.confidence_thresholds = {
            'biographical': 0.8,
            'professional': 0.75,
            'creation': 0.85,
            'temporal': 0.7,
            'location': 0.75
        }

    def extract_hotpot_facts(self, context_text: str, min_confidence: float = 0.7) -> List[Dict]:
        """
        Extract facts from HotpotQA context with enhanced pattern matching and validation.

        Args:
            context_text: The context text from HotpotQA
            min_confidence: Minimum confidence threshold for extracted rules

        Returns:
            List of extracted rules with confidence scores
        """
        extracted_rules = []

        # Process text with spaCy for enhanced NLP features
        doc = nlp(context_text)

        # Extract entities and their types
        entities = {ent.text: ent.label_ for ent in doc.ents}

        # Process each sentence for fact extraction
        for sent in doc.sents:
            # Skip short or incomplete sentences
            if len(sent.text.split()) < 4:
                continue

            # Extract facts using different pattern types
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sent.text)

                    for match in matches:
                        # Create rule from match
                        rule = self._create_rule(
                            match=match,
                            relation_type=relation_type,
                            sentence=sent.text,
                            entities=entities
                        )

                        # Validate and score the rule
                        if rule:
                            confidence = self._compute_rule_confidence(rule)
                            if confidence >= min_confidence:
                                rule['confidence'] = confidence
                                extracted_rules.append(rule)

        # Post-process rules to establish connections
        processed_rules = self._establish_rule_connections(extracted_rules)

        logger.info(f"Extracted {len(processed_rules)} rules from context")
        return processed_rules

    def _create_rule(self, match: re.Match, relation_type: str,
                     sentence: str, entities: Dict[str, str]) -> Optional[Dict]:
        """
        Create a structured rule from a regex match with enhanced metadata.
        """
        try:
            groups = match.groups()
            if not groups:
                return None

            # Basic rule structure
            rule = {
                'type': relation_type,
                'source_text': sentence,
                'groups': groups,
                'keywords': self._extract_keywords(sentence)
            }

            # Add relation-specific fields
            if relation_type == 'biographical':
                rule.update({
                    'subject': groups[0],
                    'birth_info': groups[1] if len(groups) > 1 else None,
                    'profession': groups[2] if len(groups) > 2 else None
                })
            elif relation_type == 'creation':
                rule.update({
                    'creator': groups[0],
                    'creation': groups[1],
                    'year': groups[2] if len(groups) > 2 else None
                })

            # Add entity information if available
            rule['entity_types'] = {
                entity: type_
                for entity, type_ in entities.items()
                if entity in sentence
            }

            return rule

        except Exception as e:
            logger.warning(f"Error creating rule: {str(e)}")
            return None

    def _compute_rule_confidence(self, rule: Dict) -> float:
        """
        Compute confidence score for a rule using multiple factors.
        """
        try:
            # Get base confidence from transformer model
            result = rule_scorer(rule['source_text'])
            base_confidence = next(
                (score['score'] for score in result[0]
                 if score['label'] == 'ENTAILMENT'),
                0.5
            )

            # Adjust confidence based on various factors
            adjustments = [
                # Entity presence adjustment
                0.1 if rule.get('entity_types') else -0.1,

                # Keyword richness adjustment
                min(0.1, len(rule['keywords']) * 0.02),

                # Relation type threshold
                self.confidence_thresholds.get(rule['type'], 0.0) - 0.5
            ]

            # Compute final confidence
            confidence = base_confidence + sum(adjustments)

            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.warning(f"Error computing confidence: {str(e)}")
            return 0.5

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from text using spaCy.
        """
        doc = nlp(text.lower())
        return [
            token.lemma_
            for token in doc
            if (token.pos_ in ('NOUN', 'VERB', 'ADJ') and
                not token.is_stop and
                len(token.text) > 2)
        ]

    def _establish_rule_connections(self, rules: List[Dict]) -> List[Dict]:
        """
        Establish connections between rules for multi-hop reasoning.
        """
        # Create entity index
        entity_index = defaultdict(list)
        for i, rule in enumerate(rules):
            for entity in rule.get('entity_types', {}):
                entity_index[entity].append(i)

        # Establish connections
        for rule in rules:
            connected_rules = set()
            for entity in rule.get('entity_types', {}):
                for connected_idx in entity_index[entity]:
                    connected_rules.add(connected_idx)

            rule['connected_rules'] = list(connected_rules)

            # Compute path relevance scores
            rule['path_scores'] = self._compute_path_scores(
                rule,
                [rules[idx] for idx in connected_rules]
            )

        return rules

    def _compute_path_scores(self,
                             source_rule: Dict,
                             connected_rules: List[Dict]) -> Dict[int, float]:
        """
        Compute relevance scores for connected reasoning paths.
        """
        path_scores = {}

        source_emb = self.encoder.encode(source_rule['source_text'])

        for i, target_rule in enumerate(connected_rules):
            target_emb = self.encoder.encode(target_rule['source_text'])
            similarity = util.cos_sim(source_emb, target_emb).item()

            # Adjust score based on relation types
            type_compatibility = 0.2 if source_rule['type'] == target_rule['type'] else 0.0

            # Compute final path score
            path_scores[i] = (similarity + type_compatibility) / 2

        return path_scores