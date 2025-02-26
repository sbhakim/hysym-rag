# src/utils/rule_extractor.py

import json
import re
import numpy as np
import spacy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging
from transformers import pipeline
from spacy.tokens import Token
import torch
from sentence_transformers import SentenceTransformer, util

# Import the DimensionalityManager for alignment
from src.utils.dimension_manager import DimensionalityManager

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
        self.relation_patterns_drop = {
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
            ],
            # NEW: Numeric facts common in DROP passages
            'numeric': [
                # Capture distances (e.g., "20-yard touchdown pass")
                r"(\d+)-yard\s+(touchdown pass|touchdown run|field goal)",
                # Optionally capture patterns like "a 33-yard field goal" (with an optional article)
                r"(?:a\s+)?(\d+)-yard\s+(touchdown pass|touchdown run|field goal)",
                # You could also capture plain numbers with units (e.g., "53 yard")
                r"(\d+)\s+yard",
                # And if needed, scores or other numeric events
                r"(\d+)\s+points"
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
            'location': 0.75,
            'numeric': 0.7 
        }

        # Add semantic rule types for enhanced semantic extraction
        self.semantic_relation_types = {
            'subject_object': ['nsubj', 'dobj'],
            'modifier': ['amod', 'advmod'],
            'compound': ['compound', 'nmod']
        }

        # Initialize the DimensionalityManager to ensure rule embeddings are aligned
        self.dim_manager = DimensionalityManager(target_dim=768)

    def _compute_aligned_embedding(self, text: str) -> torch.Tensor:
        """
        Computes an embedding for the given text using SentenceTransformer,
        and then aligns it to the target dimension using the DimensionalityManager.
        """
        embedding = self.encoder.encode(text, convert_to_tensor=True)
        aligned_embedding = self.dim_manager.align_embeddings(embedding, "rule")
        return aligned_embedding

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
            logger.debug(f"Processing sentence: '{doc.sents}'")
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
                        if rule:
                            confidence = self._compute_rule_confidence(rule)
                            if confidence >= min_confidence:
                                rule['confidence'] = confidence
                                # Mark rule as extracted from HotpotQA supporting context
                                rule['supporting_fact'] = True
                                extracted_rules.append(rule)

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
            elif relation_type == 'numeric':
                # For numeric facts, we might want to capture the numeric value and the event
                # e.g., groups[0] is the number, groups[1] is the event type (if available)
                rule.update({
                    'value': groups[0],
                    'event': groups[1] if len(groups) > 1 else 'numeric'
                })

            # Add entity information if available
            rule['entity_types'] = {
                entity: type_
                for entity, type_ in entities.items()
                if entity in sentence
            }

            # Compute and add the aligned embedding for the rule using the entire sentence
            rule['embedding'] = self._compute_aligned_embedding(sentence)

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
                0.1 if rule.get('entity_types') else -0.1,
                min(0.1, len(rule['keywords']) * 0.02),
                self.confidence_thresholds.get(rule['type'], 0.0) - 0.5
            ]

            confidence = base_confidence + sum(adjustments)
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
        entity_index = defaultdict(list)
        for i, rule in enumerate(rules):
            for entity in rule.get('entity_types', {}):
                entity_index[entity].append(i)

        for rule in rules:
            connected_rules = set()
            for entity in rule.get('entity_types', {}):
                for connected_idx in entity_index[entity]:
                    connected_rules.add(connected_idx)
            rule['connected_rules'] = list(connected_rules)
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
            type_compatibility = 0.2 if source_rule['type'] == target_rule['type'] else 0.0
            path_scores[i] = (similarity + type_compatibility) / 2
        return path_scores

    def _validate_rule_structure(self, rule: Dict) -> bool:
        """
        Validate if a rule dictionary has the basic required structure.
        Allows rules with either 'keywords' and 'response' or, for HotpotQA facts, a 'text' field.
        """
        if rule.get('supporting_fact'):
            return isinstance(rule, dict) and ("text" in rule or "source_text" in rule)
        required_fields = {"keywords", "response"}
        return isinstance(rule, dict) and all(field in rule for field in required_fields)

    def _track_rule_addition(self, valid_rules: List[Dict]):
        """
        Track metrics related to the addition of dynamic rules (for academic purposes).
        """
        num_rules_added = len(valid_rules)
        total_rules_now = len(self.rules)
        avg_confidence = float(np.mean([rule.get('confidence', 0.0) for rule in valid_rules])) if valid_rules else 0.0
        logger.info(
            f"Tracked Rule Addition: Added {num_rules_added} rules. Total rules: {total_rules_now}. Avg confidence: {avg_confidence:.3f}"
        )

    # --- New Functions for Semantic Rule Extraction ---

    def _create_semantic_rule(self, token, sentence) -> Optional[Dict]:
        """
        Create a rule from semantic relationships in the sentence.

        Args:
            token: The spaCy token representing subject or object.
            sentence: The full sentence containing the relationship.

        Returns:
            Dict containing the extracted semantic rule or None if extraction fails.
        """
        try:
            # Find the predicate (usually the root verb)
            predicate = [t for t in token.ancestors if t.dep_ == 'ROOT']
            if not predicate:
                return None

            rule = {
                'type': 'semantic',
                'subject': token.text if token.dep_ == 'nsubj' else None,
                'predicate': predicate[0].text,
                'object': token.text if token.dep_ == 'dobj' else None,
                'sentence': sentence.text,
                'keywords': [token.text, predicate[0].text],
                'confidence': 0.8  # Initial confidence for semantic rules
            }

            return rule if rule['subject'] and (rule['object'] or rule['predicate']) else None

        except Exception as e:
            logger.error(f"Error creating semantic rule: {e}")
            return None

    def extract_semantic_rules(self, context: str) -> List[Dict]:
        """
        Extract rules based on semantic relationships rather than just patterns.

        Args:
            context: Input text to extract rules from.

        Returns:
            List of extracted semantic rules.
        """
        doc = nlp(context)
        semantic_rules = []

        for sent in doc.sents:
            # Use dependency parsing to extract semantic relationships
            for token in sent:
                if token.dep_ in self.semantic_relation_types['subject_object']:
                    rule = self._create_semantic_rule(token, sent)
                    if rule:
                        semantic_rules.append(rule)

        return semantic_rules
    
    def extract_drop_facts(self, context_text: str, min_confidence: float = 0.7) -> List[Dict]:
        """
        Extract facts from a DROP context using enhanced pattern matching and validation.
        
        Args:
            context_text: The context text from the DROP dataset.
            min_confidence: Minimum confidence threshold for extracted rules.
        
        Returns:
            List of extracted rules with confidence scores.
        """
        extracted_rules = []
        
        logger.debug("Starting extraction of DROP facts.")
        # Process text with spaCy for enhanced NLP features
        doc = nlp(context_text)
        # sentences = list(doc.sents)
        # logger.debug(f"Processed context text with spaCy; found {len(doc.sents)} sentences.")

        # Extract entities and their types from the DROP passage
        entities = {ent.text: ent.label_ for ent in doc.ents}
        logger.debug(f"Extracted entities: {entities}")

        # Process each sentence for fact extraction
        for sent in doc.sents:
            # sentence_text = sent.text.strip()
            if len(sent.text.split()) < 4:
                # logger.debug(f"Skipping short sentence: '{sentence_text}'")
                continue
            
            logger.debug(f"Processing sentence: '{sent.text}'")
            # Apply the same pattern matching as for HotpotQA
            for relation_type, patterns in self.relation_patterns_drop.items():
                for pattern in patterns:
                    # logger.debug(f"Using pattern: '{pattern}' for relation type: '{relation_type}'")
                    matches = list(re.finditer(pattern, sent.text))
                    if not matches:
                        logger.debug(f"No matches found for pattern: '{pattern}'")
                    for match in matches:
                        logger.debug(f"Match found: '{match.group()}' in sentence: '{sent.text}'")
                        # Create rule from match
                        rule = self._create_rule(
                            match=match,
                            relation_type=relation_type,
                            sentence=sent.text,
                            entities=entities
                        )
                        if rule is None:
                            logger.debug(f"Rule creation failed for match: '{match.group()}'")
                            continue
                        confidence = self._compute_rule_confidence(rule)
                        logger.debug(f"Computed confidence: {confidence} for rule: {rule}")
                        if confidence >= min_confidence:
                            rule['confidence'] = confidence
                            # Mark rule as extracted from DROP supporting context
                            rule['supporting_fact'] = True
                            extracted_rules.append(rule)
                            logger.debug(f"Rule accepted: {rule}")
                        
            processed_rules = self._establish_rule_connections(extracted_rules)
            logger.info(f"Extracted {len(processed_rules)} rules from DROP context")
            return processed_rules