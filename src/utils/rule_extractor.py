# src/utils/rule_extractor.py

import json
import re
import numpy as np
import spacy
from typing import List, Dict, Optional, Any
from collections import defaultdict, Counter
import logging
from transformers import pipeline
from spacy.tokens import Token
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.dimension_manager import DimensionalityManager

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
    rule_scorer = pipeline("text-classification",
                           model="facebook/bart-large-mnli",
                           top_k=None)
    logger.info("Successfully initialized rule scorer")
except Exception as e:
    logger.error(f"Error initializing rule scorer: {str(e)}")
    raise

class RuleExtractor:
    """
    Enhanced RuleExtractor for both HotpotQA and DROP datasets.
    Supports sophisticated pattern matching, semantic extraction, and DROP-specific discrete reasoning.
    """

    def __init__(self):
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

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        self.confidence_thresholds = {
            'biographical': 0.8,
            'professional': 0.75,
            'creation': 0.85,
            'temporal': 0.7,
            'location': 0.75
        }

        self.semantic_relation_types = {
            'subject_object': ['nsubj', 'dobj'],
            'modifier': ['amod', 'advmod'],
            'compound': ['compound', 'nmod']
        }

        self.drop_triggers = {
            "count": ["how many", "number of"],
            "extreme_value": ["first", "last", "longest", "shortest"],
            "difference": ["difference between", "how many more", "how many less"],
            "entity_span": ["who", "which team", "what player"],
            "date": ["when", "what date", "which year"]
        }

        self.temporal_phrases = {"first half", "second half", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"}
        self.spatial_phrases = {"in the end zone", "at the 50-yard line"}

        self.dim_manager = DimensionalityManager(target_dim=768)

    def _compute_aligned_embedding(self, text: str) -> torch.Tensor:
        """
        Computes an embedding for the given text using SentenceTransformer,
        and then aligns it to the target dimension using the DimensionalityManager.
        """
        embedding = self.encoder.encode(text, convert_to_tensor=True)
        aligned_embedding = self.dim_manager.align_embeddings(embedding, "rule")
        return aligned_embedding

    # --- HotpotQA-Specific Rule Extraction ---

    def extract_hotpot_facts(self, context_text: str, min_confidence: float = 0.7) -> List[Dict]:
        """
        Extract facts from HotpotQA context with enhanced pattern matching and validation.
        """
        extracted_rules = []

        doc = nlp(context_text)
        entities = {ent.text: ent.label_ for ent in doc.ents}

        for sent in doc.sents:
            if len(sent.text.split()) < 4:
                continue

            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sent.text)
                    for match in matches:
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
                                rule['supporting_fact'] = True
                                extracted_rules.append(rule)

        processed_rules = self._establish_rule_connections(extracted_rules)
        logger.info(f"Extracted {len(processed_rules)} rules from HotpotQA context")
        return processed_rules

    def _create_rule(self, match: re.Match, relation_type: str,
                     sentence: str, entities: Dict[str, str]) -> Optional[Dict]:
        """
        Create a structured rule from a regex match with enhanced metadata (HotpotQA-specific).
        """
        try:
            groups = match.groups()
            if not groups:
                return None

            rule = {
                'type': relation_type,
                'source_text': sentence,
                'groups': groups,
                'keywords': self._extract_keywords(sentence)
            }

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

            rule['entity_types'] = {
                entity: type_
                for entity, type_ in entities.items()
                if entity in sentence
            }

            rule['embedding'] = self._compute_aligned_embedding(sentence)
            return rule

        except Exception as e:
            logger.warning(f"Error creating rule: {str(e)}")
            return None

    # --- DROP-Specific Rule Extraction Methods ---

    def extract_dep_patterns_with_temporal(self, questions: List[str], min_freq: int = 1) -> List[Dict[str, Any]]:
        """
        Extract dependency-based patterns with temporal constraints for DROP.
        Identifies operation-specific patterns (e.g., count, extreme_value) and incorporates temporal constraints.
        """
        logger.info(f"Starting extract_dep_patterns_with_temporal with {len(questions)} questions")
        trigger_to_op = {
            "how many": "count",
            "number of": "count",
            "first": "extreme_value",
            "last": "extreme_value",
            "longest": "extreme_value",
            "shortest": "extreme_value",
            "difference": "difference",
            "how many more": "difference",
            "how many less": "difference",
            "who": "entity_span",
            "which team": "entity_span",
            "what player": "entity_span",
            "when": "date",
            "what date": "date",
            "which year": "date"
        }

        patterns = []
        path_counts = Counter()
        processed_questions = 0

        for q in questions:
            if not q or not isinstance(q, str):
                logger.debug(f"Skipping invalid question: {q}")
                continue
            processed_questions += 1
            doc = nlp(q.lower())
            trigger_found = False

            for token in doc:
                trigger = None
                for trig in trigger_to_op:
                    if trig in f"{token.text} {token.nbor(1).text if token.i < len(doc) - 1 else ''}":
                        trigger = trig
                        break
                if not trigger:
                    continue

                trigger_found = True
                target = None
                for child in token.children:
                    if child.dep_ in ("dobj", "nsubj", "attr"):
                        target = child
                        break
                if not target:
                    for nc in doc.noun_chunks:
                        if token in nc or any(child in nc for child in token.children):
                            target = nc.root
                            break
                if not target:
                    target = token
                    logger.debug(f"No noun chunk found for trigger '{trigger}' in question '{q}'. Using token '{target.text}' as target.")

                temporal = None
                for nc in doc.noun_chunks:
                    if nc.text in self.temporal_phrases:
                        temporal = nc.text
                        break

                entity_text = None
                for nc in doc.noun_chunks:
                    if target in nc or any(child in nc for child in target.children):
                        entity_text = nc.text
                        break
                if not entity_text:
                    # Fallback: Construct a phrase using target and its head for more context
                    if target.dep_ in ("dobj", "nsubj", "attr") and target.head:
                        entity_text = f"{target.text} {target.head.text}"
                    else:
                        entity_text = target.text
                if temporal:
                    pattern = rf"\b{trigger}\s+([\w\s]+?)\s+(?:in|during)\s+{re.escape(temporal)}\b"
                else:
                    pattern = rf"\b{trigger}\s+([\w\s]+?)\b"
                path_counts[(trigger, entity_text, temporal)] += 1
                patterns.append({
                    'type': trigger_to_op[trigger],
                    'pattern': pattern,
                    'entity': entity_text,
                    'temporal_constraint': temporal,
                    'support': path_counts[(trigger, entity_text, temporal)]
                })
                logger.debug(f"Found pattern: Type={trigger_to_op[trigger]}, Pattern='{pattern}', Entity='{entity_text}', Temporal='{temporal}', Support={path_counts[(trigger, entity_text, temporal)]}")
            if not trigger_found:
                logger.debug(f"No trigger found in question: '{q}'")

        logger.info(f"Processed {processed_questions} questions. Found {len(patterns)} patterns before filtering (min_freq={min_freq})")
        filtered_patterns = [p for p in patterns if path_counts[(p['type'], p['entity'], p['temporal_constraint'])] >= min_freq]
        logger.info(f"After filtering with min_freq={min_freq}, retained {len(filtered_patterns)} patterns")
        return filtered_patterns

    def extract_span_context_patterns(self, drop_json_path: str, window: int = 5, min_freq: int = 1) -> List[Dict[str, Any]]:
        """
        Extract and generalize context patterns around DROP answer spans.
        Uses ground-truth spans to learn reliable extraction patterns.
        """
        logger.info(f"Starting extract_span_context_patterns with drop_json_path={drop_json_path}")
        patterns = []
        span_counts = Counter()
        processed_pairs = 0
        span_matches_found = 0

        # Load and validate dataset
        try:
            with open(drop_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load DROP dataset from {drop_json_path}: {e}")
            return []

        if not isinstance(data, dict):
            logger.error(f"Expected DROP dataset to be a dictionary, got type {type(data)}")
            return []
        logger.debug(f"DROP dataset contains {len(data)} passages")

        # Define flexible placeholders
        number_pat = r"[0-9][0-9,\. ]*"
        date_pat = r"\d{1,2}(?:/\d{1,2})?(?:/\d{2,4})?"
        span_pat = r"[\w\s]+?"

        for passage_id, passage_content in data.items():
            if not isinstance(passage_content, dict):
                logger.debug(f"Skipping passage {passage_id}: Expected dict, got {type(passage_content)}")
                continue
            if 'passage' not in passage_content or 'qa_pairs' not in passage_content:
                logger.debug(f"Skipping passage {passage_id}: Missing 'passage' or 'qa_pairs' keys")
                continue
            if not isinstance(passage_content['qa_pairs'], list):
                logger.debug(f"Skipping passage {passage_id}: 'qa_pairs' is not a list")
                continue

            passage = passage_content['passage']
            if not isinstance(passage, str) or not passage.strip():
                logger.debug(f"Skipping passage {passage_id}: Passage is not a valid string")
                continue

            doc = nlp(passage)
            tokens = [token.text for token in doc]

            for qa_pair in passage_content['qa_pairs']:
                if not isinstance(qa_pair, dict):
                    logger.debug(f"Skipping QA pair in passage {passage_id}: Expected dict, got {type(qa_pair)}")
                if 'answer' not in qa_pair:
                    logger.debug(f"Skipping QA pair in passage {passage_id}: Missing 'answer' key")
                    continue

                processed_pairs += 1
                answer = qa_pair['answer']
                span = None
                span_type = None

                if not isinstance(answer, dict):
                    logger.debug(f"Skipping QA pair in passage {passage_id}: Answer is not a dict")
                    continue

                if 'number' in answer and answer['number']:
                    span = str(answer['number'])
                    span_type = 'number'
                elif 'spans' in answer and answer['spans']:
                    span = answer['spans'][0] if isinstance(answer['spans'], list) and answer['spans'] else None
                    span_type = 'spans'
                elif 'date' in answer and isinstance(answer['date'], dict) and answer['date'].get('year'):
                    span = f"{answer['date'].get('month', '')}/{answer['date'].get('day', '')}/{answer['date']['year']}"
                    span_type = 'date'
                else:
                    logger.debug(f"Skipping QA pair in passage {passage_id}: No valid answer type (number, spans, date)")
                    continue

                if not span or not isinstance(span, str) or not span.strip():
                    logger.debug(f"Skipping QA pair in passage {passage_id}: Invalid span '{span}'")
                    continue

                # Find the span in the passage using spaCy's char_span
                start_idx = passage.find(span)
                if start_idx == -1:
                    logger.debug(f"Span '{span}' not found in passage for passage {passage_id}")
                    continue

                span_token = doc.char_span(start_idx, start_idx + len(span), alignment_mode="expand")
                if span_token is None:
                    logger.debug(f"Couldn't align span '{span}' in passage {passage_id}")
                    continue

                tok_idx = span_token.start
                span_length = span_token.end - span_token.start
                span_matches_found += 1
                left = tokens[max(0, tok_idx - window):tok_idx]
                right = tokens[tok_idx + span_length:tok_idx + span_length + window]

                if not left or not right:
                    logger.debug(f"Skipping pattern for span '{span}': Insufficient context (left={left}, right={right})")
                    continue

                placeholder = {"number": number_pat, "date": date_pat, "spans": span_pat}[span_type]
                pattern = rf"{' '.join(left)}\s+({placeholder})\s+{' '.join(right)}"
                span_counts[pattern] += 1
                patterns.append({
                    'type': span_type,
                    'pattern': pattern,
                    'support': span_counts[pattern]
                })
                logger.debug(f"Generated pattern: Type={span_type}, Pattern='{pattern}', Support={span_counts[pattern]}")

        logger.info(f"Processed {processed_pairs} QA pairs. Found {span_matches_found} span matches. Generated {len(patterns)} patterns before filtering (min_freq={min_freq})")
        filtered_patterns = [p for p in patterns if p['support'] >= min_freq]
        logger.info(f"After filtering with min_freq={min_freq}, retained {len(filtered_patterns)} patterns")
        return filtered_patterns

    def extract_semantic_drop_rules(self, questions: List[str], min_support: int = 1) -> List[Dict[str, Any]]:
        """
        Extract semantic rules for DROP by combining semantic triples with operation triggers.
        Maps subject-predicate-object triples to operations like count, extreme_value.
        """
        logger.info(f"Starting extract_semantic_drop_rules with {len(questions)} questions")
        drop_rules = []
        count_predicates = {"kick", "score", "make", "throw", "pass", "run", "gain", "attempt", "complete"}
        extreme_predicates = {"score", "throw", "pass", "run", "gain"}
        processed_questions = 0
        triples_found = 0

        for q in questions:
            if not q or not isinstance(q, str):
                logger.debug(f"Skipping invalid question: {q}")
                continue
            processed_questions += 1
            sems = self.extract_semantic_rules(q)
            triples_found += len(sems)
            q_lower = q.lower()
            operation = None
            for op, triggers in self.drop_triggers.items():
                if any(trigger in q_lower for trigger in triggers):
                    operation = op
                    break
            if not operation:
                logger.debug(f"No operation trigger found in question: '{q}'")
                continue

            for r in sems:
                pred = r.get('predicate', '').lower()
                obj = r.get('object')
                if not obj:
                    logger.debug(f"Skipping semantic triple with no object: {r}")
                    continue

                if operation == 'count' and pred in count_predicates:
                    pattern = rf"\bhow many\s+{re.escape(obj)}s?\b"
                    drop_rules.append({
                        'type': 'count',
                        'pattern': pattern,
                        'predicate': pred,
                        'support': 1
                    })
                    logger.debug(f"Generated count rule: Pattern='{pattern}', Predicate='{pred}'")
                elif operation == 'extreme_value' and pred in extreme_predicates:
                    pattern = rf"\b(first|last|longest|shortest)\s+{re.escape(obj)}s?\b"
                    drop_rules.append({
                        'type': 'extreme_value',
                        'pattern': pattern,
                        'predicate': pred,
                        'support': 1
                    })
                    logger.debug(f"Generated extreme_value rule: Pattern='{pattern}', Predicate='{pred}'")
                elif operation == 'difference':
                    pattern = rf"\b(difference between|how many more|how many less)\s+{re.escape(obj)}s?\b"
                    drop_rules.append({
                        'type': 'difference',
                        'pattern': pattern,
                        'support': 1
                    })
                    logger.debug(f"Generated difference rule: Pattern='{pattern}'")

        unique_rules = {tuple(sorted(r.items())): r for r in drop_rules}
        final_rules = list(unique_rules.values())
        filtered_rules = [r for r in final_rules if r['support'] >= min_support]
        logger.info(f"Processed {processed_questions} questions. Found {triples_found} semantic triples. Generated {len(drop_rules)} rules before filtering. Retained {len(filtered_rules)} rules after filtering (min_support={min_support})")
        return filtered_rules

    def _train_rule_classifier(self, drop_json_path: str, subset_size: int = 100) -> List[Dict[str, Any]]:
        """
        Train a rule classifier for DROP dataset using a labeled subset to improve rule specificity.
        Generates rules based on question patterns and answer types.
        """
        logger.info(f"Training rule classifier with subset_size={subset_size} from {drop_json_path}")
        rules = []
        try:
            # Load DROP dataset
            with open(drop_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.error(f"Expected DROP dataset to be a dictionary, got type {type(data)}")
                return []

            # Collect training data
            questions = []
            labels = []
            valid_pairs = 0
            for passage_id, passage_content in data.items():
                if valid_pairs >= subset_size:
                    break
                if not isinstance(passage_content, dict) or 'qa_pairs' not in passage_content:
                    continue
                for qa_pair in passage_content['qa_pairs']:
                    if valid_pairs >= subset_size:
                        break
                    if not isinstance(qa_pair, dict) or 'question' not in qa_pair or 'answer' not in qa_pair:
                        continue
                    question = qa_pair['question'].lower()
                    answer = qa_pair['answer']
                    if not isinstance(answer, dict):
                        continue

                    # Determine operation type based on triggers and answer
                    operation = None
                    for op, triggers in self.drop_triggers.items():
                        if any(trigger in question for trigger in triggers):
                            if (op == 'count' and answer.get('number')) or \
                               (op == 'extreme_value' and (answer.get('number') or answer.get('spans'))) or \
                               (op == 'difference' and answer.get('number')) or \
                               (op == 'entity_span' and answer.get('spans')) or \
                               (op == 'date' and answer.get('date')):
                                operation = op
                                break
                    if not operation:
                        continue

                    questions.append(question)
                    labels.append(operation)
                    valid_pairs += 1

            if not questions:
                logger.warning("No valid QA pairs found for training rule classifier")
                return []

            # Train classifier
            classifier = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('clf', LogisticRegression(multi_class='multinomial', max_iter=1000))
            ])
            classifier.fit(questions, labels)
            logger.info(f"Trained rule classifier on {len(questions)} QA pairs")

            # Generate rules based on classifier predictions
            operation_types = ['count', 'extreme_value', 'difference', 'entity_span', 'date']
            for question in questions[:subset_size]:
                doc = nlp(question)
                predicted_op = classifier.predict([question])[0]
                if predicted_op not in operation_types:
                    continue

                # Extract entity or temporal constraint
                entity = None
                temporal = None
                for nc in doc.noun_chunks:
                    if nc.text in self.temporal_phrases:
                        temporal = nc.text
                    elif nc.root.dep_ in ('dobj', 'nsubj', 'attr'):
                        entity = nc.text

                if not entity:
                    entity = 'entity'  # Generic placeholder

                # Generate pattern based on operation
                if predicted_op == 'count':
                    pattern = rf"\bhow many\s+{re.escape(entity)}s?\b"
                elif predicted_op == 'extreme_value':
                    pattern = rf"\b(first|last|longest|shortest)\s+{re.escape(entity)}s?\b"
                elif predicted_op == 'difference':
                    pattern = rf"\b(difference between|how many more|how many less)\s+{re.escape(entity)}s?\b"
                elif predicted_op == 'entity_span':
                    pattern = rf"\b(who|which team|what player)\s+{re.escape(entity)}s?\b"
                elif predicted_op == 'date':
                    pattern = rf"\b(when|what date|which year)\s+{re.escape(entity)}s?\b"
                else:
                    continue

                rule = {
                    'type': predicted_op,
                    'pattern': pattern,
                    'entity': entity,
                    'temporal_constraint': temporal,
                    'support': 1,
                    'confidence': classifier.predict_proba([question])[0][operation_types.index(predicted_op)]
                }
                rules.append(rule)
                logger.debug(f"Generated rule from classifier: Type={predicted_op}, Pattern='{pattern}', Entity='{entity}', Temporal='{temporal}', Confidence={rule['confidence']}")

            logger.info(f"Generated {len(rules)} rules from rule classifier")
            return rules

        except Exception as e:
            logger.error(f"Error training rule classifier: {str(e)}")
            return []

    def extract_rules_from_drop(self, drop_json_path: str, questions: List[str], passages: List[str],
                                min_support: int = 1) -> List[Dict[str, Any]]:
        """
        Extract rules for DROP dataset using multiple sophisticated strategies, including a trained rule classifier.
        Args:
            drop_json_path: Path to DROP JSON file containing passages and QA pairs.
            questions: List of DROP questions.
            passages: List of corresponding passages.
            min_support: Minimum support threshold for final rules.
        Returns:
            List of extracted rules for DROP.
        """
        logger.info(f"Starting extract_rules_from_drop with {len(questions)} questions, {len(passages)} passages, min_support={min_support}")
        rules = []
        counts = Counter()
        by_key = {}

        # Dependency-based patterns with temporal constraints
        dep_rules = self.extract_dep_patterns_with_temporal(questions, min_freq=1)
        for p in dep_rules:
            key = (p['type'], p['pattern'])
            counts[key] += p['support']
            if key not in by_key:
                by_key[key] = p
        rules.extend(dep_rules)
        logger.info(f"Dependency patterns extracted: {len(dep_rules)} rules")

        # Answer-span-guided patterns
        span_rules = self.extract_span_context_patterns(drop_json_path, min_freq=1)
        for p in span_rules:
            key = (p['type'], p['pattern'])
            counts[key] += p['support']
            if key not in by_key:
                by_key[key] = p
        rules.extend(span_rules)
        logger.info(f"Span context patterns extracted: {len(span_rules)} rules")

        # Hybrid semantic-symbolic rules
        semantic_rules = self.extract_semantic_drop_rules(questions, min_support=1)
        for p in semantic_rules:
            key = (p['type'], p['pattern'])
            counts[key] += p['support']
            if key not in by_key:
                by_key[key] = p
        rules.extend(semantic_rules)
        logger.info(f"Semantic rules extracted: {len(semantic_rules)} rules")

        # Machine learning-based rules
        classifier_rules = self._train_rule_classifier(drop_json_path, subset_size=100)
        for p in classifier_rules:
            key = (p['type'], p['pattern'])
            counts[key] += p['support']
            if key not in by_key:
                by_key[key] = p
        rules.extend(classifier_rules)
        logger.info(f"Classifier-based rules extracted: {len(classifier_rules)} rules")

        # Deduplicate and filter rules by (type, pattern)
        final_rules = []
        for key, support in counts.items():
            if support >= min_support:
                rule = by_key[key]
                rule['support'] = support
                final_rules.append(rule)

        logger.info(f"Total rules before final filtering: {len(rules)}. After filtering with min_support={min_support}, retained {len(final_rules)} rules")
        if not final_rules:
            logger.warning("No rules met the final min_support threshold. Consider lowering min_support or checking the dataset format.")
        return final_rules

    # --- Shared Methods for Both Datasets ---

    def _compute_rule_confidence(self, rule: Dict) -> float:
        """
        Compute confidence score for a rule using multiple factors.
        """
        try:
            source_text = rule.get('source_text', rule.get('sentence', ''))
            result = rule_scorer(source_text)
            base_confidence = next(
                (score['score'] for score in result[0]
                 if score['label'] == 'ENTAILMENT'),
                0.5
            )

            adjustments = [
                0.1 if rule.get('entity_types') else -0.1,
                min(0.1, len(rule.get('keywords', [])) * 0.02),
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
        Establish connections between rules for multi-hop reasoning (HotpotQA-specific).
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

    def _compute_path_scores(self, source_rule: Dict, connected_rules: List[Dict]) -> Dict[int, float]:
        """
        Compute relevance scores for connected reasoning paths (HotpotQA-specific).
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
        Validate if a rule dictionary has the required structure.
        Allows flexibility for HotpotQA and DROP rules.
        """
        if rule.get('supporting_fact'):
            return isinstance(rule, dict) and ("text" in rule or "source_text" in rule)
        required_fields = {"type", "pattern"}
        return isinstance(rule, dict) and all(field in rule for field in required_fields)

    def _create_semantic_rule(self, token, sentence) -> Optional[Dict]:
        """
        Create a rule from semantic relationships in the sentence.
        Used for both HotpotQA and DROP semantic extraction.
        """
        try:
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
                'confidence': 0.8
            }

            return rule if rule['subject'] and (rule['object'] or rule['predicate']) else None

        except Exception as e:
            logger.error(f"Error creating semantic rule: {e}")
            return None

    def extract_semantic_rules(self, context: str) -> List[Dict]:
        """
        Extract rules based on semantic relationships for both HotpotQA and DROP.
        """
        doc = nlp(context)
        semantic_rules = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ in self.semantic_relation_types['subject_object']:
                    rule = self._create_semantic_rule(token, sent)
                    if rule:
                        semantic_rules.append(rule)

        return semantic_rules