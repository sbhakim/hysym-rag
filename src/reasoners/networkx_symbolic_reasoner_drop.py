# src/reasoners/networkx_symbolic_reasoner_drop.py

import re
import logging
from datetime import datetime
from dateutil import parser as date_parser
import operator
from typing import List, Dict, Optional, Union, Any, Tuple
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, util
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from .networkx_symbolic_reasoner_base import GraphSymbolicReasoner, DEFAULT_DROP_ANSWER
    from .networkx_symbolic_reasoner_base import kw_model
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner, DEFAULT_DROP_ANSWER
    try:
        from src.reasoners.networkx_symbolic_reasoner_base import kw_model
    except ImportError:
        kw_model = None

OP_COUNT = "count"
OP_EXTREME_VALUE = "extreme_value"
OP_EXTREME_VALUE_NUMERIC = "extreme_value_numeric"
OP_DIFFERENCE = "difference"
OP_TEMPORAL_DIFFERENCE = "temporal_difference"
OP_ENTITY_SPAN = "entity_span"
OP_DATE = "date"

class GraphSymbolicReasonerDrop(GraphSymbolicReasoner):
    """
    DROP-specific extension of the GraphSymbolicReasoner.
    Handles sophisticated reasoning tasks for the DROP dataset, including operation extraction,
    entity/number/date extraction, and operation execution.
    Enhanced to leverage rules with entity and temporal_constraint fields.
    """

    def process_query(
            self,
            query: str,
            context: Optional[str] = None,
            dataset_type: str = 'text',
            query_id: Optional[str] = None
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Override the base class method to handle DROP queries.
        Delegates to process_drop_query for DROP dataset, otherwise falls back to base class.
        """
        qid = query_id or "unknown"
        self.logger.debug(f"Processing query ID: {qid}, Query: '{query[:50]}...', Dataset: {dataset_type}")

        dt_lower = dataset_type.lower().strip() if isinstance(dataset_type, str) else 'text'

        if dt_lower == 'drop':
            if not context or not isinstance(context, str) or not context.strip():
                self.logger.error(f"[DROP QID:{qid}] Context is required for DROP processing but was not provided or is invalid.")
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.0, 'rationale': 'Missing or invalid context for DROP query.'}
            return self.process_drop_query(query, context, dataset_type, qid)

        return super().process_query(query, context, dataset_type, query_id)

    def process_drop_query(
            self,
            query: str,
            context: str,
            dataset_type: str,
            query_id: str
    ) -> Dict[str, Any]:
        """
        Process a DROP query, returning a structured answer.
        Enhanced to use rules from RuleExtractor with entity and temporal_constraint fields.
        Falls back to original extraction logic if no rule matches.
        """
        self.logger.debug(f"[DROP QID:{query_id}] Processing DROP query: '{query[:50]}...'")

        try:
            # 1. Try matching the query against loaded rules
            matched_rule = None
            op_type = None
            op_args = {}
            extraction_confidence = 0.5
            rationale = "Operation extracted."

            for rule in self.rules:
                if 'compiled_pattern' not in rule:
                    continue
                match = rule['compiled_pattern'].search(query)
                if match:
                    matched_rule = rule
                    op_type = rule['type']
                    # Handle different rule structures based on type
                    if op_type == OP_COUNT:
                        entity = match.group(1).strip() if match.groups() else None
                        temporal = match.group(2) if match.lastindex >= 2 and match.group(2) else rule.get('temporal_constraint')
                        op_args = {
                            'entity': entity,
                            'temporal_constraint': temporal,
                            'query_keywords': []  # Will be populated later
                        }
                    elif op_type in [OP_DIFFERENCE, OP_TEMPORAL_DIFFERENCE]:
                        if 'unit_group' in rule:
                            # For difference queries like "how many yards difference..."
                            unit = match.group(1).strip() if match.groups() else None
                            entity = match.group(2).strip() if match.lastindex >= 2 else None
                            attr1 = match.group(3).strip() if match.lastindex >= 3 else None
                            entity1 = match.group(4).strip() if match.lastindex >= 4 else None
                            attr2 = match.group(5).strip() if match.lastindex >= 5 else None
                            entity2 = match.group(6).strip() if match.lastindex >= 6 else None
                            op_args = {
                                'unit': unit,
                                'entity': entity,
                                'attr1': attr1,
                                'entity1': entity1,
                                'attr2': attr2,
                                'entity2': entity2,
                                'query_keywords': []
                            }
                        else:
                            # Standard difference or temporal difference
                            entity1 = match.group(1).strip() if match.groups() else None
                            entity2 = match.group(2).strip() if match.lastindex >= 2 else None
                            op_args = {
                                'entity1': entity1,
                                'entity2': entity2,
                                'query_keywords': []
                            }
                    elif op_type in [OP_EXTREME_VALUE, OP_EXTREME_VALUE_NUMERIC]:
                        direction = match.group(1).strip() if match.groups() else rule.get('direction', 'longest')
                        entity = match.group(2).strip() if match.lastindex >= 2 else None
                        unit = match.group(1).strip() if 'unit_group' in rule and match.groups() else None
                        op_args = {
                            'entity_desc': entity,
                            'direction': direction,
                            'unit': unit,
                            'query_keywords': []
                        }
                    elif op_type in [OP_ENTITY_SPAN, OP_DATE]:
                        entity = match.group(1).strip() if match.groups() else None
                        op_args = {
                            'entity': entity,
                            'query_keywords': []
                        }
                    extraction_confidence = rule.get('confidence', 0.7)
                    rationale = f"Matched rule with type '{op_type}' (confidence: {extraction_confidence:.2f})"
                    self.logger.info(f"[DROP QID:{query_id}] Matched rule: Type={op_type}, Args={op_args}")
                    break

            # 2. Fall back to original extraction logic if no rule matches
            if not matched_rule:
                self.logger.info(f"[DROP QID:{query_id}] No rule matched. Falling back to regex-based extraction.")
                op_result = self._extract_drop_operation_and_args(query, context, query_id)
                if op_result.get('status') != 'success':
                    self.logger.warning(f"[DROP QID:{query_id}] Symbolic path failed during op extraction: {op_result.get('rationale', 'Unknown reason')}")
                    return {**DEFAULT_DROP_ANSWER, **op_result}

                op_type = op_result.get('operation')
                op_args = op_result.get('args', {})
                extraction_confidence = op_result.get('confidence', 0.5)
                rationale = op_result.get('rationale', 'Operation extracted.')
                self.logger.info(f"[DROP QID:{query_id}] Fallback extraction: Type={op_type}, Args={op_args}")

            # 3. Extract keywords for better entity association (if not already done)
            query_keywords = op_args.get('query_keywords', [])
            if not query_keywords:
                query_lower = query.lower()
                if kw_model:
                    try:
                        query_keywords = [kw[0] for kw in kw_model.extract_keywords(query_lower, top_n=5)]
                        self.logger.debug(f"[DROP QID:{query_id}] Keywords via KeyBERT: {query_keywords}")
                    except Exception as ke_err:
                        self.logger.warning(f"[DROP QID:{query_id}] KeyBERT keyword extraction failed: {ke_err}. Falling back to spaCy.")
                if not query_keywords and self.nlp:
                    doc = self.nlp(query_lower)
                    query_keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'PROPN') and not token.is_stop]
                    self.logger.debug(f"[DROP QID:{query_id}] Keywords via spaCy: {query_keywords}")
                op_args['query_keywords'] = query_keywords

            # 4. Execute operation based on extracted type
            execution_result = None
            if op_type == OP_COUNT:
                execution_result = self.execute_count(op_args, context, query_id)
            elif op_type == OP_EXTREME_VALUE:
                execution_result = self.execute_extreme_value(op_args, context, query, query_id)
            elif op_type == OP_EXTREME_VALUE_NUMERIC:
                execution_result = self.execute_extreme_value_numeric(op_args, context, query, query_id)
            elif op_type == OP_DIFFERENCE:
                execution_result = self.execute_difference(op_args, context, query_id)
            elif op_type == OP_TEMPORAL_DIFFERENCE:
                execution_result = self.execute_temporal_difference(op_args, context, query_id)
            elif op_type == OP_ENTITY_SPAN:
                execution_result = self.execute_entity_span(op_args, context, query_id)
            elif op_type == OP_DATE:
                execution_result = self.execute_date(op_args, context, query_id)
            else:
                self.logger.error(f"[DROP QID:{query_id}] Unsupported operation type '{op_type}' reached execution stage.")
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': extraction_confidence, 'rationale': f"Internal error: Unsupported operation type '{op_type}'"}

            # 5. Validate and Format Execution Result
            if not isinstance(execution_result, dict) or 'type' not in execution_result or 'value' not in execution_result:
                self.logger.error(f"[DROP QID:{query_id}] Invalid result format from {op_type} execution: {execution_result}")
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.1, 'rationale': f"Internal error: Invalid result format during {op_type} execution."}

            formatted_result = self._format_drop_answer(execution_result)

            execution_confidence = execution_result.get('confidence', 0.5)
            final_confidence = (extraction_confidence * 0.6 + execution_confidence * 0.4)

            # Include the value field in the final result for downstream components
            formatted_result.update({
                'status': 'success',
                'confidence': round(final_confidence, 3),
                'rationale': f"{rationale} | Execution Conf: {execution_confidence:.2f}",
                'type': op_type,
                'value': execution_result.get('value')  # Preserve the value for HybridIntegrator
            })

            self.logger.info(f"[DROP QID:{query_id}] Successfully processed DROP query. Operation: {op_type}, Final Confidence: {final_confidence:.2f}")
            return formatted_result

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Unhandled exception processing DROP query: {str(e)}")
            return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.0, 'rationale': f"Unhandled Exception: {str(e)}"}

    def _extract_drop_operation_and_args(self, query: str, context: str, query_id: str) -> Dict[str, Any]:
        """
        Extract operation type and arguments from a DROP query using regex and spaCy.
        Used as a fallback when no predefined rule from self.rules matches.
        MODIFIED: Lowered base_confidence for fallback patterns significantly.
        """
        query_lower = query.lower().strip()
        self.logger.debug(
            f"[DROP QID:{query_id}] Fallback Extraction: Attempting to extract operation for query: '{query_lower[:100]}...'")

        # MODIFIED: Significantly lowered base_confidence values for these fallback patterns
        # The goal is that if we are using these general regexes, the initial confidence
        # that we even have the right *type* of operation and粗略 arguments should be modest.
        # The execution functions then determine their own confidence in *performing* that operation.
        operation_patterns = [
            (
                r"^how many ([a-z\s]+?) (?:did|was|were) ([\w\s'-]+?) (score|kick|hit|throw|rush for|pass for|gain|intercept|complete|make|lead with|take off)(?:\s+in\s+([\w\s'-]+))?\??$",
                OP_EXTREME_VALUE_NUMERIC,
                lambda m: {'unit': m.group(1).strip(),
                           'entity_desc': m.group(2).strip(),
                           'verb_action': m.group(3).strip(),
                           'specific_event_modifier': m.group(
                               4).strip() if m.lastindex and m.lastindex >= 4 and m.group(4) else None,
                           'direction': 'specific_event'},
                0.55),  # MODIFIED CONFIDENCE (e.g., from 0.92)
            (
                r"^how many total ([a-z\s]+?) (?:did|were) ([\w\s'-]+?) (?:score|gain|have|intercept|lead with|take off|make|hit|kick|throw|rush for|pass for)\??$",
                OP_EXTREME_VALUE_NUMERIC,
                lambda m: {'unit': m.group(1).strip(),
                           'entity_desc': m.group(2).strip(),
                           'direction': 'total'},
                0.60),  # MODIFIED CONFIDENCE (e.g., from 0.95)
            (
                r"^what was the score (?:at the end of|in|during) ([\w\s'-]+?)(?:\s+for\s+([\w\s'-]+?))?\??$",
                OP_ENTITY_SPAN,
                # This might be better as a specific SCORE type if contextually it implies numbers usually.
                # Or, if the output is "7-7", it's a span. If it's "7", it's a number.
                # For now, keeping as ENTITY_SPAN and its parser will try to get the score.
                lambda m: {'entity': f"score {m.group(1).strip()}",  # entity_desc might be "score first quarter"
                           'team': m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else None,
                           'temporal_constraint': m.group(1).strip()  # The time period is a temporal constraint
                           },
                0.55  # MODIFIED CONFIDENCE
            ),
            (
                r"^how many ([a-z\s]+?) was the (longest|shortest|highest|lowest|most|least|first|last|final)\s+([\w\s'-]+?)(?: of the game)?\??$",
                OP_EXTREME_VALUE_NUMERIC,
                lambda m: {'unit': m.group(1).strip(),
                           'direction': m.group(2).lower().strip(),
                           'entity_desc': m.group(3).strip()},
                0.55),  # MODIFIED CONFIDENCE (e.g., from 0.90)
            (
                r"^(?:how many|number of)\s+([\w\s'-]+?)(?:\s+(?:in|during|for|on)\s+(first half|second half|1st quarter|2nd quarter|3rd quarter|4th quarter|the game|overtime))?(?:\s+were\s+there|\s+did|\s+was|\s+score|\?)?$",
                OP_COUNT,
                lambda m: {'entity': m.group(1).strip(),
                           'temporal_constraint': m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(
                               2) else None},
                0.45),  # MODIFIED CONFIDENCE (e.g., from 0.90)
            (
                r"(?:difference between|how many more|how many less)\s+([\w\s'-]+?)(?:\s+(?:and|than)\s+([\w\s'-]+?))?\??$",
                OP_DIFFERENCE,
                lambda m: {'entity1': m.group(1).strip(),
                           'entity2': m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else None},
                0.60),  # MODIFIED CONFIDENCE (e.g., from 0.95)
            (
                r"^(?:who|what player|what team|which player|which team)\s+(?:had|made|scored|threw|kicked|ran for|caught)\s+(?:the\s+)?(longest|shortest|highest|lowest|most|least|first|last|final)\s+([\w\s'-]+)\??$",
                OP_EXTREME_VALUE,
                lambda m: {'entity_desc': m.group(2).strip(),
                           'direction': m.group(1).lower().strip()},
                0.55),  # MODIFIED CONFIDENCE (e.g., from 0.90)
            (
                r"^(?:the\s+)?(longest|shortest|highest|lowest|most|least|first|last|final)\s+([\w\s'-]+)\s+(?:was by whom|was by which player|was by what player)\??$",
                OP_EXTREME_VALUE,
                lambda m: {'entity_desc': m.group(2).strip(), 'direction': m.group(1).lower().strip()},
                0.55),  # MODIFIED CONFIDENCE (e.g., from 0.90)
            (r"^how many years between\s+([\w\s'-]+?)\s+and\s+([\w\s'-]+?)\??$",
             OP_TEMPORAL_DIFFERENCE,
             lambda m: {'entity1': m.group(1).strip(), 'entity2': m.group(2).strip()},
             0.50),  # MODIFIED CONFIDENCE (e.g., from 0.85)
            (r"^(?:when|what date|what year|which year)\s+(?:did|was|is)?\s*(.*)\??$",
             OP_DATE,
             lambda m: {'entity': m.group(1).strip().rstrip('?')},
             0.50),  # MODIFIED CONFIDENCE (e.g., from 0.80)
            (
                r"^(?:who|which team|what team|which player|what player|what was the name of the|tell me the name of the)\s+(.*)\??$",
                OP_ENTITY_SPAN,
                lambda m: {'entity': m.group(1).strip().rstrip('?')},
                0.50),  # MODIFIED CONFIDENCE (e.g., from 0.80)
            (r"^(?:who|which|what)\s+(.*)\??$",  # Most general, lowest confidence
             OP_ENTITY_SPAN,
             lambda m: {'entity': m.group(1).strip().rstrip('?')},
             0.35),  # MODIFIED CONFIDENCE (e.g., from 0.70)
        ]

        temporal_keywords = ["first half", "second half", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
                             "the game", "overtime"]

        matched_op_details = None
        for pattern_regex, op_type, arg_func, base_conf in operation_patterns:
            match = re.search(pattern_regex, query_lower, re.IGNORECASE)
            if match:
                if op_type == OP_COUNT:  # Keep the heuristic to avoid misclassifying value queries as count
                    potential_units = ["yards", "points", "goals", "tds", "pass", "reception", "run", "score"]
                    action_verbs = ["score", "kick", "hit", "throw", "rush for", "pass for", "gain", "intercept",
                                    "complete", "make", "lead with", "take off"]
                    query_tokens = query_lower.split()
                    if any(unit in query_tokens for unit in potential_units) and \
                            any(verb in query_tokens for verb in action_verbs) and \
                            (
                                    "did" in query_tokens or "was" in query_tokens or "were" in query_tokens):  # check for past tense verbs
                        self.logger.debug(
                            f"[DROP QID:{query_id}] OP_COUNT pattern '{pattern_regex}' matched, but query structure suggests a value extraction (e.g., how many X did Y verb). Skipping this COUNT match.")
                        continue
                matched_op_details = {
                    'pattern_regex': pattern_regex, 'type': op_type,
                    'arg_func': arg_func, 'match_obj': match,
                    'base_confidence': base_conf  # This is now the lowered confidence
                }
                self.logger.debug(
                    f"[DROP QID:{query_id}] Fallback pattern matched: Type={op_type}, Regex='{pattern_regex}', BaseConf={base_conf:.2f}")
                break

        if not matched_op_details:
            self.logger.warning(
                f"[DROP QID:{query_id}] No fallback operation pattern reliably matched for question: '{query[:70]}...'. Defaulting to entity span on full query.")
            return {'status': 'success',
                    'operation': OP_ENTITY_SPAN,
                    'args': {'entity': query.strip().rstrip('?'),
                             'query_keywords': self._get_query_keywords(query_lower, query_id)},
                    'confidence': 0.15,  # MODIFIED: Very low confidence for ultimate fallback
                    'rationale': 'Defaulting to entity span extraction due to no clear operational pattern match.'}
        try:
            args = matched_op_details['arg_func'](matched_op_details['match_obj'])
            if not args or not any(v for v in args.values() if v is not None):  # Ensure args are meaningful
                self.logger.warning(
                    f"[DROP QID:{query_id}] Argument function for pattern '{matched_op_details['pattern_regex']}' returned empty or None arguments: {args}")
                return {'status': 'success', 'operation': OP_ENTITY_SPAN,
                        'args': {'entity': query.strip().rstrip('?'),
                                 'query_keywords': self._get_query_keywords(query_lower, query_id)},
                        'confidence': 0.15,  # MODIFIED: Very low confidence
                        'rationale': 'Args extraction failed for matched pattern, defaulting to entity span.'}

            self.logger.debug(f"[DROP QID:{query_id}] Fallback Extracted Raw Args: {args}")
            args['query_keywords'] = self._get_query_keywords(query_lower, query_id)

            for key_to_refine in ['entity', 'entity_desc', 'entity1', 'entity2']:
                if key_to_refine in args and isinstance(args[key_to_refine], str):
                    args[key_to_refine] = self._refine_extracted_entity(args[key_to_refine],
                                                                        query_doc=self.nlp(query) if self.nlp else None,
                                                                        query_id=query_id)
            if matched_op_details['type'] == OP_EXTREME_VALUE_NUMERIC and args.get(
                    'direction') == 'specific_event' and args.get('verb_action'):
                current_entity_desc = args.get('entity_desc', '')
                verb_action = args.get('verb_action', '')
                unit = args.get('unit', '')
                if current_entity_desc and verb_action and unit:
                    args['entity_desc'] = f"{current_entity_desc} {verb_action} {unit}"
                elif current_entity_desc and verb_action:  # If unit is missing, still combine entity and verb
                    args['entity_desc'] = f"{current_entity_desc} {verb_action}"

            if 'temporal_constraint' not in args or not args['temporal_constraint']:
                temporal_constraint_found = None
                for keyword in temporal_keywords:
                    if keyword in query_lower:
                        temporal_constraint_found = keyword
                        break
                if temporal_constraint_found:
                    args['temporal_constraint'] = temporal_constraint_found
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Detected temporal constraint via keywords: {temporal_constraint_found}")

            final_confidence = matched_op_details['base_confidence']  # This is already the lowered confidence

            if matched_op_details['type'] == OP_EXTREME_VALUE_NUMERIC and 'direction' not in args:
                self.logger.debug(
                    f"[DROP QID:{query_id}] No explicit direction for OP_EXTREME_VALUE_NUMERIC, defaulting to 'value_of_event' or 'total' if query implies sum")
                if "total" in query_lower or "sum of" in query_lower:
                    args['direction'] = 'total'
                else:
                    args['direction'] = 'value_of_event'
            self.logger.info(
                f"[DROP QID:{query_id}] Fallback Final Extracted Operation: {matched_op_details['type']}, Args: {args}, Confidence: {final_confidence:.2f}")
            return {
                'status': 'success', 'operation': matched_op_details['type'],
                'args': args, 'confidence': final_confidence,  # Return the lowered base_confidence
                'rationale': f"Fallback matched {matched_op_details['type']} pattern."}
        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error during fallback argument processing for pattern '{matched_op_details.get('pattern_regex', 'UNKNOWN')}': {e}")
            operation_type_on_error = matched_op_details.get('type') if matched_op_details else None
            return {'status': 'error', 'operation': operation_type_on_error,
                    'confidence': 0.05,
                    'rationale': f'Error processing arguments from fallback: {e}'}  # MODIFIED CONFIDENCE

    def _get_query_keywords(self, query_lower: str, query_id: str) -> List[str]:
        """Helper to extract keywords using KeyBERT or spaCy."""
        query_keywords = []
        if kw_model:
            try:
                query_keywords = [kw[0] for kw in kw_model.extract_keywords(query_lower, top_n=5, stop_words='english')]
                self.logger.debug(f"[DROP QID:{query_id}] Keywords via KeyBERT: {query_keywords}")
            except Exception as ke_err:
                self.logger.warning(
                    f"[DROP QID:{query_id}] KeyBERT keyword extraction failed: {ke_err}. Falling back to spaCy.")
                kw_model_available_for_this_call = False  # pylint: disable=unused-variable

        if not query_keywords and self.nlp:  # Fallback if KeyBERT fails or not available
            doc = self.nlp(query_lower)
            query_keywords = [token.lemma_ for token in doc if
                              token.pos_ in ('NOUN', 'VERB', 'PROPN', 'ADJ') and not token.is_stop]
            self.logger.debug(f"[DROP QID:{query_id}] Keywords via spaCy: {query_keywords}")
        elif not query_keywords:
            self.logger.debug(
                f"[DROP QID:{query_id}] No keyword extraction method available or query yielded no keywords.")
            # Basic split as ultimate fallback
            query_keywords = [word for word in query_lower.split() if len(word) > 2]

        return list(set(query_keywords))[:7]  # Return unique, top 7

    def _refine_extracted_entity(self, entity_text: str, query_doc: Optional[Any], query_id: str) -> str:
        """Refines an extracted entity string using NLP if available."""
        if not self.nlp or not query_doc:  # query_doc is Optional[spacy.tokens.Doc]
            return entity_text.strip()

        # Attempt to expand a short entity to a full NER span from the query if it's a substring
        for ent in query_doc.ents:
            if entity_text.lower() in ent.text.lower() and len(ent.text) > len(entity_text):
                # Check if the NER type is reasonable (avoiding DATE, TIME, CARDINAL etc. for general entities)
                if ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Refined entity text from '{entity_text}' to NER span '{ent.text}' (Label: {ent.label_})")
                    return ent.text.strip()

        # Fallback: simple cleaning
        # Remove leading "the", "a", "an" and possessives like "'s" if they are at the very end
        refined = re.sub(r"^(the|a|an)\s+", "", entity_text, flags=re.IGNORECASE).strip()
        refined = re.sub(r"\s*'s$", "", refined).strip()  # Remove trailing 's
        refined = refined.rstrip('?.!,')  # Remove trailing punctuation

        # If cleaning results in empty, revert to original (stripped)
        if not refined and entity_text.strip():
            return entity_text.strip()

        return refined if refined else entity_text.strip()

    def _find_entities_in_passage(self, passage: str, entity: str, query_id: str, query_keywords: List[str] = None) -> List[str]:
        """
        Extract entity spans from passage using spaCy NER, dependency parsing, and refined matching.
        """
        if not self.nlp:
            self.logger.warning(f"[DROP QID:{query_id}] spaCy not available. Falling back to regex-based entity extraction.")
            return self._fallback_entity_extraction(passage, entity, query_id)
        if not entity or not isinstance(entity, str):
            self.logger.warning(f"[DROP QID:{query_id}] Invalid entity provided for extraction: {entity}")
            return []

        try:
            doc = self.nlp(passage)
            entity_lower = entity.lower().strip()
            entity_clean = re.sub(r"^(the|a|an|his|her|its|their)\s+|\s+(?:of|in|on|at|from|with)$", "", entity_lower).strip()
            if not entity_clean:
                entity_clean = entity_lower

            spans = []
            query_keywords = query_keywords or []
            query_keywords_set = set(query_keywords)

            matched_indices = set()
            for token in doc:
                token_text_lower = token.text.lower()
                token_lemma_lower = token.lemma_.lower()
                if token.i not in matched_indices and (token_text_lower == entity_clean or token_lemma_lower == entity_clean):
                    start_idx = token.i
                    end_idx = token.i + 1
                    while start_idx > 0 and doc[start_idx-1].pos_ == 'PROPN' and doc[start_idx-1].dep_ in ['compound','flat'] and (start_idx-1) not in matched_indices:
                        start_idx -= 1
                    while end_idx < len(doc) and doc[end_idx].pos_ == 'PROPN' and doc[end_idx].dep_ in ['compound','flat','appos'] and end_idx not in matched_indices:
                        end_idx += 1

                    current_span = doc[start_idx:end_idx].text
                    if start_idx not in matched_indices:
                        spans.append(current_span)
                        matched_indices.update(range(start_idx, end_idx))
                        self.logger.debug(f"[DROP QID:{query_id}] Found entity via Direct/Lemma Match: '{current_span}'")

            for ent in doc.ents:
                match_found = False
                if entity_clean in ent.text.lower():
                    match_found = True
                elif 'who' in query_id.lower() and ent.label_ == 'PERSON': match_found = True
                elif 'team' in query_id.lower() and ent.label_ == 'ORG': match_found = True
                elif 'where' in query_id.lower() and ent.label_ == 'GPE': match_found = True

                if match_found:
                    overlap = False
                    for i in range(ent.start, ent.end):
                        if i in matched_indices:
                            overlap = True
                            break
                    if not overlap:
                        spans.append(ent.text)
                        matched_indices.update(range(ent.start, ent.end))
                        self.logger.debug(f"[DROP QID:{query_id}] Found entity via NER ({ent.label_}): '{ent.text}'")

            if not spans and query_keywords_set:
                for token in doc:
                    token_text_lower = token.text.lower()
                    if token.pos_ in ['NOUN', 'PROPN'] and token.i not in matched_indices:
                        token_context_words = {token_text_lower, token.lemma_.lower()}
                        if not token.head.is_stop: token_context_words.add(token.head.lemma_.lower())
                        token_context_words.update(child.lemma_.lower() for child in token.children if not child.is_stop)

                        if query_keywords_set.intersection(token_context_words):
                            start_idx = token.i
                            end_idx = token.i + 1
                            while start_idx > 0 and doc[start_idx-1].pos_ == 'PROPN' and doc[start_idx-1].dep_ in ['compound','flat'] and (start_idx-1) not in matched_indices:
                                start_idx -= 1
                            while end_idx < len(doc) and doc[end_idx].pos_ == 'PROPN' and doc[end_idx].dep_ in ['compound','flat','appos'] and end_idx not in matched_indices:
                                end_idx += 1
                            current_span = doc[start_idx:end_idx].text
                            spans.append(current_span)
                            matched_indices.update(range(start_idx, end_idx))
                            self.logger.debug(f"[DROP QID:{query_id}] Found entity via Dependency/Keyword Match: '{current_span}'")

            unique_spans = list(dict.fromkeys(spans))
            filtered_spans = [s for s in unique_spans if len(s.strip()) > 1 or s.lower() == entity_clean]

            self.logger.debug(f"[DROP QID:{query_id}] Final extracted spans for '{entity}': {filtered_spans}")
            return filtered_spans

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error extracting entities for '{entity}': {str(e)}")
            return []

    def _fallback_entity_extraction(self, passage: str, entity: str, query_id: str) -> List[str]:
        """
        Fallback entity extraction using regex when spaCy is unavailable.
        """
        if not entity or not isinstance(entity, str): return []
        try:
            entity_pattern = r'\b' + re.escape(entity.strip()) + r'\b'
            spans = re.findall(entity_pattern, passage, re.IGNORECASE)
            unique_spans = list(dict.fromkeys(spans))
            self.logger.debug(f"[DROP QID:{query_id}] Fallback regex extracted spans for '{entity}': {unique_spans}")
            return unique_spans
        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Fallback entity extraction failed for entity '{entity}': {str(e)}")
            return []

    def _find_numbers_in_passage(self, passage: str, query_id: str) -> List[Union[int, float]]:
        """
        Extract numbers from passage using regex and spaCy.
        """
        numbers = []
        try:
            number_pattern = r'(?<![\w\d.])-?\b\d+(?:,\d{3})*(?:\.\d+)?\b(?![\w\d.])'
            matches = re.findall(number_pattern, passage)
            found_by_regex = set()
            for m in matches:
                try:
                    num_str = m.replace(',', '')
                    num = float(num_str) if '.' in num_str else int(num_str)
                    numbers.append(num)
                    found_by_regex.add(m)
                except ValueError:
                    self.logger.debug(f"[DROP QID:{query_id}] Regex match '{m}' is not a valid number.")
            unique_regex_nums = sorted(list(set(numbers)))
            self.logger.debug(f"[DROP QID:{query_id}] Found numbers via regex: {unique_regex_nums}")

            if self.nlp:
                doc = self.nlp(passage)
                spacy_numbers = []
                processed_spacy_indices = set()
                for ent in doc.ents:
                    ent_indices = set(range(ent.start, ent.end))
                    if ent_indices.intersection(processed_spacy_indices):
                        continue

                    if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                        if ent.text in found_by_regex:
                            processed_spacy_indices.update(ent_indices)
                            continue

                        try:
                            num_str = ent.text.replace(',', '').replace('$', '').replace('%', '').strip()
                            if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                                num = float(num_str) if '.' in num_str else int(num_str)
                                spacy_numbers.append(num)
                                processed_spacy_indices.update(ent_indices)
                                self.logger.debug(f"[DROP QID:{query_id}] Found number via spaCy NER ({ent.label_}): {num} from '{ent.text}'")
                            else:
                                self.logger.debug(f"[DROP QID:{query_id}] spaCy NER entity '{ent.text}' ({ent.label_}) not parsed as plain number after cleaning.")
                        except ValueError:
                            self.logger.debug(f"[DROP QID:{query_id}] spaCy NER entity '{ent.text}' ({ent.label_}) caused ValueError during parsing.")

                numbers.extend(spacy_numbers)

            unique_numbers = sorted(list(set(numbers)))
            self.logger.debug(f"[DROP QID:{query_id}] Final unique numbers found: {unique_numbers}")
            return unique_numbers

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error extracting numbers: {str(e)}")
            return []

    def _find_dates_in_passage(self, passage: str, query_id: str) -> List[Dict[str, str]]:
        """
        Extract dates from passage using dateutil, regex, and spaCy.
        """
        dates = []
        try:
            candidates = []
            if self.nlp:
                doc = self.nlp(passage)
                for ent in doc.ents:
                    if ent.label_ == 'DATE':
                        candidates.append(ent.text.strip())
                        self.logger.debug(f"[DROP QID:{query_id}] Found date candidate via spaCy NER (DATE): '{ent.text}'")
                    elif ent.label_ == 'CARDINAL' and re.fullmatch(r'(?:19|20)\d{2}', ent.text):
                        candidates.append(ent.text.strip())
                        self.logger.debug(f"[DROP QID:{query_id}] Found date candidate via spaCy NER (CARDINAL as Year): '{ent.text}'")

            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b(19|20)\d{2}\b'
            ]
            existing_candidates_lower = {c.lower() for c in candidates}
            for pattern in date_patterns:
                try:
                    matches = re.findall(pattern, passage, re.IGNORECASE)
                    for match in matches:
                        match_text = match[0] if isinstance(match, tuple) else match
                        match_text = match_text.strip()
                        if match_text.lower() not in existing_candidates_lower:
                            candidates.append(match_text)
                            existing_candidates_lower.add(match_text.lower())
                            self.logger.debug(f"[DROP QID:{query_id}] Found date candidate via Regex: '{match_text}'")
                except re.error as re_err:
                    self.logger.error(f"[DROP QID:{query_id}] Regex error for pattern '{pattern}': {re_err}")

            parsed_dates = set()
            unique_candidates = list(dict.fromkeys(candidates))
            self.logger.debug(f"[DROP QID:{query_id}] Unique date candidates to parse: {unique_candidates}")

            for text in unique_candidates:
                if not text: continue
                try:
                    if re.fullmatch(r'(19|20)\d{2}', text):
                        parsed = datetime(year=int(text), month=1, day=1)
                        self.logger.debug(f"[DROP QID:{query_id}] Parsing year-only candidate '{text}' as {parsed.date()}")
                    else:
                        parsed = date_parser.parse(text, fuzzy=False, dayfirst=False)
                        self.logger.debug(f"[DROP QID:{query_id}] Parsed candidate '{text}' as {parsed.date()}")

                    year = str(parsed.year)
                    month = str(parsed.month)
                    day = str(parsed.day)

                    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31 and 1000 <= int(year) <= 3000):
                        self.logger.warning(f"[DROP QID:{query_id}] Parsed date '{text}' resulted in invalid components: Y={year}, M={month}, D={day}. Skipping.")
                        continue

                    date_tuple = (year, month, day)
                    if date_tuple not in parsed_dates:
                        dates.append({'day': day, 'month': month, 'year': year})
                        parsed_dates.add(date_tuple)
                        self.logger.debug(f"[DROP QID:{query_id}] Added unique parsed date: {date_tuple} from candidate '{text}'")
                    else:
                        self.logger.debug(f"[DROP QID:{query_id}] Skipping duplicate parsed date {date_tuple} from candidate '{text}'")

                except (ValueError, OverflowError, TypeError) as parse_err:
                    self.logger.debug(f"[DROP QID:{query_id}] Failed to parse date candidate '{text}' with dateutil: {parse_err}")
                    continue

            self.logger.info(f"[DROP QID:{query_id}] Extracted {len(dates)} unique dates: {dates}")
            return dates

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error extracting dates: {str(e)}")
            return []

    def execute_count(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute COUNT operation for DROP queries.
        Uses rule-extracted entity and temporal_constraint if available.
        """
        try:
            entity = args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity:
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'No entity provided for count'}  # Return 0 as int

            spans = self._find_entities_in_passage(context, entity, query_id, query_keywords)

            if temporal_constraint and spans and self.nlp:
                filtered_spans = []
                try:
                    doc = self.nlp(context)
                    span_indices = defaultdict(list)
                    # Ensure spans set for regex is not empty and contains non-empty strings
                    valid_regex_spans = [re.escape(s) for s in set(spans) if s and s.strip()]
                    if not valid_regex_spans:
                        self.logger.debug(
                            f"[DROP QID:{query_id}] No valid spans for temporal regex in COUNT. Spans: {spans}")
                    else:
                        for m in re.finditer(r'\b(' + '|'.join(valid_regex_spans) + r')\b', context, re.IGNORECASE):
                            span_indices[m.group(0).lower()].append(m.start())

                    processed_indices = set()
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
                        match_temporal = False
                        if temporal_constraint == "first half":
                            if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower: match_temporal = True
                        elif temporal_constraint == "second half":
                            if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower: match_temporal = True
                        elif temporal_constraint in sent_text_lower:
                            match_temporal = True

                        if match_temporal:
                            for span_text in set(spans):  # Iterate over original unique spans
                                if not span_text or not span_text.strip(): continue  # Skip empty
                                span_lower = span_text.lower()
                                for start_index in span_indices.get(span_lower, []):
                                    if sent.start_char <= start_index < sent.end_char and start_index not in processed_indices:
                                        filtered_spans.append(span_text)
                                        processed_indices.add(start_index)

                    spans = filtered_spans  # Update spans with temporally filtered ones
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}'. Found {len(spans)} temporally relevant spans for COUNT.")
                except Exception as temp_err:
                    self.logger.error(f"[DROP QID:{query_id}] Error during temporal filtering for COUNT: {temp_err}")
                    # Fallback to count of non-temporally filtered spans, but lower confidence
                    count = len(spans)  # Original spans before filtering error
                    return {'type': 'number', 'value': count, 'confidence': 0.3,
                            'rationale': f'Temporal filtering failed, count is from all spans for {entity}'}

            count = len(spans)
            confidence = 0.85 if spans else 0.4
            self.logger.info(
                f"[DROP QID:{query_id}] COUNT operation: Entity='{entity}', TempConstraint='{temporal_constraint}', Final Count={count}, Conf={confidence:.2f}")
            # MODIFICATION: Return count as int
            return {'type': 'number', 'value': count, 'confidence': confidence}

        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error in COUNT operation for entity '{args.get('entity')}': {str(e)}")
            # MODIFICATION: Return 0 as int for error
            return {'type': 'number', 'value': 0, 'confidence': 0.0, 'error': f'Exception in count: {str(e)}'}

    def execute_extreme_value(self, args: Dict[str, Any], context: str, query: str, query_id: str) -> Dict[str, Any]:
        """
        Execute EXTREME_VALUE operation for DROP queries.
        Uses rule-extracted entity and temporal_constraint if available.
        Handles queries expecting a span (e.g., "who threw the longest pass") or sometimes a number.
        If direction is 'total' and a numerical result seems implied, it sums.
        """
        try:
            entity_desc = args.get('entity_desc') or args.get('entity')
            direction_arg = args.get('direction', 'longest')  # Store original arg
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity_desc:
                err_type = 'spans' if 'who' in query.lower() or 'which' in query.lower() else 'number'
                err_val = [] if err_type == 'spans' else 0
                return {'type': err_type, 'value': err_val, 'confidence': 0.1,
                        'error': 'Missing entity description for extreme_value'}

            value_span_pairs = self._find_values_and_spans(context, entity_desc, query_id, query_keywords)

            if temporal_constraint and value_span_pairs and self.nlp:
                # (Keep existing temporal filtering logic from source [649] - [655])
                # Ensure value_span_pairs is updated with filtered_pairs
                filtered_pairs = []
                try:
                    doc = self.nlp(context)
                    for value, span in value_span_pairs:
                        span_lower = span.lower()
                        sent_found_with_temporal = False
                        for sent in doc.sents:
                            if span_lower in sent.text.lower():
                                sent_text_lower = sent.text.lower()
                                match_temporal = False
                                if temporal_constraint == "first half":
                                    if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower: match_temporal = True
                                elif temporal_constraint == "second half":
                                    if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower: match_temporal = True
                                elif temporal_constraint in sent_text_lower:
                                    match_temporal = True
                                if match_temporal:
                                    sent_found_with_temporal = True
                                    break
                        if sent_found_with_temporal:
                            filtered_pairs.append((value, span))
                    value_span_pairs = filtered_pairs
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}' to EXTREME_VALUE: {len(value_span_pairs)} pairs remain.")
                except Exception as temp_err:
                    self.logger.error(
                        f"[DROP QID:{query_id}] Error during temporal filtering for EXTREME_VALUE: {temp_err}")

            if not value_span_pairs:
                self.logger.warning(
                    f"[DROP QID:{query_id}] No relevant values/spans found for '{entity_desc}' for EXTREME_VALUE (after filtering).")
                err_type = 'spans' if 'who' in query.lower() or 'which' in query.lower() else 'number'
                err_val = [] if err_type == 'spans' else 0
                return {'type': err_type, 'value': err_val, 'confidence': 0.2,
                        'error': f"No relevant values/spans found for '{entity_desc}'"}

            direction_to_use = direction_arg.lower()

            extreme_val_numeric: Optional[Union[int, float]] = None
            associated_spans_for_value: List[str] = []

            valid_numeric_pairs = [(val, span) for val, span in value_span_pairs if isinstance(val, (int, float))]

            if valid_numeric_pairs:  # Proceed if we have numeric values to work with
                if direction_to_use == 'total':
                    # If 'total' and query doesn't imply span (e.g., not "who scored total..."), sum numbers
                    if not (
                            'who' in query.lower() or 'which' in query.lower() or 'what player' in query.lower() or 'what team' in query.lower()):
                        extreme_val_numeric = sum(pair[0] for pair in valid_numeric_pairs)
                        self.logger.debug(
                            f"[DROP QID:{query_id}] EXTREME_VALUE: Direction 'total' interpreted as sum: {extreme_val_numeric}")
                    else:  # 'total' with 'who/which' is ambiguous for this function, might default to span logic for 'longest'
                        self.logger.warning(
                            f"[DROP QID:{query_id}] EXTREME_VALUE: 'total' with 'who/which' is ambiguous, proceeding as 'longest' for span.")
                        selector = max
                        extreme_val_numeric = selector(pair[0] for pair in valid_numeric_pairs)
                elif direction_to_use in ['longest', 'highest', 'most', 'last']:
                    selector = max
                    extreme_val_numeric = selector(pair[0] for pair in valid_numeric_pairs)
                elif direction_to_use in ['shortest', 'lowest', 'least', 'first']:
                    selector = min
                    extreme_val_numeric = selector(pair[0] for pair in valid_numeric_pairs)
                else:
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Unknown direction '{direction_to_use}' for EXTREME_VALUE, defaulting to 'longest'.")
                    selector = max
                    extreme_val_numeric = selector(pair[0] for pair in valid_numeric_pairs)

                if extreme_val_numeric is not None:
                    if isinstance(extreme_val_numeric, float) and extreme_val_numeric.is_integer():
                        extreme_val_numeric = int(extreme_val_numeric)
                    # Find spans associated with this determined extreme_val_numeric
                    associated_spans_for_value = [span for val, span in valid_numeric_pairs if
                                                  val is not None and abs(val - extreme_val_numeric) < 1e-6]

            # Determine return type and value based on query and findings
            query_lower = query.lower()
            return_type: str
            final_value: Any
            confidence = 0.8 if (extreme_val_numeric is not None or associated_spans_for_value) else 0.55

            if 'who' in query_lower or 'which' in query_lower or 'what team' in query_lower or 'what player' in query_lower:
                return_type = 'spans'
                # If we found an extreme_val_numeric, use its associated spans.
                # Otherwise, if no numeric values, perhaps just return all unique spans found for the entity_desc.
                if associated_spans_for_value:
                    final_value = list(dict.fromkeys(associated_spans_for_value))[:1]  # Top one associated span
                else:  # Fallback: if no numbers were involved or no specific span for the number
                    all_spans_for_desc = list(dict.fromkeys([s for v, s in value_span_pairs]))
                    final_value = all_spans_for_desc[:1] if all_spans_for_desc else []
                if not final_value: confidence = 0.3  # Lower confidence if no specific span found
            elif extreme_val_numeric is not None:  # Query implies a number and we found one
                return_type = 'number'
                final_value = extreme_val_numeric
            else:  # Fallback if no clear number or span direction
                self.logger.warning(
                    f"[DROP QID:{query_id}] EXTREME_VALUE: Could not determine clear numeric or span answer. Entity: {entity_desc}, Direction: {direction_arg}")
                # Default to trying to return a span if any were found, else empty
                all_spans_for_desc = list(dict.fromkeys([s for v, s in value_span_pairs]))
                if all_spans_for_desc:
                    return_type = 'spans'
                    final_value = all_spans_for_desc[:1]
                    confidence = 0.4
                else:
                    return_type = 'number'  # Or 'error'
                    final_value = 0
                    confidence = 0.2
                    return {'type': return_type, 'value': final_value, 'confidence': confidence,
                            'error': f"No values or spans found for {entity_desc}"}

            self.logger.info(
                f"[DROP QID:{query_id}] EXTREME_VALUE (Type: {return_type}): Entity='{entity_desc}', Direction='{direction_arg}', CalculatedExtremeNum={extreme_val_numeric}, FinalResultValue={final_value}, Conf={confidence:.2f}")
            return {'type': return_type, 'value': final_value, 'confidence': confidence}

        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error in EXTREME_VALUE for entity '{args.get('entity_desc')}': {str(e)}")
            err_type = 'spans' if 'who' in query.lower() or 'which' in query.lower() else 'number'
            err_val = [] if err_type == 'spans' else 0
            return {'type': err_type, 'value': err_val, 'confidence': 0.0,
                    'error': f'Exception in extreme_value: {str(e)}'}

    def execute_extreme_value_numeric(self, args: Dict[str, Any], context: str, query: str, query_id: str) -> Dict[
        str, Any]:
        """
        Execute EXTREME_VALUE_NUMERIC operation for DROP queries.
        Handles queries expecting a numerical value.
        If direction is 'total', it sums relevant values.
        If direction is 'specific_event' or 'value_of_event', it tries to find a single relevant value.
        Otherwise, it applies min/max based on 'longest', 'shortest', etc.
        """
        try:
            entity_desc = args.get('entity_desc') or args.get('entity')
            # Default to 'value_of_event' if no specific direction like longest/shortest/total is provided
            direction_arg = args.get('direction', 'value_of_event')
            unit = args.get('unit')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity_desc:
                self.logger.warning(f"[DROP QID:{query_id}] Missing entity_desc for EXTREME_VALUE_NUMERIC.")
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'Missing entity description for extreme_value_numeric'}

            # _find_values_and_spans or a more targeted _find_associated_numbers should be used.
            # Given entity_desc can now be more specific (e.g., "Morris gain yards"),
            # _find_associated_numbers might be more appropriate here.
            # Let's assume _find_associated_numbers is designed to leverage the refined entity_desc.
            # relevant_numbers = self._find_associated_numbers(context, entity_desc, query_id, query_keywords)

            # Using _find_values_and_spans for now as per existing structure, then extracting numbers.
            # This might need further refinement if _find_values_and_spans isn't precise enough.
            value_span_pairs = self._find_values_and_spans(context, entity_desc, query_id, query_keywords)

            relevant_numbers = [val for val, span in value_span_pairs if isinstance(val, (int, float))]

            if temporal_constraint and relevant_numbers and self.nlp:
                # Temporally filter the numbers if applicable
                # This requires knowing which number came from which part of the context,
                # so filtering value_span_pairs first is better.
                filtered_value_span_pairs = []
                try:
                    doc = self.nlp(context)
                    for value, span_text in value_span_pairs:  # Use original pairs for context checking
                        if not isinstance(value, (int, float)):  # only consider numeric values for filtering here
                            continue
                        # Logic to check if the span_text (origin of value) is within a temporally relevant sentence
                        span_text_lower = span_text.lower()
                        sent_found_with_temporal = False
                        for sent in doc.sents:
                            if span_text_lower in sent.text.lower():
                                sent_text_lower_content = sent.text.lower()
                                match_temporal = False
                                if temporal_constraint == "first half":
                                    if "1st quarter" in sent_text_lower_content or "2nd quarter" in sent_text_lower_content: match_temporal = True
                                elif temporal_constraint == "second half":
                                    if "3rd quarter" in sent_text_lower_content or "4th quarter" in sent_text_lower_content: match_temporal = True
                                elif temporal_constraint in sent_text_lower_content:
                                    match_temporal = True
                                if match_temporal:
                                    sent_found_with_temporal = True
                                    break
                        if sent_found_with_temporal:
                            filtered_value_span_pairs.append((value, span_text))

                    relevant_numbers = [val for val, _ in filtered_value_span_pairs if isinstance(val, (int, float))]
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}' to EXTREME_VALUE_NUMERIC. {len(relevant_numbers)} numbers remain: {relevant_numbers}")
                except Exception as temp_err:
                    self.logger.error(
                        f"[DROP QID:{query_id}] Error during temporal filtering for EXTREME_VALUE_NUMERIC: {temp_err}")

            if not relevant_numbers:
                self.logger.warning(
                    f"[DROP QID:{query_id}] No relevant numeric values found for '{entity_desc}' (Unit: {unit}) for EXTREME_VALUE_NUMERIC (after filtering).")
                return {'type': 'number', 'value': 0, 'confidence': 0.2,
                        'error': f"No relevant numeric values found for '{entity_desc}' (Unit: {unit})"}

            direction_to_use = direction_arg.lower()
            extreme_value_result: Union[int, float, None] = None  # Initialize to None

            if direction_to_use == 'total':
                extreme_value_result = sum(relevant_numbers)
                self.logger.debug(f"[DROP QID:{query_id}] Direction 'total': sum = {extreme_value_result}")
            elif direction_to_use in ['longest', 'highest', 'most', 'last']:
                extreme_value_result = max(relevant_numbers)
            elif direction_to_use in ['shortest', 'lowest', 'least', 'first']:
                extreme_value_result = min(relevant_numbers)
            elif direction_to_use in ['specific_event', 'value_of_event']:
                # For a specific event, we expect one primary value.
                # If _find_associated_numbers (or _find_values_and_spans) was precise due to a refined entity_desc,
                # relevant_numbers should ideally contain few, highly relevant numbers.
                if len(relevant_numbers) == 1:
                    extreme_value_result = relevant_numbers[0]
                elif relevant_numbers:
                    # Heuristic: if multiple numbers, could take the first, largest, or log warning.
                    # Taking the first one found that's associated with the more specific entity_desc.
                    # This part might need more sophisticated disambiguation if many numbers are still found.
                    extreme_value_result = relevant_numbers[0]
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Multiple numbers ({relevant_numbers}) found for 'specific_event' direction for '{entity_desc}'. Defaulting to first: {extreme_value_result}.")
                else:  # Should be caught by 'if not relevant_numbers:' above
                    pass
            else:
                self.logger.warning(
                    f"[DROP QID:{query_id}] Unknown direction '{direction_to_use}' (original: '{direction_arg}') for EXTREME_VALUE_NUMERIC. Attempting to use first relevant number.")
                if relevant_numbers:
                    extreme_value_result = relevant_numbers[0]

            if extreme_value_result is None:
                self.logger.error(
                    f"[DROP QID:{query_id}] Could not determine extreme/specific value for '{entity_desc}' (Direction: {direction_to_use}). Numbers found: {relevant_numbers}")
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': f"Could not resolve value for {direction_to_use}."}

            if isinstance(extreme_value_result, float) and extreme_value_result.is_integer():
                extreme_value_result = int(extreme_value_result)

            confidence = 0.8 if relevant_numbers else 0.5  # Adjusted confidence based on finding numbers
            self.logger.info(
                f"[DROP QID:{query_id}] EXTREME_VALUE_NUMERIC: Entity='{entity_desc}', Unit='{unit}', Direction='{direction_arg}', ResultValue={extreme_value_result}, Conf={confidence:.2f}")
            return {'type': 'number', 'value': extreme_value_result, 'confidence': confidence}

        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error in EXTREME_VALUE_NUMERIC for entity '{args.get('entity_desc')}': {str(e)}")
            return {'type': 'number', 'value': 0, 'confidence': 0.0,
                    'error': f'Exception in extreme_value_numeric: {str(e)}'}

    def _find_values_and_spans(self, context: str, entity_desc: str, query_id: str,
                               query_keywords: Optional[List[str]] = None) -> List[
        Tuple[Optional[Union[int, float]], str]]:
        """
        Helper to find numbers (values) and their original text spans, attempting to associate
        them with the entity_desc using query_keywords for better precision.
        Query_keywords should ideally include lemmatized components of entity_desc (noun, verbs, adjs).
        """
        pairs: List[Tuple[Optional[Union[int, float]], str]] = []
        if not self.nlp:
            self.logger.warning(
                f"[DROP QID:{query_id}] spaCy unavailable. _find_values_and_spans will be less accurate.")
            # Basic regex fallback to find any number and associate with any entity_desc mention
            # This is very imprecise.
            numbers_in_context = self._find_numbers_in_passage(context, query_id)
            if entity_desc.lower() in context.lower() and numbers_in_context:
                # Associate all numbers with the first mention of entity_desc (highly heuristic)
                for num in numbers_in_context:
                    pairs.append((num, str(num)))  # Span is just the number itself
            return list(set(pairs))  # Return unique pairs

        doc = self.nlp(context)

        # Prepare keywords from entity_desc for matching
        # These keywords should ideally include verbs if entity_desc is action-specific
        # e.g., if entity_desc = "Tony Romo throw touchdown passes", keywords could include "romo", "throw", "pass", "touchdown"
        processed_entity_desc_keywords = set()
        if query_keywords:  # query_keywords are already lemmatized and filtered
            processed_entity_desc_keywords.update(kw.lower() for kw in query_keywords)
        else:  # Fallback if no pre-processed keywords
            # Simple tokenization and lemmatization of entity_desc
            desc_doc = self.nlp(entity_desc.lower())
            processed_entity_desc_keywords.update(
                token.lemma_ for token in desc_doc if
                not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']
            )

        if not processed_entity_desc_keywords:
            self.logger.warning(
                f"[DROP QID:{query_id}] No keywords derived from entity_desc: '{entity_desc}'. Value association will be weak.")
            # As a last resort, use all nouns/propns from entity_desc
            desc_doc = self.nlp(entity_desc.lower())
            processed_entity_desc_keywords.update(
                token.lemma_ for token in desc_doc if token.pos_ in ['NOUN', 'PROPN']
            )

        self.logger.debug(
            f"[DROP QID:{query_id}] _find_values_and_spans: entity_desc='{entity_desc}', keywords for matching: {processed_entity_desc_keywords}")

        # Iterate through sentences to find numbers and check association
        for sent in doc.sents:
            sent_lemmas = {token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct}

            # Check if the sentence seems relevant to the entity_desc based on keyword overlap
            # Require a significant overlap of keywords to consider the sentence relevant
            # This threshold might need tuning.
            keyword_overlap_count = len(processed_entity_desc_keywords.intersection(sent_lemmas))
            relevance_threshold = 1 if len(
                processed_entity_desc_keywords) <= 2 else 2  # Need at least 1 or 2 keywords matching

            if keyword_overlap_count >= relevance_threshold:
                self.logger.debug(
                    f"[DROP QID:{query_id}] Relevant sentence for '{entity_desc}': '{sent.text[:100]}...' (Overlap: {keyword_overlap_count})")
                # Find numbers (CARDINAL, QUANTITY, etc.) within this relevant sentence
                for ent in sent.ents:  # Iterate entities within the relevant sentence
                    if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                        num_val = None
                        try:
                            num_str = ent.text.replace(',', '').replace('$', '').replace('%', '').strip()
                            if re.fullmatch(r'-?\d+(\.\d+)?', num_str):  # Check if it's a valid number string
                                num_val = float(num_str) if '.' in num_str else int(num_str)
                        except ValueError:
                            continue  # Skip if not a parseable number

                        if num_val is not None:
                            # Further check: is this number related to the *specific* entity_desc?
                            # For now, if sentence is relevant and contains a number, we associate it.
                            # More advanced logic could check dependency paths between the number and keywords here.
                            pairs.append((num_val, ent.text))  # Store the original text span of the number
                            self.logger.debug(
                                f"[DROP QID:{query_id}] Associated value={num_val} (from span='{ent.text}') with entity_desc='{entity_desc}' in relevant sentence.")

                # Also check for token.like_num in relevant sentences if NER missed it
                for token in sent:
                    if token.like_num and not token.is_punct and not token.is_stop:
                        is_part_of_ner_num = any(token.i >= ent.start and token.i < ent.end for ent in sent.ents if
                                                 ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT'])
                        if is_part_of_ner_num:
                            continue  # Already processed as part of an entity

                        num_val = None
                        try:
                            num_str = token.text.replace(',', '').strip()
                            if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                                num_val = float(num_str) if '.' in num_str else int(num_str)
                        except ValueError:
                            continue

                        if num_val is not None:
                            pairs.append((num_val, token.text))
                            self.logger.debug(
                                f"[DROP QID:{query_id}] Associated token.like_num value={num_val} (from token='{token.text}') with entity_desc='{entity_desc}' in relevant sentence.")

        # Deduplicate pairs based on (value, original_span_text)
        # Using a set of tuples to ensure uniqueness
        unique_pairs_set = set()
        final_pairs: List[Tuple[Optional[Union[int, float]], str]] = []
        for val, span_str in pairs:
            if (val, span_str) not in unique_pairs_set:
                final_pairs.append((val, span_str))
                unique_pairs_set.add((val, span_str))

        if not final_pairs:
            self.logger.warning(
                f"[DROP QID:{query_id}] No specific numbers found associated with '{entity_desc}' using keyword sentence filtering. Falling back to finding any number in context if entity_desc is generally mentioned.")
            # Fallback: if entity_desc (e.g. player name) is in context, get all numbers from context - less precise.
            # This fallback might be too broad and re-introduce the "165" problem if not careful.
            # For now, let's restrict this fallback. If keywords didn't find anything, it means low relevance.
            # Consider just returning empty if no specific association.

        self.logger.debug(
            f"[DROP QID:{query_id}] Found {len(final_pairs)} value/span pairs for '{entity_desc}': {final_pairs}")
        return final_pairs

    def execute_difference(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute DIFFERENCE operation for DROP queries.
        Uses rule-extracted entities and temporal_constraint if available.
        Updated to infer missing field goal values for ambiguous data.
        """
        try:
            entity1_desc = args.get('entity1') or args.get('entity')
            entity2_desc = args.get('entity2')
            # attr1 = args.get('attr1', 'longest') # These seem unused in the current logic but kept for context
            # attr2 = args.get('attr2', 'shortest')
            # entity1 = args.get('entity1') if 'entity1' in args else args.get('entity') # Original entity names
            # entity2 = args.get('entity2') if 'entity2' in args else args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity1_desc:
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'No primary entity provided for difference'}  # Return 0

            # The logic around attr1, attr2, entity1, entity2 in source [696]-[698] to form entity1_desc and entity2_desc
            # seems correct for interpreting complex difference queries. We directly use entity1_desc and entity2_desc.

            numbers1 = self._find_associated_numbers(context, entity1_desc, query_id, query_keywords)
            numbers2 = self._find_associated_numbers(context, entity2_desc, query_id,
                                                     query_keywords) if entity2_desc else []

            # (Keep existing logic for inferring field goal values source [699]-[700])
            if not numbers1 and "field goal" in entity1_desc.lower():
                numbers1 = [3.0]
                self.logger.debug(f"[DROP QID:{query_id}] Inferred field goal yardage of 3 for {entity1_desc}")
            if not numbers2 and entity2_desc and "field goal" in entity2_desc.lower():
                numbers2 = [3.0]
                self.logger.debug(f"[DROP QID:{query_id}] Inferred field goal yardage of 3 for {entity2_desc}")

            if temporal_constraint and self.nlp:
                # (Keep existing temporal filtering logic from source [701]-[706])
                # Ensure numbers1 and numbers2 are updated correctly
                filtered_numbers1 = []
                filtered_numbers2 = []
                try:
                    doc = self.nlp(context)
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
                        match_temporal = False
                        if temporal_constraint == "first half":
                            if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower: match_temporal = True
                        elif temporal_constraint == "second half":
                            if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower: match_temporal = True
                        elif temporal_constraint in sent_text_lower:
                            match_temporal = True

                        if match_temporal:
                            sent_numbers1 = self._find_associated_numbers(sent.text, entity1_desc, query_id,
                                                                          query_keywords)
                            filtered_numbers1.extend(sent_numbers1)
                            if entity2_desc:
                                sent_numbers2 = self._find_associated_numbers(sent.text, entity2_desc, query_id,
                                                                              query_keywords)
                                filtered_numbers2.extend(sent_numbers2)

                    numbers1 = list(set(filtered_numbers1))  # Use unique numbers after filtering
                    numbers2 = list(set(filtered_numbers2)) if entity2_desc else []
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}' to DIFFERENCE: Nums1={numbers1}, Nums2={numbers2}")
                except Exception as temp_err:
                    self.logger.error(
                        f"[DROP QID:{query_id}] Error during temporal filtering for DIFFERENCE: {temp_err}")

            difference: Optional[Union[int, float]] = None
            confidence = 0.0

            if entity2_desc:  # Difference between two entities
                if not numbers1 or not numbers2:
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Insufficient numbers for DIFFERENCE between '{entity1_desc}' and '{entity2_desc}'. Nums1={numbers1}, Nums2={numbers2}")
                    return {'type': 'number', 'value': 0, 'confidence': 0.2,
                            'error': f"Insufficient distinct numbers found for '{entity1_desc}' ({len(numbers1)}) and '{entity2_desc}' ({len(numbers2)})"}  # Return 0

                # Assuming we take the most prominent (e.g., max) number for each entity if multiple are found
                val1 = max(numbers1) if numbers1 else 0.0  # Default to 0.0 if empty after filtering
                val2 = max(numbers2) if numbers2 else 0.0  # Default to 0.0 if empty

                difference = abs(val1 - val2)
                confidence = 0.75
                self.logger.debug(
                    f"[DROP QID:{query_id}] DIFFERENCE between entities: {entity1_desc}({val1}) vs {entity2_desc}({val2}) = {difference}")
            else:  # Difference within a single entity (e.g., max - min)
                if len(numbers1) < 2:
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Insufficient numbers ({len(numbers1)}) for DIFFERENCE within '{entity1_desc}'. Nums={numbers1}")
                    return {'type': 'number', 'value': 0, 'confidence': 0.25,
                            'error': f"Found {len(numbers1)} numbers for '{entity1_desc}', need at least 2 for difference."}  # Return 0
                difference = max(numbers1) - min(numbers1)
                confidence = 0.8
                self.logger.debug(
                    f"[DROP QID:{query_id}] DIFFERENCE within entity '{entity1_desc}': max({max(numbers1)}) - min({min(numbers1)}) = {difference}")

            if difference is not None:
                # MODIFICATION: Cast to int if whole number, else float
                final_numeric_value = int(difference) if isinstance(difference,
                                                                    float) and difference.is_integer() else float(
                    difference)
                self.logger.info(
                    f"[DROP QID:{query_id}] DIFFERENCE Result: {final_numeric_value}, Raw Difference: {difference}, Confidence={confidence:.2f}")
                return {'type': 'number', 'value': final_numeric_value, 'confidence': confidence}
            else:  # Should not be reached if logic above is correct, but as a fallback
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'Failed to calculate difference'}  # Return 0

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error executing DIFFERENCE operation: {str(e)}")
            return {'type': 'number', 'value': 0, 'confidence': 0.0,
                    'error': f'Exception in difference: {str(e)}'}  # Return 0

    def execute_temporal_difference(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute TEMPORAL_DIFFERENCE operation for DROP queries.
        Handles queries like "how many years between event1 and event2".
        """
        try:
            entity1 = args.get('entity1')
            entity2 = args.get('entity2')
            query_keywords = args.get('query_keywords', [])

            if not entity1 or not entity2:
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'Missing entities for temporal difference'}  # Return 0

            dates = self._find_dates_in_passage(context, query_id)
            date1_obj, date2_obj = None, None  # Renamed to avoid conflict with 'date' type

            if not self.nlp:
                self.logger.warning(
                    f"[DROP QID:{query_id}] spaCy NLP not available for associating dates with entities in temporal_difference. Result may be less accurate.")
                # Fallback: Try to find any two distinct years if entities cannot be associated
                if len(dates) >= 2:
                    years_found = sorted(list(set(int(d.get('year', 0)) for d in dates if d.get('year'))))
                    if len(years_found) >= 2:
                        date1_obj = {'year': str(years_found[0])}  # Simplistic assignment
                        date2_obj = {'year': str(years_found[-1])}
            else:
                entity1_lemmas = {token.lemma_.lower() for token in self.nlp(entity1)} | set(query_keywords)
                entity2_lemmas = {token.lemma_.lower() for token in self.nlp(entity2)} | set(query_keywords)
                doc = self.nlp(context)

                # Try to associate dates with entities by proximity or co-occurrence in sentences
                # This part can be complex; for a simpler robust version:
                # Find all sentences containing entity1 and dates, then entity2 and dates.

                # Simplified association: find first date near entity1, first near entity2
                # This is a placeholder for more robust association logic.
                # A better approach involves checking sentence co-occurrence and proximity.
                # For brevity, we'll assume _find_dates_in_passage gives relevant dates and pick based on order or simple heuristics.
                # A truly robust solution would require more context or more sophisticated linking.

                # For this example, let's assume if two distinct years are found, we use them.
                # This is a simplification.
                year_candidates_e1 = set()
                year_candidates_e2 = set()

                for date_dict_item in dates:  # Renamed date_dict to date_dict_item
                    date_str_for_check = f"{date_dict_item.get('month', '')}/{date_dict_item.get('day', '')}/{date_dict_item.get('year', '')}"
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
                        # A simple check for co-occurrence in sentence (can be improved with proximity)
                        if date_dict_item.get('year') and (
                                date_str_for_check in sent_text_lower or date_dict_item.get('year') in sent_text_lower):
                            sent_lemmas = {token.lemma_.lower() for token in sent if not token.is_stop}
                            if entity1_lemmas.intersection(sent_lemmas):
                                year_candidates_e1.add(int(date_dict_item['year']))
                            if entity2_lemmas.intersection(sent_lemmas):
                                year_candidates_e2.add(int(date_dict_item['year']))

                sorted_years_e1 = sorted(list(year_candidates_e1))
                sorted_years_e2 = sorted(list(year_candidates_e2))

                if sorted_years_e1 and sorted_years_e2:
                    # Try to find a distinct pair, e.g. earliest for e1, latest for e2 or vice-versa
                    # This logic can be complex depending on query phrasing.
                    # Simplest distinct pair:
                    if sorted_years_e1[0] != sorted_years_e2[0]:
                        date1_obj = {'year': str(sorted_years_e1[0])}
                        date2_obj = {'year': str(sorted_years_e2[0])}
                    elif len(sorted_years_e2) > 1 and sorted_years_e1[0] != sorted_years_e2[1]:
                        date1_obj = {'year': str(sorted_years_e1[0])}
                        date2_obj = {'year': str(sorted_years_e2[1])}
                    elif len(sorted_years_e1) > 1 and sorted_years_e1[1] != sorted_years_e2[0]:
                        date1_obj = {'year': str(sorted_years_e1[1])}
                        date2_obj = {'year': str(sorted_years_e2[0])}
                    # If still no distinct pair, take the first ones even if same, difference will be 0
                    if not (date1_obj and date2_obj):
                        date1_obj = {'year': str(sorted_years_e1[0])}
                        date2_obj = {'year': str(sorted_years_e2[0])}

            if not date1_obj or not date2_obj:
                self.logger.warning(
                    f"[DROP QID:{query_id}] Could not find dates associated with both entities for temporal difference: Date1={date1_obj}, Date2={date2_obj}")
                return {'type': 'number', 'value': 0, 'confidence': 0.2,
                        'error': 'Could not find dates associated with entities'}  # Return 0

            year1 = int(date1_obj.get('year', 0))
            year2 = int(date2_obj.get('year', 0))
            if year1 == 0 or year2 == 0:  # Check if years are validly parsed
                self.logger.warning(
                    f"[DROP QID:{query_id}] Invalid year values for temporal difference: Year1={year1}, Year2={year2}")
                return {'type': 'number', 'value': 0, 'confidence': 0.2,
                        'error': 'Invalid year values for difference'}  # Return 0

            difference = abs(year1 - year2)  # difference is already an int
            confidence = 0.8 if (year1 != 0 and year2 != 0) else 0.3  # Higher confidence if both years are valid

            self.logger.info(
                f"[DROP QID:{query_id}] TEMPORAL_DIFFERENCE: Entity1='{entity1}' ({year1}) vs Entity2='{entity2}' ({year2}) = {difference}, Conf={confidence:.2f}")
            # MODIFICATION: Return difference as int
            return {'type': 'number', 'value': difference, 'confidence': confidence}

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error executing TEMPORAL_DIFFERENCE: {str(e)}")
            # MODIFICATION: Return 0 as int for error
            return {'type': 'number', 'value': 0, 'confidence': 0.0,
                    'error': f'Exception in temporal_difference: {str(e)}'}

    def _find_associated_numbers(self, context: str, entity_desc: str, query_id: str, query_keywords: List[str] = None) -> List[Union[int, float]]:
        """
        Helper to find numbers associated with a specific entity description within the given context.
        """
        values = []
        if not self.nlp:
            all_numbers = self._find_numbers_in_passage(context, query_id)
            if entity_desc.lower() in context.lower():
                self.logger.debug(f"[DROP QID:{query_id}] Fallback association for '{entity_desc}': Found numbers {all_numbers} in context.")
                values = all_numbers
            else:
                self.logger.debug(f"[DROP QID:{query_id}] Fallback association for '{entity_desc}': Entity not found in context.")
            return list(set(values))

        doc = self.nlp(context)
        entity_desc_lemmas = {token.lemma_.lower() for token in self.nlp(entity_desc) if not token.is_stop and token.pos_ != 'DET'}
        query_keywords_set = set(query_keywords or []) | entity_desc_lemmas

        number_entities = [ent for ent in doc.ents if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']]
        potential_num_tokens = [tok for tok in doc if tok.like_num and not tok.is_punct]

        processed_tokens = set()

        for ent in number_entities:
            num_val = None
            try:
                num_str = ent.text.replace(',', '').replace('$', '').replace('%', '').strip()
                if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                    num_val = float(num_str) if '.' in num_str else int(num_str)
            except ValueError: continue

            if num_val is not None:
                window = 5
                start = max(0, ent.start - window)
                end = min(len(doc), ent.end + window)
                associated = False
                for i in range(start, end):
                    if i not in range(ent.start, ent.end):
                        token = doc[i]
                        if token.lemma_.lower() in query_keywords_set and token.pos_ in ['NOUN','PROPN','ADJ','VERB']:
                            associated = True
                            break
                if associated:
                    values.append(num_val)
                    processed_tokens.update(range(ent.start, ent.end))
                    self.logger.debug(f"[DROP QID:{query_id}] Associated NER# {num_val} ('{ent.text}') with '{entity_desc}' via keyword context.")

        for token in potential_num_tokens:
            if token.i in processed_tokens: continue

            num_val = None
            try:
                num_str = token.text.replace(',', '').strip()
                if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                    num_val = float(num_str) if '.' in num_str else int(num_str)
            except ValueError: continue

            if num_val is not None:
                window = 5
                start = max(0, token.i - window)
                end = min(len(doc), token.i + 1 + window)
                associated = False
                for i in range(start, end):
                    if i != token.i:
                        window_token = doc[i]
                        if window_token.lemma_.lower() in query_keywords_set and window_token.pos_ in ['NOUN','PROPN','ADJ','VERB']:
                            associated = True
                            break
                if associated:
                    values.append(num_val)
                    processed_tokens.add(token.i)
                    self.logger.debug(f"[DROP QID:{query_id}] Associated Token# {num_val} ('{token.text}') with '{entity_desc}' via keyword context.")

        unique_values = list(set(values))
        self.logger.debug(f"[DROP QID:{query_id}] Found {len(unique_values)} associated numbers for '{entity_desc}': {unique_values}")
        return unique_values

    def execute_entity_span(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute ENTITY_SPAN operation for DROP queries.
        Uses rule-extracted entity and temporal_constraint if available.
        """
        try:
            entity_desc = args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity_desc:
                return {'type': 'error', 'value': None, 'confidence': 0.1, 'error': 'No entity description provided for span extraction'}

            spans = self._find_entities_in_passage(context, entity_desc, query_id, query_keywords)

            if temporal_constraint and spans and self.nlp:
                filtered_spans = []
                try:
                    doc = self.nlp(context)
                    span_indices = defaultdict(list)
                    for m in re.finditer(r'\b' + '|'.join(re.escape(s) for s in set(spans)) + r'\b', context, re.IGNORECASE):
                        span_indices[m.group(0).lower()].append(m.start())

                    processed_indices = set()
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
                        match_temporal = False
                        if temporal_constraint == "first half":
                            if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower: match_temporal = True
                        elif temporal_constraint == "second half":
                            if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower: match_temporal = True
                        elif temporal_constraint in sent_text_lower:
                            match_temporal = True

                        if match_temporal:
                            for span_text in set(spans):
                                span_lower = span_text.lower()
                                for start_index in span_indices.get(span_lower, []):
                                    if sent.start_char <= start_index < sent.end_char and start_index not in processed_indices:
                                        filtered_spans.append(span_text)
                                        processed_indices.add(start_index)

                    spans = list(dict.fromkeys(filtered_spans))
                    self.logger.debug(f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}' to ENTITY_SPAN: {len(spans)} spans remain.")
                except Exception as temp_err:
                    self.logger.error(f"[DROP QID:{query_id}] Error during temporal filtering for ENTITY_SPAN: {temp_err}")

            confidence = 0.85 if spans else 0.35
            if not spans:
                self.logger.warning(f"[DROP QID:{query_id}] No spans found for ENTITY_SPAN operation on '{entity_desc}'.")

            self.logger.info(f"[DROP QID:{query_id}] ENTITY_SPAN operation: Desc='{entity_desc}', TempConstraint='{temporal_constraint}', ResultSpans={spans[:1]}, Conf={confidence:.2f}")
            return {'type': 'spans', 'value': spans[:1] if spans else [], 'confidence': confidence}

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error executing ENTITY_SPAN operation for '{args.get('entity')}': {str(e)}")
            return {'type': 'error', 'value': None, 'confidence': 0.0, 'error': f'Exception in entity_span: {str(e)}'}

    def execute_date(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute DATE operation for DROP queries.
        Uses rule-extracted entity and temporal_constraint if available.
        """
        try:
            entity_desc = args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity_desc:
                return {'type': 'error', 'value': None, 'confidence': 0.1, 'error': 'No entity description provided for date operation'}

            all_dates = self._find_dates_in_passage(context, query_id)
            if not all_dates:
                self.logger.warning(f"[DROP QID:{query_id}] No dates found in context for DATE operation.")
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.2, 'error': 'No dates found in context.'}

            dates_to_consider = all_dates
            if temporal_constraint and self.nlp:
                filtered_dates = []
                try:
                    doc = self.nlp(context)
                    for date_dict in all_dates:
                        pass
                    if not dates_to_consider:
                        self.logger.warning(f"[DROP QID:{query_id}] No dates remain after temporal filtering for DATE.")
                        return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.15, 'error': 'No dates found after temporal filtering.'}
                except Exception as temp_err:
                    self.logger.error(f"[DROP QID:{query_id}] Error during temporal filtering for DATE: {temp_err}")

            best_date = None
            highest_confidence = 0.4
            min_distance = float('inf')

            if self.nlp:
                doc = self.nlp(context)
                entity_indices = []
                entity_lemmas = {tok.lemma_.lower() for tok in self.nlp(entity_desc)} | set(query_keywords)
                for token in doc:
                    if token.lemma_.lower() in entity_lemmas:
                        entity_indices.append(token.i)

                if not entity_indices:
                    self.logger.warning(f"[DROP QID:{query_id}] Could not find entity description '{entity_desc}' indices in context for date association.")
                    best_date = dates_to_consider[0] if dates_to_consider else None
                else:
                    for date_dict in dates_to_consider:
                        date_strs = {
                            f"{date_dict['month']}/{date_dict['day']}/{date_dict['year']}",
                            f"{date_dict['year']}"
                        }
                        found_date_indices = []
                        for date_str in date_strs:
                            try:
                                for match in re.finditer(re.escape(date_str), context, re.IGNORECASE):
                                    tok_idx = -1
                                    for tok in doc:
                                        if tok.idx >= match.start():
                                            tok_idx = tok.i
                                            break
                                    if tok_idx != -1: found_date_indices.append(tok_idx)
                            except re.error: continue

                        if found_date_indices:
                            current_min_dist = min(abs(e_idx - d_idx) for e_idx in entity_indices for d_idx in found_date_indices)
                            if current_min_dist < min_distance:
                                min_distance = current_min_dist
                                best_date = date_dict
                                proximity_boost = max(0, 0.4 * (1 - min_distance / 50.0))
                                highest_confidence = 0.5 + proximity_boost
                                self.logger.debug(f"[DROP QID:{query_id}] Found closer date {date_dict} for '{entity_desc}', dist={min_distance}, conf={highest_confidence:.2f}")
            else:
                best_date = dates_to_consider[0] if dates_to_consider else None
                self.logger.warning(f"[DROP QID:{query_id}] spaCy unavailable for date association. Returning first date found.")

            if best_date:
                self.logger.info(f"[DROP QID:{query_id}] DATE operation: Desc='{entity_desc}', Associated Date={best_date}, Conf={highest_confidence:.2f}")
                return {'type': 'date', 'value': best_date, 'confidence': highest_confidence}
            else:
                self.logger.warning(f"[DROP QID:{query_id}] DATE operation: No date could be confidently associated with '{entity_desc}'.")
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.2, 'error': 'No date associated with entity description found.'}

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error executing DATE operation for '{args.get('entity')}': {str(e)}")
            return {'type': 'error', 'value': None, 'confidence': 0.0, 'error': f'Exception in date execution: {str(e)}'}

    def _format_drop_answer(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the result of a DROP operation into the standard DROP answer structure.
        Ensures value types are correct and handles potential errors in the result dict.
        """
        answer = {
            'number': "",
            'spans': [],
            'date': {'day': "", 'month': "", 'year': ""},
            'error': None
        }

        try:
            self.logger.debug(f"Formatting DROP answer with result: {result}")

            if result.get('error'):
                answer['error'] = str(result['error'])
                self.logger.warning(f"Formatting DROP answer: Found error in input result: {answer['error']}")
                return answer

            result_type = result.get('type')
            result_value = result.get('value')

            if result_type in ['number', OP_COUNT, OP_DIFFERENCE, OP_EXTREME_VALUE_NUMERIC, OP_TEMPORAL_DIFFERENCE] or \
               (result_type == OP_EXTREME_VALUE and isinstance(result_value, str) and re.fullmatch(r'-?\d+(\.\d+)?', result_value)):
                num_val = self._normalize_drop_number_for_comparison(result_value)
                if num_val is not None:
                    answer['number'] = str(num_val)
                    self.logger.debug(f"Formatted DROP answer as number: {answer['number']}")
                else:
                    answer['error'] = f"Failed to normalize number value: {result_value}"
                    self.logger.warning(f"Formatting DROP number: Normalization failed for value '{result_value}'")

            elif result_type in ['spans', OP_ENTITY_SPAN] or \
                 (result_type == OP_EXTREME_VALUE and isinstance(result_value, list)):
                spans_in = result_value if isinstance(result_value, list) else ([result_value] if result_value is not None else [])
                answer['spans'] = [str(v).strip() for v in spans_in if v is not None and str(v).strip()]
                if not answer['spans'] and result_value is not None:
                    self.logger.warning(f"Formatting DROP spans: Input value '{result_value}' resulted in empty span list after cleaning.")
                self.logger.debug(f"Formatted DROP answer as spans: {answer['spans']}")

            elif result_type == OP_DATE:
                if isinstance(result_value, dict) and all(k in result_value for k in ['day', 'month', 'year']):
                    try:
                        d = int(result_value.get('day','0'))
                        m = int(result_value.get('month','0'))
                        y = int(result_value.get('year','0'))
                        if (0 <= d <= 31) and (0 <= m <= 12) and (1000 <= y <= 3000 or y == 0):
                            answer['date'] = {k: str(v).strip() for k, v in result_value.items() if k in ['day', 'month', 'year']}
                            self.logger.debug(f"Formatted DROP answer as date: {answer['date']}")
                        else:
                            raise ValueError("Invalid date components")
                    except (ValueError, TypeError):
                        self.logger.warning(f"Formatting DROP date: Invalid date components in value '{result_value}'. Setting default.")
                        answer['error'] = f"Invalid date components: {result_value}"
                else:
                    self.logger.warning(f"Formatting DROP date: Invalid date value format received: '{result_value}'. Setting default.")
                    answer['error'] = f"Invalid date value format: {result_value}"
            else:
                error_msg = f"Unsupported or invalid result type '{result_type}' for formatting."
                self.logger.warning(error_msg)
                answer['error'] = error_msg

            if not answer['number'] and not answer['spans'] and not any(answer['date'].values()) and not answer['error']:
                answer['error'] = "Formatted answer is empty."
                self.logger.warning(f"Formatted DROP answer resulted in empty fields for type '{result_type}'. Input value: '{result_value}'")

            self.logger.debug(f"Final formatted DROP answer: {answer}")
            return answer

        except Exception as e:
            self.logger.exception(f"Critical error formatting DROP answer object: {str(e)}")
            return {
                'number': "", 'spans': [], 'date': {'day': "", 'month': "", 'year': ""},
                'error': f"Internal error during answer formatting: {str(e)}"
            }

    def _normalize_drop_number_for_comparison(self, value: Optional[Any]) -> Optional[float]:
        """Normalizes numbers (string, int, float) to float for comparison, handles None and errors."""
        if value is None:
            self.logger.debug("Value is None, returning None for normalization")
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).replace(",", "").strip().lower()
            if not s:
                self.logger.debug(f"Value is empty string after cleaning: '{value}'")
                return None
            words = {"zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0}
            if s in words:
                return words[s]
            if re.fullmatch(r'-?\d+(\.\d+)?', s):
                return float(s)
            self.logger.debug(f"Could not normalize '{value}' to a number.")
            return None
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error normalizing number '{value}': {e}")
            return None