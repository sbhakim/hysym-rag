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
                self.logger.error(
                    f"[DROP QID:{qid}] Context is required for DROP processing but was not provided or is invalid."
                )
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.0,
                        'rationale': 'Missing or invalid context for DROP query.'}
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
            op_args: Dict[str, Any] = {}
            extraction_confidence = 0.5
            rationale = "Operation extracted."

            for rule in self.rules:
                if 'compiled_pattern' not in rule:
                    continue
                match = rule['compiled_pattern'].search(query)
                if not match:
                    continue

                matched_rule = rule
                op_type = rule['type']
                if op_type == OP_COUNT:
                    entity = match.group(1).strip() if match.groups() else None
                    temporal = (match.group(2)
                                if match.lastindex >= 2 and match.group(2)
                                else rule.get('temporal_constraint'))
                    op_args = {
                        'entity': entity,
                        'temporal_constraint': temporal,
                        'query_keywords': []
                    }
                elif op_type in (OP_DIFFERENCE, OP_TEMPORAL_DIFFERENCE):
                    if 'unit_group' in rule:
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
                        entity1 = match.group(1).strip() if match.groups() else None
                        entity2 = match.group(2).strip() if match.lastindex >= 2 else None
                        op_args = {
                            'entity1': entity1,
                            'entity2': entity2,
                            'query_keywords': []
                        }
                elif op_type in (OP_EXTREME_VALUE, OP_EXTREME_VALUE_NUMERIC):
                    direction = match.group(1).strip() if match.groups() else rule.get('direction', 'longest')
                    entity = match.group(2).strip() if match.lastindex >= 2 else None
                    unit = (match.group(1).strip()
                            if 'unit_group' in rule and match.groups()
                            else None)
                    op_args = {
                        'entity_desc': entity,
                        'direction': direction,
                        'unit': unit,
                        'query_keywords': []
                    }
                elif op_type in (OP_ENTITY_SPAN, OP_DATE):
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
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Symbolic path failed during op extraction: "
                        f"{op_result.get('rationale', 'Unknown reason')}"
                    )
                    return {**DEFAULT_DROP_ANSWER, **op_result}

                op_type = op_result['operation']
                op_args = op_result['args']
                extraction_confidence = op_result.get('confidence', 0.5)
                rationale = op_result.get('rationale', 'Operation extracted.')
                self.logger.info(f"[DROP QID:{query_id}] Fallback extraction: Type={op_type}, Args={op_args}")

            # 3. Extract keywords for better entity association
            if not op_args.get('query_keywords'):
                query_lower = query.lower()
                try:
                    if kw_model:
                        kws = kw_model.extract_keywords(query_lower, top_n=5)
                        op_args['query_keywords'] = [kw[0] for kw in kws]
                        self.logger.debug(f"[DROP QID:{query_id}] Keywords via KeyBERT: {op_args['query_keywords']}")
                except Exception:
                    pass
                if not op_args['query_keywords'] and self.nlp:
                    doc = self.nlp(query_lower)
                    op_args['query_keywords'] = [
                        token.lemma_ for token in doc
                        if token.pos_ in ('NOUN', 'VERB', 'PROPN') and not token.is_stop
                    ]
                    self.logger.debug(f"[DROP QID:{query_id}] Keywords via spaCy: {op_args['query_keywords']}")

            # 4. Execute operation
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
                self.logger.error(
                    f"[DROP QID:{query_id}] Unsupported operation type '{op_type}' reached execution stage."
                )
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': extraction_confidence,
                        'rationale': f"Internal error: Unsupported operation type '{op_type}'"}

            # 5. Validate and format
            if not isinstance(execution_result, dict) or 'type' not in execution_result or 'value' not in execution_result:
                self.logger.error(
                    f"[DROP QID:{query_id}] Invalid result format from {op_type} execution: {execution_result}"
                )
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.1,
                        'rationale': f"Internal error: Invalid result format during {op_type} execution."}

            formatted = self._format_drop_answer(execution_result)
            exec_conf = execution_result.get('confidence', 0.5)
            final_conf = round(extraction_confidence * 0.6 + exec_conf * 0.4, 3)

            formatted.update({
                'status': 'success',
                'confidence': final_conf,
                'rationale': f"{rationale} | Execution Conf: {exec_conf:.2f}",
                'type': op_type,
                'value': execution_result['value']
            })

            self.logger.info(
                f"[DROP QID:{query_id}] Successfully processed DROP query. Operation: {op_type}, "
                f"Final Confidence: {final_conf:.2f}"
            )
            return formatted

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Unhandled exception: {e}")
            return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.0,
                    'rationale': f"Unhandled Exception: {e}"}

    def _extract_drop_operation_and_args(self, query: str, context: str, query_id: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        self.logger.debug(
            f"[DROP QID:{query_id}] Fallback Extraction: Attempting to extract operation for query: '{query_lower[:100]}...'")

        # Ensure query_doc is created once if nlp is available
        query_doc_for_refine = self.nlp(query) if self.nlp else None

        # Corrected operation_patterns with is_fallback_arg=True for _refine_extracted_entity
        operation_patterns = [
            (
                r"^how many ([a-z\s]+?) (?:did|was|were) ([\w\s'-]+?) (score|kick|hit|throw|rush for|pass for|gain|intercept|complete|make|lead with|take off)(?:\s+in\s+([\w\s'-]+))?\??$",
                OP_EXTREME_VALUE_NUMERIC,
                lambda m: {'unit': m.group(1).strip(),
                           'entity_desc': self._refine_extracted_entity(f"{m.group(2).strip()} {m.group(3).strip()}", query_doc_for_refine, query_id, True),
                           'verb_action': m.group(3).strip(),
                           'specific_event_modifier': m.group(4).strip() if m.lastindex and m.lastindex >= 4 and m.group(4) else None,
                           'direction': 'specific_event'},
                0.60),
            (
                r"^how many total ([a-z\s]+?) (?:did|were) ([\w\s'-]+?) (?:score|gain|have|intercept|lead with|take off|make|hit|kick|throw|rush for|pass for)\??$",
                OP_EXTREME_VALUE_NUMERIC,
                lambda m: {'unit': m.group(1).strip(),
                           'entity_desc': self._refine_extracted_entity(m.group(2).strip(), query_doc_for_refine, query_id, True),
                           'direction': 'total'},
                0.60),
            (
                r"^how many ([\w\s'-]+?) was the (longest|shortest|highest|lowest|most|least|first|last|final)\s+([\w\s'-]+?)(?: of the game)?\??$",
                OP_EXTREME_VALUE_NUMERIC,
                lambda m: {'unit': m.group(1).strip(),
                           'direction': m.group(2).lower().strip(),
                           'entity_desc': self._refine_extracted_entity(m.group(3).strip(), query_doc_for_refine, query_id, True)},
                0.55),
            (
                r"^(?:how many|number of)\s+([\w\s'-]+?)(?:\s+(?:in|during|for|on)\s+(first half|second half|1st quarter|2nd quarter|3rd quarter|4th quarter|the game|overtime))?(?:\s+were\s+there|\s+did|\s+was|\s+score|\?)?$",
                OP_COUNT,
                lambda m: {'entity': self._refine_extracted_entity(m.group(1).strip(), query_doc_for_refine, query_id, True),
                           'temporal_constraint': m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else None},
                0.45),
            (
                r"(?:difference between|how many more|how many less)\s+([\w\s'-]+?)(?:\s+(?:and|than)\s+([\w\s'-]+?))?\??$",
                OP_DIFFERENCE,
                lambda m: {'entity1': self._refine_extracted_entity(m.group(1).strip(), query_doc_for_refine, query_id, True),
                           'entity2': self._refine_extracted_entity(m.group(2).strip(), query_doc_for_refine, query_id, True) if m.lastindex and m.lastindex >= 2 and m.group(2) else None},
                0.60),
            (
                r"^(?:who|what player|what team|which player|which team)\s+(?:had|made|scored|threw|kicked|ran for|caught)\s+(?:the\s+)?(longest|shortest|highest|lowest|most|least|first|last|final)\s+([\w\s'-]+)\??$",
                OP_EXTREME_VALUE,
                lambda m: {'entity_desc': self._refine_extracted_entity(m.group(2).strip(), query_doc_for_refine, query_id, True),
                           'direction': m.group(1).lower().strip()},
                0.55),
            (
                r"^(?:the\s+)?(longest|shortest|highest|lowest|most|least|first|last|final)\s+([\w\s'-]+)\s+(?:was by whom|was by which player|was by what player)\??$",
                OP_EXTREME_VALUE,
                lambda m: {'entity_desc': self._refine_extracted_entity(m.group(2).strip(), query_doc_for_refine, query_id, True),
                           'direction': m.group(1).lower().strip()},
                0.55),
            (
                r"^how many years between\s+([\w\s'-]+?)\s+and\s+([\w\s'-]+?)\??$",
                OP_TEMPORAL_DIFFERENCE,
                lambda m: {'entity1': self._refine_extracted_entity(m.group(1).strip(), query_doc_for_refine, query_id, True),
                           'entity2': self._refine_extracted_entity(m.group(2).strip(), query_doc_for_refine, query_id, True)},
                0.50),
            (
                r"^(?:when|what date|what year|which year)\s+(?:did|was|is)?\s*(.+?)\??$",
                OP_DATE,
                lambda m: {'entity': self._refine_extracted_entity(m.group(1).strip().rstrip('?'), query_doc_for_refine, query_id, True)},
                0.50),
            (
                r"^(?:what team|which team|name the team|the team that)\s+(.+)\??$",
                OP_ENTITY_SPAN,
                lambda m: {'entity': self._refine_extracted_entity(m.group(1).strip().rstrip('?'), query_doc_for_refine, query_id, True)},
                0.45),
            (
                r"^(?:who|what player|which player|name the player|the player who)\s+(.+)\??$",
                OP_ENTITY_SPAN,
                lambda m: {'entity': self._refine_extracted_entity(m.group(1).strip().rstrip('?'), query_doc_for_refine, query_id, True)},
                0.45),
            (
                r"^(?:what is the score|what was the score)(?:\s+at\s+(?:the\s+)?(end of|start of)?\s*(.*?))?\??$",
                OP_ENTITY_SPAN,
                lambda m: {'entity': self._refine_extracted_entity(f"score {m.group(2).strip() if m.group(2) else m.group(1).strip() if m.group(1) else 'final'}", query_doc_for_refine, query_id, True),
                           'temporal_constraint': m.group(2).strip() if m.group(2) else m.group(1).strip() if m.group(1) else None},
                0.50),
            (
                r"^(?:what|which)\s+(?:is|was|are|were)\s+(?:the\s+)?(.+?)\??$",
                OP_ENTITY_SPAN,
                lambda m: {'entity': self._refine_extracted_entity(m.group(1).strip().rstrip('?'), query_doc_for_refine, query_id, True)},
                0.40),
            (
                r"^(?:who|which|what)\s+(.*)\??$",
                OP_ENTITY_SPAN,
                lambda m: {'entity': self._refine_extracted_entity(m.group(1).strip().rstrip('?'), query_doc_for_refine, query_id, True)},
                0.35),
        ]

        temporal_keywords = ["first half", "second half", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
                             "the game", "overtime"]
        matched_op_details = None

        for pattern_regex, op_type, arg_func, base_conf in operation_patterns:
            match = re.search(pattern_regex, query_lower, re.IGNORECASE)
            if match:
                if op_type == OP_COUNT:
                    potential_units = ["yards", "points", "goals", "tds", "pass", "reception", "run", "score"]
                    action_verbs = ["score", "kick", "hit", "throw", "rush for", "pass for", "gain", "intercept",
                                    "complete", "make", "lead with", "take off"]
                    query_tokens = query_lower.split()
                    is_value_query_structure = any(unit in query_tokens for unit in potential_units) and \
                                              any(verb in query_tokens for verb in action_verbs) and \
                                              ("did" in query_tokens or "was" in query_tokens or "were" in query_tokens)
                    if is_value_query_structure:
                        self.logger.debug(f"[DROP QID:{query_id}] OP_COUNT pattern '{pattern_regex}' matches query '{query_lower}', but structure resembles value extraction. Relying on pattern order.")

                matched_op_details = {
                    'pattern_regex': pattern_regex, 'type': op_type,
                    'arg_func': arg_func, 'match_obj': match,
                    'base_confidence': base_conf
                }
                self.logger.debug(f"[DROP QID:{query_id}] Fallback pattern matched: Type={op_type}, Regex='{pattern_regex}', BaseConf={base_conf:.2f}")
                break

        if not matched_op_details:
            self.logger.warning(f"[DROP QID:{query_id}] No fallback operation pattern reliably matched. Defaulting to entity span on full query.")
            return {'status': 'success', 'operation': OP_ENTITY_SPAN,
                    'args': {'entity': query.strip().rstrip('?'), 'query_keywords': self._get_query_keywords(query_lower, query_id)},
                    'confidence': 0.15,
                    'rationale': 'Defaulting to entity span extraction due to no clear operational pattern match.'}
        try:
            args = matched_op_details['arg_func'](matched_op_details['match_obj'])
            if not args or not any(v for k, v in args.items() if v is not None and k != 'query_keywords'):
                self.logger.warning(f"[DROP QID:{query_id}] Arg func for pattern '{matched_op_details['pattern_regex']}' returned empty/None args: {args}")
                return {'status': 'success', 'operation': OP_ENTITY_SPAN,
                        'args': {'entity': query.strip().rstrip('?'), 'query_keywords': self._get_query_keywords(query_lower, query_id)},
                        'confidence': 0.15,
                        'rationale': 'Args extraction failed for matched pattern, defaulting to entity span.'}

            args['query_keywords'] = self._get_query_keywords(query_lower, query_id)

            for key_to_refine in ['entity', 'entity_desc', 'entity1', 'entity2']:
                if key_to_refine in args and isinstance(args[key_to_refine], str):
                    args[key_to_refine] = self._refine_extracted_entity(
                        args[key_to_refine],
                        query_doc_for_refine,
                        query_id,
                        is_fallback_arg=True
                    )

            if not any(args.get(k) for k in ['entity', 'entity_desc', 'entity1'] if k in args):
                self.logger.warning(f"[DROP QID:{query_id}] Entities became empty after refinement: {args}. Defaulting.")
                return {'status': 'success', 'operation': OP_ENTITY_SPAN,
                        'args': {'entity': query.strip().rstrip('?'), 'query_keywords': self._get_query_keywords(query_lower, query_id)},
                        'confidence': 0.10,
                        'rationale': 'Entities became empty after refinement, defaulting.'}

            if matched_op_details['type'] == OP_EXTREME_VALUE_NUMERIC and args.get('direction') == 'specific_event' and args.get('verb_action'):
                current_entity_desc = args.get('entity_desc', '')
                verb_action = args.get('verb_action', '')
                unit = args.get('unit', '')
                combined_desc = current_entity_desc
                if verb_action and verb_action.lower() not in combined_desc.lower():
                    combined_desc = f"{combined_desc} {verb_action}".strip()
                if unit and unit.lower() not in combined_desc.lower():
                    combined_desc = f"{combined_desc} {unit}".strip()
                args['entity_desc'] = combined_desc

            if 'temporal_constraint' not in args or not args['temporal_constraint']:
                temporal_constraint_found = None
                for keyword in temporal_keywords:
                    if keyword in query_lower:
                        temporal_constraint_found = keyword
                        break
                if temporal_constraint_found:
                    args['temporal_constraint'] = temporal_constraint_found

            final_confidence = matched_op_details['base_confidence']

            if matched_op_details['type'] == OP_EXTREME_VALUE_NUMERIC and 'direction' not in args:
                if "total" in query_lower or "sum of" in query_lower:
                    args['direction'] = 'total'
                else:
                    args['direction'] = 'value_of_event'

            self.logger.info(f"[DROP QID:{query_id}] Fallback Final Extracted Operation: {matched_op_details['type']}, Args: {args}, Confidence: {final_confidence:.2f}")
            return {'status': 'success', 'operation': matched_op_details['type'],
                    'args': args, 'confidence': final_confidence,
                    'rationale': f"Fallback matched {matched_op_details['type']} pattern."}
        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error processing fallback args for pattern '{matched_op_details.get('pattern_regex', 'UNKNOWN')}': {e}")
            return {'status': 'error', 'operation': matched_op_details.get('type') if matched_op_details else None,
                    'confidence': 0.05, 'rationale': f'Error processing arguments from fallback: {e}'}

    def _refine_extracted_entity(self, entity_text: str, query_doc: Optional[Any], query_id: str, is_fallback_arg: bool = False) -> str:
        """
        Refines an extracted entity string using NLP if available, with more care for fallback arguments.
        """
        original_entity_text = entity_text.strip()
        qid_log_prefix = f"[DROP QID:{query_id}] _refine_extracted_entity"

        if not original_entity_text:
            return ""

        # Minimal cleaning first
        refined = re.sub(r"^(the|a|an|any|what|which|who)\s+", "", original_entity_text, flags=re.IGNORECASE).strip()
        refined = re.sub(r"\s*'s$", "", refined).strip()
        refined = refined.rstrip('?.!,:')

        if is_fallback_arg:
            common_prefixes = ["name of the", "the name of the", "the name of", "name of", "scored by", "kicked by",
                               "thrown by", "gained by", "caught by", "number of", "how many total", "how many"]
            common_suffixes = ["was by whom", "was by which player", "was by what player", "did", "was", "is", "were",
                               "score", "kick", "hit", "throw", "rush for", "pass for", "gain", "intercept", "complete",
                               "make", "lead with", "take off", "of the game"]
            for _ in range(2):
                for prefix in common_prefixes:
                    if refined.lower().startswith(prefix.lower()):
                        refined = refined[len(prefix):].strip()
                for suffix in common_suffixes:
                    if refined.lower().endswith(suffix.lower()):
                        refined = refined[:-len(suffix)].strip()
            refined = refined.rstrip('?.!,:')

        if not refined:
            self.logger.debug(f"{qid_log_prefix} Entity '{original_entity_text}' became empty after initial/fallback cleaning, reverting.")
            return original_entity_text

        if not self.nlp:
            self.logger.debug(f"{qid_log_prefix} spaCy unavailable, returning basic cleaned: '{refined}'")
            return refined

        doc_to_refine = self.nlp(refined)

        if is_fallback_arg and len(doc_to_refine) > 4:
            self.logger.debug(f"{qid_log_prefix} Fallback arg '{original_entity_text}' (cleaned: '{refined}') is long, attempting deeper NLP refinement.")
            ner_entities_in_arg = [ent.text.strip() for ent in doc_to_refine.ents
                                   if ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'] and len(ent.text.split()) < 5]
            if ner_entities_in_arg:
                best_ner_arg = max(ner_entities_in_arg, key=len)
                self.logger.debug(f"{qid_log_prefix} Refined fallback arg to internal NER '{best_ner_arg}'")
                return best_ner_arg

            noun_chunks_in_arg = [chunk.text.strip() for chunk in doc_to_refine.noun_chunks
                                  if not all(tok.is_stop or tok.is_punct for tok in chunk) and len(chunk.text.split()) < 5]
            if noun_chunks_in_arg:
                best_noun_chunk_arg = sorted(noun_chunks_in_arg, key=lambda nc: (sum(1 for t in self.nlp(nc) if t.pos_ == 'PROPN'), len(nc)), reverse=True)
                if best_noun_chunk_arg:
                    self.logger.debug(f"{qid_log_prefix} Refined fallback arg to Noun Chunk '{best_noun_chunk_arg[0]}'")
                    return best_noun_chunk_arg[0]
            self.logger.debug(f"{qid_log_prefix} No significant NER/NounChunk refinement for long fallback arg: '{refined}'")

        if query_doc:
            for ent in query_doc.ents:
                if refined.lower() in ent.text.lower() and len(ent.text) > len(refined) and len(ent.text.split()) < 5:
                    if ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                        self.logger.debug(f"{qid_log_prefix} Expanded '{refined}' to query NER span '{ent.text}' (Label: {ent.label_})")
                        return ent.text.strip()
                elif ent.text.lower() in refined.lower() and len(ent.text.split()) < 5 and len(ent.text.lower()) >= 3 and ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    self.logger.debug(f"{qid_log_prefix} Refined '{refined}' by matching query NER span '{ent.text}' (Label: {ent.label_})")
                    return ent.text.strip()

        return refined

    def _get_query_keywords(self, query_lower: str, query_id: str) -> List[str]:
        """
        Helper to extract keywords using KeyBERT or spaCy.
        """
        kws: List[str] = []
        if kw_model:
            try:
                kws = [kw[0] for kw in kw_model.extract_keywords(
                    query_lower, top_n=5, stop_words='english'
                )]
            except Exception:
                pass
        if not kws and self.nlp:
            doc = self.nlp(query_lower)
            kws = [
                token.lemma_ for token in doc
                if token.pos_ in ('NOUN', 'VERB', 'PROPN', 'ADJ') and not token.is_stop
            ]
        if not kws:
            kws = [w for w in query_lower.split() if len(w) > 2]
        return list(dict.fromkeys(kws))[:7]

    def _find_entities_in_passage(self, passage: str, entity: str, query_id: str,
                                  query_keywords: List[str] = None) -> List[str]:
        """
        Extract entity spans from passage using spaCy NER, dependency parsing, and refined matching.
        Enhanced with NER label specificity, scoring, and generic span filtering.
        [FIXED: Changed self.encoder to self.embedder and ensured correct tensor operations for util.cos_sim]
        """
        qid_log_prefix = f"[DROP QID:{query_id}] _find_entities_in_passage" # [cite: 2425]
        if not self.nlp:
            self.logger.warning(f"{qid_log_prefix} spaCy not available. Using fallback regex entity extraction.") # [cite: 2425]
            return self._fallback_entity_extraction(passage, entity, query_id) # [cite: 2425]
        if not entity or not isinstance(entity, str):
            self.logger.warning(f"{qid_log_prefix} Invalid entity provided: {entity}") # [cite: 2427]
            return [] # [cite: 2427]

        try:
            doc = self.nlp(passage) # [cite: 2428]
            entity_lower = entity.lower().strip() # [cite: 2428]
            entity_clean = re.sub(r"^(the|a|an|his|her|its|their|any|some)\s+", "", entity_lower).strip() # [cite: 2428]
            entity_clean = entity_clean.split(" of ")[0].strip() # [cite: 2428]
            if not entity_clean:
                entity_clean = entity_lower # [cite: 2428]

            query_keywords_set = set(kw.lower() for kw in (query_keywords or [])) # [cite: 2429]
            processed_entity_desc_keywords = query_keywords_set # [cite: 2429]
            if not processed_entity_desc_keywords:
                desc_doc = self.nlp(entity_clean) # [cite: 2429]
                processed_entity_desc_keywords.update(
                    token.lemma_ for token in desc_doc # [cite: 2429]
                    if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] # [cite: 2430]
                )
            if not processed_entity_desc_keywords and entity_clean:
                processed_entity_desc_keywords.update(
                    token.lemma_ for token in self.nlp(entity_clean) if token.pos_ in ['NOUN', 'PROPN'] # [cite: 2431]
                )

            candidate_spans_with_scores: List[Tuple[str, int, int, float]] = [] # [cite: 2431]

            # 1. Direct/Lemma Match with NER expansion
            for token in doc:
                token_text_lower = token.text.lower() # [cite: 2431]
                token_lemma_lower = token.lemma_.lower() # [cite: 2432]
                is_core_match = (token_text_lower == entity_clean or # [cite: 2432]
                                 token_lemma_lower == entity_clean or # [cite: 2432]
                                 (len(entity_clean.split()) > 1 and (token_text_lower in entity_clean.split() or token_lemma_lower in entity_clean.split()))) # [cite: 2433]

                if is_core_match:
                    expanded_span_text = token.text # [cite: 2433]
                    score = 0.6 # [cite: 2433]
                    for ent in doc.ents: # [cite: 2434]
                        if token.i >= ent.start and token.i < ent.end: # [cite: 2434]
                            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'NORP'}: # [cite: 2434]
                                expanded_span_text = ent.text # [cite: 2435]
                                score = 0.85 # [cite: 2435]
                                break # [cite: 2435]
                    if expanded_span_text == token.text: # [cite: 2435]
                        for chunk in doc.noun_chunks: # [cite: 2436]
                            if token.i >= chunk.start and token.i < chunk.end: # [cite: 2436]
                                expanded_span_text = chunk.text # [cite: 2436]
                                score = 0.7 # [cite: 2437]
                                break # [cite: 2437]
                    candidate_spans_with_scores.append((expanded_span_text.strip(), token.idx, token.idx + len(expanded_span_text), score)) # [cite: 2437]
                    self.logger.debug(f"{qid_log_prefix} Found potential entity via direct/lemma: '{expanded_span_text}' (Score: {score})") # [cite: 2438]

            # 2. General NER pass
            for ent in doc.ents:
                if ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'] or \
                   (entity_lower.isdigit() and ent.label_ == 'CARDINAL'): # [cite: 2438]
                    score = 0.5 # [cite: 2439]
                    ent_text_lower = ent.text.lower() # [cite: 2439]
                    if entity_clean in ent_text_lower:
                        score += 0.2 # [cite: 2439]
                    # UPDATED BLOCK: Changed self.encoder to self.embedder and ensured tensor operations
                    elif self.embedder and util and processed_entity_desc_keywords: # [cite: 2440]
                        try:
                            # Ensure inputs to embedder are strings and then convert to tensor and move to device
                            entity_lower_emb = self.embedder.encode(entity_lower, convert_to_tensor=True).to(self.device)
                            ent_text_emb = self.embedder.encode(ent.text, convert_to_tensor=True).to(self.device)
                            sim = util.cos_sim(entity_lower_emb, ent_text_emb).item() # [cite: 2440]
                            if sim > 0.6: # [cite: 2440]
                                score += sim * 0.2 # [cite: 2441]
                        except Exception as e_enc:
                            self.logger.warning(f"{qid_log_prefix} Could not compute similarity for NER span '{ent.text}': {e_enc}") # [cite: 2441]
                    # END OF UPDATED BLOCK
                    candidate_spans_with_scores.append((ent.text.strip(), ent.start_char, ent.end_char, score)) # [cite: 2442]
                    self.logger.debug(f"{qid_log_prefix} Found entity via NER ({ent.label_}): '{ent.text}' (Score: {score})") # [cite: 2442]

            # Sort by score (desc) then by start_char (asc)
            candidate_spans_with_scores.sort(key=lambda x: (x[3], -x[1]), reverse=True) # [cite: 2442]

            # Deduplicate based on character indices
            final_spans_text_only = [] # [cite: 2443]
            added_char_indices = set() # [cite: 2443]
            for text, start_char, end_char, score in candidate_spans_with_scores:
                has_major_overlap = False # [cite: 2443]
                for i in range(start_char, end_char): # [cite: 2443]
                    if i in added_char_indices: # [cite: 2444]
                        has_major_overlap = True # [cite: 2444]
                        break # [cite: 2444]
                if not has_major_overlap:
                    final_spans_text_only.append(text) # [cite: 2444]
                    for i in range(start_char, end_char): # [cite: 2445]
                        added_char_indices.add(i) # [cite: 2445]

            # Deduplicate case-insensitive
            seen_final = set() # [cite: 2445]
            deduplicated_final_spans = [s for s in final_spans_text_only if s.lower() not in seen_final and not seen_final.add(s.lower())] # [cite: 2445]

            # Filter out generic spans
            if len(entity_lower) > 3: # [cite: 2446]
                deduplicated_final_spans = [s for s in deduplicated_final_spans if len(s.strip()) > 2 or s.lower() == entity_clean] # [cite: 2446]

            self.logger.debug(f"{qid_log_prefix} Final refined & deduplicated spans for '{entity}': {deduplicated_final_spans}") # [cite: 2446]
            return deduplicated_final_spans # [cite: 2446]

        except Exception as e: # [cite: 2446]
            self.logger.exception(f"{qid_log_prefix} Error extracting entities for '{entity}': {str(e)}") # [cite: 2447]
            return []

    def _fallback_entity_extraction(self, passage: str, entity: str, query_id: str) -> List[str]:
        """
        Fallback entity extraction using regex when spaCy is unavailable.
        """
        if not entity or not isinstance(entity, str):
            return []
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
                        self.logger.debug(
                            f"[DROP QID:{query_id}] Found date candidate via spaCy NER (DATE): '{ent.text}'")
                    elif ent.label_ == 'CARDINAL' and re.fullmatch(r'(?:19|20)\d{2}', ent.text):
                        candidates.append(ent.text.strip())
                        self.logger.debug(
                            f"[DROP QID:{query_id}] Found date candidate via spaCy NER (CARDINAL as Year): '{ent.text}'")

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
                if not text:
                    continue
                try:
                    if re.fullmatch(r'(19|20)\d{2}', text):
                        parsed = datetime(year=int(text), month=1, day=1)
                        self.logger.debug(
                            f"[DROP QID:{query_id}] Parsing year-only candidate '{text}' as {parsed.date()}")
                    else:
                        parsed = date_parser.parse(text, fuzzy=False, dayfirst=False)
                        self.logger.debug(f"[DROP QID:{query_id}] Parsed candidate '{text}' as {parsed.date()}")

                    year = str(parsed.year)
                    month = str(parsed.month)
                    day = str(parsed.day)

                    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31 and 1000 <= int(year) <= 3000):
                        self.logger.warning(
                            f"[DROP QID:{query_id}] Parsed date '{text}' resulted in invalid components: Y={year}, M={month}, D={day}. Skipping.")
                        continue

                    date_tuple = (year, month, day)
                    if date_tuple not in parsed_dates:
                        dates.append({'day': day, 'month': month, 'year': year})
                        parsed_dates.add(date_tuple)
                        self.logger.debug(
                            f"[DROP QID:{query_id}] Added unique parsed date: {date_tuple} from candidate '{text}'")
                    else:
                        self.logger.debug(
                            f"[DROP QID:{query_id}] Skipping duplicate parsed date {date_tuple} from candidate '{text}'")

                except (ValueError, OverflowError, TypeError) as parse_err:
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Failed to parse date candidate '{text}' with dateutil: {parse_err}")
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
                        'error': 'No entity provided for count'}

            spans = self._find_entities_in_passage(context, entity, query_id, query_keywords)

            if temporal_constraint and spans and self.nlp:
                filtered_spans = []
                try:
                    doc = self.nlp(context)
                    span_indices = defaultdict(list)
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
                            if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                match_temporal = True
                        elif temporal_constraint == "second half":
                            if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                match_temporal = True
                        elif temporal_constraint in sent_text_lower:
                            match_temporal = True

                        if match_temporal:
                            for span_text in set(spans):
                                if not span_text or not span_text.strip():
                                    continue
                                span_lower = span_text.lower()
                                for start_index in span_indices.get(span_lower, []):
                                    if sent.start_char <= start_index < sent.end_char and start_index not in processed_indices:
                                        filtered_spans.append(span_text)
                                        processed_indices.add(start_index)

                    spans = filtered_spans
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}'. Found {len(spans)} temporally relevant spans for COUNT.")
                except Exception as temp_err:
                    self.logger.error(f"[DROP QID:{query_id}] Error during temporal filtering for COUNT: {temp_err}")
                    count = len(spans)
                    return {'type': 'number', 'value': count, 'confidence': 0.3,
                            'rationale': f'Temporal filtering failed, count is from all spans for {entity}'}

            count = len(spans)
            confidence = 0.85 if spans else 0.4
            self.logger.info(
                f"[DROP QID:{query_id}] COUNT operation: Entity='{entity}', TempConstraint='{temporal_constraint}', Final Count={count}, Conf={confidence:.2f}")
            return {'type': 'number', 'value': count, 'confidence': confidence}

        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error in COUNT operation for entity '{args.get('entity')}': {str(e)}")
            return {'type': 'number', 'value': 0, 'confidence': 0.0, 'error': f'Exception in count: {str(e)}'}

    def execute_extreme_value(self, args: Dict[str, Any], context: str, query: str, query_id: str) -> Dict[str, Any]:
        """
        Execute EXTREME_VALUE operation for DROP queries.
        Uses rule-extracted entity and temporal_constraint if available.
        Handles queries expecting a span (e.g., "who threw the longest pass") or sometimes a number.
        """
        try:
            entity_desc = args.get('entity_desc') or args.get('entity')
            direction_arg = args.get('direction', 'longest')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity_desc:
                err_type = 'spans' if 'who' in query.lower() or 'which' in query.lower() else 'number'
                err_val = [] if err_type == 'spans' else 0
                return {'type': err_type, 'value': err_val, 'confidence': 0.1,
                        'error': 'Missing entity description for extreme_value'}

            value_span_pairs = self._find_values_and_spans(context, entity_desc, query_id, query_keywords)

            if temporal_constraint and value_span_pairs and self.nlp:
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
                                    if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                        match_temporal = True
                                elif temporal_constraint == "second half":
                                    if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                        match_temporal = True
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

            if valid_numeric_pairs:
                if direction_to_use == 'total':
                    if not (
                            'who' in query.lower() or 'which' in query.lower() or 'what player' in query.lower() or 'what team' in query.lower()):
                        extreme_val_numeric = sum(pair[0] for pair in valid_numeric_pairs)
                        self.logger.debug(
                            f"[DROP QID:{query_id}] EXTREME_VALUE: Direction 'total' interpreted as sum: {extreme_val_numeric}")
                    else:
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
                    associated_spans_for_value = [span for val, span in valid_numeric_pairs if
                                                  val is not None and abs(val - extreme_val_numeric) < 1e-6]

            query_lower = query.lower()
            return_type: str
            final_value: Any
            confidence = 0.8 if (extreme_val_numeric is not None or associated_spans_for_value) else 0.55

            if 'who' in query_lower or 'which' in query_lower or 'what team' in query_lower or 'what player' in query_lower:
                return_type = 'spans'
                if associated_spans_for_value:
                    final_value = list(dict.fromkeys(associated_spans_for_value))[:1]
                else:
                    all_spans_for_desc = list(dict.fromkeys([s for v, s in value_span_pairs]))
                    final_value = all_spans_for_desc[:1] if all_spans_for_desc else []
                if not final_value:
                    confidence = 0.3
            elif extreme_val_numeric is not None:
                return_type = 'number'
                final_value = extreme_val_numeric
            else:
                self.logger.warning(
                    f"[DROP QID:{query_id}] EXTREME_VALUE: Could not determine clear numeric or span answer. Entity: {entity_desc}, Direction: {direction_arg}")
                all_spans_for_desc = list(dict.fromkeys([s for v, s in value_span_pairs]))
                if all_spans_for_desc:
                    return_type = 'spans'
                    final_value = all_spans_for_desc[:1]
                    confidence = 0.4
                else:
                    return_type = 'number'
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
        Enhanced with improved disambiguation for 'specific_event' and confidence scoring.
        """
        qid_log_prefix = f"[DROP QID:{query_id}] execute_extreme_value_numeric"
        try:
            entity_desc = args.get('entity_desc') or args.get('entity')
            direction_arg = args.get('direction', 'value_of_event')
            unit = args.get('unit')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity_desc:
                self.logger.warning(f"{qid_log_prefix} Missing entity_desc.")
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'Missing entity_desc for extreme_value_numeric'}

            value_span_pairs = self._find_values_and_spans(context, entity_desc, query_id, query_keywords)
            numeric_value_span_pairs = [(val, span) for val, span in value_span_pairs if isinstance(val, (int, float))]

            if temporal_constraint and numeric_value_span_pairs and self.nlp:
                filtered_numeric_pairs = []
                doc = self.nlp(context)
                for value, span_text in numeric_value_span_pairs:
                    span_text_lower = span_text.lower()
                    sent_found_with_temporal = False
                    for sent in doc.sents:
                        if span_text_lower in sent.text.lower():
                            sent_text_lower_content = sent.text.lower()
                            match_temporal = False
                            if temporal_constraint == "first half" and (
                                    "1st quarter" in sent_text_lower_content or "2nd quarter" in sent_text_lower_content):
                                match_temporal = True
                            elif temporal_constraint == "second half" and (
                                    "3rd quarter" in sent_text_lower_content or "4th quarter" in sent_text_lower_content):
                                match_temporal = True
                            elif temporal_constraint and temporal_constraint != "the game" and temporal_constraint in sent_text_lower_content:
                                match_temporal = True
                            elif temporal_constraint == "the game":
                                match_temporal = True
                            if match_temporal:
                                sent_found_with_temporal = True
                                break
                    if sent_found_with_temporal:
                        filtered_numeric_pairs.append((value, span_text))

                if numeric_value_span_pairs and not filtered_numeric_pairs:
                    self.logger.warning(
                        f"{qid_log_prefix} Temporal constraint '{temporal_constraint}' removed all numeric candidates. Original pairs: {numeric_value_span_pairs}")
                numeric_value_span_pairs = filtered_numeric_pairs
                self.logger.debug(
                    f"{qid_log_prefix} After temporal constraint '{temporal_constraint}', {len(numeric_value_span_pairs)} numeric pairs remain.")

            relevant_numbers = [val for val, _ in numeric_value_span_pairs]

            if not relevant_numbers:
                self.logger.warning(
                    f"{qid_log_prefix} No relevant numeric values found for '{entity_desc}' (Unit: {unit}) after all filtering.")
                return {'type': 'number', 'value': 0, 'confidence': 0.2,
                        'error': f"No relevant numeric values found for '{entity_desc}' (Unit: {unit})"}

            direction_to_use = direction_arg.lower()
            extreme_value_result: Optional[Union[int, float]] = None
            confidence = 0.75

            if direction_to_use == 'total':
                extreme_value_result = sum(relevant_numbers)
                confidence = 0.85
            elif direction_to_use in ['longest', 'highest', 'most', 'last']:
                extreme_value_result = max(relevant_numbers)
            elif direction_to_use in ['shortest', 'lowest', 'least', 'first']:
                extreme_value_result = min(relevant_numbers)
            elif direction_to_use in ['specific_event', 'value_of_event']:
                if len(relevant_numbers) == 1:
                    extreme_value_result = relevant_numbers[0]
                    confidence = 0.85
                elif relevant_numbers:
                    extreme_value_result = relevant_numbers[0]
                    confidence = 0.60
                    self.logger.warning(
                        f"{qid_log_prefix} Multiple numbers {relevant_numbers} for '{direction_to_use}' on '{entity_desc}'. Defaulted to first. Confidence lowered.")
            else:
                self.logger.warning(
                    f"{qid_log_prefix} Unknown direction '{direction_arg}'. Defaulting to first relevant number.")
                extreme_value_result = relevant_numbers[0] if relevant_numbers else None
                confidence = 0.5

            if extreme_value_result is None:
                self.logger.error(f"{qid_log_prefix} Failed to determine extreme/specific value for '{entity_desc}'.")
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': f"Could not resolve value for {direction_arg}"}

            if isinstance(extreme_value_result, float) and extreme_value_result.is_integer():
                extreme_value_result = int(extreme_value_result)

            self.logger.info(
                f"{qid_log_prefix}: Entity='{entity_desc}', Unit='{unit}', Dir='{direction_arg}', Result={extreme_value_result}, Conf={confidence:.2f}")
            return {'type': 'number', 'value': extreme_value_result, 'confidence': round(confidence, 3)}

        except Exception as e:
            self.logger.exception(f"{qid_log_prefix} Error for entity '{args.get('entity_desc')}': {e}")
            return {'type': 'number', 'value': 0, 'confidence': 0.0, 'error': f'Exception: {e}'}

    def _find_values_and_spans(self, context: str, entity_desc: str, query_id: str,
                               query_keywords: Optional[List[str]] = None) -> List[
        Tuple[Optional[Union[int, float]], str]]:
        """
        Helper to find numbers (values) and their original text spans, attempting to associate
        them with the entity_desc using query_keywords and NLP for better precision.
        [FIXED: Changed access to entity text to be more robust, avoiding direct token.ent_.text if problematic]
        """
        pairs: List[Tuple[Optional[Union[int, float]], str]] = []  #
        qid_log_prefix = f"[DROP QID:{query_id}] _find_values_and_spans"  #

        if not self.nlp:
            self.logger.warning(f"{qid_log_prefix} spaCy unavailable. Fallback will be less accurate.")  #
            numbers_in_context = self._find_numbers_in_passage(context, query_id)  #
            if entity_desc.lower() in context.lower() and numbers_in_context:
                for num in numbers_in_context:
                    pairs.append((num, str(num)))  #
            return list(set(pairs))  #

        doc = self.nlp(context)  #
        entity_desc_lower = entity_desc.lower()  #

        processed_entity_desc_keywords = set(kw.lower() for kw in (query_keywords or []))  #
        if not processed_entity_desc_keywords:
            desc_doc = self.nlp(entity_desc_lower)  #
            processed_entity_desc_keywords.update(
                token.lemma_ for token in desc_doc if  #
                not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']  #
            )
        if not processed_entity_desc_keywords and entity_desc_lower:
            processed_entity_desc_keywords.update(
                token.lemma_ for token in self.nlp(entity_desc_lower) if token.pos_ in ['NOUN', 'PROPN'])  #

        self.logger.debug(
            f"{qid_log_prefix}: entity_desc='{entity_desc}', keywords for matching: {processed_entity_desc_keywords}")  #

        PROXIMITY_WINDOW = 7  #

        # Keep track of tokens already processed as part of an NER entity to avoid double counting
        processed_ner_tokens = set()

        for sent in doc.sents:
            sent_lemmas = {token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct}  #
            keyword_overlap_count = len(processed_entity_desc_keywords.intersection(sent_lemmas))  #
            relevance_threshold = 1 if len(processed_entity_desc_keywords) <= 3 else 2  #

            if keyword_overlap_count >= relevance_threshold:  #
                self.logger.debug(
                    f"{qid_log_prefix} Relevant sentence for '{entity_desc}': '{sent.text[:100]}...' (Overlap: {keyword_overlap_count})")  #

                # First, process recognized NER entities in the sentence
                for ent in sent.ents:
                    if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                        num_val = None
                        original_num_span_text = ent.text  # Use ent.text for the full entity span
                        try:
                            num_str = original_num_span_text.replace(',', '').replace('$', '').replace('%',
                                                                                                       '').strip()  #
                            if re.fullmatch(r'-?\d+(\.\d+)?', num_str):  #
                                num_val = float(num_str) if '.' in num_str else int(num_str)  #
                                # Mark tokens within this entity as processed
                                for i in range(ent.start, ent.end):
                                    processed_ner_tokens.add(i)
                        except ValueError:
                            pass  # num_val remains None

                        if num_val is not None:
                            # Association logic (check proximity to keywords)
                            is_associated = False
                            # Check keywords within the entity itself or in proximity
                            ent_lemmas = {t.lemma_.lower() for t in ent if not t.is_stop and not t.is_punct}
                            if processed_entity_desc_keywords.intersection(ent_lemmas):
                                is_associated = True
                            else:  # Check proximity if not in entity itself
                                window_tokens_indices = set(
                                    range(max(sent.start, ent.start - PROXIMITY_WINDOW), ent.start)) | \
                                                        set(range(ent.end, min(sent.end, ent.end + PROXIMITY_WINDOW)))
                                for token_idx in window_tokens_indices:
                                    if doc[token_idx].lemma_.lower() in processed_entity_desc_keywords:
                                        is_associated = True
                                        break

                            if is_associated:
                                pairs.append((num_val, original_num_span_text))  #
                                self.logger.debug(
                                    f"{qid_log_prefix} Associated NER value={num_val} (from span='{original_num_span_text}') with entity_desc='{entity_desc}' via NER & proximity.")  #

                # Then, process like_num tokens that were NOT part of an already processed NER entity
                for token in sent:
                    if token.i in processed_ner_tokens:  # Skip if already handled by NER
                        continue

                    num_val = None
                    original_num_span_text = token.text  #

                    if token.like_num and not token.is_punct and not token.is_stop:  #
                        try:
                            num_str = token.text.replace(',', '').strip()  #
                            if re.fullmatch(r'-?\d+(\.\d+)?', num_str):  #
                                num_val = float(num_str) if '.' in num_str else int(num_str)  #
                        except ValueError:
                            pass  #

                    if num_val is not None:
                        is_associated = False  #
                        # Check proximity for like_num tokens
                        context_window_start = max(0, token.i - sent.start - PROXIMITY_WINDOW)  #
                        context_window_end = min(len(sent), token.i - sent.start + 1 + PROXIMITY_WINDOW)  #

                        for i in range(context_window_start, context_window_end):  #
                            # Check relative to sentence start
                            if sent[i].i == token.i:  # Don't compare token to itself
                                continue  #
                            if sent[i].lemma_.lower() in processed_entity_desc_keywords:  #
                                is_associated = True  #
                                break  #

                        if is_associated:
                            pairs.append((num_val, original_num_span_text))  #
                            self.logger.debug(
                                f"{qid_log_prefix} Associated like_num value={num_val} (from span='{original_num_span_text}') with entity_desc='{entity_desc}' due to keyword proximity.")  #

        unique_pairs_set = set()  #
        final_pairs: List[Tuple[Optional[Union[int, float]], str]] = []  #
        for val, span_str in pairs:
            if (val, span_str) not in unique_pairs_set:  #
                final_pairs.append((val, span_str))  #
                unique_pairs_set.add((val, span_str))  #

        if not final_pairs:
            self.logger.warning(
                f"{qid_log_prefix} No numbers found specifically associated with '{entity_desc}' using keywords.")  #

        self.logger.debug(
            f"{qid_log_prefix} Found {len(final_pairs)} value/span pairs for '{entity_desc}': {final_pairs}")  #
        return final_pairs

    def execute_difference(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute DIFFERENCE operation for DROP queries.
        Uses rule-extracted entities and temporal_constraint if available.
        """
        try:
            entity1_desc = args.get('entity1') or args.get('entity')
            entity2_desc = args.get('entity2')
            temporal_constraint = args.get('temporal_constraint')
            query_keywords = args.get('query_keywords', [])

            if not entity1_desc:
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'No primary entity provided for difference'}

            numbers1 = self._find_associated_numbers(context, entity1_desc, query_id, query_keywords)
            numbers2 = self._find_associated_numbers(context, entity2_desc, query_id,
                                                     query_keywords) if entity2_desc else []

            if not numbers1 and "field goal" in entity1_desc.lower():
                numbers1 = [3.0]
                self.logger.debug(f"[DROP QID:{query_id}] Inferred field goal yardage of 3 for {entity1_desc}")
            if not numbers2 and entity2_desc and "field goal" in entity2_desc.lower():
                numbers2 = [3.0]
                self.logger.debug(f"[DROP QID:{query_id}] Inferred field goal yardage of 3 for {entity2_desc}")

            if temporal_constraint and self.nlp:
                filtered_numbers1 = []
                filtered_numbers2 = []
                try:
                    doc = self.nlp(context)
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
                        match_temporal = False
                        if temporal_constraint == "first half":
                            if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                match_temporal = True
                        elif temporal_constraint == "second half":
                            if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                match_temporal = True
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

                    numbers1 = list(set(filtered_numbers1))
                    numbers2 = list(set(filtered_numbers2)) if entity2_desc else []
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}' to DIFFERENCE: Nums1={numbers1}, Nums2={numbers2}")
                except Exception as temp_err:
                    self.logger.error(
                        f"[DROP QID:{query_id}] Error during temporal filtering for DIFFERENCE: {temp_err}")

            difference: Optional[Union[int, float]] = None
            confidence = 0.0

            if entity2_desc:
                if not numbers1 or not numbers2:
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Insufficient numbers for DIFFERENCE between '{entity1_desc}' and '{entity2_desc}'. Nums1={numbers1}, Nums2={numbers2}")
                    return {'type': 'number', 'value': 0, 'confidence': 0.2,
                            'error': f"Insufficient distinct numbers found for '{entity1_desc}' ({len(numbers1)}) and '{entity2_desc}' ({len(numbers2)})"}
                val1 = max(numbers1) if numbers1 else 0.0
                val2 = max(numbers2) if numbers2 else 0.0
                difference = abs(val1 - val2)
                confidence = 0.75
                self.logger.debug(
                    f"[DROP QID:{query_id}] DIFFERENCE between entities: {entity1_desc}({val1}) vs {entity2_desc}({val2}) = {difference}")
            else:
                if len(numbers1) < 2:
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Insufficient numbers ({len(numbers1)}) for DIFFERENCE within '{entity1_desc}'. Nums={numbers1}")
                    return {'type': 'number', 'value': 0, 'confidence': 0.25,
                            'error': f"Found {len(numbers1)} numbers for '{entity1_desc}', need at least 2 for difference."}
                difference = max(numbers1) - min(numbers1)
                confidence = 0.8
                self.logger.debug(
                    f"[DROP QID:{query_id}] DIFFERENCE within entity '{entity1_desc}': max({max(numbers1)}) - min({min(numbers1)}) = {difference}")

            if difference is not None:
                final_numeric_value = int(difference) if isinstance(difference,
                                                                    float) and difference.is_integer() else float(
                    difference)
                self.logger.info(
                    f"[DROP QID:{query_id}] DIFFERENCE Result: {final_numeric_value}, Raw Difference: {difference}, Confidence={confidence:.2f}")
                return {'type': 'number', 'value': final_numeric_value, 'confidence': confidence}
            else:
                return {'type': 'number', 'value': 0, 'confidence': 0.1,
                        'error': 'Failed to calculate difference'}

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error executing DIFFERENCE operation: {str(e)}")
            return {'type': 'number', 'value': 0, 'confidence': 0.0,
                    'error': f'Exception in difference: {str(e)}'}

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
                        'error': 'Missing entities for temporal difference'}

            dates = self._find_dates_in_passage(context, query_id)
            date1_obj, date2_obj = None, None

            if not self.nlp:
                self.logger.warning(
                    f"[DROP QID:{query_id}] spaCy NLP not available for associating dates with entities in temporal_difference. Result may be less accurate.")
                if len(dates) >= 2:
                    years_found = sorted(list(set(int(d.get('year', 0)) for d in dates if d.get('year'))))
                    if len(years_found) >= 2:
                        date1_obj = {'year': str(years_found[0])}
                        date2_obj = {'year': str(years_found[-1])}
            else:
                entity1_lemmas = {token.lemma_.lower() for token in self.nlp(entity1)} | set(query_keywords)
                entity2_lemmas = {token.lemma_.lower() for token in self.nlp(entity2)} | set(query_keywords)
                doc = self.nlp(context)
                year_candidates_e1 = set()
                year_candidates_e2 = set()

                for date_dict_item in dates:
                    date_str_for_check = f"{date_dict_item.get('month', '')}/{date_dict_item.get('day', '')}/{date_dict_item.get('year', '')}"
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
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
                    if sorted_years_e1[0] != sorted_years_e2[0]:
                        date1_obj = {'year': str(sorted_years_e1[0])}
                        date2_obj = {'year': str(sorted_years_e2[0])}
                    elif len(sorted_years_e2) > 1 and sorted_years_e1[0] != sorted_years_e2[1]:
                        date1_obj = {'year': str(sorted_years_e1[0])}
                        date2_obj = {'year': str(sorted_years_e2[1])}
                    elif len(sorted_years_e1) > 1 and sorted_years_e1[1] != sorted_years_e2[0]:
                        date1_obj = {'year': str(sorted_years_e1[1])}
                        date2_obj = {'year': str(sorted_years_e2[0])}
                    if not (date1_obj and date2_obj):
                        date1_obj = {'year': str(sorted_years_e1[0])}
                        date2_obj = {'year': str(sorted_years_e2[0])}

            if not date1_obj or not date2_obj:
                self.logger.warning(
                    f"[DROP QID:{query_id}] Could not find dates associated with both entities for temporal difference: Date1={date1_obj}, Date2={date2_obj}")
                return {'type': 'number', 'value': 0, 'confidence': 0.2,
                        'error': 'Could not find dates associated with entities'}

            year1 = int(date1_obj.get('year', 0))
            year2 = int(date2_obj.get('year', 0))
            if year1 == 0 or year2 == 0:
                self.logger.warning(
                    f"[DROP QID:{query_id}] Invalid year values for temporal difference: Year1={year1}, Year2={year2}")
                return {'type': 'number', 'value': 0, 'confidence': 0.2,
                        'error': 'Invalid year values for difference'}

            difference = abs(year1 - year2)
            confidence = 0.8 if (year1 != 0 and year2 != 0) else 0.3

            self.logger.info(
                f"[DROP QID:{query_id}] TEMPORAL_DIFFERENCE: Entity1='{entity1}' ({year1}) vs Entity2='{entity2}' ({year2}) = {difference}, Conf={confidence:.2f}")
            return {'type': 'number', 'value': difference, 'confidence': confidence}

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error executing TEMPORAL_DIFFERENCE: {str(e)}")
            return {'type': 'number', 'value': 0, 'confidence': 0.0,
                    'error': f'Exception in temporal_difference: {str(e)}'}

    def _find_associated_numbers(self, context: str, entity_desc: str, query_id: str,
                                 query_keywords: List[str] = None) -> List[Union[int, float]]:
        """
        Find numbers associated with an entity description in the context.
        """
        qid_log_prefix = f"[DROP QID:{query_id}] _find_associated_numbers"
        values: List[Union[int, float]] = []
        if not self.nlp:
            self.logger.warning(f"{qid_log_prefix} spaCy unavailable. Number association will be basic regex based.")
            all_numbers_in_context = self._find_numbers_in_passage(context, query_id)
            if entity_desc.lower() in context.lower():
                self.logger.debug(
                    f"{qid_log_prefix} Fallback: entity '{entity_desc}' in context, returning all numbers: {all_numbers_in_context}")
                return list(set(all_numbers_in_context))
            return []

        doc = self.nlp(context)
        entity_desc_lemmas = {token.lemma_.lower() for token in self.nlp(entity_desc) if
                              not token.is_stop and token.pos_ != 'DET'}
        query_keywords_set = {kw.lower() for kw in (query_keywords or [])} | entity_desc_lemmas
        if not query_keywords_set and entity_desc:
            query_keywords_set.update(token.lemma_.lower() for token in self.nlp(entity_desc) if
                                      token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'])

        self.logger.debug(
            f"{qid_log_prefix}: entity_desc='{entity_desc}', keywords for association: {query_keywords_set}")

        processed_number_token_indices = set()

        for sent in doc.sents:
            sent_keyword_match = any(tok.lemma_.lower() in query_keywords_set for tok in sent)
            if not sent_keyword_match:
                continue

            self.logger.debug(f"{qid_log_prefix} Analyzing relevant sentence: '{sent.text[:100]}...'")

            for token in sent:
                if token.i in processed_number_token_indices:
                    continue

                num_val: Optional[Union[int, float]] = None
                num_token_span_for_debug = token.text

                if token.ent_type_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                    try:
                        num_str = token.ent_.text.replace(',', '').replace('$', '').replace('%', '').strip()
                        if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                            num_val = float(num_str) if '.' in num_str else int(num_str)
                            num_token_span_for_debug = token.ent_.text
                            for ent_tok_idx in range(token.ent_.start, token.ent_.end):
                                processed_number_token_indices.add(ent_tok_idx)
                    except ValueError:
                        pass

                if num_val is None and token.like_num and not token.is_punct:
                    try:
                        num_str = token.text.replace(',', '').strip()
                        if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                            num_val = float(num_str) if '.' in num_str else int(num_str)
                            processed_number_token_indices.add(token.i)
                    except ValueError:
                        pass

                if num_val is not None:
                    is_strongly_associated = False
                    head_token = token.head
                    if head_token.lemma_.lower() in query_keywords_set:
                        is_strongly_associated = True
                        self.logger.debug(
                            f"{qid_log_prefix} Number '{num_val}' from '{num_token_span_for_debug}' associated with head '{head_token.text}'({head_token.lemma_})")

                    if not is_strongly_associated:
                        for child in token.children:
                            if child.lemma_.lower() in query_keywords_set:
                                is_strongly_associated = True
                                self.logger.debug(
                                    f"{qid_log_prefix} Number '{num_val}' from '{num_token_span_for_debug}' associated with child '{child.text}'({child.lemma_})")
                                break

                    if not is_strongly_associated:
                        window_tokens = [sent[j] for j in range(max(0, token.i - 3), min(len(sent), token.i + 4)) if
                                         j != token.i]
                        if any(wt.lemma_.lower() in query_keywords_set for wt in window_tokens):
                            is_strongly_associated = True
                            self.logger.debug(
                                f"{qid_log_prefix} Number '{num_val}' from '{num_token_span_for_debug}' associated by proximity to keywords.")

                    if is_strongly_associated:
                        values.append(num_val)

        unique_values = sorted(list(set(values)))
        self.logger.debug(
            f"{qid_log_prefix} Found {len(unique_values)} associated numbers for '{entity_desc}': {unique_values}")
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
                return {'type': 'error', 'value': None, 'confidence': 0.1,
                        'error': 'No entity description provided for span extraction'}

            spans = self._find_entities_in_passage(context, entity_desc, query_id, query_keywords)

            if temporal_constraint and spans and self.nlp:
                filtered_spans = []
                try:
                    doc = self.nlp(context)
                    span_indices = defaultdict(list)
                    for m in re.finditer(r'\b' + '|'.join(re.escape(s) for s in set(spans)) + r'\b', context,
                                         re.IGNORECASE):
                        span_indices[m.group(0).lower()].append(m.start())

                    processed_indices = set()
                    for sent in doc.sents:
                        sent_text_lower = sent.text.lower()
                        match_temporal = False
                        if temporal_constraint == "first half":
                            if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                match_temporal = True
                        elif temporal_constraint == "second half":
                            if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                match_temporal = True
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
                    self.logger.debug(
                        f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}' to ENTITY_SPAN: {len(spans)} spans remain.")
                except Exception as temp_err:
                    self.logger.error(
                        f"[DROP QID:{query_id}] Error during temporal filtering for ENTITY_SPAN: {temp_err}")

            confidence = 0.85 if spans else 0.35
            if not spans:
                self.logger.warning(
                    f"[DROP QID:{query_id}] No spans found for ENTITY_SPAN operation on '{entity_desc}'.")

            self.logger.info(
                f"[DROP QID:{query_id}] ENTITY_SPAN operation: Desc='{entity_desc}', TempConstraint='{temporal_constraint}', ResultSpans={spans[:1]}, Conf={confidence:.2f}")
            return {'type': 'spans', 'value': spans[:1] if spans else [], 'confidence': confidence}

        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error executing ENTITY_SPAN operation for '{args.get('entity')}': {str(e)}")
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
                return {'type': 'error', 'value': None, 'confidence': 0.1,
                        'error': 'No entity description provided for date operation'}

            all_dates = self._find_dates_in_passage(context, query_id)
            if not all_dates:
                self.logger.warning(f"[DROP QID:{query_id}] No dates found in context for DATE operation.")
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.2,
                        'error': 'No dates found in context.'}

            dates_to_consider = all_dates
            if temporal_constraint and self.nlp:
                filtered_dates = []
                try:
                    doc = self.nlp(context)
                    for date_dict in all_dates:
                        pass
                    if not dates_to_consider:
                        self.logger.warning(f"[DROP QID:{query_id}] No dates remain after temporal filtering for DATE.")
                        return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.15,
                                'error': 'No dates found after temporal filtering.'}
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
                    self.logger.warning(
                        f"[DROP QID:{query_id}] Could not find entity description '{entity_desc}' indices in context for date association.")
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
                                    if tok_idx != -1:
                                        found_date_indices.append(tok_idx)
                            except re.error:
                                continue

                        if found_date_indices:
                            current_min_dist = min(
                                abs(e_idx - d_idx) for e_idx in entity_indices for d_idx in found_date_indices)
                            if current_min_dist < min_distance:
                                min_distance = current_min_dist
                                best_date = date_dict
                                proximity_boost = max(0, 0.4 * (1 - min_distance / 50.0))
                                highest_confidence = 0.5 + proximity_boost
                                self.logger.debug(
                                    f"[DROP QID:{query_id}] Found closer date {date_dict} for '{entity_desc}', dist={min_distance}, conf={highest_confidence:.2f}")
            else:
                best_date = dates_to_consider[0] if dates_to_consider else None
                self.logger.warning(
                    f"[DROP QID:{query_id}] spaCy unavailable for date association. Returning first date found.")

            if best_date:
                self.logger.info(
                    f"[DROP QID:{query_id}] DATE operation: Desc='{entity_desc}', Associated Date={best_date}, Conf={highest_confidence:.2f}")
                return {'type': 'date', 'value': best_date, 'confidence': highest_confidence}
            else:
                self.logger.warning(
                    f"[DROP QID:{query_id}] DATE operation: No date could be confidently associated with '{entity_desc}'.")
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.2,
                        'error': 'No date associated with entity description found.'}

        except Exception as e:
            self.logger.exception(
                f"[DROP QID:{query_id}] Error executing DATE operation for '{args.get('entity')}': {str(e)}")
            return {'type': 'error', 'value': None, 'confidence': 0.0,
                    'error': f'Exception in date execution: {str(e)}'}

    def _format_drop_answer(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the execution result into the DROP answer structure.
        """
        answer = {
            'number': "", 'spans': [],
            'date': {'day': "", 'month': "", 'year': ""},
            'error': None
        }
        query_id_for_log = result.get('query_id', 'unknown_format_qid')
        log_prefix = f"[DROP QID:{query_id_for_log}] _format_drop_answer"

        try:
            self.logger.debug(f"{log_prefix} Formatting result: {result}")

            if result.get('error'):
                answer['error'] = str(result['error'])
                self.logger.warning(f"{log_prefix} Input result contained error: {answer['error']}")
                return answer

            result_type = result.get('type')
            result_value = result.get('value')

            if result_value is None and result_type not in ['error']:
                answer['error'] = f"No value provided for type '{result_type}' to format."
                self.logger.warning(f"{log_prefix} {answer['error']}")
                return answer

            if result_type in ['number', OP_COUNT, OP_DIFFERENCE, OP_EXTREME_VALUE_NUMERIC, OP_TEMPORAL_DIFFERENCE] or \
                    (result_type == OP_EXTREME_VALUE and isinstance(result_value, (str, int, float)) and str(
                        result_value).strip() and self._normalize_drop_number_for_comparison(
                        str(result_value)) is not None):
                num_val = self._normalize_drop_number_for_comparison(result_value)
                if num_val is not None:
                    answer['number'] = str(int(num_val) if num_val.is_integer() else float(num_val))
                else:
                    answer['error'] = f"Failed to normalize number value: '{result_value}'"
                    self.logger.warning(f"{log_prefix} Number normalization failed for value '{result_value}'")

            elif result_type in ['spans', OP_ENTITY_SPAN] or \
                    (result_type == OP_EXTREME_VALUE and isinstance(result_value, list)):
                spans_in = result_value if isinstance(result_value, list) else (
                    [str(result_value)] if result_value is not None else [])
                seen_spans = set()
                cleaned_spans = []
                for s_val in spans_in:
                    s_strip = str(s_val).strip()
                    if s_strip and s_strip.lower() not in seen_spans:
                        cleaned_spans.append(s_strip)
                        seen_spans.add(s_strip.lower())
                answer['spans'] = cleaned_spans
                if not answer['spans'] and spans_in:
                    self.logger.warning(f"{log_prefix} Input spans '{spans_in}' resulted in empty list after cleaning.")

            elif result_type == OP_DATE:
                if isinstance(result_value, dict) and all(k in result_value for k in ['day', 'month', 'year']):
                    try:
                        d_str = str(result_value.get('day', '')).strip()
                        m_str = str(result_value.get('month', '')).strip()
                        y_str = str(result_value.get('year', '')).strip()
                        final_date = {}
                        if d_str:
                            final_date['day'] = d_str
                        if m_str:
                            final_date['month'] = m_str
                        if y_str:
                            final_date['year'] = y_str
                        if final_date:
                            answer['date'] = {'day': final_date.get('day', ''),
                                              'month': final_date.get('month', ''),
                                              'year': final_date.get('year', '')}
                        else:
                            answer['error'] = f"Date value provided was empty: {result_value}"
                            self.logger.warning(f"{log_prefix} {answer['error']}")
                    except (ValueError, TypeError) as e:
                        answer['error'] = f"Invalid date components in value '{result_value}': {e}"
                        self.logger.warning(f"{log_prefix} {answer['error']}")
                else:
                    answer['error'] = f"Invalid date value format: '{result_value}'"
                    self.logger.warning(f"{log_prefix} {answer['error']}")
            else:
                answer['error'] = f"Unsupported or invalid result type '{result_type}' for formatting."
                self.logger.warning(f"{log_prefix} {answer['error']}")

            if not answer['number'] and not answer['spans'] and not any(answer['date'].values()) and not answer[
                'error']:
                answer['error'] = "Formatted answer is empty despite successful operation type."
                self.logger.warning(f"{log_prefix} {answer['error']} Type: '{result_type}', Value: '{result_value}'")

            self.logger.debug(f"{log_prefix} Final formatted DROP answer: {answer}")
            return answer

        except Exception as e:
            self.logger.exception(f"{log_prefix} Critical error formatting DROP answer object: {e}")
            return {
                'number': "", 'spans': [], 'date': {'day': "", 'month': "", 'year': ""},
                'error': f"Internal error during answer formatting: {e}"
            }

    def _normalize_drop_number_for_comparison(self, value: Optional[Any]) -> Optional[float]:
        """
        Normalizes numbers (string, int, float) to float for comparison, handles None and errors.
        """
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
            words = {"zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0,
                     "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0}
            if s in words:
                return words[s]
            if re.fullmatch(r'-?\d+(\.\d+)?', s):
                return float(s)
            self.logger.debug(f"Could not normalize '{value}' to a number.")
            return None
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error normalizing number '{value}': {e}")
            return None