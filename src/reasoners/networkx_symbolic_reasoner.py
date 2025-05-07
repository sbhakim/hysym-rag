# src/reasoners/networkx_symbolic_reasoner.py

import json
import spacy
import time
import networkx as nx
import logging
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from collections import defaultdict
from datetime import datetime
from dateutil import parser as date_parser
import operator  # For sorting/finding max/min associated spans

from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager

# For improved keyword extraction using KeyBERT
try:
    from keybert import KeyBERT
    kw_model = KeyBERT()
except ImportError:
    logging.getLogger(__name__).warning("KeyBERT not installed. Keyword extraction will use fallback methods.")
    kw_model = None
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to initialize KeyBERT model: {e}")
    kw_model = None

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Operation type constants
OP_COUNT = "count"
OP_EXTREME_VALUE = "extreme_value"
OP_DIFFERENCE = "difference"
OP_ENTITY_SPAN = "entity_span"
OP_DATE = "date"

# Default DROP answer structure
DEFAULT_DROP_ANSWER = {"number": "", "spans": [], "date": {"day": "", "month": "", "year": ""}}

class GraphSymbolicReasoner:
    """
    Enhanced graph-based symbolic reasoner for HySym-RAG.
    Includes logic for text-based QA (HotpotQA) via semantic matching/graph traversal
    and discrete reasoning capabilities for DROP dataset with improved entity extraction.
    """

    def __init__(
            self,
            rules_file: str,
            match_threshold: float = 0.1,
            max_hops: int = 5,  # For graph traversal in text-based reasoning
            embedding_model: str = 'all-MiniLM-L6-v2',
            device: Optional[torch.device] = None,
            dim_manager: Optional[DimensionalityManager] = None
    ):
        self.logger = logger
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.device = device or DeviceManager.get_device()
        self.query_cache = {}  # Cache for query results

        if dim_manager is None:
            self.logger.warning("DimensionalityManager not provided to GraphSymbolicReasoner. Creating a default one.")
            self.dim_manager = DimensionalityManager(target_dim=768, device=self.device)  # Default target_dim
        else:
            self.dim_manager = dim_manager

        try:
            with torch.no_grad():
                self.embedder = SentenceTransformer(embedding_model).to(self.device)
            self.logger.info("Successfully loaded SentenceTransformer model.")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model '{embedding_model}': {e}. Symbolic reasoner embedding features will be impacted.")
            self.embedder = None

        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Successfully loaded spaCy model for GraphSymbolicReasoner.")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model in GraphSymbolicReasoner: {str(e)}. Some NLP functionalities might be affected.")
            self.nlp = None  # Allow fallback if spaCy is not critical for all paths

        self.rules = self._load_and_validate_rules(rules_file=rules_file)
        self.semantic_rule_index, self.rule_embeddings = self._build_semantic_rule_index()
        self.knowledge_graph = self._build_knowledge_graph()

        self.logger.info(f"GraphSymbolicReasoner initialized. Rules loaded: {len(self.rules)}. Rules indexed for semantic match: {len(self.semantic_rule_index)}.")

    def _load_and_validate_rules(self, rules_file: str) -> List[Dict]:
        """
        Load rules from file and validate their structure.
        Supports both HotpotQA and DROP-specific rules with types like 'count', 'extreme_value'.
        """
        if not os.path.exists(rules_file):
            self.logger.warning(f"Rules file {rules_file} does not exist. Returning empty rule list.")
            return []

        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                raw_rules = json.load(f)
            self.logger.info(f"Successfully loaded {len(raw_rules)} raw rules/definitions from {rules_file}")
        except Exception as e:
            self.logger.error(f"Failed to load rules from {rules_file}: {e}. Returning empty rule list.")
            return []

        valid_rules = []
        for rule in raw_rules:
            try:
                # Validate required fields
                if not isinstance(rule, dict):
                    self.logger.debug(f"Skipping invalid rule: not a dictionary")
                    continue
                if 'keywords' not in rule or 'response' not in rule:
                    self.logger.debug(f"Skipping rule with missing keywords or response: {rule}")
                    continue
                # Ensure keywords is a list of strings
                if not isinstance(rule['keywords'], list) or not all(isinstance(k, str) for k in rule['keywords']):
                    self.logger.debug(f"Skipping rule with invalid keywords format: {rule['keywords']}")
                    continue
                # Ensure response is a string
                if not isinstance(rule['response'], str):
                    self.logger.debug(f"Skipping rule with invalid response format: {rule['response']}")
                    continue
                # Optional fields for DROP
                rule.setdefault('confidence', 0.7)
                rule.setdefault('type', 'general')
                valid_rules.append(rule)
            except Exception as e:
                self.logger.debug(f"Error validating rule {rule}: {e}")
                continue

        self.logger.info(f"Loaded {len(valid_rules)} valid rules/definitions out of {len(raw_rules)} raw entries.")
        return valid_rules

    def _build_semantic_rule_index(self) -> Tuple[List[Dict], Optional[torch.Tensor]]:
        """
        Build an index of rules with precomputed embeddings for semantic matching.
        Uses batch processing and torch.no_grad() to optimize GPU usage.
        """
        semantic_rule_index = []
        embeddings = []

        if not self.embedder:
            self.logger.warning("Embedder not available. Semantic rule index will be empty.")
            return semantic_rule_index, None

        try:
            # Prepare all keywords text for batch processing
            keywords_texts = []
            valid_rules = []
            for rule in self.rules:
                keywords_text = " ".join(rule.get('keywords', []))
                if not keywords_text.strip():
                    self.logger.debug(f"Skipping rule with empty keywords: {rule}")
                    continue
                keywords_texts.append(keywords_text)
                rule_copy = rule.copy()
                rule_copy['keywords_set'] = set(rule['keywords'])
                # Use keywords as response tokens if response is trivial (e.g., 'yes')
                response_text = rule['response'].lower()
                if len(response_text.split()) <= 1:
                    rule_copy['response_tokens'] = set(rule['keywords'])
                else:
                    rule_copy['response_tokens'] = set(re.findall(r'\b\w+\b', response_text))

                semantic_rule_index.append(rule_copy)
                valid_rules.append(rule_copy)  # Keep track of rules actually indexed

            if not keywords_texts:
                self.logger.warning("No valid keywords for embedding. Semantic rule index will be empty.")
                return [], None  # Return empty list and None

            # Batch encode embeddings
            with torch.no_grad():
                batch_embeddings = self.embedder.encode(
                    keywords_texts,
                    convert_to_tensor=True,
                    batch_size=32,  # Optimize for GPU
                    show_progress_bar=False
                ).to(self.device)

            # Align embeddings using DimensionalityManager
            for idx, emb in enumerate(batch_embeddings):
                # Make sure index is within bounds of semantic_rule_index
                if idx < len(semantic_rule_index):
                    try:
                        aligned_emb = self.dim_manager.align_embeddings(emb.unsqueeze(0), f"rule_{idx}").squeeze(0)
                        embeddings.append(aligned_emb)
                        semantic_rule_index[idx]['embedding'] = aligned_emb
                    except Exception as align_err:
                        self.logger.error(f"Error aligning embedding for rule index {idx}: {align_err}")
                        # Remove rule from index if embedding fails
                        semantic_rule_index[idx] = None  # Mark for removal or handle differently

            # Filter out rules where embedding failed
            semantic_rule_index = [rule for rule in semantic_rule_index if rule is not None and 'embedding' in rule]
            # Ensure embeddings list matches filtered index
            if len(embeddings) != len(semantic_rule_index):
                self.logger.warning(f"Mismatch between embeddings ({len(embeddings)}) and indexed rules ({len(semantic_rule_index)}) after alignment errors.")
                embeddings_tensor = None  # Indicate potential issue
            else:
                embeddings_tensor = torch.stack(embeddings) if embeddings else None

            self.logger.info(f"Semantic rule embeddings tensor created with shape: {embeddings_tensor.shape if embeddings_tensor is not None else 'None'}")

        except Exception as e:
            self.logger.error(f"Error building semantic index: {str(e)}")
            return [], None  # Return empty list and None

        indexed_keywords_count = len(set().union(*(r['keywords_set'] for r in semantic_rule_index))) if semantic_rule_index else 0
        self.logger.info(f"Rule index built. Total valid rules: {len(self.rules)}. Rules in semantic index: {len(semantic_rule_index)}. Keywords indexed: {indexed_keywords_count}")
        return semantic_rule_index, embeddings_tensor

    def _build_knowledge_graph(self) -> nx.DiGraph:
        """
        Build a directed acyclic graph (DAG) for rule-based inference.
        Note: Edge creation is currently disabled due to a previous 0-node issue.
        """
        G = nx.DiGraph()
        indexed_rule_map = {id(rule): f"rule_{idx}" for idx, rule in enumerate(self.semantic_rule_index)}

        for idx, rule in enumerate(self.semantic_rule_index):
            try:
                rule_id_str = indexed_rule_map.get(id(rule))
                if not rule_id_str:
                    continue  # Skip if not in semantic index map

                G.add_node(rule_id_str, keywords=rule['keywords_set'], response=rule['response'], type=rule.get('type', 'general'))

                # Edge creation could be re-enabled with improved logic, such as:
                # - Linking rules based on shared entities using spaCy NER.
                # - Using semantic similarity between rule responses and keywords.
                # For now, edges are disabled to focus on DROP improvements.

            except Exception as e:
                self.logger.debug(f"Error adding rule {idx} to knowledge graph: {e}")
                continue
        self.logger.info(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def process_query(
            self,
            query: str,
            context: Optional[str] = None,
            dataset_type: str = 'text',
            query_id: Optional[str] = None
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Process a query based on dataset type (text for HotpotQA, drop for DROP).
        Delegates to process_drop_query for DROP dataset.
        """
        qid = query_id or "unknown"
        self.logger.debug(f"Processing query ID: {qid}, Query: '{query[:50]}...', Dataset: {dataset_type}")

        # Use lower case dataset type for comparison
        dt_lower = dataset_type.lower() if dataset_type else 'text'

        if dt_lower == 'drop':
            return self.process_drop_query(query, context, dataset_type, qid)

        # Default to text processing otherwise
        return self._process_text_query(query, context, qid)

    def process_drop_query(
            self,
            query: str,
            context: Optional[str],
            dataset_type: str,  # Kept for consistency, but assumed 'drop' here
            query_id: str
    ) -> Dict[str, Any]:
        """
        Process a DROP query, returning a structured answer.
        Improved operation extraction and execution logic.
        """
        self.logger.debug(f"[DROP QID:{query_id}] Processing DROP query: '{query[:50]}...'")

        if not context:
            self.logger.warning(f"[DROP QID:{query_id}] No context provided. Returning default answer.")
            return {**DEFAULT_DROP_ANSWER, 'status': 'cannot_answer_no_context', 'confidence': 0.0, 'rationale': 'No context provided'}

        try:
            # Extract operation and arguments
            op_result = self._extract_drop_operation_and_args(query, context, query_id)
            if op_result['status'] != 'success':
                self.logger.warning(f"[DROP QID:{query_id}] DROP Symbolic path failed: {op_result['rationale']}")
                return {**DEFAULT_DROP_ANSWER, **op_result}

            op_type = op_result.get('operation')
            op_args = op_result.get('args', {})  # Arguments are now in 'args'
            confidence = op_result.get('confidence', 0.5)  # Use extracted confidence
            rationale = op_result.get('rationale', 'Operation extracted.')

            # Execute operation based on extracted type
            result = None
            if op_type == OP_COUNT:
                result = self.execute_count(op_args, context, query_id)
            elif op_type == OP_EXTREME_VALUE:
                result = self.execute_extreme_value(op_args, context, query, query_id)
            elif op_type == OP_DIFFERENCE:
                result = self.execute_difference(op_args, context, query_id)
            elif op_type == OP_ENTITY_SPAN:
                result = self.execute_entity_span(op_args, context, query_id)
            elif op_type == OP_DATE:
                result = self.execute_date(op_args, context, query_id)
            else:
                self.logger.warning(f"[DROP QID:{query_id}] Unsupported operation type extracted: {op_type}")
                return {**DEFAULT_DROP_ANSWER, 'status': 'unsupported_operation', 'confidence': confidence, 'rationale': f"Unsupported operation: {op_type}"}

            # Check execution result structure
            if not isinstance(result, dict) or 'type' not in result or 'value' not in result:
                self.logger.error(f"[DROP QID:{query_id}] Invalid result format from execution function for type {op_type}: {result}")
                return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.1, 'rationale': f"Internal error during {op_type} execution."}

            # Format result
            formatted_result = self._format_drop_answer(result)
            # Combine execution confidence with extraction confidence
            execution_confidence = result.get('confidence', 0.5)
            final_confidence = (confidence + execution_confidence) / 2

            formatted_result.update({
                'status': 'success',
                'confidence': final_confidence,
                'rationale': rationale + f" | Execution Conf: {execution_confidence:.2f}",
                'type': op_type  # Ensure final type is set
            })
            self.logger.info(f"[DROP QID:{query_id}] Successfully processed DROP query. Operation: {op_type}, Confidence: {final_confidence:.2f}")
            return formatted_result

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Unhandled error processing DROP query: {str(e)}")
            return {**DEFAULT_DROP_ANSWER, 'status': 'error', 'confidence': 0.0, 'rationale': f"Exception: {str(e)}"}

    def _process_text_query(self, query: str, context: Optional[str], query_id: str) -> List[str]:
        """
        Process a text-based query (HotpotQA) using semantic matching and graph traversal.
        """
        query_fingerprint = hash(query + (context or ""))  # Include context in cache key if provided
        if query_fingerprint in self.query_cache:
            self.logger.info(f"[Text QID:{query_id}] Cache hit for query: {query_fingerprint}")
            return self.query_cache[query_fingerprint]

        # Use semantic matching first
        matched_rules = self._match_rule_to_query(query, query_id)
        if not matched_rules:
            self.logger.info(f"[Text QID:{query_id}] No symbolic match found via semantic search.")
            responses = ["No symbolic match found."]

        else:
            responses = []
            visited = set()
            # Use a context dictionary that persists across chained calls for a single query
            current_query_context = {"subject": None}

            # Process top N matched rules directly and potentially chain from them
            top_matches = matched_rules[:min(len(matched_rules), 3)]  # Process top 3 matches

            for similarity, rule in top_matches:
                try:
                    rule_id = id(rule)
                    if rule_id in visited:
                        continue
                    visited.add(rule_id)
                    self.logger.debug(f"[Text QID:{query_id}] Processing top symbolic match (score={similarity:.2f}): {rule['response'][:50]}...")
                    response_list = self._process_rule(rule, current_query_context)  # Pass persistent context
                    responses.extend(response_list)
                except Exception as e:
                    self.logger.debug(f"[Text QID:{query_id}] Error processing rule {rule.get('response', 'N/A')}: {e}")
                    continue

            if not responses:
                self.logger.info(f"[Text QID:{query_id}] No valid responses after processing top rules.")
                responses = ["No symbolic match found."]

        responses = self._filter_responses(responses)
        self.query_cache[query_fingerprint] = responses
        self.logger.info(f"[Text QID:{query_id}] Returning {len(responses)} responses.")
        return responses

    def _extract_drop_operation_and_args(self, query: str, context: str, query_id: str) -> Dict[str, Any]:
        """
        Extract operation type and arguments from a DROP query using regex and spaCy.
        Improved pattern matching, priority, and argument extraction.
        """
        query_lower = query.lower().strip()

        # Define patterns with priority (more specific first)
        operation_patterns = [
            # Difference (e.g., "difference between X and Y", "how many more yards")
            (r"(?:difference between|how many more|how many less) ([\w\s]+)(?: and | than )([\w\s]+)", OP_DIFFERENCE, lambda m: {'entity1': m.group(1).strip(), 'entity2': m.group(2).strip()}, 0.95),
            (r"(?:difference between|how many more|how many less) ([\w\s]+)", OP_DIFFERENCE, lambda m: {'entity': m.group(1).strip()}, 0.90),

            # Extreme Value (e.g., "longest pass", "first touchdown")
            (r"(longest|shortest|highest|lowest|most|least|first|last) ([\w\s]+)(?: in| of | from | between | was | did)?", OP_EXTREME_VALUE, lambda m: {'entity_desc': m.group(2).strip(), 'direction': m.group(1).lower()}, 0.90),

            # Count (e.g., "how many field goals")
            (r"^(?:how many|number of)\s+([\w\s]+?)(?:\s+were\s+there|\s+did|\s+was|\s+in|\s+score)?", OP_COUNT, lambda m: {'entity': m.group(1).strip()}, 0.85),

            # Date (e.g., "when did X happen")
            (r"^(?:when|what date|what year|which year)\s+(.*)", OP_DATE, lambda m: {'entity': m.group(1).strip()}, 0.80),

            # Entity Span (e.g., "who scored", "which team")
            (r"^(?:who|which|what team|what player|what was the name of the)\s+(.*)", OP_ENTITY_SPAN, lambda m: {'entity': m.group(1).strip()}, 0.80),
            (r"^(?:who|which|what)\s+(.*)", OP_ENTITY_SPAN, lambda m: {'entity': m.group(1).strip()}, 0.75),  # Generic fallback
        ]

        # Temporal keywords to detect time constraints
        temporal_keywords = ["first half", "second half", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]

        # Try matching patterns in order of definition
        matched_op = None
        for pattern, op_type, arg_func, base_conf in operation_patterns:
            match = re.search(pattern, query_lower)
            if match:
                matched_op = {
                    'pattern': pattern,
                    'type': op_type,
                    'arg_func': arg_func,
                    'match_obj': match,
                    'base_confidence': base_conf
                }
                self.logger.debug(f"[DROP QID:{query_id}] Pattern matched: Type={op_type}, Pattern='{pattern}'")
                break  # Stop after first match (due to priority order)

        if not matched_op:
            self.logger.info(f"[DROP QID:{query_id}] No operation pattern matched for question: '{query[:50]}...'")
            return {'status': 'cannot_answer_mapping', 'confidence': 0.0, 'rationale': 'No operation pattern matched.'}

        # Extract arguments based on matched pattern
        try:
            args = matched_op['arg_func'](matched_op['match_obj'])
            # Basic validation of extracted args
            if not any(args.values()):
                raise ValueError("Argument function returned empty arguments.")
            self.logger.debug(f"[DROP QID:{query_id}] Extracted Args: {args}")

            # Refine entity and detect temporal constraints using spaCy
            doc = self.nlp(query_lower) if self.nlp else None
            query_keywords = []
            if kw_model:
                try:
                    query_keywords = [kw[0] for kw in kw_model.extract_keywords(query_lower, top_n=5)]
                except Exception:
                    pass
            elif doc:
                query_keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'PROPN') and not token.is_stop]

            # Check for temporal constraints
            temporal_constraint = None
            for keyword in temporal_keywords:
                if keyword in query_lower:
                    temporal_constraint = keyword
                    break
            if temporal_constraint:
                args['temporal_constraint'] = temporal_constraint
                self.logger.debug(f"[DROP QID:{query_id}] Detected temporal constraint: {temporal_constraint}")

            # Enhance arguments using spaCy
            if doc:
                # Extract subject for better entity refinement
                for token in doc:
                    if token.dep_ == "nsubj" and "entity" not in args:
                        args["entity"] = token.text
                    elif token.ent_type_ in ("PERSON", "ORG"):
                        if "entity" in args:
                            args["entity"] += f" {token.text}"
                        else:
                            args["entity"] = token.text

            # Calculate confidence
            confidence = matched_op['base_confidence']
            if self.embedder:
                try:
                    with torch.no_grad():
                        query_emb = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
                        context_sentences = [s.text for s in self.nlp(context).sents] if self.nlp else [context]
                        context_emb = self.embedder.encode(context_sentences, convert_to_tensor=True, batch_size=32).to(self.device)
                        sim = util.torch_max_dot_product(query_emb, context_emb)[0].item()
                        confidence = min(0.95, confidence + (sim - 0.4) * 0.5)  # Scale similarity
                except Exception as e:
                    self.logger.debug(f"[DROP QID:{query_id}] Confidence calculation failed: {e}")

            self.logger.debug(f"[DROP QID:{query_id}] Matched operation: {matched_op['type']}, Args: {args}, Confidence: {confidence:.2f}")
            return {
                'status': 'success',
                'operation': matched_op['type'],
                'args': args,
                'confidence': confidence,
                'rationale': f"Matched {matched_op['type']} pattern."
            }

        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Error extracting/refining args for pattern '{matched_op['pattern']}': {e}")
            return {'status': 'cannot_answer_args', 'confidence': 0.1, 'rationale': f'Error extracting arguments: {e}'}

    def _find_entities_in_passage(self, passage: str, entity: str, query_id: str, query_keywords: List[str] = None) -> List[str]:
        """
        Extract entity spans from passage using spaCy NER, dependency parsing, and refined matching.
        """
        if not self.nlp:
            self.logger.warning(f"[DROP QID:{query_id}] spaCy not available. Falling back to regex-based entity extraction.")
            return self._fallback_entity_extraction(passage, entity, query_id)

        try:
            doc = self.nlp(passage)
            # Normalize the target entity for robust matching
            entity_lower = entity.lower().strip()
            # Remove common leading/trailing words that might confuse matching
            entity_clean = re.sub(r"^(the|a|an|his|her|its|their)\s+|\s+(?:of|in|on|at|from|with)$", "", entity_lower).strip()
            if not entity_clean:
                entity_clean = entity_lower  # Use original if cleaning removed everything

            spans = []
            query_keywords = query_keywords or []
            query_keywords_set = set(query_keywords)

            # Priority 1: Exact/Lemma/NER Match
            matched_indices = set()  # Track matched token indices to avoid duplicates
            for token in doc:
                token_text_lower = token.text.lower()
                token_lemma_lower = token.lemma_.lower()
                # Check exact, lemma, or clean entity match
                if token_text_lower == entity_clean or token_lemma_lower == entity_clean:
                    # Try to expand to cover full phrase (e.g., multi-word proper noun)
                    current_span = token.text
                    # Check preceding/following tokens if they form part of the entity (e.g., Proper Noun sequence)
                    if token.i > 0 and doc[token.i - 1].pos_ == 'PROPN' and doc[token.i-1].i not in matched_indices:
                        current_span = doc[token.i - 1].text + " " + current_span
                        matched_indices.add(token.i-1)
                    if token.i < len(doc) - 1 and doc[token.i + 1].pos_ == 'PROPN' and doc[token.i+1].i not in matched_indices:
                        current_span = current_span + " " + doc[token.i + 1].text
                        matched_indices.add(token.i+1)

                    if token.i not in matched_indices:
                        spans.append(current_span)
                        matched_indices.add(token.i)
                        self.logger.debug(f"[DROP QID:{query_id}] Found entity via Direct/Lemma Match: {current_span}")

            # Check NER matches separately to capture full entity names
            for ent in doc.ents:
                # Check if the cleaned entity is part of the NER span
                if entity_clean in ent.text.lower():
                    if ent.start not in matched_indices:  # Avoid adding if already covered by token match
                        spans.append(ent.text)
                        matched_indices.update(range(ent.start, ent.end))
                        self.logger.debug(f"[DROP QID:{query_id}] Found entity via NER ({ent.label_}): {ent.text}")

            # Priority 2: Dependency/Keyword Match (if no direct match found yet)
            if not spans:
                for token in doc:
                    token_text_lower = token.text.lower()
                    # Check if token is a relevant POS tag and related to query keywords
                    if token.pos_ in ['NOUN', 'PROPN'] and token.i not in matched_indices:
                        # Check direct keyword overlap or overlap in dependency children/head
                        token_context_words = {token_text_lower, token.lemma_.lower()}
                        token_context_words.add(token.head.lemma_.lower())
                        token_context_words.update(child.lemma_.lower() for child in token.children)

                        if query_keywords_set.intersection(token_context_words):
                            spans.append(token.text)
                            matched_indices.add(token.i)
                            self.logger.debug(f"[DROP QID:{query_id}] Found entity via Dependency/Keyword Match: {token.text}")

            # Final Filtering
            unique_spans = list(dict.fromkeys(spans))
            # Filter very short spans unless it's the specific target entity
            filtered_spans = [s for s in unique_spans if len(s.strip()) > 1 or s.lower() == entity_clean]

            self.logger.debug(f"[DROP QID:{query_id}] Final extracted spans for '{entity}': {filtered_spans}")
            return filtered_spans

        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Error extracting entities for '{entity}': {str(e)}")
            return []

    def _fallback_entity_extraction(self, passage: str, entity: str, query_id: str) -> List[str]:
        """
        Fallback entity extraction using regex when spaCy is unavailable.
        Improved to handle word boundaries.
        """
        try:
            entity_pattern = r'\b' + re.escape(entity) + r'\b'
            spans = re.findall(entity_pattern, passage, re.IGNORECASE)
            unique_spans = list(dict.fromkeys(spans))  # Simple uniqueness
            self.logger.debug(f"[DROP QID:{query_id}] Fallback regex extracted spans for '{entity}': {unique_spans}")
            return unique_spans
        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Fallback entity extraction failed: {str(e)}")
            return []

    def _find_numbers_in_passage(self, passage: str, query_id: str) -> List[Union[int, float]]:
        """
        Extract numbers from passage using regex and spaCy.
        Improved handling of context (nearby words) if spaCy is available.
        """
        numbers = []
        try:
            # Regex first for broad coverage
            number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'  # Handles commas
            matches = re.findall(number_pattern, passage)
            for m in matches:
                try:
                    num_str = m.replace(',', '')
                    num = float(num_str) if '.' in num_str else int(num_str)
                    numbers.append(num)
                except ValueError:
                    self.logger.debug(f"[DROP QID:{query_id}] Regex match '{m}' is not a valid number.")
            self.logger.debug(f"[DROP QID:{query_id}] Found numbers via regex: {list(set(numbers))}")  # Show unique numbers found

            # spaCy for context and validation (if available)
            if self.nlp:
                doc = self.nlp(passage)
                spacy_numbers = []
                for ent in doc.ents:
                    if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                        try:
                            num_str = ent.text.replace(',', '').replace('$', '').replace('%', '').strip()
                            # Basic check if the string is potentially numeric before conversion
                            if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                                num = float(num_str) if '.' in num_str else int(num_str)
                                spacy_numbers.append(num)
                                self.logger.debug(f"[DROP QID:{query_id}] Found number via NER ({ent.label_}): {num} from '{ent.text}'")
                        except ValueError:
                            self.logger.debug(f"[DROP QID:{query_id}] NER entity '{ent.text}' ({ent.label_}) not parsed as number.")
                # Combine and ensure uniqueness
                numbers.extend(spacy_numbers)

            # Remove duplicates
            unique_numbers = sorted(list(set(numbers)))
            self.logger.debug(f"[DROP QID:{query_id}] Final unique numbers found: {unique_numbers}")
            return unique_numbers

        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Error extracting numbers: {str(e)}")
            return []  # Return empty list on error

    def _find_dates_in_passage(self, passage: str, query_id: str) -> List[Dict[str, str]]:
        """
        Extract dates from passage using dateutil, regex, and spaCy.
        """
        dates = []
        try:
            # Use spaCy for initial candidate identification if available
            candidates = []
            if self.nlp:
                doc = self.nlp(passage)
                for ent in doc.ents:
                    if ent.label_ == 'DATE':
                        candidates.append(ent.text)
                        self.logger.debug(f"[DROP QID:{query_id}] Found date candidate via NER: {ent.text}")

            # Use regex as fallback or supplement
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b\d{4}\b',  # Year only
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',  # Month Day, Year
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'  # Day Month Year
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, passage, re.IGNORECASE)
                candidates.extend(matches)

            # Parse candidates using dateutil
            parsed_dates = set()  # Use set of tuples to store unique dates
            for text in list(dict.fromkeys(candidates)):  # Process unique candidates
                try:
                    # Handle year-only case explicitly
                    if re.fullmatch(r'\d{4}', text):
                        parsed = datetime(year=int(text), month=1, day=1)  # Default month/day
                    else:
                        parsed = date_parser.parse(text, fuzzy=True)

                    # Store as tuple for uniqueness check
                    date_tuple = (str(parsed.year), str(parsed.month), str(parsed.day))
                    if date_tuple not in parsed_dates:
                        dates.append({
                            'day': str(parsed.day),
                            'month': str(parsed.month),
                            'year': str(parsed.year)
                        })
                        parsed_dates.add(date_tuple)
                        self.logger.debug(f"[DROP QID:{query_id}] Successfully parsed date: {date_tuple} from '{text}'")
                except (ValueError, OverflowError) as parse_err:
                    self.logger.debug(f"[DROP QID:{query_id}] Failed to parse date candidate '{text}': {parse_err}")
                    continue

            self.logger.debug(f"[DROP QID:{query_id}] Extracted unique dates: {dates}")
            return dates

        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Error extracting dates: {str(e)}")
            return []

    def execute_count(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute COUNT operation for DROP queries.
        Handles temporal constraints and refined entity matching.
        """
        try:
            entity = args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            if not entity:
                return {'type': 'number', 'value': '0', 'confidence': 0.1, 'error': 'No entity provided for count'}

            # Extract keywords for better matching
            query_keywords = []
            if self.nlp:
                doc = self.nlp(entity.lower())
                query_keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'PROPN') and not token.is_stop]

            # Use the refined entity finder
            spans = self._find_entities_in_passage(context, entity, query_id, query_keywords)

            # Apply temporal filtering if constraint is present
            if temporal_constraint and spans:
                filtered_spans = []
                doc = self.nlp(context)
                for span in spans:
                    # Find sentences containing the span
                    span_lower = span.lower()
                    for sent in doc.sents:
                        if span_lower in sent.text.lower():
                            sent_text_lower = sent.text.lower()
                            # Check for temporal constraint
                            if temporal_constraint == "first half":
                                if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                    filtered_spans.append(span)
                            elif temporal_constraint == "second half":
                                if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                    filtered_spans.append(span)
                            elif temporal_constraint in sent_text_lower:
                                filtered_spans.append(span)
                            break
                spans = filtered_spans
                self.logger.debug(f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}': {spans}")

            count = len(spans)
            # Confidence based on whether spans were found related to the entity
            confidence = 0.8 if spans else 0.4
            self.logger.debug(f"[DROP QID:{query_id}] COUNT operation: Entity '{entity}', Found Spans: {spans}, Count: {count}, Confidence: {confidence:.2f}")
            return {'type': 'number', 'value': str(count), 'confidence': confidence}
        except Exception as e:
            self.logger.error(f"[DROP QID:{query_id}] Error in COUNT operation for entity '{args.get('entity')}': {str(e)}")
            return {'type': 'number', 'value': '0', 'confidence': 0.0, 'error': str(e)}

    def execute_extreme_value(self, args: Dict[str, Any], context: str, query: str, query_id: str) -> Dict[str, Any]:
        """
        Execute EXTREME_VALUE operation (longest/shortest/first/last etc.) for DROP queries.
        Determines return type based on question intent (number or spans).
        Handles temporal constraints.
        """
        try:
            entity_desc = args.get('entity_desc')  # Description like "pass", "field goal"
            direction = args.get('direction', 'longest')  # longest, shortest, first, last, etc.
            temporal_constraint = args.get('temporal_constraint')
            if not entity_desc:
                return {'type': 'number', 'value': '0', 'confidence': 0.1, 'error': 'Missing entity for extreme_value'}

            # Find relevant numbers and associated text spans in the context
            value_span_pairs = self._find_values_and_spans(context, entity_desc, query_id)

            # Apply temporal filtering if constraint is present
            if temporal_constraint and value_span_pairs:
                filtered_pairs = []
                doc = self.nlp(context)
                for value, span in value_span_pairs:
                    span_lower = span.lower()
                    for sent in doc.sents:
                        if span_lower in sent.text.lower():
                            sent_text_lower = sent.text.lower()
                            if temporal_constraint == "first half":
                                if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                    filtered_pairs.append((value, span))
                            elif temporal_constraint == "second half":
                                if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                    filtered_pairs.append((value, span))
                            elif temporal_constraint in sent_text_lower:
                                filtered_pairs.append((value, span))
                            break
                value_span_pairs = filtered_pairs
                self.logger.debug(f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}': {value_span_pairs}")

            if not value_span_pairs:
                self.logger.warning(f"[DROP QID:{query_id}] No relevant values/spans found for '{entity_desc}' for EXTREME_VALUE.")
                return {'type': 'number', 'value': '0', 'confidence': 0.2, 'error': f"No relevant values/spans found for '{entity_desc}'"}

            # Determine the comparison key and operator based on direction
            if direction in ['longest', 'highest', 'most', 'last']:
                op = operator.ge  # Greater than or equal
                selector = max
            elif direction in ['shortest', 'lowest', 'least', 'first']:
                op = operator.le  # Less than or equal
                selector = min
            else:
                self.logger.warning(f"[DROP QID:{query_id}] Unknown direction '{direction}', defaulting to 'longest'.")
                op = operator.ge
                selector = max

            # Find the extreme value
            try:
                valid_pairs = [pair for pair in value_span_pairs if pair[0] is not None]
                if not valid_pairs:
                    raise ValueError("No valid numeric values found.")
                extreme_value = selector(pair[0] for pair in valid_pairs)
            except ValueError as ve:
                self.logger.error(f"[DROP QID:{query_id}] Could not determine extreme value for '{entity_desc}': {ve}")
                return {'type': 'number', 'value': '0', 'confidence': 0.1, 'error': str(ve)}

            # Find spans associated with the extreme value
            associated_spans = [span for val, span in valid_pairs if val is not None and abs(val - extreme_value) < 1e-6]
            confidence = 0.75 if associated_spans else 0.5

            # Decide return type: if question implies WHO/WHICH, return spans, else return number
            return_type = 'number'
            query_lower = query.lower()
            if 'who' in query_lower or 'which' in query_lower or 'what team' in query_lower or 'what player' in query_lower:
                return_type = 'spans'

            if return_type == 'spans':
                self.logger.debug(f"[DROP QID:{query_id}] EXTREME_VALUE (span): Entity='{entity_desc}', Direction={direction}, Value={extreme_value}, Spans={associated_spans}, Conf={confidence:.2f}")
                return {'type': 'spans', 'value': associated_spans[:1] if associated_spans else [], 'confidence': confidence}
            else:
                self.logger.debug(f"[DROP QID:{query_id}] EXTREME_VALUE (number): Entity='{entity_desc}', Direction={direction}, Value={extreme_value}, Conf={confidence:.2f}")
                return {'type': 'number', 'value': str(extreme_value), 'confidence': confidence}

        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error in EXTREME_VALUE for entity '{args.get('entity_desc')}': {str(e)}")
            return {'type': 'number', 'value': '0', 'confidence': 0.0, 'error': str(e)}

    def _find_values_and_spans(self, context: str, entity_desc: str, query_id: str) -> List[Tuple[Optional[float], str]]:
        """
        Helper to find numbers and associated entity spans.
        E.g., find "yards" near "pass" and associate with a player.
        """
        pairs = []
        if not self.nlp:
            return []
        doc = self.nlp(context)
        entity_desc_tokens = set(entity_desc.lower().split())

        for ent in doc.ents:
            # Look for numeric entities (CARDINAL, QUANTITY, MONEY, PERCENT)
            if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                num_val = None
                try:
                    num_str = ent.text.replace(',', '').replace('$', '').replace('%', '').strip()
                    if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                        num_val = float(num_str) if '.' in num_str else int(num_str)
                except ValueError:
                    continue

                if num_val is not None:
                    # Check nearby context for the entity description
                    window = 5  # Check N tokens before and after
                    start = max(0, ent.start - window)
                    end = min(len(doc), ent.end + window)
                    context_window_tokens = {token.lemma_.lower() for token in doc[start:end] if token.pos_ in ['NOUN', 'PROPN']}
                    # If the entity description is found nearby, associate the number
                    if entity_desc_tokens.intersection(context_window_tokens):
                        # Try to find a relevant associated span (e.g., the person/team involved)
                        associated_span = ent.text  # Default to the number itself
                        # Look for nearby proper nouns or nouns that might be the subject/object
                        for token in doc[start:end]:
                            if token.pos_ == 'PROPN' and token.i >= ent.start - 3 and token.i <= ent.end + 3:
                                # Capture multi-word proper nouns if possible
                                pn_span = token.text
                                if token.i > 0 and doc[token.i-1].pos_ == 'PROPN':
                                    pn_span = doc[token.i-1].text + " " + pn_span
                                if token.i < len(doc)-1 and doc[token.i+1].pos_ == 'PROPN':
                                    pn_span = pn_span + " " + doc[token.i+1].text
                                associated_span = pn_span
                                break
                            elif token.pos_ == 'NOUN' and token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.i != ent.start:
                                associated_span = token.text

                        pairs.append((num_val, associated_span))
                        self.logger.debug(f"[DROP QID:{query_id}] Found value={num_val} associated with '{entity_desc}' near span '{associated_span}'")

        return pairs

    def execute_difference(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute DIFFERENCE operation for DROP queries.
        Handles cases with one or two entities mentioned, and temporal constraints.
        """
        try:
            entity1 = args.get('entity1') or args.get('entity')  # Primary entity or first entity
            entity2 = args.get('entity2')  # Optional second entity
            temporal_constraint = args.get('temporal_constraint')
            if not entity1:
                return {'type': 'number', 'value': '0', 'confidence': 0.1, 'error': 'No entity provided for difference'}

            numbers1 = self._find_associated_numbers(context, entity1, query_id)
            numbers2 = self._find_associated_numbers(context, entity2, query_id) if entity2 else []

            # Apply temporal filtering if constraint is present
            if temporal_constraint:
                filtered_numbers1 = []
                filtered_numbers2 = []
                doc = self.nlp(context)
                for sent in doc.sents:
                    sent_text_lower = sent.text.lower()
                    if temporal_constraint == "first half":
                        if "1st quarter" not in sent_text_lower and "2nd quarter" not in sent_text_lower:
                            continue
                    elif temporal_constraint == "second half":
                        if "3rd quarter" not in sent_text_lower and "4th quarter" not in sent_text_lower:
                            continue
                    elif temporal_constraint not in sent_text_lower:
                        continue
                    # Re-extract numbers from this sentence
                    sent_numbers1 = self._find_associated_numbers(sent.text, entity1, query_id)
                    filtered_numbers1.extend(sent_numbers1)
                    if entity2:
                        sent_numbers2 = self._find_associated_numbers(sent.text, entity2, query_id)
                        filtered_numbers2.extend(sent_numbers2)
                numbers1 = filtered_numbers1
                numbers2 = filtered_numbers2
                self.logger.debug(f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}': Numbers1={numbers1}, Numbers2={numbers2}")

            if entity2:  # Difference between two entities
                if not numbers1 or not numbers2:
                    self.logger.warning(f"[DROP QID:{query_id}] Insufficient numbers for DIFFERENCE between '{entity1}' and '{entity2}'.")
                    return {'type': 'number', 'value': '0', 'confidence': 0.2, 'error': f"Insufficient numbers for '{entity1}' and '{entity2}'"}
                # Assume difference between max/min or first found value of each entity
                val1 = max(numbers1)
                val2 = max(numbers2)
                difference = abs(val1 - val2)
                confidence = 0.7
            else:  # Difference within a single entity description (e.g., longest and shortest)
                if len(numbers1) < 2:
                    self.logger.warning(f"[DROP QID:{query_id}] Insufficient numbers for DIFFERENCE within '{entity1}'.")
                    return {'type': 'number', 'value': '0', 'confidence': 0.2, 'error': f"Insufficient numbers for '{entity1}'"}
                difference = max(numbers1) - min(numbers1)
                confidence = 0.75

            self.logger.debug(f"[DROP QID:{query_id}] DIFFERENCE operation: Entities=('{entity1}', '{entity2}'), Diff={difference}, Confidence={confidence:.2f}")
            return {'type': 'number', 'value': str(difference), 'confidence': confidence}
        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error in DIFFERENCE operation: {str(e)}")
            return {'type': 'number', 'value': '0', 'confidence': 0.0, 'error': str(e)}

    def _find_associated_numbers(self, context: str, entity: str, query_id: str) -> List[float]:
        """
        Helper to find numbers associated with a specific entity description.
        """
        values = []
        if not self.nlp:
            # Fallback to regex if no spaCy
            numbers = self._find_numbers_in_passage(context, query_id)
            if entity.lower() in context.lower():
                values = numbers  # Less precise
            return list(set(values))

        doc = self.nlp(context)
        entity_tokens = set(entity.lower().split())
        for ent in doc.ents:
            if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']:
                # Check context window
                window = 5
                start = max(0, ent.start - window)
                end = min(len(doc), ent.end + window)
                context_window_tokens = {token.lemma_.lower() for token in doc[start:end]}
                if entity_tokens.intersection(context_window_tokens):
                    try:
                        num_str = ent.text.replace(',', '').replace('$', '').replace('%', '').strip()
                        if re.fullmatch(r'-?\d+(\.\d+)?', num_str):
                            num = float(num_str) if '.' in num_str else int(num_str)
                            values.append(num)
                    except ValueError:
                        continue
        return list(set(values))

    def execute_entity_span(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute ENTITY_SPAN operation for DROP queries.
        Prioritizes finding spans related to the specific entity.
        """
        try:
            entity = args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            if not entity:
                return {'type': 'spans', 'value': [], 'confidence': 0.1, 'error': 'No entity provided for span extraction'}

            # Extract keywords for better matching
            query_keywords = []
            if self.nlp:
                doc = self.nlp(entity.lower())
                query_keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'PROPN') and not token.is_stop]

            spans = self._find_entities_in_passage(context, entity, query_id, query_keywords)

            # Apply temporal filtering if constraint is present
            if temporal_constraint and spans:
                filtered_spans = []
                doc = self.nlp(context)
                for span in spans:
                    span_lower = span.lower()
                    for sent in doc.sents:
                        if span_lower in sent.text.lower():
                            sent_text_lower = sent.text.lower()
                            if temporal_constraint == "first half":
                                if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                    filtered_spans.append(span)
                            elif temporal_constraint == "second half":
                                if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                    filtered_spans.append(span)
                            elif temporal_constraint in sent_text_lower:
                                filtered_spans.append(span)
                            break
                spans = filtered_spans
                self.logger.debug(f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}': {spans}")

            confidence = 0.8 if spans else 0.3
            if not spans:
                self.logger.warning(f"[DROP QID:{query_id}] No spans found for ENTITY_SPAN operation on '{entity}'.")

            self.logger.debug(f"[DROP QID:{query_id}] ENTITY_SPAN operation: Entity '{entity}', Spans: {spans}, Confidence: {confidence:.2f}")
            # Return only the first/most relevant span typically for DROP 'who/which' questions
            return {'type': 'spans', 'value': spans[:1] if spans else [], 'confidence': confidence}
        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error in ENTITY_SPAN operation for entity '{args.get('entity')}': {str(e)}")
            return {'type': 'spans', 'value': [], 'confidence': 0.0, 'error': str(e)}

    def execute_date(self, args: Dict[str, Any], context: str, query_id: str) -> Dict[str, Any]:
        """
        Execute DATE operation for DROP queries.
        Finds dates potentially associated with the entity.
        """
        try:
            entity = args.get('entity')
            temporal_constraint = args.get('temporal_constraint')
            if not entity:
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.1, 'error': 'No entity for date'}

            # Find all dates first
            all_dates = self._find_dates_in_passage(context, query_id)
            if not all_dates:
                self.logger.warning(f"[DROP QID:{query_id}] No dates found in context for DATE operation.")
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.2}

            # Apply temporal filtering if constraint is present
            if temporal_constraint:
                filtered_dates = []
                doc = self.nlp(context)
                for date_dict in all_dates:
                    # Try reconstructing common formats to find in text
                    formats_to_try = [f"{date_dict['month']}/{date_dict['day']}/{date_dict['year']}"]
                    for fmt_str in formats_to_try:
                        for sent in doc.sents:
                            if fmt_str in sent.text:
                                sent_text_lower = sent.text.lower()
                                if temporal_constraint == "first half":
                                    if "1st quarter" in sent_text_lower or "2nd quarter" in sent_text_lower:
                                        filtered_dates.append(date_dict)
                                elif temporal_constraint == "second half":
                                    if "3rd quarter" in sent_text_lower or "4th quarter" in sent_text_lower:
                                        filtered_dates.append(date_dict)
                                elif temporal_constraint in sent_text_lower:
                                    filtered_dates.append(date_dict)
                                break
                all_dates = filtered_dates
                self.logger.debug(f"[DROP QID:{query_id}] Applied temporal constraint '{temporal_constraint}': {all_dates}")

            if not all_dates:
                return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.2, 'error': 'No dates found after temporal filtering'}

            # Find the date most closely associated with the entity
            value = all_dates[0]  # Default to first date found
            confidence = 0.6  # Base confidence
            if self.nlp:
                entity_indices = [m.start() for m in re.finditer(re.escape(entity), context, re.IGNORECASE)]
                if entity_indices:
                    first_entity_idx = entity_indices[0]
                    closest_date = None
                    min_dist = float('inf')
                    for date_dict in all_dates:
                        formats_to_try = [f"{date_dict['month']}/{date_dict['day']}/{date_dict['year']}"]
                        for fmt_str in formats_to_try:
                            date_indices = [m.start() for m in re.finditer(re.escape(fmt_str), context, re.IGNORECASE)]
                            if date_indices:
                                dist = abs(date_indices[0] - first_entity_idx)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_date = date_dict
                                    break
                    if closest_date:
                        value = closest_date
                        confidence = 0.8
                        self.logger.debug(f"[DROP QID:{query_id}] Associated date {value} with entity '{entity}' based on proximity.")

            self.logger.debug(f"[DROP QID:{query_id}] DATE operation: Entity '{entity}', Date: {value}, Confidence: {confidence:.2f}")
            return {'type': 'date', 'value': value, 'confidence': confidence}
        except Exception as e:
            self.logger.exception(f"[DROP QID:{query_id}] Error in DATE operation for entity '{args.get('entity')}': {str(e)}")
            return {'type': 'date', 'value': DEFAULT_DROP_ANSWER['date'], 'confidence': 0.0, 'error': str(e)}

    def _format_drop_answer(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the result of a DROP operation into the standard DROP answer structure.
        Ensures value types are correct.
        """
        answer = DEFAULT_DROP_ANSWER.copy()
        try:
            result_type = result.get('type')
            result_value = result.get('value')  # This is the raw value from execute_*

            if result.get('error'):  # Propagate error if execution failed
                answer['error'] = result.get('error')
                return answer

            if result_type in ['number', OP_COUNT, OP_DIFFERENCE, OP_EXTREME_VALUE]:
                # Ensure value is stringified number or empty string
                answer['number'] = str(result_value).strip() if result_value is not None else ''
            elif result_type in ['spans', OP_ENTITY_SPAN]:
                # Ensure value is a list of non-empty strings
                if isinstance(result_value, list):
                    answer['spans'] = [str(v).strip() for v in result_value if str(v).strip()]
                elif result_value is not None:  # Handle single string case
                    answer['spans'] = [str(result_value).strip()] if str(result_value).strip() else []
                else:
                    answer['spans'] = []
            elif result_type == OP_DATE:
                # Ensure value is a dict with day, month, year strings
                if isinstance(result_value, dict) and all(k in result_value for k in ['day', 'month', 'year']):
                    answer['date'] = {k: str(v).strip() for k, v in result_value.items() if k in ['day', 'month', 'year']}
                else:
                    self.logger.warning(f"Invalid date value format received: {result_value}")
                    answer['date'] = DEFAULT_DROP_ANSWER['date']
            else:
                self.logger.warning(f"Unsupported result type '{result_type}' for DROP answer formatting.")
                answer['error'] = f'unsupported_result_type: {result_type}'

            return answer

        except Exception as e:
            self.logger.error(f"Error formatting DROP answer: {str(e)}")
            return {**DEFAULT_DROP_ANSWER, 'error': str(e)}

    def _match_rule_to_query(self, query: str, query_id: str) -> List[Tuple[float, Dict]]:
        """
        Match query to rules using semantic similarity on indexed embeddings.
        """
        if not self.embedder:
            self.logger.warning(f"[Text QID:{query_id}] Embedder not available. Cannot perform semantic matching.")
            return []
        if self.rule_embeddings is None or len(self.semantic_rule_index) == 0 or self.rule_embeddings.shape[0] != len(self.semantic_rule_index):
            self.logger.warning(f"[Text QID:{query_id}] Rule embeddings missing or mismatched ({self.rule_embeddings.shape[0] if self.rule_embeddings is not None else 'None'} embeddings vs {len(self.semantic_rule_index)} rules). Cannot perform semantic matching.")
            return []

        try:
            query_embedding = self._encode_query(query)
            if query_embedding is None or query_embedding.nelement() == 0:
                self.logger.error(f"[Text QID:{query_id}] Failed to encode query. Cannot match rules.")
                return []

            # Ensure dimensions match for similarity calculation
            if query_embedding.shape[0] != self.rule_embeddings.shape[1]:
                self.logger.error(f"[Text QID:{query_id}] Dimension mismatch between query ({query_embedding.shape}) and rule ({self.rule_embeddings.shape}) embeddings.")
                if self.dim_manager:
                    query_embedding = self.dim_manager.align_embeddings(query_embedding.unsqueeze(0), f"query_{query_id}").squeeze(0)
                    if query_embedding.shape[0] != self.rule_embeddings.shape[1]:
                        self.logger.error(f"[Text QID:{query_id}] Alignment failed. Dimension mismatch persists.")
                        return []
                else:
                    return []

            similarities = util.cos_sim(query_embedding, self.rule_embeddings).squeeze(0)
            matched_rules = []

            # Use torch to find indices above threshold efficiently
            indices = torch.where(similarities >= self.match_threshold)[0]

            for idx in indices.tolist():
                if idx < len(self.semantic_rule_index):
                    sim = similarities[idx].item()
                    matched_rules.append((sim, self.semantic_rule_index[idx]))
                else:
                    self.logger.warning(f"Index {idx} out of bounds for semantic_rule_index (len {len(self.semantic_rule_index)})")

            matched_rules.sort(key=lambda x: x[0], reverse=True)
            self.logger.debug(f"[Text QID:{query_id}] Found {len(matched_rules)} matching rules with similarities >= {self.match_threshold}: {[f'{s:.2f}' for s, _ in matched_rules[:5]]}...")
            return matched_rules

        except Exception as e:
            self.logger.exception(f"[Text QID:{query_id}] Error matching query to rules: {str(e)}")
            return []

    def _encode_query(self, query: str) -> Optional[torch.Tensor]:
        """
        Encode query using SentenceTransformer with torch.no_grad(). Returns None on error.
        """
        if not self.embedder:
            self.logger.error("Embedder not available for query encoding.")
            return None
        try:
            with torch.no_grad():
                return self.embedder.encode(query, convert_to_tensor=True).to(self.device)
        except Exception as e:
            self.logger.error(f"Error encoding query: {str(e)}")
            return None

    def _process_rule(self, rule: Dict, context: Dict) -> List[str]:
        """
        Process a single rule in a context-aware manner for text queries.
        """
        response = rule['response']
        try:
            # Basic pronoun replacement (can be improved with coreference resolution)
            if 'subject' in context and context['subject']:
                response = re.sub(r'\b(it|this)\b', context['subject'], response, flags=re.IGNORECASE)

            # Attempt to update context with the new subject from this response
            if self.nlp:
                doc = self.nlp(response)
                subjects = [tok.text for tok in doc if tok.dep_ == 'nsubj']
                if subjects:
                    first_subj_token = next((tok for tok in doc if tok.text == subjects[0]), None)
                    if first_subj_token:
                        full_subject = first_subj_token.text
                        head = first_subj_token
                        while head.head.i != head.i and head.head.dep_ in ['compound', 'flat'] and head.head.pos_ == 'PROPN':
                            full_subject = head.head.text + " " + full_subject
                            head = head.head
                        context['subject'] = full_subject

            return [response]
        except Exception as e:
            self.logger.error(f"Error processing rule response '{rule.get('response', '')[:50]}...': {str(e)}")
            return [rule['response']]

    def _chain_rules_with_context(self, current_rule: Dict, context: Dict, visited: Set, depth: int) -> List[str]:
        """
        Recursively chain to other rules using context and response tokens.
        Note: Graph traversal might be more robust if graph edges were reliably built.
        """
        if depth >= self.max_hops:
            return []

        responses = []
        next_rules_candidates = self._find_related_rules(current_rule, context)

        for rule in next_rules_candidates:
            try:
                rule_id = id(rule)
                if rule_id not in visited:
                    visited.add(rule_id)
                    response_list = self._process_rule(rule, context)
                    if response_list:
                        responses.extend(response_list)
                    chain_responses = self._chain_rules_with_context(rule, context, visited, depth + 1)
                    responses.extend(chain_responses)
            except Exception as e:
                self.logger.debug(f"Error chaining rule: {str(e)}")
                continue
        return responses

    def _find_related_rules(self, current_rule: Dict, context: Dict) -> List[Dict]:
        """
        Find rules related to the current rule's response tokens or context subject.
        Uses the semantic index for matching.
        """
        potential_query_tokens = set(current_rule.get('response_tokens', set()))
        if 'subject' in context and context['subject']:
            subject_tokens = set(re.findall(r'\b\w+\b', context['subject'].lower()))
            potential_query_tokens.update(subject_tokens)

        if not potential_query_tokens:
            return []

        related = []
        for rule in self.semantic_rule_index:
            if id(rule) == id(current_rule):
                continue
            if potential_query_tokens.intersection(rule.get('keywords_set', set())):
                related.append(rule)

        return related

    def _filter_responses(self, responses: List[str]) -> List[str]:
        """
        Remove duplicate responses and sort by length (proxy for detail).
        Also attempts basic sentence completion/truncation.
        """
        try:
            seen = set()
            unique_responses = []
            for resp in responses:
                cleaned_resp = self._truncate_to_last_sentence(resp.strip())
                if cleaned_resp and cleaned_resp not in seen:
                    seen.add(cleaned_resp)
                    unique_responses.append(cleaned_resp)
            return unique_responses
        except Exception as e:
            self.logger.error(f"Error filtering responses: {str(e)}")
            return [r.strip() for r in responses if r and r.strip()]

    def add_dynamic_rules(self, new_rules: List[Dict]):
        """
        Add new rules dynamically and update indices.
        Handles potential errors during index rebuilding.
        """
        if not new_rules:
            return

        try:
            valid_rules_added = 0
            current_rule_count = len(self.rules)
            temp_new_rules = []

            for rule in new_rules:
                is_valid = True
                if not isinstance(rule, dict) or 'keywords' not in rule or 'response' not in rule:
                    is_valid = False
                elif not isinstance(rule['keywords'], list) or not all(isinstance(k, str) for k in rule['keywords']):
                    is_valid = False
                elif not isinstance(rule['response'], str):
                    is_valid = False

                if is_valid:
                    rule.setdefault('confidence', 0.7)
                    rule.setdefault('type', 'dynamic')
                    temp_new_rules.append(rule)
                    valid_rules_added += 1
                else:
                    self.logger.debug(f"Skipping invalid dynamic rule: {rule}")

            if valid_rules_added > 0:
                self.rules.extend(temp_new_rules)
                self.semantic_rule_index, self.rule_embeddings = self._build_semantic_rule_index()
                self.knowledge_graph = self._build_knowledge_graph()
                self.logger.info(f"Added {valid_rules_added} dynamic rules. Total rules now: {len(self.rules)}. Indices rebuilt.")
            else:
                self.logger.info("No valid dynamic rules provided to add.")

        except Exception as e:
            self.logger.error(f"Error adding dynamic rules and rebuilding indices: {str(e)}")

    def _truncate_to_last_sentence(self, text: str) -> str:
        """
        Truncate text to the last complete sentence for cleaner responses.
        Returns original text if no clear sentence ending is found near the end.
        """
        if not text or len(text.strip()) < 5:
            return text

        end_indices = [text.rfind(p) for p in '.!?']
        last_punctuation_index = max(end_indices) if any(i != -1 for i in end_indices) else -1

        if last_punctuation_index != -1 and last_punctuation_index >= len(text) - 5:
            if last_punctuation_index > 0 and text[last_punctuation_index-1].isalpha():
                truncated_text = text[:last_punctuation_index + 1].strip()
                if len(truncated_text.split()) > 2:
                    return truncated_text

        return text