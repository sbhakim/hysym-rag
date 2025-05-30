# src/reasoners/networkx_symbolic_reasoner_base.py

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

from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager

try:
    from keybert import KeyBERT
    kw_model = KeyBERT()
except ImportError:
    logging.getLogger(__name__).warning("KeyBERT not installed. Keyword extraction will use fallback methods.")
    kw_model = None
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to initialize KeyBERT model: {e}")
    kw_model = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_DROP_ANSWER = {"number": "", "spans": [], "date": {"day": "", "month": "", "year": ""}}

class GraphSymbolicReasoner:
    """
    Base class for the graph-based symbolic reasoner in SymRAG.
    Handles static components like rule loading, semantic indexing, and text-based QA (HotpotQA).
    DROP-specific functionality is implemented in a separate module.
    """

    def __init__(
            self,
            rules_file: str,
            match_threshold: float = 0.1,
            max_hops: int = 5,
            embedding_model: str = 'all-MiniLM-L6-v2',
            device: Optional[torch.device] = None,
            dim_manager: Optional[DimensionalityManager] = None
    ):
        self.logger = logger
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.device = device or DeviceManager.get_device()
        self.query_cache = {}

        if dim_manager is None:
            self.logger.warning("DimensionalityManager not provided to GraphSymbolicReasoner. Creating a default one.")
            self.dim_manager = DimensionalityManager(target_dim=768, device=self.device)
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
            self.nlp = None

        self.rules = self._load_and_validate_rules(rules_file=rules_file)
        self.semantic_rule_index, self.rule_embeddings = self._build_semantic_rule_index()
        self.knowledge_graph = self._build_knowledge_graph()

        self.logger.info(f"GraphSymbolicReasoner initialized. Rules loaded: {len(self.rules)}. Rules indexed for semantic match: {len(self.semantic_rule_index)}.")

    def _load_and_validate_rules(self, rules_file: str, min_support: int = 0) -> List[Dict]:
        """
        Load rules from file and validate their structure.
        Supports both HotpotQA (keywords, response) and DROP (type, pattern) rule formats.
        Optionally filters DROP rules by minimum support.
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
                if not isinstance(rule, dict):
                    self.logger.debug(f"Skipping invalid rule: not a dictionary")
                    continue

                # Check if this is a HotpotQA rule (requires keywords and response)
                is_hotpotqa_rule = 'keywords' in rule and 'response' in rule
                # Check if this is a DROP rule (requires type and pattern)
                is_drop_rule = 'type' in rule and 'pattern' in rule

                if not (is_hotpotqa_rule or is_drop_rule):
                    self.logger.debug(f"Skipping rule with missing required fields: {rule}")
                    continue

                if is_hotpotqa_rule:
                    # Validate HotpotQA rule
                    if not isinstance(rule['keywords'], list) or not all(isinstance(k, str) for k in rule['keywords']):
                        self.logger.debug(f"Skipping rule with invalid keywords format: {rule.get('keywords', 'N/A')}")
                        continue
                    if not isinstance(rule['response'], str):
                        self.logger.debug(f"Skipping rule with invalid response format: {rule.get('response', 'N/A')}")
                        continue
                    rule.setdefault('confidence', 0.7)
                    rule.setdefault('type', 'general')
                    valid_rules.append(rule)

                elif is_drop_rule:
                    # Validate DROP rule
                    if not isinstance(rule['type'], str):
                        self.logger.debug(f"Skipping DROP rule with invalid type format: {rule.get('type', 'N/A')}")
                        continue
                    if not isinstance(rule['pattern'], str):
                        self.logger.debug(f"Skipping DROP rule with invalid pattern format: {rule.get('pattern', 'N/A')}")
                        continue
                    # Filter by support if specified
                    if 'support' in rule and rule['support'] < min_support:
                        self.logger.debug(f"Skipping DROP rule with support {rule['support']} below threshold {min_support}")
                        continue
                    # Compile the regex pattern
                    try:
                        rule['compiled_pattern'] = re.compile(rule['pattern'])
                    except re.error as e:
                        self.logger.warning(f"Invalid regex pattern in DROP rule {rule}: {e}")
                        continue
                    # Ensure optional DROP fields are present
                    rule.setdefault('entity', None)
                    rule.setdefault('temporal_constraint', None)
                    rule.setdefault('support', 0)
                    rule.setdefault('confidence', 0.7)
                    valid_rules.append(rule)

            except Exception as e:
                self.logger.debug(f"Error validating rule {rule}: {e}")
                continue

        self.logger.info(f"Loaded {len(valid_rules)} valid rules/definitions out of {len(raw_rules)} raw entries.")
        return valid_rules

    def _build_semantic_rule_index(self) -> Tuple[List[Dict], Optional[torch.Tensor]]:
        """
        Build an index of rules with precomputed embeddings for semantic matching.
        Handles both HotpotQA (keywords, response) and DROP (type, pattern) rules.
        """
        semantic_rule_index = []
        embeddings = []

        if not self.embedder:
            self.logger.warning("Embedder not available. Semantic rule index will be empty.")
            return semantic_rule_index, None

        try:
            # Prepare text for embedding based on rule type
            texts_to_embed = []
            valid_rules = []
            for rule in self.rules:
                # Determine if this is a HotpotQA or DROP rule
                is_hotpotqa_rule = 'keywords' in rule and 'response' in rule
                is_drop_rule = 'type' in rule and 'pattern' in rule

                if not (is_hotpotqa_rule or is_drop_rule):
                    self.logger.debug(f"Skipping rule with unrecognized format: {rule}")
                    continue

                rule_copy = rule.copy()

                if is_hotpotqa_rule:
                    # HotpotQA rule: use keywords for embedding and token sets
                    keywords_text = " ".join(rule.get('keywords', []))
                    if not keywords_text.strip():
                        self.logger.debug(f"Skipping HotpotQA rule with empty keywords: {rule}")
                        continue
                    texts_to_embed.append(keywords_text)
                    rule_copy['keywords_set'] = set(rule['keywords'])
                    response_text = rule['response'].lower()
                    if len(response_text.split()) <= 1:
                        rule_copy['response_tokens'] = set(rule['keywords'])
                    else:
                        rule_copy['response_tokens'] = set(re.findall(r'\b\w+\b', response_text))

                else:
                    # DROP rule: use pattern for embedding and token sets
                    pattern_text = rule['pattern']
                    if not pattern_text.strip():
                        self.logger.debug(f"Skipping DROP rule with empty pattern: {rule}")
                        continue
                    texts_to_embed.append(pattern_text)
                    # Extract tokens from the pattern (simplified; could use spaCy for better tokenization)
                    pattern_tokens = set(re.findall(r'\b\w+\b', pattern_text.lower()))
                    rule_copy['keywords_set'] = pattern_tokens  # For compatibility with matching logic
                    rule_copy['response_tokens'] = pattern_tokens  # Simplified; DROP rules don't have a response

                semantic_rule_index.append(rule_copy)
                valid_rules.append(rule_copy)

            if not texts_to_embed:
                self.logger.warning("No valid texts for embedding. Semantic rule index will be empty.")
                return [], None

            with torch.no_grad():
                batch_embeddings = self.embedder.encode(
                    texts_to_embed,
                    convert_to_tensor=True,
                    batch_size=32,
                    show_progress_bar=False
                ).to(self.device)

            for idx, emb in enumerate(batch_embeddings):
                if idx < len(semantic_rule_index):
                    try:
                        aligned_emb = self.dim_manager.align_embeddings(emb.unsqueeze(0), f"rule_{idx}").squeeze(0)
                        embeddings.append(aligned_emb)
                        semantic_rule_index[idx]['embedding'] = aligned_emb
                    except Exception as align_err:
                        self.logger.error(f"Error aligning embedding for rule index {idx}: {align_err}")
                        semantic_rule_index[idx] = None

            semantic_rule_index = [rule for rule in semantic_rule_index if rule is not None and 'embedding' in rule]
            if len(embeddings) != len(semantic_rule_index):
                self.logger.warning(f"Mismatch between embeddings ({len(embeddings)}) and indexed rules ({len(semantic_rule_index)}) after alignment errors.")
                embeddings_tensor = None
            else:
                embeddings_tensor = torch.stack(embeddings) if embeddings else None

            self.logger.info(f"Semantic rule embeddings tensor created with shape: {embeddings_tensor.shape if embeddings_tensor is not None else 'None'}")

        except Exception as e:
            self.logger.error(f"Error building semantic index: {str(e)}")
            return [], None

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
                    continue

                G.add_node(rule_id_str, keywords=rule['keywords_set'], response=rule.get('response', ''), type=rule.get('type', 'general'))

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
        Delegates to process_drop_query for DROP dataset, which should be implemented in the DROP-specific module.
        """
        qid = query_id or "unknown"
        self.logger.debug(f"Processing query ID: {qid}, Query: '{query[:50]}...', Dataset: {dataset_type}")

        dt_lower = dataset_type.lower() if dataset_type else 'text'

        if dt_lower == 'drop':
            raise NotImplementedError("DROP query processing is not implemented in the base class. Use the DROP-specific module.")

        return self._process_text_query(query, context, qid)

    def _process_text_query(self, query: str, context: Optional[str], query_id: str) -> List[str]:
        """
        Process a text-based query (HotpotQA) using semantic matching and graph traversal.
        """
        query_fingerprint = hash(query + (context or ""))
        if query_fingerprint in self.query_cache:
            self.logger.info(f"[Text QID:{query_id}] Cache hit for query: {query_fingerprint}")
            return self.query_cache[query_fingerprint]

        matched_rules = self._match_rule_to_query(query, query_id)
        if not matched_rules:
            self.logger.info(f"[Text QID:{query_id}] No symbolic match found via semantic search.")
            responses = ["No symbolic match found."]
        else:
            responses = []
            visited = set()
            current_query_context = {"subject": None}

            top_matches = matched_rules[:min(len(matched_rules), 3)]

            for similarity, rule in top_matches:
                try:
                    rule_id = id(rule)
                    if rule_id in visited:
                        continue
                    visited.add(rule_id)
                    self.logger.debug(f"[Text QID:{query_id}] Processing top symbolic match (score={similarity:.2f}): {rule.get('response', 'N/A')[:50]}...")
                    response_list = self._process_rule(rule, current_query_context)
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
        response = rule.get('response', '')
        if not response:  # DROP rules may not have a response
            return []
        try:
            if 'subject' in context and context['subject']:
                response = re.sub(r'\b(it|this)\b', context['subject'], response, flags=re.IGNORECASE)

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
            self.logger.error(f"Error processing rule response '{response[:50]}...': {str(e)}")
            return [response]

    def _chain_rules_with_context(self, current_rule: Dict, context: Dict, visited: Set, depth: int) -> List[str]:
        """
        Recursively chain to other rules using context and response tokens.
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