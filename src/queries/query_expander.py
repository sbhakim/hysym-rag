# src/queries/query_expander.py

import yaml
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import os
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
import logging


class QueryExpander:
    """
    Enhanced query expander with specific support for multi-hop reasoning queries
    and HotpotQA-style questions.
    """

    def __init__(self,
                 complexity_config: Optional[str] = None,
                 expansion_rules: Optional[Dict] = None,
                 embedder: Optional[SentenceTransformer] = None,
                 hop_threshold: float = 0.6):
        """
        Initialize with enhanced multi-hop support.
        """
        # Initialize base expansion rules
        self.expansion_rules = expansion_rules or {
            "comparison": ["compare", "difference between", "versus", "similar to"],
            "bridge": ["related to", "connected with", "involved in"],
            "temporal": ["when", "before", "after", "during"],
            "causal": ["cause", "effect", "result in", "lead to"],
            "composite": ["and", "both", "together with"]
        }

        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_md")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny", output_attentions=True)

        # Initialize embedding model
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')

        # Set configuration parameters
        self.hop_threshold = hop_threshold
        self.config = {}
        if complexity_config and os.path.exists(complexity_config):
            with open(complexity_config, "r") as f:
                self.config = yaml.safe_load(f)

        # Set up pattern recognition
        self.multi_hop_patterns = {
            'bridge': [
                r".*(?:what|who|where).*(?:of|by|from).*(?:who|that|which).*",
                r".*(?:involved in|related to|connected with).*"
            ],
            'comparison': [
                r".*(?:compare|difference between|versus|or).*(?:and).*",
                r".*(?:which|who).*(?:more|less|better|worse).*"
            ],
            'temporal': [
                r".*(?:before|after|when|during).*(?:what|who|where).*",
                r".*(?:first|last|previous|next).*(?:to|in|at).*"
            ]
        }

        # Initialize caching
        self.pattern_cache = {}
        self.embedding_cache = {}

        # Set up logging
        self.logger = logging.getLogger("QueryExpander")
        self.logger.setLevel(logging.INFO)

    def expand_query(self, query: str) -> str:
        """
        Expand query with enhanced multi-hop awareness.
        """
        # Get query complexity
        complexity = self.get_query_complexity(query)
        self.logger.info(f"Query complexity score: {complexity:.4f}")

        if complexity > 1.0:
            self.logger.info("Skipping expansion due to high complexity.")
            return query

        # Determine query type
        query_type = self._determine_query_type(query)

        # Perform type-specific expansion
        if query_type == 'bridge':
            return self._expand_bridge_query(query)
        elif query_type == 'comparison':
            return self._expand_comparison_query(query)
        else:
            return self._expand_standard_query(query)

    def get_query_complexity(self, query: str) -> float:
        """
        Calculate query complexity with multi-hop consideration.
        """
        # Get basic attention-based complexity
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        attn = outputs.attentions[-1].mean().item()

        # Base complexity score
        base_complexity = attn * len(query.split())

        # Adjust for multi-hop characteristics
        hop_adjustment = self._calculate_hop_complexity(query)

        return base_complexity * (1 + hop_adjustment)

    def _determine_query_type(self, query: str) -> str:
        """
        Determine query type with pattern matching.
        """
        # Check cache
        cache_key = hash(query)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        # Check patterns
        for q_type, patterns in self.multi_hop_patterns.items():
            for pattern in patterns:
                if any(re.match(p, query.lower()) for p in patterns):
                    self.pattern_cache[cache_key] = q_type
                    return q_type

        return 'standard'

    def _expand_bridge_query(self, query: str) -> str:
        """
        Expand bridge-type multi-hop query.
        """
        # Extract potential bridge entities
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]

        # Get related terms for bridge entities
        expanded_terms = []
        for entity in entities:
            related_terms = self._get_related_terms(entity)
            if related_terms:
                expanded_terms.append(f"({entity} OR {' OR '.join(related_terms)})")

        # Combine original query with expansions
        expanded = query
        if expanded_terms:
            expanded += " AND " + " AND ".join(expanded_terms)

        return expanded

    def _expand_comparison_query(self, query: str) -> str:
        """
        Expand comparison-type multi-hop query.
        """
        doc = self.nlp(query)

        # Extract comparison entities
        comparison_pairs = self._extract_comparison_pairs(doc)

        # Expand each entity in the comparison
        expanded_parts = []
        for entity1, entity2 in comparison_pairs:
            exp1 = self._get_related_terms(entity1)
            exp2 = self._get_related_terms(entity2)

            if exp1 and exp2:
                expanded_parts.append(
                    f"({entity1} OR {' OR '.join(exp1)}) compared to ({entity2} OR {' OR '.join(exp2)})"
                )

        return query + " " + " AND ".join(expanded_parts) if expanded_parts else query

    def _expand_standard_query(self, query: str) -> str:
        """
        Expand standard query with semantic similarity.
        """
        doc = self.nlp(query)
        expanded_terms = []

        for token in doc:
            if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop:
                similar_terms = self._get_similar_terms(token.text)
                if similar_terms:
                    expanded_terms.append(f"({token.text} OR {' OR '.join(similar_terms)})")

        return query + " " + " ".join(expanded_terms) if expanded_terms else query

    def _calculate_hop_complexity(self, query: str) -> float:
        """
        Calculate complexity adjustment for multi-hop characteristics.
        """
        doc = self.nlp(query)

        # Count potential hops
        hop_indicators = sum(1 for token in doc if token.text.lower()
                             in ['and', 'or', 'then', 'after', 'before'])

        # Count named entities
        entity_count = len([ent for ent in doc.ents])

        # Calculate adjustment
        return min(0.5, (hop_indicators * 0.1 + entity_count * 0.05))

    def _get_related_terms(self, term: str) -> List[str]:
        """
        Get semantically related terms using embeddings.
        """
        cache_key = hash(term)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            # Get term embedding
            term_emb = self.embedder.encode(term, convert_to_tensor=True)

            # Get related terms from expansion rules
            related = []
            for category, terms in self.expansion_rules.items():
                terms_emb = self.embedder.encode(terms, convert_to_tensor=True)
                similarities = util.cos_sim(term_emb, terms_emb)

                # Add highly similar terms
                for idx, sim in enumerate(similarities[0]):
                    if sim > self.hop_threshold:
                        related.append(terms[idx])

            self.embedding_cache[cache_key] = related
            return related

        except Exception as e:
            self.logger.error(f"Error getting related terms for {term}: {str(e)}")
            return []

    def _extract_comparison_pairs(self, doc) -> List[Tuple[str, str]]:
        """
        Extract entity pairs for comparison queries.
        """
        pairs = []
        entities = list(doc.ents)

        for i in range(len(entities) - 1):
            for j in range(i + 1, len(entities)):
                # Check if entities are being compared
                between_tokens = doc[entities[i].end:entities[j].start]
                if any(token.text.lower() in ['versus', 'vs', 'or', 'and']
                       for token in between_tokens):
                    pairs.append((entities[i].text, entities[j].text))

        return pairs
