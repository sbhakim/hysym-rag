# src/queries/query_expander.py
import yaml
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import os
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict


class QueryExpander:
    def __init__(self, complexity_config=None, expansion_rules=None, embedder=None):
        # Original expansion rules (if not provided, use default)
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }
        # Use a medium-size spaCy model for structural analysis
        self.nlp = spacy.load("en_core_web_md")
        # Initialize tokenizer and model for computing query complexity
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny", output_attentions=True)
        # Load additional configuration if provided
        self.config = {}
        if complexity_config and os.path.exists(complexity_config):
            with open(complexity_config, "r") as f:
                self.config = yaml.safe_load(f)
        # Set a similarity threshold for expansion
        self.sim_threshold = 0.6
        # Initialize embedder for rule guidance (use SentenceTransformer)
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize cache and rule performance tracking for dynamic rule selection
        self.rule_cache = {}  # Caches relevant rules for each query
        self.rule_performance = defaultdict(list)  # Tracks historical performance for each rule
        self.rule_embeddings = None
        self._initialize_rule_embeddings()

    def _initialize_rule_embeddings(self):
        """Pre-compute embeddings for all rules to improve efficiency."""
        if not self.expansion_rules:
            return
        rule_texts = []
        # Use both rule keys and their synonyms for computing embeddings
        for rule_key, synonyms in self.expansion_rules.items():
            rule_texts.append(rule_key)
            rule_texts.extend(synonyms)
        self.rule_embeddings = self.embedder.encode(
            rule_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def _get_rule_guidance(self, query):
        """
        Get relevant rules for query expansion using a basic similarity approach.
        (This method is kept for backward compatibility.)
        """
        cache_key = hash(query)
        if cache_key in self.rule_cache:
            return self.rule_cache[cache_key]
        query_embedding = self.embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.cos_sim(query_embedding.unsqueeze(0), self.rule_embeddings).squeeze()
        relevant_indices = (similarities >= self.sim_threshold).nonzero(as_tuple=False).squeeze()
        if relevant_indices.dim() == 0:
            if similarities.item() >= self.sim_threshold:
                relevant_indices = [int(relevant_indices.item())]
            else:
                relevant_indices = []
        else:
            relevant_indices = relevant_indices.tolist()
        rule_keys = list(self.expansion_rules.keys())
        relevant_rules = [rule_keys[i] for i in relevant_indices if i < len(rule_keys)]
        self.rule_cache[cache_key] = relevant_rules
        return relevant_rules

    def _get_relevant_rules(self, query, top_k=3):
        """
        Get most relevant rules for a query using semantic similarity and historical performance.
        """
        cache_key = hash(query)
        if cache_key in self.rule_cache:
            return self.rule_cache[cache_key]

        # Encode query
        query_emb = self.embedder.encode(query, convert_to_tensor=True)

        scored_rules = []
        # Interpret each rule as a dict with an id and keywords
        for rule_key in self.expansion_rules.keys():
            rule = {"id": rule_key, "keywords": [rule_key] + self.expansion_rules[rule_key]}
            rule_text = " ".join(rule.get("keywords", []))
            rule_emb = self.embedder.encode(rule_text, convert_to_tensor=True)
            sim_score = util.cos_sim(query_emb, rule_emb).item()

            # Get historical performance
            rule_id = rule.get("id", str(rule))
            perf_score = self._get_rule_performance(rule_id)

            # Combine scores (70% semantic similarity, 30% performance)
            final_score = 0.7 * sim_score + 0.3 * perf_score
            scored_rules.append((final_score, rule))

        # Select top_k rules
        top_rules = sorted(scored_rules, key=lambda x: x[0], reverse=True)[:top_k]
        self.rule_cache[cache_key] = [rule for _, rule in top_rules]
        return self.rule_cache[cache_key]

    def _get_rule_performance(self, rule_id):
        """Calculate rule performance score based on history."""
        if not self.rule_performance[rule_id]:
            return 0.5  # Default score
        recent_performance = self.rule_performance[rule_id][-10:]
        return sum(recent_performance) / len(recent_performance)

    def update_rule_performance(self, rule_id, success_score):
        """Update rule performance history."""
        self.rule_performance[rule_id].append(success_score)
        if len(self.rule_performance[rule_id]) > 100:  # Keep last 100 entries
            self.rule_performance[rule_id].pop(0)

    def get_query_complexity(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        attn = outputs.attentions[-1].mean().item()
        complexity = attn * len(query.split())
        return complexity

    def _analyze_structural_patterns(self, doc):
        if not self.config.get("structural_patterns"):
            return 0.0
        score = 0.0
        text = [t.text for t in doc]
        multi_part = self.config["structural_patterns"].get("multi_part", [])
        for item in multi_part:
            pattern = item["pattern"]
            weight = item["weight"]
            if pattern in text:
                score += weight
        dependency = self.config["structural_patterns"].get("dependency", [])
        for item in dependency:
            pattern = item["pattern"]
            weight = item["weight"]
            if pattern in " ".join(text):
                score += weight
        return score

    def _semantic_expansion(self, query):
        """
        Perform semantic query expansion using spaCy vectors.
        For each term in our expansion rules that appears in the query,
        find similar words (from the provided candidate list) that exceed a similarity threshold.
        """
        doc = self.nlp(query)
        expanded_terms = []
        for term, candidates in self.expansion_rules.items():
            term_doc = self.nlp(term)
            if term.lower() in query.lower():
                similar = []
                for cand in candidates:
                    cand_doc = self.nlp(cand)
                    sim = term_doc.similarity(cand_doc)
                    if sim >= self.sim_threshold:
                        similar.append(cand)
                if similar:
                    expanded_terms.append(f"({term} OR {' OR '.join(similar)})")
        return " ".join(expanded_terms)

    def expand_query(self, query):
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity}")
        if complexity > 1.0:
            print("Skipping expansion due to high complexity.")
            return query
        # Perform semantic expansion and basic string-based expansion
        semantic_expansion = self._semantic_expansion(query)
        basic_expansion = query
        for term, synonyms in self.expansion_rules.items():
            if term.lower() in query.lower():
                expansion_group = f"({term} OR {' OR '.join(synonyms)})"
                basic_expansion = basic_expansion.replace(term, expansion_group)
        # Get rule-guided expansion using dynamic rule selection
        rule_guidance = self._get_relevant_rules(query)
        if rule_guidance:
            expanded_terms = set(basic_expansion.split())
            # Add rule IDs from the dynamic selection (or you might want to add rule responses)
            expanded_terms.update([rule.get("id", rule) for rule in rule_guidance])
            combined_expansion = " ".join(expanded_terms)
        else:
            combined_expansion = basic_expansion
        if semantic_expansion:
            combined_expansion = combined_expansion + " " + semantic_expansion
        return combined_expansion.strip()


# For testing purposes:
if __name__ == "__main__":
    expander = QueryExpander(complexity_config="src/config/complexity_rules.yaml")
    test_query = "What are the environmental effects of deforestation?"
    print("Original Query:", test_query)
    print("Expanded Query:", expander.expand_query(test_query))
