# src/query_expander.py
import yaml
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import os

class QueryExpander:
    def __init__(self, complexity_config=None, expansion_rules=None):
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny", output_attentions=True)
        self.config = {}
        if complexity_config and os.path.exists(complexity_config):
            with open(complexity_config, "r") as f:
                self.config = yaml.safe_load(f)

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

    def expand_query(self, query):
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity}")
        if complexity > 1.0:
            print("Skipping expansions due to high complexity.")
            return query
        expanded = query
        for term, synonyms in self.expansion_rules.items():
            if term.lower() in query.lower():
                expansion_group = f"({term} OR {' OR '.join(synonyms)})"
                expanded = expanded.replace(term, expansion_group)
        return expanded
