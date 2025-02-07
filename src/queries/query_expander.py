# src/queries/query_expander.py
# src/queries/query_expander.py
import os
import yaml
import torch
import spacy
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class AdaptiveQueryComplexityEstimator:
    def __init__(self):
        # Base linguistic features with fixed weights
        self.analytical_terms = {
            'why': 0.8, 'how': 0.7, 'explain': 0.9, 'analyze': 0.9,
            'compare': 0.8, 'evaluate': 0.85, 'synthesize': 0.9
        }
        self.domain_terms = {
            'biodiversity': 0.6, 'climate': 0.7, 'environment': 0.6,
            'ecosystem': 0.7, 'sustainability': 0.8
        }
        # Runtime learning components
        self.term_weights = {}  # Dynamically adjusted term weights
        self.complexity_history = []  # Track historical complexity scores

        # Adaptive learning rate parameters
        self.base_learning_rate = 0.1
        self.min_learning_rate = 0.01
        self.learning_decay = 0.995
        self.current_learning_rate = self.base_learning_rate

        # Initialize language analysis components
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def analyze_linguistic_structure(self, query):
        """
        Analyze syntactic and semantic properties of the query.
        """
        doc = self.nlp(query)
        clauses = list(doc.sents)
        clause_count = len(clauses)
        dependency_depth = max((len(list(token.ancestors)) for token in doc), default=0)
        entity_count = len(doc.ents)
        verb_complexity = sum(1 for token in doc if token.pos_ == "VERB")
        return {
            'clause_count': clause_count,
            'dependency_depth': dependency_depth,
            'entity_count': entity_count,
            'verb_complexity': verb_complexity
        }

    def calculate_semantic_complexity(self, query):
        """
        Enhanced semantic complexity calculation with attention to semantic density
        and relationship patterns.
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        doc = self.nlp(query)

        # Calculate semantic density (norm per word)
        semantic_density = torch.norm(query_embedding).item() / len(query.split())

        # Analyze semantic relationships: count tokens with key dependency roles
        total_tokens = len(doc)
        semantic_relationships = sum(1 for token in doc if token.dep_ in ['nsubj', 'dobj', 'pobj'])
        relationship_ratio = semantic_relationships / total_tokens if total_tokens > 0 else 0

        # Consider named entity complexity: average number of words per entity
        if doc.ents:
            entity_complexity = sum(len(ent.text.split()) for ent in doc.ents) / len(doc.ents)
        else:
            entity_complexity = 0

        # Combine these measures using weighted average
        combined_semantic = 0.5 * semantic_density + 0.3 * relationship_ratio + 0.2 * entity_complexity
        return combined_semantic, {
            'density': semantic_density,
            'relationships': relationship_ratio,
            'entity_complexity': entity_complexity
        }

    def update_learning_rate(self, success_rate):
        """Adjust learning rate based on success of previous predictions."""
        if success_rate > 0.8:
            self.current_learning_rate *= self.learning_decay
        else:
            self.current_learning_rate = min(
                self.base_learning_rate,
                self.current_learning_rate / self.learning_decay
            )
        self.current_learning_rate = max(self.min_learning_rate, self.current_learning_rate)

    def update_term_weights(self, query, execution_time=None, context=None):
        """
        Update term weights with context awareness and confidence tracking.
        """
        words = set(query.lower().split())
        doc = self.nlp(query)

        for word in words:
            # Initialize weight structure if not present
            if word not in self.term_weights:
                self.term_weights[word] = {
                    'weight': 0.5,
                    'updates': 0,
                    'confidence': 0.5
                }
            weight_data = self.term_weights[word]

            # Find token corresponding to the word in context
            word_token = None
            for token in doc:
                if token.text.lower() == word:
                    word_token = token
                    break

            context_factor = 1.0
            if word_token:
                # Increase factor if word holds key syntactic roles
                if word_token.dep_ in ['ROOT', 'nsubj', 'dobj']:
                    context_factor *= 1.2
                # Consider neighboring adjectives/adverbs
                for neighbor in word_token.children:
                    if neighbor.pos_ in ['ADJ', 'ADV']:
                        context_factor *= 1.1

            if execution_time:
                normalized_time = min(execution_time / 10.0, 1.0)
                new_weight = (weight_data['weight'] * (1 - self.current_learning_rate) +
                              normalized_time * self.current_learning_rate * context_factor)

                weight_change = abs(new_weight - weight_data['weight'])
                confidence_adjust = -0.1 if weight_change > 0.2 else 0.1
                new_confidence = min(1.0, max(0.1, weight_data['confidence'] + confidence_adjust))

                self.term_weights[word] = {
                    'weight': new_weight,
                    'updates': weight_data['updates'] + 1,
                    'confidence': new_confidence
                }

    def estimate_complexity(self, query):
        """
        Estimates query complexity using multiple factors and runtime learning.
        Returns a normalized complexity score between 0 and 1.
        """
        linguistic_features = self.analyze_linguistic_structure(query)
        base_complexity = (
                linguistic_features['clause_count'] * 0.2 +
                linguistic_features['dependency_depth'] * 0.15 +
                linguistic_features['entity_count'] * 0.1 +
                linguistic_features['verb_complexity'] * 0.15
        )
        semantic_complexity, semantic_details = self.calculate_semantic_complexity(query)
        words = query.lower().split()
        term_complexity = sum(
            self.analytical_terms.get(word, 0) +
            self.domain_terms.get(word, 0) +
            self.term_weights.get(word, {'weight': 0})['weight']
            for word in words
        ) / len(words)
        final_complexity = base_complexity * 0.3 + semantic_complexity * 0.3 + term_complexity * 0.4
        normalized_complexity = min(max(final_complexity, 0), 1)
        self.complexity_history.append(normalized_complexity)
        return normalized_complexity

    def get_complexity_analysis(self, query):
        """
        Provides detailed analysis of query complexity factors.
        """
        overall = self.estimate_complexity(query)
        linguistic = self.analyze_linguistic_structure(query)
        semantic, semantic_details = self.calculate_semantic_complexity(query)
        term_info = {word: self.term_weights.get(word, {'weight': 0}) for word in query.lower().split()}
        return {
            'overall_complexity': overall,
            'linguistic_features': linguistic,
            'semantic_complexity': semantic,
            'semantic_details': semantic_details,
            'term_weights': term_info
        }


class QueryExpander:
    def __init__(self, complexity_estimator=None, complexity_config=None, expansion_rules=None):
        self.expansion_rules = expansion_rules or {
            "deforestation": ["forest loss", "tree cutting", "logging"],
            "climate change": ["global warming", "greenhouse gases"],
            "biodiversity": ["species diversity", "ecosystem diversity"]
        }
        self.nlp = spacy.load("en_core_web_md")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny", output_attentions=True)
        self.config = {}
        if complexity_config and os.path.exists(complexity_config):
            with open(complexity_config, "r") as f:
                self.config = yaml.safe_load(f)
        self.sim_threshold = 0.6
        # Use adaptive estimator if none is provided
        self.complexity_estimator = complexity_estimator or AdaptiveQueryComplexityEstimator()
        # Performance metrics for self-adjustment
        self.performance_metrics = {
            'expansions': 0,
            'successful_expansions': 0,
            'average_expansion_ratio': 0,
            'expansion_times': []
        }

    def get_query_complexity(self, query):
        """
        Gets complexity score and detailed analysis for a query.
        """
        analysis = self.complexity_estimator.get_complexity_analysis(query)
        print("Query complexity analysis:")
        print(f"  Overall complexity: {analysis['overall_complexity']:.2f}")
        print(f"  Linguistic features: {analysis['linguistic_features']}")
        print(f"  Semantic complexity: {analysis['semantic_complexity']:.2f}")
        print(f"  Semantic details: {analysis['semantic_details']}")
        print(f"  Term weights: {analysis['term_weights']}")
        return analysis['overall_complexity']

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
        Generate additional expansion terms based on semantic similarity.
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

    def monitor_expansion_performance(self, original_query, expanded_query, success_score):
        """
        Track and analyze query expansion performance.
        """
        expansion_ratio = len(expanded_query.split()) / len(original_query.split())
        self.performance_metrics['expansions'] += 1
        if success_score > 0.7:
            self.performance_metrics['successful_expansions'] += 1
        n = self.performance_metrics['expansions']
        old_avg = self.performance_metrics['average_expansion_ratio']
        self.performance_metrics['average_expansion_ratio'] = ((old_avg * (n - 1)) + expansion_ratio) / n
        if self.performance_metrics['expansions'] > 10:
            success_rate = self.performance_metrics['successful_expansions'] / self.performance_metrics['expansions']
            if success_rate < 0.6:
                self.sim_threshold *= 1.1  # become more selective
            elif success_rate > 0.8:
                self.sim_threshold *= 0.9  # allow more expansions
            self.sim_threshold = min(0.9, max(0.3, self.sim_threshold))

    def expand_query(self, query):
        """
        Expand the query based on complexity analysis and semantic relationships.
        """
        complexity = self.get_query_complexity(query)
        print(f"Query complexity score: {complexity:.2f}")
        if complexity > 1.0:
            print("Skipping expansion due to high complexity.")
            return query
        semantic_expansion = self._semantic_expansion(query)
        basic_expansion = query
        for term, synonyms in self.expansion_rules.items():
            if term.lower() in query.lower():
                expansion_group = f"({term} OR {' OR '.join(synonyms)})"
                basic_expansion = basic_expansion.replace(term, expansion_group)
        if semantic_expansion:
            expanded = basic_expansion + " " + semantic_expansion
        else:
            expanded = basic_expansion

        # Monitor performance (using a dummy success score for illustration)
        self.monitor_expansion_performance(query, expanded, success_score=0.75)

        return expanded.strip()


if __name__ == "__main__":
    expander = QueryExpander(complexity_config="src/config/complexity_rules.yaml")
    test_query = "What are the environmental effects of deforestation?"
    print("Original Query:", test_query)
    print("Expanded Query:", expander.expand_query(test_query))
