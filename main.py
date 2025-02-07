# main.py

# !/usr/bin/env python

import os
import json
import time
import torch
import psutil  # For memory monitoring

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import project modules
from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.utils.rule_extractor import RuleExtractor, DynamicRuleGenerator
from src.queries.query_logger import QueryLogger
from src.resources.resource_manager import ResourceManager
from src.feedback.feedback_manager import FeedbackManager
from src.feedback.feedback_handler import FeedbackHandler
from src.config.config_manager import ConfigManager  # Updated configuration manager
from src.queries.query_expander import QueryExpander, AdaptiveQueryComplexityEstimator
from src.system.system_control_manager import SystemControlManager, UnifiedResponseAggregator
from src.utils.model_manager import ModelManager


# ---------------- Helper Functions for Memory Monitoring and Batch Processing ----------------

def check_memory_status():
    """Monitor memory status at critical points. Returns True if below threshold."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    process = psutil.Process()
    memory_percent = process.memory_percent()
    if memory_percent > 85:  # 85% threshold
        return False
    return True


def process_query_batch(queries, context, system_manager, resource_manager):
    """Process queries with active resource monitoring."""
    results = []
    for query in queries:
        if not check_memory_status():
            print("Performing emergency cleanup before processing query...")
            torch.cuda.empty_cache()
            time.sleep(1)
        try:
            result = system_manager.process_query_with_fallback(
                query["query"],
                context,
                max_retries=3
            )
            results.append(result)
        except Exception as e:
            print(f"Query processing failed: {e}")
            results.append({"error": str(e)})
        torch.cuda.empty_cache()  # Cleanup after each query
    return results


# ---------------- End Helper Functions ----------------

# ---------------- Define NeuralRetriever using shared models ----------------

class NeuralRetriever:
    """
    NeuralRetriever for retrieving answers from a neural language model.
    Provides an encode() method for obtaining neural embeddings.
    """

    def __init__(self, model_name, model_manager):
        print(f"Initializing Neural Retriever with model: {model_name}...")
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.neural_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)
        print("Neural Retriever model loaded successfully!")

    def retrieve_answer(self, context, question):
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        torch.cuda.empty_cache()
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def encode(self, text):
        try:
            with torch.no_grad():
                emb = self.encoder.encode(text, convert_to_tensor=True)
            if torch.cuda.is_available():
                emb = emb.to('cuda')
            return emb
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            raise


# ---------------- Main Initialization and Query Processing ----------------

if __name__ == "__main__":
    try:
        print("\n=== Initializing HySym-RAG System ===")

        # Step 1: Load configuration using the new ConfigManager
        print("Loading configuration...")
        config_manager = ConfigManager("src/config/config.yaml")
        config = config_manager.get_config()
        model_name = config["model_name"]

        # Step 2: Initialize Resource Manager
        print("Initializing Resource Manager...")
        resource_manager = ResourceManager("src/config/resource_config.yaml")

        # Step 3: Initialize Model Manager singleton
        print("Initializing Model Manager...")
        try:
            model_manager = ModelManager()
        except Exception as e:
            print(f"Warning: Full model initialization failed: {e}")
            print("Falling back to CPU-only mode...")
            config['use_gpu'] = False
            model_manager = ModelManager()

        # Step 4: Extract symbolic rules
        print("Extracting rules from deforestation.txt...")
        RuleExtractor.extract_rules("data/deforestation.txt", "data/rules.json")

        # Step 5: Dynamically update rules from context
        print("Generating dynamic rules from current context...")
        with open("data/small_knowledge_base.txt", "r") as kb_file:
            kb_text = kb_file.read()
        dynamic_rule_generator = DynamicRuleGenerator()
        new_rules = dynamic_rule_generator.generate_rules_from_embeddings(kb_text.split(". "))
        dynamic_rule_generator.update_rule_base(new_rules, "data/rules.json")

        # Step 6: Initialize Graph-Based Symbolic Reasoner
        print("Initializing Graph-Based Symbolic Reasoner...")
        symbolic = GraphSymbolicReasoner("data/rules.json", match_threshold=0.25, max_hops=5)

        # Step 7: Initialize Neural Retriever using Model Manager
        print("Initializing Neural Retriever...")
        neural = NeuralRetriever(model_name, model_manager)

        # Step 8: Initialize support components
        print("Initializing support components...")
        logger = QueryLogger()
        feedback_manager = FeedbackManager()
        complexity_estimator = AdaptiveQueryComplexityEstimator()
        expander = QueryExpander(complexity_estimator=complexity_estimator,
                                 complexity_config="src/config/complexity_rules.yaml")
        with open("data/ground_truths.json", "r") as gt_file:
            ground_truths = json.load(gt_file)
        from src.utils.evaluation import Evaluation

        evaluator = Evaluation()

        # Step 9: Create Hybrid Integrator (using energy-aware scheduling and batching)
        print("Creating Hybrid Integrator...")
        integrator = HybridIntegrator(symbolic, neural, resource_manager, expander)

        # Step 10: Initialize System Control Manager with error recovery
        print("Initializing System Control Components...")
        try:
            aggregator = UnifiedResponseAggregator(include_explanations=True)
            system_manager = SystemControlManager(
                hybrid_integrator=integrator,
                resource_manager=resource_manager,
                aggregator=aggregator,
                error_retry_limit=2,
                max_query_time=10
            )
        except Exception as e:
            print(f"Warning: System control initialization failed: {e}")
            print("Falling back to basic control mode...")
            system_manager = SystemControlManager.create_basic_mode(integrator=integrator,
                                                                    resource_manager=resource_manager)

        # Step 11: Initialize Application and Feedback Handling
        print("Initializing Application...")
        from src.app import App

        feedback_handler = FeedbackHandler(feedback_manager)
        app = App(symbolic=symbolic, neural=neural, logger=logger, feedback=resource_manager,
                  evaluator=evaluator, expander=expander, ground_truths=ground_truths, system_manager=system_manager)

        # Step 12: Load knowledge base
        print("Loading knowledge base...")
        with open("data/small_knowledge_base.txt", "r") as kb_file:
            context = kb_file.read()

        # Step 13: Process test queries using resource-aware batch processing
        print("\n=== Testing System with Various Queries ===")
        test_queries = [
            {"query": "What are the environmental effects of deforestation?", "type": "ground_truth_available"},
            {"query": "What is the social impact of deforestation?", "type": "ground_truth_available"},
            {"query": "What is deforestation?", "type": "exploratory"},
            {"query": "How does deforestation cause climate change?", "type": "ground_truth_available"}
        ]

        results = process_query_batch(test_queries, context, system_manager, resource_manager)

        for query_info, result in zip(test_queries, results):
            query = query_info["query"]
            print(f"\nProcessing Results for Query: {query}")
            print("-" * 20)
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
            print(result)
            print("-" * 20)
            if query_info["type"] == "ground_truth_available":
                print("\nEvaluation Metrics:")
                eval_metrics = evaluator.evaluate({query: result}, ground_truths)
                print(f"Similarity Score: {eval_metrics['average_similarity']:.2f}")
            if input("\nWould you like to provide feedback? (yes/no): ").lower() == 'yes':
                feedback_handler.collect_feedback(query, result)

        print("\n=== System Performance Summary ===")
        final_resources = resource_manager.check_resources()
        print("\nResource Utilization:")
        print(f"- CPU Usage: {final_resources['cpu'] * 100:.1f}%")
        print(f"- Memory Usage: {final_resources['memory'] * 100:.1f}%")
        print("\nSystem Information:")
        print(f"Model Path: {neural.model.config._name_or_path}")
        print(f"Cache Location: {os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")
        print("=== End of Run ===")

    except Exception as main_error:
        print(f"Critical initialization error: {main_error}")
        exit(1)
