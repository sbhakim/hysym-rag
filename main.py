# main.py
from src.symbolic_reasoner import SymbolicReasoner
from src.hybrid_integrator import HybridIntegrator
from src.rule_extractor import RuleExtractor
from src.query_logger import QueryLogger
from src.resource_manager import ResourceManager
from src.feedback_manager import FeedbackManager
from src.feedback_handler import FeedbackHandler
from src.config_loader import ConfigLoader
from src.query_expander import QueryExpander
from src.evaluation import Evaluation
from src.app import App

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

class NeuralRetriever:
    """
    NeuralRetriever class for retrieving answers from a neural language model.
    """
    def __init__(self, model_name):
        print(f"Initializing Neural Retriever with model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto"
        )
        print(f"Model {model_name} loaded successfully!")

    def retrieve_answer(self, context, question):
        """
        Generate an answer from the neural model given a context and a question.
        """
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Step 1: Initialize system components
    print("\n=== Initializing HySym-RAG System ===")
    # Load configuration first
    print("Loading configuration...")
    config = ConfigLoader.load_config("config.yaml")
    model_name = config["model_name"]

    # Initialize core components in the correct order
    print("Initializing core components...")
    # First, initialize the resource manager as other components depend on it
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager()

    # Initialize other fundamental components
    print("Extracting rules from deforestation.txt...")
    RuleExtractor.extract_rules("data/deforestation.txt", "data/rules.json")

    print("Initializing Symbolic Reasoner...")
    symbolic = SymbolicReasoner("data/rules.json", match_threshold=0.1)

    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(model_name)

    print("Initializing support components...")
    logger = QueryLogger()
    feedback_manager = FeedbackManager()
    expander = QueryExpander()

    # Load evaluation components
    print("Loading evaluation dataset...")
    with open("data/ground_truths.json", "r") as gt_file:
        ground_truths = json.load(gt_file)
    evaluator = Evaluation()

    # Create the hybrid integrator (note: App will pass the expander too)
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(symbolic, neural, resource_manager)

    # Initialize application components
    print("Initializing Application...")
    feedback_handler = FeedbackHandler(feedback_manager)

    # Create the main application with all components
    app = App(
        symbolic=symbolic,
        neural=neural,
        logger=logger,
        feedback=resource_manager,  # Pass resource_manager for resource-aware processing
        evaluator=evaluator,
        expander=expander,
        ground_truths=ground_truths
    )

    # Load knowledge base
    print("Loading knowledge base...")
    with open("data/small_knowledge_base.txt", "r") as kb_file:
        context = kb_file.read()

    # Step 2: Process multiple queries to demonstrate different complexity levels
    print("\n=== Testing System with Various Queries ===")
    test_queries = [
        {
            "query": "What are the environmental effects of deforestation?",
            "type": "ground_truth_available",
            "complexity": "moderate"
        },
        {
            "query": "What is the social impact of deforestation?",
            "type": "ground_truth_available",
            "complexity": "moderate"
        },
        {
            "query": "What is deforestation?",
            "type": "exploratory",
            "complexity": "simple"
        }
    ]

    for query_info in test_queries:
        query = query_info["query"]
        print(f"\nProcessing Query: {query}")
        print(f"Query Type: {query_info['type']}")
        print("-" * 50)

        try:
            # First measure query complexity
            complexity = expander.get_query_complexity(query)
            print(f"Query Complexity Score: {complexity:.4f}")

            # Wrap query processing in resource monitoring
            def process_query():
                return app.run(query, context)  # Now returns (result, source)

            usage = resource_manager.monitor_resource_usage(process_query)

            # Unpack the tuple (result, source)
            result, source = process_query()

            logger.log_query(query, result, source)

            # Output comprehensive results
            print(f"\nProcessing Results:")
            print(f"Source: {source}")
            print(f"Resource Usage: {usage}")
            print("\nResult Preview:")
            print("-" * 20)
            # Assuming result is a list; print first 200 characters of its first element
            print(f"{result[0][:200]}...")
            print("-" * 20)

            if query_info["type"] == "ground_truth_available":
                print("\nEvaluation Metrics:")
                evaluation = evaluator.evaluate({query: result}, ground_truths)
                print(f"Similarity Score: {evaluation['average_similarity']:.2f}")

            if input("\nWould you like to provide feedback? (yes/no): ").lower() == 'yes':
                feedback_handler.collect_feedback(query, result)

        except KeyError as e:
            print(f"Error: Missing ground truth for query evaluation - {str(e)}")
            continue
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Continuing with next query...")
            continue

    # Output final system statistics
    print("\n=== System Performance Summary ===")
    final_resources = resource_manager.check_resources()

    print("\nResource Utilization:")
    print(f"- CPU Usage: {final_resources['cpu_usage'] * 100:.1f}%")
    print(f"- Memory Usage: {final_resources['memory_utilization'] * 100:.1f}%")

    print("\nSystem Information:")
    print(f"Model Path: {neural.model.config._name_or_path}")
    print(f"Cache Location: {os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")
