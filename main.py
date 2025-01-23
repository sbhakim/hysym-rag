from src.symbolic_reasoner import SymbolicReasoner
from src.hybrid_integrator import HybridIntegrator
from src.rule_extractor import RuleExtractor
from src.query_logger import QueryLogger
from src.resource_manager import ResourceManager
from src.feedback_manager import FeedbackManager
from src.feedback_handler import FeedbackHandler
from src.config_loader import ConfigLoader
from src.app import App

from transformers import AutoTokenizer, AutoModelForCausalLM
import os


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
    # Step 1: Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config("config.yaml")
    model_name = config["model_name"]

    # Step 2: Extract rules from the input text
    print("Extracting rules from deforestation.txt...")
    RuleExtractor.extract_rules("data/deforestation.txt", "data/rules.json")
    print("Rules extracted and saved to rules.json.")

    # Step 3: Initialize components
    print("Initializing Symbolic Reasoner...")
    symbolic = SymbolicReasoner("data/rules.json")
    neural = NeuralRetriever(model_name)
    logger = QueryLogger()
    feedback_manager = FeedbackManager()
    resource_manager = ResourceManager()

    # Step 4: Create Hybrid Integrator
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(symbolic, neural)

    # Step 5: Initialize App and FeedbackHandler
    print("Initializing App and FeedbackHandler...")
    feedback_handler = FeedbackHandler(feedback_manager)
    app = App(symbolic, neural, logger, feedback_manager, None, None)

    # Step 6: Load knowledge base
    print("Loading knowledge base...")
    with open("data/small_knowledge_base.txt", "r") as kb_file:
        context = kb_file.read()

    # Step 7: Process query
    query = "What are the environmental effects of deforestation?"
    print(f"Processing query: {query}")

    # Monitor resource usage during query processing
    usage = resource_manager.monitor_resource_usage(
        lambda: app.run(query, context)
    )

    # Process query through App
    result = app.run(query, context)
    source = "symbolic" if "No symbolic match found." not in result else "neural"

    # Log the query and result
    logger.log_query(query, result, source)

    # Output the result and resource usage
    print(f"Query Result: {result}")
    print(f"Result source: {source}")
    print(f"Resource usage during query processing: {usage}")
    print(f"Model is loaded from: {neural.model.config._name_or_path}")

    # Safely retrieve the cache directory
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Cache directory: {cache_dir}")

    # Step 8: Collect user feedback (optional)
    feedback_handler.collect_feedback(query, result)
