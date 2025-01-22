from src.symbolic_reasoner import SymbolicReasoner
from src.hybrid_integrator import HybridIntegrator
from src.rule_extractor import RuleExtractor
from src.query_logger import QueryLogger
from src.resource_manager import ResourceManager
from src.feedback_manager import FeedbackManager
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
    # Step 1: Extract rules from the input text
    print("Extracting rules from deforestation.txt...")
    RuleExtractor.extract_rules("data/deforestation.txt", "data/rules.json")
    print("Rules extracted and saved to rules.json.")

    # Step 2: Initialize the Symbolic Reasoner
    print("Initializing Symbolic Reasoner...")
    symbolic = SymbolicReasoner("data/rules.json")

    # Step 3: Initialize the Neural Retriever with a pre-trained LLAMA model
    model_name = "meta-llama/Llama-3.2-1B"  # Replace with the desired model
    neural = NeuralRetriever(model_name)

    # Step 4: Create the Hybrid Integrator
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(symbolic, neural)

    # Step 5: Initialize Query Logger and Resource Manager
    logger = QueryLogger()
    resource_manager = ResourceManager()

    # Step 6: Load knowledge base and process a sample query
    print("Loading knowledge base...")
    with open("data/small_knowledge_base.txt", "r") as kb_file:
        context = kb_file.read()

    query = "What are the environmental effects of deforestation?"
    print(f"Processing query: {query}")

    # Monitor resource usage during query processing
    usage = resource_manager.monitor_resource_usage(
        lambda: integrator.process_query(query, context)
    )

    # Get the result and determine its source
    result = integrator.process_query(query, context)
    source = "symbolic" if "No symbolic match found." not in result else "neural"

    # Log the query, result, and source
    logger.log_query(query, result, source)

    # Output the result
    print(f"Query Result: {result}")
    print(f"Result source: {source}")
    print(f"Resource usage during query processing: {usage}")
    print(f"Model is loaded from: {neural.model.config._name_or_path}")

    # Safely retrieve the cache directory
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Cache directory: {cache_dir}")

    # Step 7: Collect user feedback
    feedback = FeedbackManager()
    rating = int(input("Rate the result (1-5): "))
    comments = input("Comments (optional): ")
    feedback.submit_feedback(query, result, rating, comments)

    print("Thank you for your feedback!")
