# main.py

from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.utils.rule_extractor import RuleExtractor
from src.queries.query_logger import QueryLogger
from src.resources.resource_manager import ResourceManager
from src.feedback.feedback_manager import FeedbackManager
from src.feedback.feedback_handler import FeedbackHandler
from src.config.config_loader import ConfigLoader
from src.queries.query_expander import QueryExpander
from src.utils.evaluation import Evaluation
from src.app import App
from src.system.system_control_manager import SystemControlManager, UnifiedResponseAggregator

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os
import json
import torch

class NeuralRetriever:
    """
    NeuralRetriever class for retrieving answers from a neural language model.
    Also provides an encode() method for obtaining neural embeddings.
    """
    def __init__(self, model_name):
        print(f"Initializing Neural Retriever with model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto"
        )
        # Load a SentenceTransformer for encoding
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Model {model_name} loaded successfully!")

    def retrieve_answer(self, context, question):
        """
        Generate an answer from the neural model given a context and a question.
        """
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def encode(self, text):
        """
        Encode the text into an embedding tensor using the SentenceTransformer.
        Force the tensor onto CUDA if available.
        """
        emb = self.encoder.encode(text, convert_to_tensor=True)
        if torch.cuda.is_available():
            emb = emb.to('cuda')
        return emb


if __name__ == "__main__":
    print("\n=== Initializing HySym-RAG System ===")
    # 1. Load config (contains LLM model name, etc.)
    print("Loading configuration...")
    config = ConfigLoader.load_config("src/config/config.yaml")
    model_name = config["model_name"]

    # 2. Initialize ResourceManager (others depend on it)
    print("Initializing Resource Manager...")
    resource_manager = ResourceManager("src/config/resource_config.yaml")

    # 3. Extract symbolic rules into data/rules.json
    print("Extracting rules from deforestation.txt...")
    RuleExtractor.extract_rules("data/deforestation.txt", "data/rules.json")

    # 4. Initialize the graph-based symbolic reasoner (max_hops set to 5)
    print("Initializing Graph-Based Symbolic Reasoner...")
    symbolic = GraphSymbolicReasoner(
        "data/rules.json",
        match_threshold=0.25,  # modest threshold
        max_hops=5             # allow deeper multi-hop chaining
    )

    # 5. Initialize the Neural Retriever
    print("Initializing Neural Retriever...")
    neural = NeuralRetriever(model_name)

    # 6. Support components: logger, feedback manager, query expander, evaluation
    print("Initializing support components...")
    logger = QueryLogger()
    feedback_manager = FeedbackManager()
    print("Initializing QueryExpander...")
    expander = QueryExpander(
        complexity_config="src/config/complexity_rules.yaml"
    )
    print("Loading evaluation dataset...")
    with open("data/ground_truths.json", "r") as gt_file:
        ground_truths = json.load(gt_file)
    evaluator = Evaluation()

    # 7. Hybrid integrator (decides symbolic vs. neural calls)
    print("Creating Hybrid Integrator...")
    integrator = HybridIntegrator(symbolic, neural, resource_manager, expander)

    # 8. Initialize UnifiedResponseAggregator and SystemControlManager
    print("Initializing System Control Components...")
    aggregator = UnifiedResponseAggregator(include_explanations=True)
    system_manager = SystemControlManager(
        hybrid_integrator=integrator,
        resource_manager=resource_manager,
        aggregator=aggregator,
        error_retry_limit=2,
        max_query_time=10
    )

    # 9. Main application + feedback handling
    print("Initializing Application...")
    feedback_handler = FeedbackHandler(feedback_manager)
    app = App(
        symbolic=symbolic,
        neural=neural,
        logger=logger,
        feedback=resource_manager,  # Resource-aware scheduling
        evaluator=evaluator,
        expander=expander,
        ground_truths=ground_truths,
        system_manager=system_manager  # Add system_manager to App
    )

    # 10. Load knowledge base (for neural context)
    print("Loading knowledge base...")
    with open("data/small_knowledge_base.txt", "r") as kb_file:
        context = kb_file.read()

    # 11. Test queries using SystemControlManager
    print("\n=== Testing System with Various Queries ===")
    test_queries = [
        {"query": "What are the environmental effects of deforestation?", "type": "ground_truth_available"},
        {"query": "What is the social impact of deforestation?", "type": "ground_truth_available"},
        {"query": "What is deforestation?", "type": "exploratory"}
    ]

    for q_info in test_queries:
        query = q_info["query"]
        print(f"\nProcessing Query: {query}")
        print(f"Query Type: {q_info['type']}")
        print("-" * 50)
        try:
            # 1) Compute query complexity
            complexity = expander.get_query_complexity(query)
            print(f"Query Complexity Score: {complexity:.4f}")

            # 2) Process query through SystemControlManager
            final_answer = system_manager.process_query_with_fallback(query, context)

            # 3) Resource usage monitoring
            usage = resource_manager.check_resources()

            # 4) Logging
            logger.log_query(query, final_answer, "hybrid", complexity, usage)

            # 5) Output result
            print("\nProcessing Results:")
            print("-" * 20)
            print(final_answer)
            print("-" * 20)

            # 6) Evaluate if ground truth is available
            if q_info["type"] == "ground_truth_available":
                print("\nEvaluation Metrics:")
                eval_metrics = evaluator.evaluate({query: final_answer}, ground_truths)
                print(f"Similarity Score: {eval_metrics['average_similarity']:.2f}")

            # 7) Collect feedback if user desires
            if input("\nWould you like to provide feedback? (yes/no): ").lower() == 'yes':
                feedback_handler.collect_feedback(query, final_answer)

        except KeyError as e:
            print(f"Error: Missing ground truth for query evaluation - {str(e)}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

    # Final summary
    print("\n=== System Performance Summary ===")
    final_resources = resource_manager.check_resources()
    print("\nResource Utilization:")
    print(f"- CPU Usage: {final_resources['cpu'] * 100:.1f}%")
    print(f"- Memory Usage: {final_resources['memory'] * 100:.1f}%")
    print("\nSystem Information:")
    print(f"Model Path: {neural.model.config._name_or_path}")
    print(f"Cache Location: {os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")
    print("=== End of Run ===")