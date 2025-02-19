# tests/test_hotpotqa.py

import os
import torch
import logging
from datasets import load_dataset
from tqdm import tqdm
from src.config.config_loader import ConfigLoader
from src.reasoners.neural_retriever import NeuralRetriever
from src.reasoners.networkx_symbolic_reasoner import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.resources.resource_manager import ResourceManager
from src.utils.evaluation import Evaluation

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class HotpotQAEvaluator:
    def __init__(self, batch_size=16, num_samples=10):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup paths
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = os.path.join(self.ROOT_DIR, "src/config/config.yaml")
        self.config = ConfigLoader.load_config(self.config_path)

        # Initialize components
        self.initialize_components()
        self.evaluator = Evaluation()
        self.metrics = {
            'exact_match': 0,
            'f1_score': 0,
            'total_samples': 0,
            'success_rate': 0
        }

    def initialize_components(self):
        try:
            logger.info("Initializing Resource Manager...")
            self.resource_manager = ResourceManager(
                config_path=os.path.join(self.ROOT_DIR, "src/config/resource_config.yaml"),
                enable_performance_tracking=True
            )

            logger.info("Initializing Symbolic Reasoner...")
            self.symbolic_reasoner = GraphSymbolicReasoner(
                rules_file=os.path.join(self.ROOT_DIR, "data/rules.json"),
                match_threshold=0.25,
                max_hops=5
            )

            logger.info("Initializing Neural Retriever...")
            self.neural_retriever = NeuralRetriever(
                self.config["model_name"],
                use_quantization=True
            )

            logger.info("Initializing Hybrid Integrator...")
            self.hybrid_integrator = HybridIntegrator(
                self.symbolic_reasoner,
                self.neural_retriever,
                self.resource_manager
            )

            logger.info("All components initialized successfully!")
        except Exception as e:
            logger.error(f"Error during component initialization: {str(e)}")
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def prepare_context(self, example):
        try:
            logger.debug(f"Processing example with ID: {example.get('id', 'unknown')}")
            context_list = example.get('context', [])
            logger.debug(f"Example['context'] type: {type(context_list)}")  # Log type
            logger.debug(f"Example['context'] length: {len(context_list)}")  # Log length
            if context_list:
                logger.debug(f"Example['context'][0] type: {type(context_list[0])}")  # Log type of first element
                logger.debug(f"Example['context'][0] content: {context_list[0]}")  # Log content of first element

            sf = example.get('supporting_facts', [])
            supporting_facts_set = set()
            if isinstance(sf, list) and sf:  # Check if sf is not empty
                if isinstance(sf[0], (list, tuple)) and len(sf[0]) == 2:  # Check if sf[0] is a list/tuple of length 2
                    supporting_facts_set = {tuple(item) for item in sf if len(item) == 2}

            contexts = []
            for document_group in context_list:  # Iterate through the outer list, now document_group
                if isinstance(document_group, list) and len(
                        document_group) == 2:  # Check if document_group is a list of length 2
                    document_info, sentences = document_group  # Unpack document_info and sentences
                    if isinstance(document_info, list) and len(
                            document_info) >= 1:  # Check if document_info is a list and has at least a title
                        title = document_info[0]  # Extract title from document_info[0]
                        if isinstance(sentences, list):  # Check if sentences is a list
                            formatted_sentences = ["* " + s if (title, idx) in supporting_facts_set else s for idx, s in
                                                   enumerate(sentences)]
                            document_text = f"Document: {title}\n{'-' * (len(title) + 10)}\n{' '.join(formatted_sentences)}"
                            contexts.append(document_text)
                        else:
                            logger.warning(f"Skipping document due to sentences not being a list: {document_group}")
                    else:
                        logger.warning(f"Skipping document due to malformed document_info: {document_group}")
                else:
                    logger.warning(f"Skipping malformed document group: {document_group}")

            final_context = "\n\n".join(contexts)

            if supporting_facts_set:
                supporting_docs = {title for title, _ in supporting_facts_set}
                final_context = f"Supporting Documents: {', '.join(supporting_docs)}\n{'-' * 50}\n{final_context}"

            logger.debug(f"Created context with {len(contexts)} documents")
            return final_context

        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return ""


    def evaluate_sample(self, example):
        try:
            logger.info(f"Processing example ID: {example.get('id', 'unknown')}")
            question = example.get('question', '')
            answer = example.get('answer', '')

            if not question or not answer:
                logger.warning("Missing question or answer")
                return None

            context = self.prepare_context(example)
            if not context:
                logger.warning("Failed to generate context")
                return None

            # Fix: Calling process_query with positional arguments
            result = self.hybrid_integrator.process_query(question, context, 0.7)

            if isinstance(result, tuple) and len(result) == 2:
                response, source = result
                if isinstance(response, list):
                    response = " ".join(response)
                eval_result = self.evaluator.evaluate(
                    predictions={question: response},
                    ground_truths={question: answer}
                )
                return {
                    'id': example.get('id', 'unknown'),
                    'question': question,
                    'predicted': response,
                    'actual': answer,
                    'source': source,
                    'metrics': eval_result
                }
            else:
                logger.error("Unexpected result format from hybrid integrator")
                return None
        except Exception as e:
            logger.error(f"Error processing sample: {str(e)}")
            return None

    def run_evaluation(self):
        try:
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            eval_set = dataset.select(range(min(self.num_samples, len(dataset))))
            results = []
            success_count = 0

            for example in tqdm(eval_set, desc="Evaluating samples"):
                result = self.evaluate_sample(example)
                if result:
                    results.append(result)
                    success_count += 1

            self.metrics['total_samples'] = len(eval_set)
            self.metrics['success_rate'] = (success_count / len(eval_set)) * 100
            return results
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise


def main():
    torch.manual_seed(42)
    evaluator = HotpotQAEvaluator(num_samples=10)
    try:
        results = evaluator.run_evaluation()
        print("\n=== Evaluation Results ===")
        print(f"Total Samples: {evaluator.metrics['total_samples']}")
        print(f"Success Rate: {evaluator.metrics['success_rate']:.2f}%")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()
