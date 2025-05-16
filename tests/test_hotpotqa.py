# tests/test_hotpotqa.py

import os
from datetime import time

import torch
import logging
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from src.config.config_loader import ConfigLoader
from src.reasoners.neural_retriever import NeuralRetriever
from src.reasoners.networkx_symbolic_reasoner_base import GraphSymbolicReasoner
from src.integrators.hybrid_integrator import HybridIntegrator
from src.resources.resource_manager import ResourceManager
from src.utils.evaluation import Evaluation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HotpotExample:
    """Structure for HotpotQA examples with type hints"""
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: List[tuple]
    context: List[tuple]


class HotpotQAEvaluator:
    def __init__(self, batch_size: int = 16, num_samples: int = 10):
        """
        Initialize HotpotQA evaluator with configurable parameters.

        Args:
            batch_size: Size of batches for processing
            num_samples: Number of samples to evaluate
        """
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup paths and configuration
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = os.path.join(self.ROOT_DIR, "src/config/config.yaml")
        self.config = ConfigLoader.load_config(self.config_path)

        # Initialize components
        self.initialize_components()
        self.evaluator = Evaluation()

        # Initialize performance tracking
        self.performance_metrics = {
            'answer_accuracy': defaultdict(list),
            'supporting_facts_accuracy': defaultdict(list),
            'processing_time': defaultdict(list),
            'resource_usage': defaultdict(list),
            'question_types': defaultdict(int)
        }

    def initialize_components(self):
        """Initialize all required system components with proper error handling."""
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

        except Exception as e:
            logger.error(f"Error during component initialization: {str(e)}")
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def prepare_context(self, example: Dict[str, Any]) -> str:
        """
        Prepare context from HotpotQA example with supporting facts highlighting.

        Args:
            example: HotpotQA example dictionary

        Returns:
            Processed context string with supporting facts marked
        """
        try:
            # Extract supporting facts
            supporting_facts = {
                (title, sent_idx)
                for title, sent_idx in example.get('supporting_facts', [])
            }

            # Process each document
            processed_documents = []
            for doc_title, sentences in example.get('context', []):
                # Format sentences with supporting facts marked
                formatted_sentences = []
                for idx, sentence in enumerate(sentences):
                    if (doc_title, idx) in supporting_facts:
                        formatted_sentences.append(f"[SUPPORT] {sentence}")
                    else:
                        formatted_sentences.append(sentence)

                # Create document block
                document_text = (
                    f"\nDocument: {doc_title}\n"
                    f"{'-' * (len(doc_title) + 10)}\n"
                    f"{' '.join(formatted_sentences)}"
                )
                processed_documents.append(document_text)

            # Combine all documents
            final_context = "\n\n".join(processed_documents)

            # Add supporting documents summary
            supporting_docs = {
                title for title, _ in supporting_facts
            }
            if supporting_docs:
                final_context = (
                    f"Supporting Documents: {', '.join(supporting_docs)}\n"
                    f"{'-' * 50}\n{final_context}"
                )

            return final_context

        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return ""

    def process_batch(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of HotpotQA examples.

        Args:
            examples: List of HotpotQA examples

        Returns:
            List of processed results
        """
        results = []
        try:
            # Prepare batch data
            contexts = []
            questions = []

            for example in examples:
                context = self.prepare_context(example)
                if context:
                    contexts.append(context)
                    questions.append(example['question'])

            # Process batch through hybrid integrator
            if contexts and questions:
                batch_results = [
                    self.hybrid_integrator.process_query(q, c, 0.7)
                    for q, c in zip(questions, contexts)
                ]
                results.extend(batch_results)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")

        return results

    def evaluate_sample(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single HotpotQA example. **Added ROUGE and BLEU metrics to output in console.**

        Args:
            example: HotpotQA example dictionary

        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info(f"Processing example ID: {example.get('id', 'unknown')}")

            # Basic validation
            question = example.get('question', '')
            answer = example.get('answer', '')
            if not question or not answer:
                logger.warning("Missing question or answer")
                return None

            # Process context
            context = self.prepare_context(example)
            if not context:
                logger.warning("Failed to generate context")
                return None

            # Track resource usage
            initial_resources = self.resource_manager.check_resources()

            # Process query
            start_time = time.time()
            result = self.hybrid_integrator.process_query(
                question,
                context,
                query_complexity=0.7
            )
            processing_time = time.time() - start_time

            # Calculate resource usage
            final_resources = self.resource_manager.check_resources()
            resource_delta = {
                key: final_resources[key] - initial_resources[key]
                for key in final_resources
            }

            # Extract response
            if isinstance(result, tuple) and len(result) == 2:
                response, source = result
                if isinstance(response, list):
                    response = " ".join(response)

                # Evaluate results
                eval_result = self.evaluator.evaluate(
                    predictions={question: response},
                    ground_truths={question: answer}
                )

                # Prepare result dictionary
                return {
                    'id': example.get('id', 'unknown'),
                    'question': question,
                    'predicted': response,
                    'actual': answer,
                    'source': source,
                    'metrics': eval_result,
                    'processing_time': processing_time,
                    'resource_usage': resource_delta,
                    'question_type': example.get('type', 'unknown')
                }
            else:
                logger.error("Unexpected result format from hybrid integrator")
                return None

        except Exception as e:
            logger.error(f"Error processing sample: {str(e)}")
            return None

    def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Run evaluation on HotpotQA dataset.

        Returns:
            List of evaluation results
        """
        try:
            # Load dataset
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            eval_set = dataset.select(range(min(self.num_samples, len(dataset))))

            results = []
            success_count = 0

            # Process examples in batches
            for i in range(0, len(eval_set), self.batch_size):
                batch = eval_set[i:min(i + self.batch_size, len(eval_set))]

                # Process batch
                for example in tqdm(batch, desc=f"Processing batch {i // self.batch_size + 1}"):
                    result = self.evaluate_sample(example)
                    if result:
                        results.append(result)
                        success_count += 1

                        # Update metrics
                        self._update_metrics(result)

            # Calculate final metrics
            self._calculate_final_metrics(success_count, len(eval_set))

            return results

        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise

    def _update_metrics(self, result: Dict[str, Any]):
        """Update running metrics with new result."""
        question_type = result['question_type']
        self.performance_metrics['question_types'][question_type] += 1
        self.performance_metrics['answer_accuracy'][question_type].append(
            result['metrics']['average_f1']
        )
        self.performance_metrics['processing_time'][question_type].append(
            result['processing_time']
        )
        self.performance_metrics['resource_usage'][question_type].append(
            result['resource_usage']
        )

    def _calculate_final_metrics(self, success_count: int, total_count: int):
        """Calculate and store final evaluation metrics."""
        self.final_metrics = {
            'total_samples': total_count,
            'success_rate': (success_count / total_count) * 100,
            'average_accuracy': {
                qtype: sum(scores) / len(scores)
                for qtype, scores in self.performance_metrics['answer_accuracy'].items()
            },
            'average_processing_time': {
                qtype: sum(times) / len(times)
                for qtype, times in self.performance_metrics['processing_time'].items()
            },
            'question_type_distribution': {
                qtype: count / total_count * 100
                for qtype, count in self.performance_metrics['question_types'].items()
            }
        }


def main():
    """Main evaluation function."""
    torch.manual_seed(42)
    evaluator = HotpotQAEvaluator(num_samples=10)

    try:
        results = evaluator.run_evaluation()

        print("\n=== Evaluation Results ===")
        print(f"Total Samples: {evaluator.final_metrics['total_samples']}")
        print(f"Success Rate: {evaluator.final_metrics['success_rate']:.2f}%")

        print("\nPerformance by Question Type:")
        for qtype, accuracy in evaluator.final_metrics['average_accuracy'].items():
            print(f"{qtype}: {accuracy:.2f}% accuracy")

        # Print ROUGE and BLEU scores (example - you might want to aggregate these in final_metrics as well)
        avg_rougeL = np.mean([res['metrics']['average_rougeL'] for res in results if 'metrics' in res and 'average_rougeL' in res['metrics']])
        avg_bleu = np.mean([res['metrics']['average_bleu'] for res in results if 'metrics' in res and 'average_bleu' in res['metrics']])
        print(f"\nAverage ROUGE-L Score: {avg_rougeL:.2f}")
        print(f"Average BLEU Score: {avg_bleu:.2f}")


    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()