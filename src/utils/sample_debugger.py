# src/utils/sample_debugger.py
import random
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SampleDebugger:
    def __init__(self, num_samples_to_print: int = 3):
        self.num_samples_to_print = num_samples_to_print
        self.selected_query_ids: List[str] = []
        self.printed_count: int = 0
        logger.info(f"SampleDebugger initialized to print {self.num_samples_to_print} random samples.")

    def select_random_query_ids(self, all_query_ids: List[str]):
        """
        Randomly selects a specified number of query IDs from the given list.
        """
        if not all_query_ids:
            logger.warning("SampleDebugger: No query IDs provided for random selection.")
            return

        # Ensure we don't try to select more samples than available
        num_to_select = min(self.num_samples_to_print, len(all_query_ids))

        if num_to_select > 0:
            self.selected_query_ids = random.sample(all_query_ids, num_to_select)
            logger.info(f"SampleDebugger: Selected Query IDs for detailed printing: {self.selected_query_ids}")
        else:
            logger.info("SampleDebugger: Not enough query IDs to select any samples.")

    def print_debug_if_selected(self,
                                query_id: str,
                                query_text: str,
                                ground_truth_answer: Any,
                                system_prediction_value: Any,
                                actual_reasoning_path: str,
                                eval_metrics: Optional[Dict[str, float]] = None,
                                dataset_type: Optional[str] = None):
        """
        Prints detailed debug information for the current query if its ID was randomly selected.
        """
        if query_id in self.selected_query_ids and self.printed_count < self.num_samples_to_print:
            print("\n" + "*" * 15 + f" RANDOM SAMPLE DEBUG QID: {query_id} " + "*" * 15)
            print(f"  QUERY           : {query_text}")

            gt_answer_display = str(ground_truth_answer)
            prediction_display = str(system_prediction_value)

            if dataset_type == 'drop':
                # Format DROP ground truth for better readability
                if isinstance(ground_truth_answer, dict):
                    temp_gt_display = {}
                    if ground_truth_answer.get('number') is not None and str(ground_truth_answer.get('number')).strip():
                        temp_gt_display['number'] = ground_truth_answer['number']
                    if ground_truth_answer.get('spans'):
                        temp_gt_display['spans'] = ground_truth_answer['spans']
                    date_gt = ground_truth_answer.get('date', {})
                    if any(str(v).strip() for v in date_gt.values()):
                        temp_gt_display['date'] = date_gt
                    gt_answer_display = str(temp_gt_display if temp_gt_display else ground_truth_answer)

                # System prediction for DROP is expected to be a dict
                prediction_display = str(system_prediction_value)  # Already a dict, str() is for consistency in print

            print(f"  GROUND TRUTH    : {gt_answer_display}")
            print(f"  HySymRAG Output : {prediction_display}")
            print(f"  REASONING PATH  : {actual_reasoning_path}")

            if eval_metrics:
                # Use .get with a default for individual metrics to avoid KeyError if a specific metric wasn't calculated
                print(
                    f"  EM              : {eval_metrics.get('average_exact_match', eval_metrics.get('exact_match', 0.0)):.3f}")
                print(f"  F1              : {eval_metrics.get('average_f1', eval_metrics.get('f1', 0.0)):.3f}")
                if dataset_type != 'drop':
                    print(f"  ROUGE-L         : {eval_metrics.get('average_rougeL', 0.0):.3f}")
            else:
                print("  (Evaluation metrics not available for this sample printout)")

            print("*" * (30 + len(str(query_id)) + 27) + "\n")  # Adjusted length for "RANDOM SAMPLE DEBUG"

            self.printed_count += 1
            # To ensure we print exactly num_samples_to_print distinct queries,
            # we can remove the ID after printing. This means if a query is retried
            # and happens to be selected, it won't be printed again unless selection logic changes.
            # self.selected_query_ids.remove(query_id) # Optional: if you want to ensure distinct QIDs printed