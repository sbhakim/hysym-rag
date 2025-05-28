# src/utils/data_loaders.py

import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def load_hotpotqa(hotpotqa_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the HotpotQA dataset.
    Each sample includes a query, ground-truth answer,
    combined context, and a 'type' = 'ground_truth_available_hotpotqa'.
    """
    if not os.path.exists(hotpotqa_path):
        logger.error(f"HotpotQA dataset file not found at: {hotpotqa_path}")
        return []
    dataset: List[Dict[str, Any]] = []
    try:
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse HotpotQA JSON from {hotpotqa_path}: {e}")
        return []

    count = 0
    for example in data:
        if not all(k in example for k in ['question', 'answer', 'supporting_facts', 'context', '_id']):
            logger.warning(f"Skipping invalid HotpotQA example (missing keys): {example.get('_id', 'Unknown ID')}")
            continue

        question = example['question']
        answer = example['answer']
        supporting_facts = example['supporting_facts']

        context_str_parts = []
        for title, sents in example.get('context', []):
            if isinstance(title, str) and isinstance(sents, list):
                combined_sents = " ".join(str(s) for s in sents)
                context_str_parts.append(f"{title}: {combined_sents}")
        context_str = "\n".join(context_str_parts)

        dataset.append({
            "query_id": example['_id'],
            "query": question,
            "answer": answer,
            "context": context_str,
            "type": "ground_truth_available_hotpotqa",
            "supporting_facts": supporting_facts
        })
        count += 1
        if max_samples and count >= max_samples:
            logger.info(f"Loaded {count} HotpotQA samples (max requested: {max_samples}).")
            break
    logger.info(f"Finished loading HotpotQA. Total samples: {len(dataset)}.")
    return dataset


def load_drop_dataset(drop_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads a portion of the DROP dataset.
    """
    if not os.path.exists(drop_path):
        logger.error(f"DROP dataset file not found at: {drop_path}")
        return []
    dataset: List[Dict[str, Any]] = []
    try:
        with open(drop_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse DROP JSON from {drop_path}: {e}")
        return []

    total_loaded_qas = 0
    for passage_id, passage_content in data.items():
        if not isinstance(passage_content,
                          dict) or 'passage' not in passage_content or 'qa_pairs' not in passage_content:
            logger.warning(f"Skipping invalid passage structure for passage_id: {passage_id}")
            continue

        passage_text = passage_content['passage']
        if not isinstance(passage_content['qa_pairs'], list):
            logger.warning(f"Invalid qa_pairs format (not a list) for passage_id: {passage_id}")
            continue

        for qa_pair_idx, qa_pair in enumerate(passage_content['qa_pairs']):
            if not isinstance(qa_pair, dict) or 'question' not in qa_pair or 'answer' not in qa_pair:
                logger.warning(f"Skipping invalid qa_pair structure in passage_id {passage_id}, index {qa_pair_idx}")
                continue

            question = qa_pair['question']
            answer_obj = qa_pair['answer']
            query_id = qa_pair.get("query_id", f"{passage_id}-{qa_pair_idx}")

            dataset.append({
                "query_id": query_id,
                "query": question,
                "context": passage_text,
                "answer": answer_obj,
                "type": "ground_truth_available_drop"
            })
            total_loaded_qas += 1
            if max_samples and total_loaded_qas >= max_samples:
                logger.info(f"Loaded {total_loaded_qas} DROP samples (max requested: {max_samples}).")
                return dataset

        if max_samples and total_loaded_qas >= max_samples:
            break

    logger.info(f"Finished loading DROP. Total samples: {len(dataset)}.")
    return dataset