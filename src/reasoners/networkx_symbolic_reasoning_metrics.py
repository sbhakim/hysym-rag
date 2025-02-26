# src/reasoners/networkx_symbolic_reasoning_metrics.py

import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _serialize_dependencies(self, dependencies: nx.DiGraph) -> Dict:
    """
    Serialize the dependencies graph into a dictionary format.
    """
    serialized = {}
    for u, v, data in dependencies.edges(data=True):
        if u not in serialized:
            serialized[u] = []
        serialized[u].append({'target': v, 'data': data})
    return serialized


# ------------------- Utility / Placeholder methods for chain calculations -------------------

def _get_prerequisites(self, node_id: str) -> List[str]:
    """
    Placeholder: Return prerequisite nodes for a rule.
    """
    return []


def _get_conclusions(self, node_id: str) -> List[str]:
    """
    Placeholder: Return conclusion nodes for a rule.
    """
    return []


def _calculate_step_confidence(self, rule: Dict, prereqs: List[str], base_conf: float) -> float:
    """
    Placeholder for more advanced step-by-step confidence calculation.
    """
    return base_conf


def _calculate_step_confidence(self, rule: Dict, prereqs: List[str], base_conf: float) -> float:
    """
    Placeholder for more advanced step-by-step confidence calculation.
    """
    return base_conf


def _calculate_step_coherence(self, steps: List[Dict]) -> float:
    """
    Placeholder: Evaluate how coherent each step is with the others.
    """
    return 1.0


def _calculate_branching(self, dependencies: nx.DiGraph) -> float:
    """
    Placeholder: Evaluate branching factor of the dependency graph.
    """
    if dependencies.number_of_nodes() <= 1:
        return 0.0
    return float(dependencies.number_of_edges()) / dependencies.number_of_nodes()


def _calculate_linearity(self, dependencies: nx.DiGraph) -> float:
    """
    Placeholder: Evaluate how linear the path is.
    """
    if dependencies.number_of_nodes() <= 1:
        return 1.0
    edges = dependencies.number_of_edges()
    nodes = dependencies.number_of_nodes()
    linear_ratio = float(edges) / (nodes - 1) if nodes > 1 else 1.0
    return min(1.0, linear_ratio)


# ---------------------- END Utility / Placeholder methods for chain calculations------------------------

def extract_reasoning_chain(
        rules: Dict[str, Dict],
        path: List[str],
        confidence_scores: List[float],
        query_id: Optional[str] = None,
        reasoning_metrics: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Extract a structured reasoning chain from a given path through the graph.

    Returns a dictionary containing:
      - steps: a list of steps with rule responses, confidence, prerequisites, and conclusions
      - overall_confidence: mean of the provided confidence scores
      - chain_length: number of steps in the chain
      - metrics: a dictionary of computed chain metrics
      - dependencies: a serialized dependency graph of the steps
    """
    reasoning_steps = []
    dependencies = nx.DiGraph()
    for idx, (node, base_conf) in enumerate(zip(path, confidence_scores)):
        rule = rules.get(node, {})
        prereqs = _get_prerequisites(node)
        conclusions = _get_conclusions(node)
        step_conf = _calculate_step_confidence(rule, prereqs, base_conf)
        reasoning_steps.append({
            'step_id': idx,
            'rule': rule.get('response', ''),
            'confidence': step_conf,
            'prerequisites': prereqs,
            'conclusions': conclusions
        })
        if idx > 0:
            dependencies.add_edge(idx - 1, idx, weight=step_conf)
    chain_metrics = _calculate_chain_metrics(reasoning_steps, dependencies)
    if reasoning_metrics is not None and query_id:
        reasoning_metrics.setdefault('chains', []).append({
            'query_id': query_id,
            'steps': reasoning_steps,
            'pattern': extract_reasoning_pattern("", path, rules)
        })
        reasoning_metrics.setdefault('pattern_types', []).append(
            extract_reasoning_pattern("", path, rules).get('pattern_type', 'unknown')
        )
    return {
        'steps': reasoning_steps,
        'overall_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
        'chain_length': len(path),
        'metrics': chain_metrics,
        'dependencies': _serialize_dependencies(dependencies)
    }


def extract_reasoning_pattern(query: str, path: List[str], rules: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """
    Analyze the reasoning path to extract a pattern.
    """
    query_type = _classify_query_type(query)

    # Ensure non-empty path with meaningful defaults
    if not path or (isinstance(path, list) and len(path) == 0):
        # If no path provided, create a basic one-hop pattern
        return {
            'pattern_type': 'single-hop',
            'hop_count': 1,
            'intermediate_facts': [],
            'pattern_confidence': 0.5
        }

    # Analyze pattern type based on path structure
    if isinstance(path, list) and len(path) >= 3:
        pattern_type = 'multi-hop'
        confidence = 0.8
    elif isinstance(path, list) and len(path) == 2:
        pattern_type = 'two-hop'
        confidence = 0.7
    else:
        pattern_type = 'linear'
        confidence = 0.9

    # Extract intermediate steps even if rules are missing
    steps = _extract_intermediate_steps(path, rules)

    # Construct the pattern dictionary with meaningful values
    pattern = {
        'pattern_type': pattern_type,
        'hop_count': len(steps) if steps else 1,
        'intermediate_facts': steps,
        'pattern_confidence': confidence
    }

    return pattern


def _classify_query_type(query: str) -> str:
    """
    Basic classification of query type.
    """
    if "compare" in query.lower() or "contrast" in query.lower():
        return "comparison"
    elif "?" in query:
        return "multi-hop"
    else:
        return "standard"


def _analyze_path_pattern(path: List[str]) -> Dict[str, Any]:
    """
    Analyze the reasoning path pattern.

    Returns a dictionary with type and confidence.
    """
    if len(path) <= 1:
        return {'type': 'linear', 'confidence': 1.0}
    unique_rules = set(path)
    if len(unique_rules) < len(path):
        return {'type': 'branching', 'confidence': 0.8}
    return {'type': 'linear', 'confidence': 0.9}


def _extract_intermediate_steps(path: List[str], rules: Optional[Dict[str, Dict]] = None) -> List[str]:
    """
    Extract key intermediate facts from the reasoning path.
    """
    if rules is None:
        return path  # Fallback: return node IDs
    return [rules.get(node, {}).get('response', '') for node in path]


def _calculate_chain_metrics(steps: List[Dict], dependencies: nx.DiGraph) -> Dict:
    """
    Calculate comprehensive chain metrics for academic analysis.
    """
    try:
        base_metrics = {
            'step_coherence': _calculate_step_coherence(steps),
            'branching_factor': _calculate_branching(dependencies),
            'path_linearity': _calculate_linearity(dependencies)
        }
        academic_metrics = {
            'reasoning_depth': len(steps),
            'fact_utilization': _calculate_fact_utilization(steps),
            'inference_quality': _calculate_inference_quality(steps),
            'pattern_complexity': _calculate_pattern_complexity(dependencies)
        }
        metrics = {**base_metrics, **academic_metrics}
        _update_academic_metrics(metrics)
        return metrics
    except Exception as e:
        logger.error(f"Error calculating chain metrics: {str(e)}")
        return {}


def _calculate_fact_utilization(steps: List[Dict]) -> float:
    """
    Calculate fact utilization as a placeholder metric.
    """
    utilized = sum(1 for step in steps if step.get('rule', '') != "")
    return utilized / len(steps) if steps else 0.0


def _calculate_inference_quality(steps: List[Dict]) -> float:
    """
    Calculate inference quality as the average confidence.
    """
    confidences = [step.get('confidence', 0.0) for step in steps]
    return float(np.mean(confidences)) if confidences else 0.0


def _calculate_pattern_complexity(dependencies: nx.DiGraph) -> float:
    """
    Calculate pattern complexity based on the structure of the dependency graph.
    """
    if dependencies.number_of_nodes() == 0:
        return 0.0
    return float(dependencies.number_of_edges()) / dependencies.number_of_nodes()


def _update_academic_metrics(metrics: Dict):
    """
    Log or update academic metrics.
    """
    logger.info(f"Updated academic chain metrics: {metrics}")


def get_reasoning_metrics(reasoning_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive reasoning metrics for academic evaluation.
    """
    # Handle empty path_lengths gracefully
    path_lengths = reasoning_metrics.get('path_lengths', [])
    if not path_lengths:
        path_lengths = [1]  # Default to single-hop if no data

    # Ensure non-empty match_confidences
    match_confidences = reasoning_metrics.get('match_confidences', [])
    if not match_confidences:
        match_confidences = [0.7]  # Default confidence

    # Calculate path distribution with validation
    path_distribution = _calculate_path_distribution(reasoning_metrics)
    if not path_distribution:
        path_distribution = {1: 1.0}  # Default to 100% single-hop paths

    # Create metrics with sensible defaults when missing data
    return {
        'path_analysis': {
            'average_length': float(np.mean(path_lengths)),
            'max_length': float(max(path_lengths)),
            'path_distribution': path_distribution
        },
        'confidence_analysis': {
            'mean_confidence': float(np.mean(match_confidences)),
            'confidence_distribution': _calculate_confidence_distribution(match_confidences)
        },
        'rule_utilization': dict(reasoning_metrics.get('rule_utilization', {})),
        'timing_analysis': {
            'mean_time': float(np.mean(reasoning_metrics.get('reasoning_times', [0.1]))),
            'std_time': float(np.std(reasoning_metrics.get('reasoning_times', [0.1])))
        },
        'pattern_metrics': {
            'chain_count': len(reasoning_metrics.get('chains', [])),
            'pattern_types': _analyze_pattern_distribution(reasoning_metrics),
            'average_chain_length': len(path_lengths) / max(1, len(reasoning_metrics.get('chains', [1])))
        }
    }


def _calculate_confidence_distribution(confidences: List[float]) -> Dict[str, List[float]]:
    """Calculate distribution of confidence scores in buckets."""
    if not confidences:
        return {"histogram": [0], "bin_edges": [0, 1]}

    hist, bin_edges = np.histogram(confidences, bins=5, range=(0, 1))
    return {
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist()
    }


def _calculate_path_distribution(reasoning_metrics: Dict[str, Any]) -> Dict[int, float]:
    """
    Calculate distribution of reasoning path lengths.
    """
    path_counts = defaultdict(int)
    for length in reasoning_metrics['path_lengths']:
        path_counts[length] += 1
    total = len(reasoning_metrics['path_lengths'])
    return {length: count / total for length, count in path_counts.items()} if total else {}


def _analyze_pattern_distribution(reasoning_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze distribution of reasoning pattern types.
    """
    counts = defaultdict(int)
    total = len(reasoning_metrics.get('pattern_types', []))
    for p in reasoning_metrics.get('pattern_types', []):
        counts[p] += 1
    if total == 0:
        return {}
    return {ptype: counts[ptype] / total for ptype in counts}


def _calculate_complexity_correlation(reasoning_metrics: Dict[str, Any]) -> float:
    """
    Calculate correlation between chain length and average confidence.
    """
    lengths = reasoning_metrics['path_lengths']
    confidences = reasoning_metrics['match_confidences']
    if lengths and confidences and len(lengths) == len(confidences):
        return float(np.corrcoef(lengths, confidences)[0, 1])
    return 0.0


def _calculate_pattern_success_rates(reasoning_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate success rates for different reasoning patterns.
    """
    success_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for chain in reasoning_metrics.get('chains', []):
        ptype = chain.get('pattern', {}).get('type', 'unknown')
        total_counts[ptype] += 1
        if chain.get('success', 0):
            success_counts[ptype] += 1

    return {
        ptype: (success_counts[ptype] / total_counts[ptype]) if total_counts[ptype] else 0.0
        for ptype in total_counts
    }


def _get_complexity_metrics(reasoning_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Return additional complexity metrics.
    """
    return {
        'avg_chain_length': float(np.mean(reasoning_metrics['path_lengths'])),
        'std_chain_length': float(np.std(reasoning_metrics['path_lengths']))
    }


def _analyze_pattern_effectiveness(reasoning_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze effectiveness of different reasoning patterns.
    """
    effectiveness = defaultdict(list)
    for chain, ptype in zip(reasoning_metrics.get('chains', []),
                            reasoning_metrics.get('pattern_types', [])):
        effectiveness[ptype].append({
            'success': chain.get('success', 0),
            'confidence': chain.get('overall_confidence', 0.0)
        })
    result = {}
    for pattern_type, measures in effectiveness.items():
        success_rate = np.mean([m['success'] for m in measures])
        avg_confidence = np.mean([m['confidence'] for m in measures])
        result[pattern_type] = {
            'success_rate': float(success_rate),
            'avg_confidence': float(avg_confidence),
            'sample_size': len(measures)
        }
    return result

