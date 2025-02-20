import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import jsonlines
from collections import defaultdict

class QueryLogger:
    """
    Enhanced query logger that provides comprehensive logging capabilities for HySym-RAG.
    This logger tracks detailed information about query processing, resource utilization,
    and reasoning paths, with special attention to multi-hop reasoning patterns.
    """

    def __init__(self,
                 log_dir: str = "logs",
                 main_log_file: str = "query_log.json",
                 performance_log_file: str = "performance_metrics.jsonl",
                 error_log_file: str = "error_log.json",
                 max_log_size: int = 10_000):
        """
        Initialize the enhanced query logger with multiple specialized log files.

        Args:
            log_dir: Directory for log files
            main_log_file: Main query log filename
            performance_log_file: Performance metrics log filename
            error_log_file: Error tracking log filename
            max_log_size: Maximum number of entries per log file
        """
        # Create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log file paths
        self.main_log_path = self.log_dir / main_log_file
        self.performance_log_path = self.log_dir / performance_log_file
        self.error_log_path = self.log_dir / error_log_file

        # Set configuration parameters
        self.max_log_size = max_log_size

        # Initialize performance tracking
        self.performance_stats = defaultdict(list)

        # Set up logging
        self.logger = logging.getLogger("QueryLogger")
        self.logger.setLevel(logging.INFO)

        # Initialize log files if they don't exist
        self._initialize_log_files()

        self.logger.info("QueryLogger initialized successfully")

    def _initialize_log_files(self):
        """
        Initialize log files with proper structure if they don't exist.
        """
        # Initialize main query log
        if not self.main_log_path.exists():
            with open(self.main_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
                f.flush()

        # Initialize performance log with header
        if not self.performance_log_path.exists():
            with jsonlines.open(self.performance_log_path, mode='w') as writer:
                writer.write({
                    "timestamp": datetime.now().isoformat(),
                    "log_initialized": True,
                    "version": "2.0"
                })

        # Initialize error log
        if not self.error_log_path.exists():
            with open(self.error_log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "error_counts": {},
                    "error_logs": []
                }, f)
                f.flush()

    def log_query(self,
                  query: str,
                  result: Any,
                  source: str,
                  complexity: Optional[float] = None,
                  resource_usage: Optional[Dict] = None,
                  reasoning_path: Optional[Dict] = None,
                  processing_time: Optional[float] = None,
                  metadata: Optional[Dict] = None) -> None:
        """
        Log a query with comprehensive metadata and performance information.

        Args:
            query: The input query
            result: Query result
            source: Source of the answer (symbolic/neural/hybrid)
            complexity: Query complexity score
            resource_usage: Resource utilization metrics
            reasoning_path: Details about the reasoning path taken
            processing_time: Query processing time
            metadata: Additional metadata about the query
        """
        timestamp = datetime.now().isoformat()

        # Construct the log entry with enhanced metadata
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "source": source,
            "complexity": complexity,
            "processing_time": processing_time,
            "success": self._determine_success(result),
        }

        # Add optional information if available
        if resource_usage:
            log_entry["resource_usage"] = self._format_resource_usage(resource_usage)

        if reasoning_path:
            log_entry["reasoning_path"] = self._format_reasoning_path(reasoning_path)

        if metadata:
            log_entry["metadata"] = metadata

        # Add performance metrics
        log_entry["performance_metrics"] = self._calculate_performance_metrics(log_entry)

        # Write to main log file
        self._write_to_main_log(log_entry)

        # Update performance statistics
        self._update_performance_stats(log_entry)

        # Log performance metrics separately
        self._log_performance_metrics(log_entry)

    def _determine_success(self, result: Any) -> bool:
        """
        Determine if the query was successful based on the result.
        """
        if isinstance(result, str):
            return not any(error_indicator in result.lower()
                           for error_indicator in ["error", "failed", "no match"])
        return bool(result)

    def _format_resource_usage(self, resource_usage: Dict) -> Dict:
        """
        Format resource usage metrics with additional derived metrics.
        """
        formatted = {
            k: round(v * 100, 2) if isinstance(v, float) else v
            for k, v in resource_usage.items()
        }

        # Calculate efficiency metrics
        formatted["efficiency_score"] = self._calculate_efficiency_score(resource_usage)

        return formatted

    def _format_reasoning_path(self, reasoning_path: Dict) -> Dict:
        """
        Format reasoning path information with enhanced metadata.
        """
        formatted_path = {
            "path_type": reasoning_path.get("type", "unknown"),
            "steps": reasoning_path.get("steps", []),
            "confidence": reasoning_path.get("confidence", 0.0)
        }

        # Add hop analysis for multi-hop reasoning
        if "steps" in reasoning_path:
            formatted_path["hop_count"] = len(reasoning_path["steps"])
            formatted_path["hop_types"] = self._analyze_hop_types(reasoning_path["steps"])

        return formatted_path

    def _analyze_hop_types(self, steps: List[Dict]) -> Dict:
        """
        Analyze the types of reasoning hops in a multi-hop path.
        """
        hop_types = defaultdict(int)
        for step in steps:
            hop_type = step.get("type", "unknown")
            hop_types[hop_type] += 1
        return dict(hop_types)

    def _calculate_performance_metrics(self, entry: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics for the query.
        """
        metrics = {
            "processing_time": entry.get("processing_time", 0),
            "success": entry.get("success", False),
        }

        # Add resource efficiency if available
        if "resource_usage" in entry:
            metrics["resource_efficiency"] = self._calculate_efficiency_score(
                entry["resource_usage"]
            )

        # Add reasoning complexity metrics if available
        if "reasoning_path" in entry:
            metrics["reasoning_complexity"] = self._calculate_reasoning_complexity(
                entry["reasoning_path"]
            )

        return metrics

    def _calculate_efficiency_score(self, resource_usage: Dict) -> float:
        """
        Calculate a normalized efficiency score based on resource usage.
        """
        weights = {
            "cpu": 0.3,
            "memory": 0.3,
            "gpu": 0.4
        }

        score = 0.0
        for resource, weight in weights.items():
            if resource in resource_usage:
                # Lower resource usage means higher efficiency
                score += (1 - resource_usage[resource]) * weight

        return round(score, 4)

    def _calculate_reasoning_complexity(self, reasoning_path: Dict) -> float:
        """
        Calculate reasoning complexity based on path characteristics.
        """
        base_complexity = 0.0

        # Factor in hop count
        hop_count = reasoning_path.get("hop_count", 1)
        base_complexity += min(hop_count * 0.2, 0.6)

        # Factor in path type
        path_type_weights = {
            "symbolic": 0.2,
            "neural": 0.3,
            "hybrid": 0.4
        }
        path_type = reasoning_path.get("path_type", "unknown")
        base_complexity += path_type_weights.get(path_type, 0.1)

        return round(base_complexity, 4)

    def _write_to_main_log(self, log_entry: Dict):
        """
        Write to main log file with rotation if needed.
        """
        try:
            current_logs = []
            if self.main_log_path.exists():
                with open(self.main_log_path, 'r', encoding='utf-8') as f:
                    current_logs = json.load(f)

            # Rotate logs if necessary
            if len(current_logs) >= self.max_log_size:
                current_logs = current_logs[-(self.max_log_size - 1):]

            current_logs.append(log_entry)

            with open(self.main_log_path, 'w', encoding='utf-8') as f:
                json.dump(current_logs, f, indent=4)
                f.flush()
        except Exception as e:
            self.logger.error(f"Error writing to main log: {str(e)}")
            self._log_error("main_log_write_error", str(e))

    def _update_performance_stats(self, log_entry: Dict):
        """
        Update running performance statistics.
        """
        metrics = log_entry["performance_metrics"]

        # Update basic stats
        self.performance_stats["processing_times"].append(metrics["processing_time"])
        self.performance_stats["success_rate"].append(int(metrics["success"]))

        # Update resource efficiency stats if available
        if "resource_efficiency" in metrics:
            self.performance_stats["resource_efficiency"].append(
                metrics["resource_efficiency"]
            )

        # Maintain reasonable list sizes
        max_history = 1000
        for key in self.performance_stats:
            if len(self.performance_stats[key]) > max_history:
                self.performance_stats[key] = self.performance_stats[key][-max_history:]

    def _log_performance_metrics(self, log_entry: Dict):
        """
        Log detailed performance metrics to the performance log file.
        """
        try:
            with jsonlines.open(self.performance_log_path, mode='a') as writer:
                writer.write({
                    "timestamp": log_entry["timestamp"],
                    "query_id": hash(log_entry["query"]),
                    "metrics": log_entry["performance_metrics"],
                    "resource_usage": log_entry.get("resource_usage", {}),
                    "reasoning_path": log_entry.get("reasoning_path", {})
                })
        except Exception as e:
            self.logger.error(f"Error writing performance metrics: {str(e)}")
            self._log_error("performance_log_write_error", str(e))

    def _log_error(self, error_type: str, error_message: str):
        """
        Log errors with categorization and tracking.
        """
        try:
            with open(self.error_log_path, 'r+', encoding='utf-8') as f:
                error_log = json.load(f)
                error_log["error_counts"][error_type] = error_log["error_counts"].get(error_type, 0) + 1
                error_log["error_logs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": error_type,
                    "message": error_message
                })
                f.seek(0)
                json.dump(error_log, f, indent=4)
                f.truncate()
        except Exception as e:
            self.logger.error(f"Error logging error: {str(e)}")

    def get_performance_summary(self) -> Dict:
        """
        Get a summary of system performance metrics.
        """
        summary = {}

        for metric, values in self.performance_stats.items():
            if values:
                summary[metric] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }

        return summary
