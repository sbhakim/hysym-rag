# src/feedback/feedback_manager.py

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
import jsonlines


class FeedbackManager:
    """
    Enhanced FeedbackManager that manages storage, analysis, and application of system feedback.
    Specifically designed to handle complex feedback patterns in multi-hop reasoning scenarios
    and integrate with HySym-RAG's improved components.
    """

    def __init__(self,
                 feedback_dir: str = "logs/feedback",
                 max_storage: int = 10000,
                 analysis_window: int = 100,
                 backup_interval: int = 24):
        """
        Initialize the enhanced feedback manager.

        Args:
            feedback_dir: Directory for feedback storage
            max_storage: Maximum number of feedback entries to store
            analysis_window: Window size for trend analysis
            backup_interval: Hours between feedback data backups
        """
        # Initialize storage paths
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Set up specific feedback files
        self.main_feedback_file = self.feedback_dir / "feedback_main.jsonl"
        self.analysis_file = self.feedback_dir / "feedback_analysis.json"
        self.metrics_file = self.feedback_dir / "feedback_metrics.json"

        # Configuration parameters
        self.max_storage = max_storage
        self.analysis_window = analysis_window
        self.backup_interval = timedelta(hours=backup_interval)

        # Initialize logging
        self.logger = logging.getLogger("FeedbackManager")
        self.logger.setLevel(logging.INFO)

        # Initialize feedback tracking structures
        self.feedback_stats = {
            'path_performance': defaultdict(list),
            'query_types': defaultdict(list),
            'resource_usage': defaultdict(list),
            'reasoning_patterns': defaultdict(list)
        }

        # Initialize analysis components
        self.performance_metrics = {
            'symbolic': {'scores': [], 'timestamps': []},
            'neural': {'scores': [], 'timestamps': []},
            'hybrid': {'scores': [], 'timestamps': []}
        }

        # Initialize storage
        self._initialize_storage()

        self.logger.info("FeedbackManager initialized successfully")

    def _initialize_storage(self):
        """
        Initialize feedback storage with proper structure.
        """
        # Initialize main feedback file if needed
        if not self.main_feedback_file.exists():
            with jsonlines.open(self.main_feedback_file, mode='w') as writer:
                writer.write({
                    "initialized": datetime.now().isoformat(),
                    "version": "2.0"
                })

        # Initialize analysis file
        if not self.analysis_file.exists():
            self._save_analysis({
                "last_analysis": None,
                "trends": {},
                "patterns": {},
                "recommendations": []
            })

        # Initialize metrics file
        if not self.metrics_file.exists():
            self._save_metrics({
                "path_metrics": defaultdict(dict),
                "resource_metrics": defaultdict(dict),
                "performance_metrics": defaultdict(dict)
            })

    def _save_analysis(self, analysis_data: Dict[str, Any]):
        """
        Save analysis data to the analysis file.
        """
        try:
            with open(self.analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving analysis data: {str(e)}")

    def _save_metrics(self, metrics_data: Dict[str, Any]):
        """
        Save metrics data to the metrics file.
        """
        try:
            # Convert defaultdict to normal dict so it’s JSON-serializable
            def dictify(obj):
                if isinstance(obj, defaultdict):
                    return {k: dictify(v) for k, v in obj.items()}
                elif isinstance(obj, dict):
                    return {k: dictify(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [dictify(x) for x in obj]
                return obj

            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(dictify(metrics_data), f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving metrics data: {str(e)}")

    def _handle_storage_error(self, error: Exception):
        """
        Handle any storage errors gracefully.
        """
        self.logger.error(f"Storage error encountered: {error}")

    def submit_feedback(self,
                        query: str,
                        result: Any,
                        feedback_data: Dict[str, Any],
                        metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit and process new feedback with enhanced analysis.

        Args:
            query: Original query
            result: System response
            feedback_data: Structured feedback information
            metadata: Additional context and processing information

        Returns:
            Processed feedback with analysis results
        """
        try:
            # Process and validate feedback
            processed_feedback = self._process_feedback(
                query,
                result,
                feedback_data,
                metadata
            )

            # Store feedback
            self._store_feedback(processed_feedback)

            # Update statistics
            self._update_statistics(processed_feedback)

            # Analyze impact
            impact_analysis = self._analyze_feedback_impact(processed_feedback)

            # Generate recommendations
            recommendations = self._generate_recommendations(processed_feedback)

            # Create comprehensive feedback report
            feedback_report = {
                'timestamp': datetime.now().isoformat(),
                'processed_feedback': processed_feedback,
                'impact_analysis': impact_analysis,
                'recommendations': recommendations
            }

            return feedback_report

        except Exception as e:
            self.logger.error(f"Error submitting feedback: {str(e)}")
            return {'error': str(e), 'status': 'failed'}

    def _process_feedback(self,
                          query: str,
                          result: Any,
                          feedback_data: Dict[str, Any],
                          metadata: Optional[Dict]) -> Dict[str, Any]:
        """
        Process and validate feedback data with enhanced validation.
        """
        processed = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result': result,
            'feedback': feedback_data,
            'feedback_type': self._determine_feedback_type(feedback_data)
        }

        # Add reasoning path analysis if available
        if metadata and 'reasoning_path' in metadata:
            processed['reasoning_analysis'] = self._analyze_reasoning_path(
                feedback_data,
                metadata['reasoning_path']
            )

        # Add performance metrics
        processed['performance_metrics'] = self._calculate_performance_metrics(
            feedback_data,
            metadata
        )

        return processed

    def _determine_feedback_type(self, feedback_data: Dict[str, Any]) -> str:
        """
        Determine the type of feedback based on its characteristics.
        """
        if 'reasoning' in feedback_data and 'connection' in feedback_data:
            return 'multi_hop'
        elif 'accuracy' in feedback_data:
            return 'factual'
        else:
            return 'general'

    def _analyze_reasoning_path(self,
                                feedback_data: Dict[str, Any],
                                reasoning_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feedback specific to reasoning paths.
        """
        analysis = {
            'path_type': reasoning_path.get('type', 'unknown'),
            'hop_count': reasoning_path.get('hop_count', 1),
            'effectiveness': self._calculate_path_effectiveness(
                feedback_data,
                reasoning_path
            )
        }

        # Add pattern analysis
        if reasoning_path.get('steps'):
            analysis['pattern_analysis'] = self._analyze_reasoning_pattern(
                reasoning_path['steps']
            )

        return analysis

    def _calculate_path_effectiveness(self,
                                      feedback_data: Dict[str, Any],
                                      reasoning_path: Dict[str, Any]) -> float:
        """
        Calculate effectiveness score for reasoning path.
        """
        base_score = 0.0
        count = 0

        # Calculate from relevant feedback metrics
        if 'reasoning' in feedback_data:
            base_score += feedback_data['reasoning']
            count += 1

        if 'connection' in feedback_data:
            base_score += feedback_data['connection']
            count += 1

        if count == 0:
            return 0.0

        # Adjust for path complexity
        hop_count = reasoning_path.get('hop_count', 1)
        complexity_factor = 1 + (0.1 * (hop_count - 1))

        return (base_score / count) * complexity_factor

    def _analyze_reasoning_pattern(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in the reasoning steps (placeholder).
        """
        return {
            "step_count": len(steps),
            "step_types": [s.get('type', 'unknown') for s in steps]
        }

    def _calculate_performance_metrics(self,
                                       feedback_data: Dict[str, Any],
                                       metadata: Optional[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics from feedback.
        """
        metrics = {}

        # Example: incorporate accuracy or timing
        if 'accuracy' in feedback_data:
            metrics['accuracy'] = feedback_data['accuracy']

        if metadata and 'processing_time' in metadata:
            metrics['processing_time'] = metadata['processing_time']

        return metrics

    def _store_feedback(self, processed_feedback: Dict[str, Any]):
        """
        Store processed feedback with rotation handling.
        """
        try:
            # Write to main feedback file
            with jsonlines.open(self.main_feedback_file, mode='a') as writer:
                writer.write(processed_feedback)

            # Rotate if needed
            self._check_and_rotate_storage()

        except Exception as e:
            self.logger.error(f"Error storing feedback: {str(e)}")
            self._handle_storage_error(e)

    def _check_and_rotate_storage(self):
        """
        Check and rotate feedback storage if needed.
        """
        try:
            # Count entries
            count = sum(1 for _ in jsonlines.open(self.main_feedback_file))

            if count > self.max_storage:
                self._rotate_storage()

        except Exception as e:
            self.logger.error(f"Error checking storage: {str(e)}")

    def _rotate_storage(self):
        """
        Rotate feedback storage, maintaining most recent entries.
        """
        try:
            # Read existing feedback
            entries = []
            with jsonlines.open(self.main_feedback_file) as reader:
                entries = list(reader)

            # Keep most recent entries
            recent_entries = entries[-(self.max_storage - 1000):]

            # Backup old entries
            self._backup_old_entries(entries[:-len(recent_entries)])

            # Write recent entries to main file
            with jsonlines.open(self.main_feedback_file, mode='w') as writer:
                for entry in recent_entries:
                    writer.write(entry)

        except Exception as e:
            self.logger.error(f"Error rotating storage: {str(e)}")

    def _backup_old_entries(self, old_entries: List[Dict[str, Any]]):
        """
        Backup old entries somewhere else (placeholder).
        """
        # For demonstration, just log them or write to a separate file
        backup_path = self.feedback_dir / f"feedback_backup_{datetime.now().isoformat()}.jsonl"
        try:
            with jsonlines.open(backup_path, mode='w') as writer:
                for entry in old_entries:
                    writer.write(entry)
        except Exception as e:
            self.logger.error(f"Error backing up old entries: {str(e)}")

    def _update_statistics(self, processed_feedback: Dict[str, Any]):
        """
        Update running statistics with new feedback.
        """
        try:
            # Update path performance
            if 'reasoning_analysis' in processed_feedback:
                path_type = processed_feedback['reasoning_analysis']['path_type']
                effectiveness = processed_feedback['reasoning_analysis']['effectiveness']
                self.feedback_stats['path_performance'][path_type].append(effectiveness)

            # Update query type stats
            feedback_type = processed_feedback['feedback_type']
            self.feedback_stats['query_types'][feedback_type].append(
                processed_feedback['performance_metrics']
            )

            # Maintain window size
            self._maintain_stats_window()

        except Exception as e:
            self.logger.error(f"Error updating statistics: {str(e)}")

    def _maintain_stats_window(self):
        """
        Maintain the analysis window size for statistics.
        """
        for category in self.feedback_stats:
            for key in self.feedback_stats[category]:
                if len(self.feedback_stats[category][key]) > self.analysis_window:
                    self.feedback_stats[category][key] = \
                        self.feedback_stats[category][key][-self.analysis_window:]

    def _analyze_feedback_impact(self, processed_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for more advanced feedback impact analysis.
        """
        return {
            "impact_score": 0.0
        }

    def _generate_recommendations(self, processed_feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on feedback.
        """
        # Placeholder
        return []

    def get_feedback_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive feedback analysis.
        """
        return {
            'path_performance': self._analyze_path_performance(),
            'query_patterns': self._analyze_query_patterns(),
            'resource_efficiency': self._analyze_resource_efficiency(),
            'recommendations': self._generate_system_recommendations()
        }

    def _analyze_path_performance(self) -> Dict[str, Any]:
        """
        Analyze performance patterns for different reasoning paths.
        """
        analysis = {}

        for path_type, scores in self.feedback_stats['path_performance'].items():
            if scores:
                analysis[path_type] = {
                    'average_effectiveness': np.mean(scores),
                    'trend': self._calculate_trend(scores),
                    'stability': np.std(scores)
                }

        return analysis

    def _analyze_query_patterns(self) -> Dict[str, Any]:
        """
        Placeholder for analyzing query patterns across feedback.
        """
        return {}

    def _analyze_resource_efficiency(self) -> Dict[str, Any]:
        """
        Placeholder for analyzing resource usage from feedback.
        """
        return {}

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend in a series of values.
        """
        if len(values) < 2:
            return 0.0

        try:
            x = np.arange(len(values))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return 0.0

    def _generate_system_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate system improvement recommendations based on feedback analysis.
        """
        recommendations = []

        # Analyze path performance
        path_analysis = self._analyze_path_performance()
        for path_type, metrics in path_analysis.items():
            if metrics['average_effectiveness'] < 0.7:
                recommendations.append({
                    'component': path_type,
                    'issue': 'low_effectiveness',
                    'priority': 'high',
                    'suggestion': f"Improve {path_type} reasoning effectiveness"
                })

        # Add other recommendation logic if needed
        return recommendations
