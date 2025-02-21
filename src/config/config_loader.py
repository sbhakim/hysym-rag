# src/config/config_loader.py

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from collections import defaultdict


class ConfigLoader:
    """
    Enhanced configuration loader for HySym-RAG that supports hierarchical configs,
    environment overrides, and dynamic validation. Specifically designed to support
    multi-hop reasoning and HotpotQA processing configurations.
    """

    def __init__(self):
        self.logger = logging.getLogger("ConfigLoader")
        self.logger.setLevel(logging.INFO)

        # Track loaded configurations
        self.loaded_configs = {}

        # Define configuration schemas
        self.config_schemas = {
            'main': {
                'required': ['model_name', 'data_dir'],
                'optional': ['rules_file', 'knowledge_base']
            },
            'resources': {
                'required': ['resource_thresholds', 'monitoring'],
                'optional': ['adaptation', 'recovery']
            },
            'reasoning': {
                'required': ['symbolic_config', 'neural_config'],
                'optional': ['hybrid_config']
            }
        }

    @staticmethod
    def load_config(config_file: str = "config.yaml",
                    env_prefix: str = "HYSYM_") -> Dict[str, Any]:
        """
        Load and validate configuration with environment variable overrides.

        Args:
            config_file: Path to configuration file
            env_prefix: Prefix for environment variable overrides

        Returns:
            Validated configuration dictionary
        """
        try:
            # Load base configuration
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)

            # Apply environment overrides
            config = ConfigLoader._apply_env_overrides(config, env_prefix)

            # Validate configuration
            ConfigLoader._validate_config(config)

            # Add derived configurations
            config = ConfigLoader._add_derived_configs(config)

            return config

        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise

    @staticmethod
    def _apply_env_overrides(config: Dict[str, Any],
                             prefix: str) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        """
        for key in config.keys():
            env_key = f"{prefix}{key.upper()}"
            if env_key in os.environ:
                try:
                    # Parse environment value
                    env_value = ConfigLoader._parse_env_value(os.environ[env_key])
                    config[key] = env_value
                except Exception as e:
                    logging.warning(f"Error parsing environment override {env_key}: {str(e)}")

        return config

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value with type inference.
        """
        # Try parsing as JSON
        try:
            import json
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try parsing as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try parsing as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string if no other type matches
        return value

    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """
        Validate configuration structure and required fields.
        """
        required_fields = {
            'model_name',
            'data_dir',
            'embeddings'
        }

        # Check required fields
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")

        # Validate specific sections
        ConfigLoader._validate_embeddings_config(config.get('embeddings', {}))
        ConfigLoader._validate_resource_config(config.get('resource_thresholds', {}))

    @staticmethod
    def _validate_embeddings_config(embeddings_config: Dict[str, Any]):
        """
        Validate embeddings configuration section.
        """
        required_embedding_fields = {
            'symbolic_dim',
            'neural_dim',
            'target_dim',
            'model_name'
        }

        missing_fields = required_embedding_fields - set(embeddings_config.keys())
        if missing_fields:
            raise ValueError(f"Missing required embeddings configuration fields: {missing_fields}")

    @staticmethod
    def _validate_resource_config(resource_config: Dict[str, Any]):
        """
        Validate resource configuration section.
        """
        required_resources = {'cpu', 'memory', 'gpu'}

        for resource in required_resources:
            if resource not in resource_config:
                raise ValueError(f"Missing resource configuration for: {resource}")

            config = resource_config[resource]
            if not isinstance(config, dict) or 'base_threshold' not in config:
                raise ValueError(f"Invalid configuration for resource: {resource}")

    @staticmethod
    def _add_derived_configs(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add derived configuration values based on base config.
        """
        # Add HotpotQA specific configurations
        if 'hotpotqa' not in config:
            config['hotpotqa'] = {
                'max_hops': 3,
                'min_confidence': 0.7,
                'chunk_size': 512,
                'overlap': 128
            }

        # Add reasoning configurations
        if 'reasoning' not in config:
            config['reasoning'] = {
                'symbolic_weight': 0.4,
                'neural_weight': 0.6,
                'min_confidence': 0.3,
                'max_reasoning_time': 10.0
            }

        return config

    def load_multiple_configs(self,
                              config_files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Load multiple configuration files with cross-validation.
        """
        configs = {}

        for config_type, file_path in config_files.items():
            try:
                configs[config_type] = self.load_config(file_path)
                self.loaded_configs[config_type] = configs[config_type]
            except Exception as e:
                self.logger.error(f"Error loading {config_type} configuration: {str(e)}")
                raise

        # Validate cross-configuration dependencies
        self._validate_config_dependencies(configs)

        return configs

    def _validate_config_dependencies(self, configs: Dict[str, Dict[str, Any]]):
        """
        Validate dependencies between different configuration sections.
        """
        if 'main' in configs and 'resources' in configs:
            main_config = configs['main']
            resource_config = configs['resources']

            # Validate model requirements against resource limits
            if 'model_name' in main_config:
                self._validate_model_resources(
                    main_config['model_name'],
                    resource_config
                )

    def _validate_model_resources(self,
                                  model_name: str,
                                  resource_config: Dict[str, Any]):
        """
        Validate if resource configuration meets model requirements.
        """
        # Add model-specific validation logic here
        pass