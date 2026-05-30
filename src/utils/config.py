"""
Configuration management utilities.

Handles loading and merging YAML configuration files.
"""

import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config, override_config):
    """Merge override config into base config."""
    logger.info("Merging configurations")
    # Implementation placeholder
    pass


def save_config(config, output_path):
    """Save configuration to YAML file."""
    logger.info(f"Saving config to {output_path}")
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    logger.info("Config utilities ready")
