"""
Logging utilities for PromptGFM-Bio.

Provides consistent logging across modules with Weights & Biases integration.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def init_wandb(config, project_name='promptgfm-bio'):
    """Initialize Weights & Biases logging."""
    logger.info("W&B initialization placeholder - will be implemented in Phase 4")
    pass


if __name__ == "__main__":
    logger.info("Logger utilities ready")
