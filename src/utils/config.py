"""
Configuration loading and management utilities.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration has all required keys.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If required keys are missing.
    """
    required_sections = ["training", "data", "checkpoint", "lr_scheduler", "logging"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    training_required = [
        "batch_size",
        "d_model",
        "output_dim",
        "num_layers",
        "nhead",
        "learning_rate",
        "num_epochs",
    ]

    for key in training_required:
        if key not in config["training"]:
            raise ValueError(f"Missing required training config: {key}")


def get_config_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in default values for optional config keys.

    Args:
        config: Configuration dictionary.

    Returns:
        Configuration with defaults filled in.
    """
    defaults = {
        "training": {
            "weig": 100.0,
            "test_step": 5,
            "T_scaling": 1209600.0,
        },
        "lr_scheduler": {
            "type": "cosine",
            "static_max_seq_len": 5246,
            "step": 8000,
        },
    }

    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = {}
        for key, value in section_defaults.items():
            if key not in config[section]:
                config[section][key] = value

    return config
