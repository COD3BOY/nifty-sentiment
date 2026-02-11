"""Configuration loader: YAML config + .env overlay."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_config: dict[str, Any] | None = None


def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> dict[str, Any]:
    """Load config.yaml and overlay environment variables from .env.

    Validates the config against ``core.config_schema`` on first load.
    """
    global _config
    if _config is not None:
        return _config

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / env_path)

    with open(project_root / config_path) as f:
        _config = yaml.safe_load(f)

    # Validate config structure on first load
    try:
        from core.config_schema import validate_config_dict
        validate_config_dict(_config)
        logger.info("Config validation passed")
    except Exception as e:
        logger.error("Config validation failed: %s", e)
        # Don't crash — log and continue with potentially invalid config
        # This allows legacy configs to still work during migration

    return _config


def get_source_config(source_name: str) -> dict[str, Any]:
    """Get config for a specific source."""
    config = load_config()
    return config.get("sources", {}).get(source_name, {})


def get_env(key: str) -> str:
    """Get an environment variable (reads from os.getenv, not cached in config)."""
    load_config()  # ensure .env is loaded
    return os.getenv(key, "")
