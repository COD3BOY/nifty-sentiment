"""Configuration loader: YAML config + .env overlay."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_config: dict[str, Any] | None = None


def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> dict[str, Any]:
    """Load config.yaml and overlay environment variables from .env."""
    global _config
    if _config is not None:
        return _config

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / env_path)

    with open(project_root / config_path) as f:
        _config = yaml.safe_load(f)

    return _config


def get_source_config(source_name: str) -> dict[str, Any]:
    """Get config for a specific source."""
    config = load_config()
    return config.get("sources", {}).get(source_name, {})


def get_env(key: str) -> str:
    """Get an environment variable (reads from os.getenv, not cached in config)."""
    load_config()  # ensure .env is loaded
    return os.getenv(key, "")
