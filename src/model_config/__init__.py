"""
Package containing default model configurations.
"""

from pathlib import Path


def get_default_config_dir() -> str:
    """Get the path to the default model_config directory."""
    return str(Path(__file__).resolve().parent)


def get_default_configs() -> list[str]:
    """Get a list of default configuration files."""
    config_dir = get_default_config_dir()
    return [str(f) for f in Path(config_dir).glob("*.yaml")]
