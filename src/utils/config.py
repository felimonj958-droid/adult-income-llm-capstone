from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigError(Exception):
    pass


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config file must contain a top-level mapping: {path}")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ["project", "paths"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required config section: '{section}'")

    required_paths = ["model_output"]
    for key in required_paths:
        if key not in config["paths"]:
            raise ConfigError(f"Missing required config path: 'paths.{key}'")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
