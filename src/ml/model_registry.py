from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.core.logging import logger
from src.utils.config import ConfigError, load_config, resolve_path

_MODEL: Any | None = None


class ModelRegistryError(Exception):
    """Raised when the prediction model cannot be loaded."""


def get_model_path() -> Path:
    fallback_path = Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"

    try:
        config = load_config()
        return resolve_path(config["paths"]["model_output"])
    except (ConfigError, KeyError, TypeError) as exc:
        logger.warning("Using fallback model path because config lookup failed: %s", exc)
        return fallback_path


def load_model(model_path: Path | None = None) -> Any:
    path = model_path or get_model_path()

    if not path.exists():
        raise ModelRegistryError(f"Model file not found: {path}")

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ModelRegistryError(f"Failed to load model from {path}: {exc}") from exc

    if not hasattr(model, "predict"):
        raise ModelRegistryError(f"Loaded artifact does not implement predict(): {path}")

    logger.info("Loaded prediction model from %s", path)
    return model


def get_model(force_reload: bool = False) -> Any:
    global _MODEL

    if _MODEL is None or force_reload:
        _MODEL = load_model()

    return _MODEL


def clear_model_cache() -> None:
    global _MODEL
    _MODEL = None