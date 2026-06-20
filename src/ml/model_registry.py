from pathlib import Path
from typing import Any

import joblib

from src.utils.config import load_config

_config = load_config()
_model_path = Path(_config["paths"]["model_output"])
_model: Any | None = None


def get_model_path() -> Path:
    return _model_path


def load_model() -> Any:
    model_path = get_model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python -m src.train` first."
        )

    return joblib.load(model_path)


def get_model() -> Any:
    global _model

    if _model is None:
        _model = load_model()

    return _model
