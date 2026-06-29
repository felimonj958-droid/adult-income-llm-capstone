from __future__ import annotations

from typing import Any

import pandas as pd

from src.ml.model_registry import get_model
from src.schemas.inference import Features, PredictionResult

FEATURE_COLUMN_MAP = {
    "age": "age",
    "workclass": "workclass",
    "education": "education",
    "education_num": "education-num",
    "marital_status": "marital-status",
    "occupation": "occupation",
    "relationship": "relationship",
    "race": "race",
    "sex": "sex",
    "capital_gain": "capital-gain",
    "capital_loss": "capital-loss",
    "hours_per_week": "hours-per-week",
    "native_country": "native-country",
}

TARGET_POSITIVE_CLASS = ">50K"


class PredictionServiceError(Exception):
    """Raised when prediction cannot be completed safely."""


def make_input_dataframe(features: Features) -> pd.DataFrame:
    row = {
        output_column: getattr(features, field_name)
        for field_name, output_column in FEATURE_COLUMN_MAP.items()
    }
    return pd.DataFrame([row], columns=list(FEATURE_COLUMN_MAP.values()))


def _extract_single_prediction(model: Any, df: pd.DataFrame) -> str:
    if not hasattr(model, "predict"):
        raise PredictionServiceError("Loaded model does not implement predict().")

    try:
        predictions = model.predict(df)
    except Exception as exc:
        raise PredictionServiceError(f"Model prediction failed: {exc}") from exc

    if predictions is None or len(predictions) != 1:
        raise PredictionServiceError("Model did not return exactly one prediction.")

    prediction = str(predictions[0]).strip()
    if not prediction:
        raise PredictionServiceError("Model returned an empty prediction label.")

    return prediction


def _get_positive_class_probability(model: Any, df: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.0

    if not hasattr(model, "classes_"):
        raise PredictionServiceError("Model exposes predict_proba() but not classes_.")

    try:
        probabilities = model.predict_proba(df)
    except Exception as exc:
        raise PredictionServiceError(f"Model probability prediction failed: {exc}") from exc

    if probabilities is None or len(probabilities) != 1:
        raise PredictionServiceError("Model did not return exactly one probability row.")

    classes = [str(label).strip() for label in model.classes_]
    if TARGET_POSITIVE_CLASS not in classes:
        raise PredictionServiceError(
            f"Positive class '{TARGET_POSITIVE_CLASS}' not found in model classes."
        )

    class_index = classes.index(TARGET_POSITIVE_CLASS)

    try:
        probability = float(probabilities[0][class_index])
    except (IndexError, TypeError, ValueError) as exc:
        raise PredictionServiceError("Could not extract positive-class probability.") from exc

    if not 0.0 <= probability <= 1.0:
        raise PredictionServiceError("Predicted probability is out of bounds.")

    return probability


def get_model_name(model: Any) -> str:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].__class__.__name__

    name = model.__class__.__name__.strip()
    if not name:
        raise PredictionServiceError("Unable to determine model name.")
    return name


def predict_from_features(features: Features) -> PredictionResult:
    model = get_model()
    df = make_input_dataframe(features)

    return PredictionResult(
        prediction=_extract_single_prediction(model, df),
        probability_gt_50k=_get_positive_class_probability(model, df),
        model_name=get_model_name(model),
    )