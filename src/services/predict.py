from typing import Tuple

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


def make_input_dataframe(features: Features) -> pd.DataFrame:
    data = {
        output_column: getattr(features, field_name)
        for field_name, output_column in FEATURE_COLUMN_MAP.items()
    }
    return pd.DataFrame([data])


def predict_from_features(features: Features) -> PredictionResult:
    model = get_model()
    df = make_input_dataframe(features)

    prediction = model.predict(df)[0]
    probability_gt_50k = get_probability_gt_50k(model, df)
    model_name = get_model_name(model)

    return PredictionResult(
        prediction=prediction,
        probability_gt_50k=probability_gt_50k,
        model_name=model_name,
    )


def get_probability_gt_50k(model, df: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.0

    probabilities = model.predict_proba(df)[0]
    classes = list(model.classes_)
    gt_50k_index = classes.index(">50K")
    return float(probabilities[gt_50k_index])


def get_model_name(model) -> str:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].__class__.__name__
    return model.__class__.__name__


def get_raw_prediction(features: Features) -> Tuple[str, float, str]:
    result = predict_from_features(features)
    return result.prediction, result.probability_gt_50k, result.model_name
