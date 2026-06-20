from pathlib import Path

import joblib
from sklearn.metrics import f1_score

from src.utils.config import load_config
from src.preprocess import load_raw_data, clean_data, split_features_target


def _load_best_model():
    config = load_config()
    model_path = Path(config["paths"]["model_output"])
    assert model_path.exists(), f"Model file not found at {model_path}. Run training first."

    model = joblib.load(model_path)
    return model, config


def test_model_predictions_shape_and_labels():
    model, config = _load_best_model()

    df = load_raw_data(config["paths"]["data_raw"])
    df = clean_data(df, drop_columns=config["data"]["drop_columns"])
    X, y = split_features_target(df, config["project"]["target_column"])

    X_sample = X.head(200)
    y_sample = y.head(200)

    y_pred = model.predict(X_sample)

    assert len(y_pred) == len(X_sample)
    assert set(y_pred).issubset({"<=50K", ">50K"})


def test_model_minimum_f1_on_sample():
    model, config = _load_best_model()

    df = load_raw_data(config["paths"]["data_raw"])
    df = clean_data(df, drop_columns=config["data"]["drop_columns"])
    X, y = split_features_target(df, config["project"]["target_column"])

    # Use a reasonably sized sample
    X_sample = X.head(1000)
    y_sample = y.head(1000)

    y_pred = model.predict(X_sample)

    score = f1_score(y_sample, y_pred, pos_label=">50K")

    # Minimum acceptable F1 for sanity; adjust if needed
    assert score >= 0.6
