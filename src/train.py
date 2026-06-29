from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
)
from src.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


SUPPORTED_SCORING_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


def build_model(model_name: str, params: dict[str, Any], random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state, **params)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(model: Pipeline, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=">50K")),
        "recall": float(recall_score(y_test, y_pred, pos_label=">50K")),
        "f1": float(f1_score(y_test, y_pred, pos_label=">50K")),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_true_binary = (y_test == ">50K").astype(int)
        metrics["roc_auc"] = float(roc_auc_score(y_true_binary, y_proba))

    return metrics


def log_common_run_info(config: dict[str, Any], model_name: str, params: dict[str, Any]) -> None:
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("random_state", config["project"]["random_state"])
    mlflow.log_param("test_size", config["data"]["test_size"])
    mlflow.log_param("target_column", config["project"]["target_column"])
    mlflow.log_param("data_path", str(resolve_path(config["paths"]["data_raw"])))

    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

    mlflow.set_tag("project", config["project"]["name"])
    mlflow.set_tag("experiment_type", "tabular_classification")
    mlflow.set_tag("dataset", "adult_income")
    mlflow.set_tag("stage", "capstone")


def normalize_tracking_uri(tracking_uri: str) -> str:
    if tracking_uri.startswith("file:"):
        raw_path = tracking_uri[5:]
        resolved = resolve_path(raw_path)
        return f"file:{resolved}"
    return tracking_uri


def validate_training_config(config: dict[str, Any]) -> None:
    scoring_metric = config["training"]["scoring_metric"]
    if scoring_metric not in SUPPORTED_SCORING_METRICS:
        raise ValueError(
            f"Unsupported scoring metric '{scoring_metric}'. "
            f"Supported values: {sorted(SUPPORTED_SCORING_METRICS)}"
        )

    enabled_models = [
        model_name
        for model_name, model_config in config["models"].items()
        if model_config.get("enabled", False)
    ]
    if not enabled_models:
        raise ValueError("No models are enabled in config['models'].")


def train_single_pipeline(
    model_name: str,
    model_config: dict[str, Any],
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, float]]:
    params = model_config.get("params", {})
    estimator = build_model(
        model_name=model_name,
        params=params,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)

    with mlflow.start_run(run_name=model_name):
        log_common_run_info(config, model_name, params)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
        )

        config_path = resolve_path("configs/config.yaml")
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")

    return pipeline, metrics


def main() -> None:
    config = load_config()
    validate_training_config(config)

    tracking_uri = normalize_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    scoring_metric = config["training"]["scoring_metric"]
    random_state = config["project"]["random_state"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow tracking URI set to %s", tracking_uri)
    logger.info("MLflow experiment set to %s", experiment_name)

    df = load_raw_data(str(resolve_path(config["paths"]["data_raw"])))
    df = clean_data(df, drop_columns=config["data"]["drop_columns"])

    X, y = split_features_target(df, config["project"]["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        categorical_features=config["data"]["categorical_features"],
        numeric_features=config["data"]["numeric_features"],
        scale_numeric=config["preprocessing"]["scale_numeric"],
        missing_strategy_categorical=config["preprocessing"]["missing_strategy_categorical"],
        missing_strategy_numeric=config["preprocessing"]["missing_strategy_numeric"],
        handle_unknown_categories=config["preprocessing"]["handle_unknown_categories"],
    )

    best_model: Pipeline | None = None
    best_model_name: str | None = None
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")

    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        logger.info("Training model: %s", model_name)

        pipeline, metrics = train_single_pipeline(
            model_name=model_name,
            model_config=model_config,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
            config=config,
        )

        score_value = metrics.get(scoring_metric)
        if score_value is None:
            raise ValueError(
                f"Scoring metric '{scoring_metric}' not found in metrics for model '{model_name}'."
            )

        logger.info(
            "Completed model=%s %s=%.6f metrics=%s",
            model_name,
            scoring_metric,
            score_value,
            metrics,
        )

        if score_value > best_score:
            best_score = score_value
            best_model = pipeline
            best_model_name = model_name
            best_metrics = metrics

    if best_model is None or best_model_name is None or best_metrics is None:
        raise RuntimeError("No model was trained. Check config.yaml model settings.")

    model_output_path = resolve_path(config["paths"]["model_output"])
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_output_path)
    logger.info("Saved best model to %s", model_output_path)

    metrics_output_path = resolve_path(config["paths"]["metrics_output"])
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model_name": best_model_name,
                "best_score_metric": scoring_metric,
                "best_score": best_score,
                "metrics": best_metrics,
            },
            file,
            indent=2,
        )
    logger.info("Saved best metrics summary to %s", metrics_output_path)

    print(f"Best model: {best_model_name}")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
