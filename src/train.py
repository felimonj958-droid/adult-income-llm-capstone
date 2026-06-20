from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils.config import load_config
from src.preprocess import load_raw_data, clean_data, split_features_target, build_preprocessor


def build_model(model_name: str, params: dict, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state, **params)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=">50K"),
        "recall": recall_score(y_test, y_pred, pos_label=">50K"),
        "f1": f1_score(y_test, y_pred, pos_label=">50K"),
    }

    if y_proba is not None:
        y_true_binary = (y_test == ">50K").astype(int)
        metrics["roc_auc"] = roc_auc_score(y_true_binary, y_proba)

    return metrics


def main():
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_raw_data(config["paths"]["data_raw"])
    df = clean_data(df, drop_columns=config["data"]["drop_columns"])

    X, y = split_features_target(df, config["project"]["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
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

    best_model = None
    best_model_name = None
    best_metrics = None
    best_score = -1

    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            continue

        params = model_config.get("params", {})
        estimator = build_model(
            model_name=model_name,
            params=params,
            random_state=config["project"]["random_state"],
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", config["project"]["random_state"])
            mlflow.log_param("test_size", config["data"]["test_size"])
            mlflow.log_param("target_column", config["project"]["target_column"])

            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            input_example = X_train.head(5)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=input_example,
            )


            score_name = config["training"]["scoring_metric"]
            score_value = metrics[score_name]

            if score_value > best_score:
                best_score = score_value
                best_model = pipeline
                best_model_name = model_name
                best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model was trained. Check config.yaml model settings.")

    model_output_path = resolve_path(config["paths"]["model_output"])
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_output_path)

    metrics_output_path = resolve_path(config["paths"]["metrics_output"])
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_model_name,
                "best_score_metric": config["training"]["scoring_metric"],
                "best_score": best_score,
                "metrics": best_metrics,
            },
            f,
            indent=2,
        )

    print(f"Best model: {best_model_name}")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
