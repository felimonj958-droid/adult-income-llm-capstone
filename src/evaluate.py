import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, clean_data, load_raw_data, split_features_target
from src.utils.config import load_config, resolve_path



def main():
    config = load_config()
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    scoring_metric = config["training"]["scoring_metric"]

    # Normalize file-based tracking URIs to absolute paths
    if isinstance(tracking_uri, str) and tracking_uri.startswith("file:"):
        raw_path = tracking_uri[5:]
        tracking_uri = f"file:{resolve_path(raw_path)}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{scoring_metric} DESC"],
    )

    if runs_df.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    best_run = runs_df.iloc[0]

    summary = {
        "experiment_name": experiment_name,
        "scoring_metric": scoring_metric,
        "best_run_id": best_run["run_id"],
        "best_run_name": best_run.get("tags.mlflow.runName", "unknown"),
        "best_model_name": best_run.get("params.model_name", "unknown"),
        "metrics": {
            "accuracy": best_run.get("metrics.accuracy"),
            "precision": best_run.get("metrics.precision"),
            "recall": best_run.get("metrics.recall"),
            "f1": best_run.get("metrics.f1"),
            "roc_auc": best_run.get("metrics.roc_auc"),
        },
    }

    print(json.dumps(summary, indent=2))

    output_path = Path("artifacts/best_run_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
