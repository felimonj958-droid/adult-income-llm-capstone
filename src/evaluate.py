from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from src.utils.config import load_config, resolve_path


def _normalize_tracking_uri(tracking_uri: str) -> str:
    if isinstance(tracking_uri, str) and tracking_uri.startswith("file:"):
        raw_path = tracking_uri[5:]
        return f"file:{resolve_path(raw_path)}"
    return tracking_uri


def _get_best_run(
    experiment_name: str,
    scoring_metric: str,
) -> dict[str, Any]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{scoring_metric} DESC"],
    )

    if runs_df.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    return runs_df.iloc[0].to_dict()


def _build_summary(
    experiment_name: str,
    scoring_metric: str,
    best_run: dict[str, Any],
    metrics_to_report: list[str],
) -> dict[str, Any]:
    metrics_summary: dict[str, Any] = {}
    for metric_name in metrics_to_report:
        metrics_summary[metric_name] = best_run.get(f"metrics.{metric_name}")

    return {
        "experiment_name": experiment_name,
        "scoring_metric": scoring_metric,
        "best_run_id": best_run.get("run_id"),
        "best_run_name": best_run.get("tags.mlflow.runName", "unknown"),
        "best_model_name": best_run.get("params.model_name", "unknown"),
        "metrics": metrics_summary,
    }


def main() -> None:
    config = load_config()

    tracking_uri = _normalize_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    scoring_metric = config["training"]["scoring_metric"]
    metrics_to_report = config.get("evaluation", {}).get("metrics", [])

    if not metrics_to_report:
        metrics_to_report = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    mlflow.set_tracking_uri(tracking_uri)

    best_run = _get_best_run(
        experiment_name=experiment_name,
        scoring_metric=scoring_metric,
    )

    summary = _build_summary(
        experiment_name=experiment_name,
        scoring_metric=scoring_metric,
        best_run=best_run,
        metrics_to_report=metrics_to_report,
    )

    print(json.dumps(summary, indent=2))

    output_path = resolve_path("artifacts/best_run_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
