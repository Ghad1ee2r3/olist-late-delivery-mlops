"""Log the notebook 6 result and register the saved model.

This does not train again. It records the run we already have
and puts the model in the local MLflow registry.

From Task[03]:
    python -m src.register_model
"""

import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import load_config, resolve_path
from src.logging_setup import get_logger, setup_logging


def tracking_uri(cfg: dict) -> str:
    folder = resolve_path(cfg["mlflow"]["tracking_uri"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder.as_uri()


def current_production(client: MlflowClient, name: str, stage: str):
    try:
        versions = client.get_latest_versions(name, stages=[stage])
    except Exception:
        return None
    if not versions:
        return None
    return versions[0]


def register() -> None:
    setup_logging()
    log = get_logger("mlflow")
    cfg = load_config()
    ml_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(ml_cfg["experiment"])

    client = MlflowClient()
    name = ml_cfg["registered_name"]
    stage = ml_cfg["stage"]
    existing = current_production(client, name, stage)
    if existing is not None:
        log.info(
            "model already registered | name=%s version=%s stage=%s",
            name,
            existing.version,
            stage,
        )
        return

    results = json.loads(resolve_path(ml_cfg["results_file"]).read_text())
    test_metrics = results["model_test"]
    model = joblib.load(resolve_path(cfg["model"]["model_file"]))

    with mlflow.start_run(run_name="notebook6_" + results["chosen_model"]) as run:
        mlflow.log_param("chosen_model", results["chosen_model"])
        mlflow.log_param("primary_metric", cfg["problem"]["metric"])
        mlflow.log_param("source_file", cfg["model"]["model_file"])
        for key, value in test_metrics.items():
            if key == "model" or value is None:
                continue
            mlflow.log_metric("test_" + key, float(value))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=name,
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{name}'")
    latest = max(versions, key=lambda item: int(item.version))
    client.transition_model_version_stage(
        name=name,
        version=latest.version,
        stage=stage,
        archive_existing_versions=True,
    )
    log.info(
        "registered model | name=%s version=%s stage=%s run_id=%s metric=%s value=%s",
        name,
        latest.version,
        stage,
        run_id,
        cfg["problem"]["metric"],
        test_metrics.get("f1_late"),
    )


if __name__ == "__main__":
    register()
