"""Load the model from the MLflow registry, not from the notebook folder.

Local MLflow stores absolute laptop paths in meta.yaml. Inside Docker those
paths do not exist. We resolve the model under the mounted mlruns folder
by run_id so the same files work on the laptop and in the container.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import load_config, resolve_path


def _set_uri(cfg: dict) -> None:
    folder = resolve_path(cfg["mlflow"]["tracking_uri"])
    if not folder.exists():
        raise FileNotFoundError(
            "mlflow store missing at {}. From Task[03] run: python -m src.register_model".format(
                folder
            )
        )
    mlflow.set_tracking_uri(folder.as_uri())


def _model_dir_for_run(tracking_dir: Path, run_id: str) -> Path:
    hits = list(tracking_dir.glob("*/{}/artifacts/model".format(run_id)))
    if not hits:
        raise FileNotFoundError(
            "model artifacts missing under {}. Re-run: python -m src.register_model".format(
                tracking_dir
            )
        )
    return hits[0]


def load_registered_model(cfg: dict = None):
    if cfg is None:
        cfg = load_config()
    _set_uri(cfg)
    name = cfg["mlflow"]["registered_name"]
    stage = cfg["mlflow"]["stage"]
    tracking_dir = resolve_path(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(name, stages=[stage])
    except Exception as exc:
        raise FileNotFoundError(
            "no registered model. From Task[03] run: python -m src.register_model"
        ) from exc
    if not versions:
        raise FileNotFoundError(
            "no model in stage {}. From Task[03] run: python -m src.register_model".format(stage)
        )
    chosen = versions[0]
    model_dir = _model_dir_for_run(tracking_dir, chosen.run_id)
    model = mlflow.sklearn.load_model(str(model_dir))
    return model, str(chosen.version)
