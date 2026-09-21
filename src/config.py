"""Load settings from config/config.yaml.

I keep paths here so I do not write /Users/... inside the code.
If I move the project, I only change the yaml file.
Docker can override a few values with environment variables.
"""

import os
from pathlib import Path

import yaml

# Task[03] folder = parent of src/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config file must be a yaml mapping")
    return _apply_env(cfg)


def _apply_env(cfg: dict) -> dict:
    """Let the container inject secrets and host settings without editing yaml.

    Use OLIST_* names so we do not pick up MLFLOW_TRACKING_URI that the
    mlflow library writes into the process environment after set_tracking_uri.
    """
    mlflow_cfg = cfg.setdefault("mlflow", {})
    api = cfg.setdefault("api", {})
    database = cfg.setdefault("database", {})

    if os.getenv("OLIST_MLFLOW_TRACKING_URI"):
        mlflow_cfg["tracking_uri"] = os.environ["OLIST_MLFLOW_TRACKING_URI"]
    if os.getenv("API_HOST"):
        api["host"] = os.environ["API_HOST"]
    if os.getenv("API_PORT"):
        api["port"] = int(os.environ["API_PORT"])
    if os.getenv("DATABASE_URL"):
        database["url"] = os.environ["DATABASE_URL"]
    return cfg


def resolve_path(relative: str) -> Path:
    """Turn a path from the yaml into a real path under Task[03]."""
    text = str(relative)
    if text.startswith("file:"):
        # mlflow sometimes stores file:///... URIs
        from urllib.parse import unquote, urlparse

        parsed = urlparse(text)
        return Path(unquote(parsed.path))
    p = Path(text)
    if p.is_absolute():
        return p
    return ROOT / p
