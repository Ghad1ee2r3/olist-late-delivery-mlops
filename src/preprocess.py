"""Load the objects notebook 5 saved, then transform. Never fit again."""

import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features import add_features, apply_rare_maps
from src.registry import load_registered_model
from src.validate import check_columns, drop_leakage


def load_bundle(cfg: Optional[dict] = None) -> dict:
    if cfg is None:
        cfg = load_config()
    model_cfg = cfg["model"]
    tdir = resolve_path(model_cfg["transformers_dir"])
    feature_list = json.loads(resolve_path(model_cfg["feature_list"]).read_text())
    model, version = load_registered_model(cfg)
    return {
        "num_features": feature_list["num_features"],
        "cat_features": feature_list["cat_features"],
        "rare_maps": joblib.load(tdir / "rare_maps.joblib"),
        "num_imputer": joblib.load(tdir / "num_imputer.joblib"),
        "cat_imputer": joblib.load(tdir / "cat_imputer.joblib"),
        "onehot": joblib.load(tdir / "onehot.joblib"),
        "model": model,
        "model_name": model_cfg["name"],
        "model_version": version,
    }


def to_matrix(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    check_columns(df)
    clean = drop_leakage(df)
    featured = add_features(clean)
    featured = apply_rare_maps(featured, bundle["rare_maps"])

    x_num = bundle["num_imputer"].transform(featured[bundle["num_features"]])
    x_cat_raw = bundle["cat_imputer"].transform(featured[bundle["cat_features"]])
    x_cat = bundle["onehot"].transform(x_cat_raw)
    return np.hstack([x_num, x_cat])
