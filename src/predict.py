"""Predict late vs on time. Loads the saved model. Does not train.

Every call is logged: input, output, latency, model version.
Bad input returns an error dict. It does not raise out of try_predict.
"""

import time
from typing import Optional

import pandas as pd

from src.expectations import run_expectations
from src.logging_setup import get_logger
from src.monitoring import record_error, record_success
from src.preprocess import load_bundle, to_matrix
from src.validate import BadInputError, missing_counts

log = get_logger("predict")


def predict_orders(df: pd.DataFrame, bundle: Optional[dict] = None) -> pd.DataFrame:
    if bundle is None:
        bundle = load_bundle()

    started = time.perf_counter()
    order_ids = df["order_id"].astype(str).tolist() if "order_id" in df.columns else []
    gaps = missing_counts(df)
    if gaps:
        log.warning("missing values will be imputed | counts=%s", gaps)

    try:
        run_expectations(df)
        x = to_matrix(df, bundle)
        pred = bundle["model"].predict(x)
        proba = bundle["model"].predict_proba(x)[:, 1]
    except BadInputError:
        raise
    except Exception as exc:
        ms = (time.perf_counter() - started) * 1000
        log.error(
            "prediction failed | input_n=%s ids=%s | error=%s | latency_ms=%.1f | model=%s version=%s",
            len(df),
            order_ids,
            exc,
            ms,
            bundle["model_name"],
            bundle["model_version"],
        )
        raise BadInputError("could not score this input") from exc

    out = pd.DataFrame(
        {
            "prediction": pred.astype(int),
            "label": ["late" if p == 1 else "on_time" for p in pred],
            "probability_late": proba.round(4),
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
        }
    )
    if "order_id" in df.columns:
        out.insert(0, "order_id", df["order_id"].values)

    ms = (time.perf_counter() - started) * 1000
    log.info(
        "prediction request | input_n=%s ids=%s | output=%s | latency_ms=%.1f | model=%s version=%s",
        len(out),
        order_ids,
        out[["label", "probability_late"]].to_dict(orient="records"),
        ms,
        bundle["model_name"],
        bundle["model_version"],
    )
    record_success(out.to_dict(orient="records"), latency_ms=ms)
    return out


def try_predict(df: pd.DataFrame, bundle: Optional[dict] = None) -> dict:
    """Service-style wrapper. Bad input is a result, not a crash."""
    started = time.perf_counter()
    try:
        rows = predict_orders(df, bundle=bundle)
        return {"ok": True, "rows": rows.to_dict(orient="records")}
    except BadInputError as exc:
        ms = (time.perf_counter() - started) * 1000
        record_error(latency_ms=ms)
        log.error("rejected input | error=%s", exc)
        return {"ok": False, "error": str(exc)}
