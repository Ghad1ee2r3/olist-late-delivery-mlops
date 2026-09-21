"""FastAPI service for late vs on time.

Routes:
  GET  /health
  GET  /model
  GET  /metrics
  POST /predict
  POST /predict/batch

The model loads once at startup from the MLflow registry.
"""

from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from src.config import load_config
from src.logging_setup import get_logger, setup_logging
from src.monitoring import snapshot
from src.predict import try_predict
from src.preprocess import load_bundle

from app.schemas import (
    BatchIn,
    ErrorResponse,
    HealthOut,
    ModelInfoOut,
    OrderIn,
    PredictionOut,
    PredictResponse,
)

log = get_logger("api")

# filled in lifespan. one load, many requests.
STATE = {"bundle": None, "cfg": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    cfg = load_config()
    STATE["cfg"] = cfg
    STATE["bundle"] = load_bundle(cfg)
    log.info(
        "api ready | model=%s version=%s",
        STATE["bundle"]["model_name"],
        STATE["bundle"]["model_version"],
    )
    yield
    STATE["bundle"] = None
    STATE["cfg"] = None


app = FastAPI(
    title="Olist late delivery",
    description="Inference only. ",
    version="0.1.0",
    lifespan=lifespan,
)


def _orders_to_frame(orders: List[OrderIn]) -> pd.DataFrame:
    return pd.DataFrame([o.model_dump() for o in orders])


def _predict_or_400(df: pd.DataFrame) -> PredictResponse:
    result = try_predict(df, bundle=STATE["bundle"])
    if not result["ok"]:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(ok=False, error=result["error"]).model_dump(),
        )
    rows = [PredictionOut(**row) for row in result["rows"]]
    return PredictResponse(ok=True, rows=rows)


@app.get("/health", response_model=HealthOut)
def health():
    cfg = STATE["cfg"] or load_config()
    return HealthOut(status="ok", project=cfg["project"]["name"])


@app.get("/model", response_model=ModelInfoOut)
def model_info():
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    cfg = STATE["cfg"]
    bundle = STATE["bundle"]
    n_in = int(getattr(bundle["model"], "n_features_in_", 63))
    return ModelInfoOut(
        model_name=bundle["model_name"],
        model_version=str(bundle["model_version"]),
        registered_name=cfg["mlflow"]["registered_name"],
        stage=cfg["mlflow"]["stage"],
        metric=cfg["problem"]["metric"],
        n_features=n_in,
    )


@app.get("/metrics")
def metrics():
    """Request count, latency, errors, prediction mix, drift hint."""
    return snapshot(STATE["cfg"] or load_config())


@app.post("/predict", response_model=PredictResponse)
def predict_one(order: OrderIn):
    return _predict_or_400(_orders_to_frame([order]))


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(body: BatchIn):
    return _predict_or_400(_orders_to_frame(body.orders))
