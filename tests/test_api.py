"""Integration tests for the FastAPI routes."""

import json

import pytest
from fastapi.testclient import TestClient
from src.config import ROOT


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_order():
    path = ROOT / "data" / "sample_order.json"
    row = json.loads(path.read_text())
    row.pop("is_late", None)
    return row


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "project" in body


def test_model_info(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "gbdt_weighted"
    assert body["stage"] == "Production"
    assert str(body["model_version"]).isdigit()
    assert body["n_features"] == 63


def test_predict_one(client, sample_order):
    r = client.post("/predict", json=sample_order)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["label"] == "on_time"
    assert abs(float(row["probability_late"]) - 0.3356) < 0.01
    assert str(row["model_version"]).isdigit()


def test_predict_batch(client, sample_order):
    r = client.post("/predict/batch", json={"orders": [sample_order, sample_order]})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["rows"]) == 2


def test_predict_rejects_bad_payload(client):
    r = client.post("/predict", json={"order_id": "x"})
    assert r.status_code == 422


def test_predict_rejects_negative_price(client, sample_order):
    bad = dict(sample_order)
    bad["items_price_sum"] = -5
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["ok"] is False
    assert "error" in detail


def test_metrics_endpoint(client, sample_order):
    client.post("/predict", json=sample_order)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "requests_total" in body
    assert "latency_ms_avg" in body
    assert "predictions" in body
    assert "drift" in body
    assert "alerts_triggered" in body
