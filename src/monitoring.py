"""Service metrics and prediction logs.

I keep counters in memory for the live process.
I also append each scored row to a jsonl file so later, when the real
delivery date arrives, I can compare prediction vs truth.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import load_config, resolve_path

_lock = threading.Lock()

# live counters for this process
STATS = {
    "requests_total": 0,
    "requests_ok": 0,
    "requests_error": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
    "pred_late": 0,
    "pred_on_time": 0,
    "probability_sum": 0.0,
    "probability_n": 0,
    "started_at": time.time(),
}


def _predictions_path(cfg: Optional[dict] = None) -> Path:
    if cfg is None:
        cfg = load_config()
    mon = cfg.get("monitoring", {})
    path = resolve_path(mon.get("predictions_file", "logs/predictions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_success(
    rows: List[Dict[str, Any]],
    latency_ms: float,
    cfg: Optional[dict] = None,
) -> None:
    """Count a good request and store each prediction line."""
    if cfg is None:
        cfg = load_config()
    path = _predictions_path(cfg)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with _lock:
        STATS["requests_total"] += 1
        STATS["requests_ok"] += 1
        STATS["latency_ms_sum"] += float(latency_ms)
        STATS["latency_ms_max"] = max(STATS["latency_ms_max"], float(latency_ms))

        with path.open("a") as f:
            for row in rows:
                label = row.get("label", "on_time")
                proba = float(row.get("probability_late", 0.0))
                if label == "late":
                    STATS["pred_late"] += 1
                else:
                    STATS["pred_on_time"] += 1
                STATS["probability_sum"] += proba
                STATS["probability_n"] += 1

                line = {
                    "ts": now,
                    "order_id": row.get("order_id"),
                    "prediction": row.get("prediction"),
                    "label": label,
                    "probability_late": proba,
                    "model_name": row.get("model_name"),
                    "model_version": str(row.get("model_version")),
                    "latency_ms": round(float(latency_ms), 2),
                    # filled later when the real delivery date is known
                    "actual_is_late": None,
                }
                f.write(json.dumps(line) + "\n")


def record_error(latency_ms: float = 0.0) -> None:
    with _lock:
        STATS["requests_total"] += 1
        STATS["requests_error"] += 1
        if latency_ms:
            STATS["latency_ms_sum"] += float(latency_ms)
            STATS["latency_ms_max"] = max(STATS["latency_ms_max"], float(latency_ms))


def snapshot(cfg: Optional[dict] = None) -> dict:
    """Numbers for GET /metrics. Includes a simple drift hint."""
    if cfg is None:
        cfg = load_config()
    mon = cfg.get("monitoring", {})
    baseline_late_rate = float(mon.get("baseline_late_rate", 0.066))
    alert_late_rate_high = float(mon.get("alert_late_rate_high", 0.15))
    alert_error_rate_high = float(mon.get("alert_error_rate_high", 0.05))
    alert_latency_ms = float(mon.get("alert_latency_ms", 2000))

    with _lock:
        total = STATS["requests_total"]
        ok = STATS["requests_ok"]
        err = STATS["requests_error"]
        scored = STATS["pred_late"] + STATS["pred_on_time"]
        avg_latency = (STATS["latency_ms_sum"] / total) if total else 0.0
        error_rate = (err / total) if total else 0.0
        late_rate = (STATS["pred_late"] / scored) if scored else 0.0
        avg_proba = (
            STATS["probability_sum"] / STATS["probability_n"]
            if STATS["probability_n"]
            else 0.0
        )
        out = {
            "uptime_sec": round(time.time() - STATS["started_at"], 1),
            "requests_total": total,
            "requests_ok": ok,
            "requests_error": err,
            "error_rate": round(error_rate, 4),
            "latency_ms_avg": round(avg_latency, 2),
            "latency_ms_max": round(STATS["latency_ms_max"], 2),
            "predictions": {
                "late": STATS["pred_late"],
                "on_time": STATS["pred_on_time"],
                "late_rate": round(late_rate, 4),
                "avg_probability_late": round(avg_proba, 4),
            },
            "drift": {
                "baseline_late_rate": baseline_late_rate,
                "current_late_rate": round(late_rate, 4),
                "delta": round(late_rate - baseline_late_rate, 4),
            },
            "alerts_triggered": [],
            "predictions_file": str(_predictions_path(cfg)),
        }

    alerts = []
    if total > 0 and error_rate >= alert_error_rate_high:
        alerts.append("error_rate_high")
    if total > 0 and avg_latency >= alert_latency_ms:
        alerts.append("latency_high")
    if scored >= 20 and late_rate >= alert_late_rate_high:
        alerts.append("late_rate_drift")
    out["alerts_triggered"] = alerts
    return out


def reset_stats_for_tests() -> None:
    """Only for pytest. Do not call from the API."""
    with _lock:
        STATS["requests_total"] = 0
        STATS["requests_ok"] = 0
        STATS["requests_error"] = 0
        STATS["latency_ms_sum"] = 0.0
        STATS["latency_ms_max"] = 0.0
        STATS["pred_late"] = 0
        STATS["pred_on_time"] = 0
        STATS["probability_sum"] = 0.0
        STATS["probability_n"] = 0
        STATS["started_at"] = time.time()
