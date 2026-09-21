"""Monitoring tests: counters and prediction log file."""

import json

import pandas as pd
from src.monitoring import reset_stats_for_tests, snapshot
from src.predict import try_predict
from src.preprocess import load_bundle


def test_metrics_count_success_and_error(tmp_path, monkeypatch, sample_df):
    reset_stats_for_tests()
    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("MONITORING_PREDICTIONS_FILE", str(log_file))

    # patch resolve through config override
    from src import monitoring as mon

    monkeypatch.setattr(
        mon,
        "_predictions_path",
        lambda cfg=None: log_file,
    )

    bundle = load_bundle()
    ok = try_predict(sample_df, bundle=bundle)
    assert ok["ok"] is True

    bad = try_predict(pd.DataFrame([{"order_id": "x"}]), bundle=bundle)
    assert bad["ok"] is False

    stats = snapshot()
    assert stats["requests_total"] == 2
    assert stats["requests_ok"] == 1
    assert stats["requests_error"] == 1
    assert stats["predictions"]["on_time"] + stats["predictions"]["late"] == 1

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "probability_late" in row
    assert row["actual_is_late"] is None
