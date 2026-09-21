"""Model tests: load from registry, shape, known sample."""

import pytest
from src.predict import try_predict
from src.preprocess import load_bundle, to_matrix


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_model_loads_from_registry(bundle):
    assert bundle["model"] is not None
    assert bundle["model_version"] is not None
    assert str(bundle["model_version"]).isdigit()


def test_matrix_shape_is_63_features(sample_df, bundle):
    # saved imputer drops dist_km, so 63 columns, not 64
    x = to_matrix(sample_df, bundle)
    assert x.shape == (1, 63)


def test_predict_returns_one_row_and_probability(sample_df, bundle):
    result = try_predict(sample_df, bundle=bundle)
    assert result["ok"] is True
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["label"] in ("late", "on_time")
    assert 0.0 <= float(row["probability_late"]) <= 1.0
    assert str(row["model_version"]) == str(bundle["model_version"])


def test_known_sample_stays_on_time(sample_df, bundle):
    # same order we checked by hand in step 2 and 5
    result = try_predict(sample_df, bundle=bundle)
    row = result["rows"][0]
    assert row["label"] == "on_time"
    assert abs(float(row["probability_late"]) - 0.3356) < 0.01


def test_bad_input_does_not_crash(bundle):
    import pandas as pd

    result = try_predict(pd.DataFrame([{"order_id": "x"}]), bundle=bundle)
    assert result["ok"] is False
    assert "error" in result
