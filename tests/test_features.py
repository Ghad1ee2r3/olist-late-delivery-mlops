"""Unit tests for feature building. No model needed."""

import pandas as pd
from src.features import add_features, apply_rare_maps


def test_add_features_creates_expected_columns(sample_df):
    out = add_features(sample_df)
    for col in [
        "est_lead_days",
        "approve_hours",
        "purchase_dow",
        "purchase_month",
        "same_state",
        "dist_km",
        "weight_missing",
        "dist_missing",
    ]:
        assert col in out.columns


def test_same_state_is_one_when_states_match(sample_df):
    out = add_features(sample_df)
    assert float(out["same_state"].iloc[0]) == 1.0


def test_est_lead_days_is_positive(sample_df):
    out = add_features(sample_df)
    assert int(out["est_lead_days"].iloc[0]) > 0


def test_apply_rare_maps_keeps_known_and_maps_unknown():
    df = pd.DataFrame({"payment_type_main": ["boleto", "crypto_coin"]})
    rare_maps = {"payment_type_main": ["boleto", "credit_card"]}
    out = apply_rare_maps(df, rare_maps)
    assert out["payment_type_main"].tolist() == ["boleto", "other"]
