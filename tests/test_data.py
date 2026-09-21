"""Data tests: schema, ranges, nulls, expectations."""

import pytest
from src.expectations import run_expectations
from src.validate import REQUIRED_COLUMNS, BadInputError, check_columns


def test_sample_has_required_columns(sample_df):
    for col in REQUIRED_COLUMNS:
        assert col in sample_df.columns


def test_sample_passes_expectations(sample_df):
    run_expectations(sample_df)


def test_negative_price_is_rejected(sample_df):
    bad = sample_df.copy()
    bad["items_price_sum"] = -10.0
    with pytest.raises(BadInputError) as err:
        run_expectations(bad)
    assert "items_price_sum" in str(err.value)


def test_unknown_payment_type_is_rejected(sample_df):
    bad = sample_df.copy()
    bad["payment_type_main"] = "banana_pay"
    with pytest.raises(BadInputError) as err:
        run_expectations(bad)
    assert "payment_type_main" in str(err.value)


def test_null_weight_still_passes_schema(sample_df):
    # nulls are ok at the schema step. the imputer fills them later.
    soft = sample_df.copy()
    soft["product_weight_g"] = None
    check_columns(soft)
    run_expectations(soft)


def test_too_many_nulls_are_rejected(sample_df):
    bad = sample_df.copy()
    for col in REQUIRED_COLUMNS[:10]:
        bad[col] = None
    with pytest.raises(BadInputError) as err:
        run_expectations(bad)
    assert "missing rate" in str(err.value)
