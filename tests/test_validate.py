"""Unit tests for input checks and leakage drop."""

import pandas as pd
import pytest
from src.validate import BadInputError, check_columns, drop_leakage, missing_counts


def test_check_columns_rejects_empty():
    with pytest.raises(BadInputError):
        check_columns(pd.DataFrame())


def test_check_columns_rejects_missing_column(sample_df):
    bad = sample_df.drop(columns=["product_weight_g"])
    with pytest.raises(BadInputError) as err:
        check_columns(bad)
    assert "product_weight_g" in str(err.value)


def test_check_columns_accepts_sample(sample_df):
    check_columns(sample_df)


def test_drop_leakage_removes_label_and_delivery(sample_df):
    dirty = sample_df.copy()
    dirty["is_late"] = 1
    dirty["order_delivered_customer_date"] = "2018-07-01"
    dirty["review_score"] = 5
    clean = drop_leakage(dirty)
    assert "is_late" not in clean.columns
    assert "order_delivered_customer_date" not in clean.columns
    assert "review_score" not in clean.columns
    assert "order_id" in dirty.columns


def test_missing_counts_reports_nulls(sample_df):
    counts = missing_counts(sample_df)
    assert "seller_lat" in counts
    assert counts["seller_lat"] == 1
