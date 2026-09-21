"""Shared fixtures. One sample order for many checks."""

import json

import pandas as pd
import pytest
from src.config import ROOT


@pytest.fixture
def sample_row() -> dict:
    path = ROOT / "data" / "sample_order.json"
    return json.loads(path.read_text())


@pytest.fixture
def sample_df(sample_row) -> pd.DataFrame:
    row = dict(sample_row)
    row.pop("is_late", None)
    return pd.DataFrame([row])
