"""Check a new order against the expectation file before scoring.

Decision: reject on fail. A warning is not enough if the value cannot be real
(negative price, unknown payment type, too many empty fields).
Nulls under the missing-rate limit still pass. The imputer fills those.
"""

from typing import Optional

import pandas as pd
import yaml

from src.config import load_config, resolve_path
from src.validate import REQUIRED_COLUMNS, BadInputError, missing_counts


def load_expectations(cfg: Optional[dict] = None) -> dict:
    if cfg is None:
        cfg = load_config()
    path = resolve_path(cfg["validation"]["expectations_file"])
    with path.open() as f:
        suite = yaml.safe_load(f)
    if not isinstance(suite, dict):
        raise BadInputError("expectations file is empty")
    return suite


def run_expectations(df: pd.DataFrame, cfg: Optional[dict] = None) -> None:
    if cfg is None:
        cfg = load_config()
    suite = load_expectations(cfg)
    max_rate = float(suite.get("max_missing_rate", cfg["validation"]["max_missing_rate"]))
    problems = []

    gaps = missing_counts(df)
    rate = len(gaps) / float(len(REQUIRED_COLUMNS))
    if rate > max_rate:
        problems.append("missing rate {:.2f} is above {:.2f}".format(rate, max_rate))

    for col, bounds in (suite.get("ranges") or {}).items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        low = bounds.get("min")
        high = bounds.get("max")
        bad = values.notna() & ((values < low) | (values > high))
        if bad.any():
            problems.append("{} out of range {} to {}".format(col, low, high))

    for col, allowed in (suite.get("allowed") or {}).items():
        if col not in df.columns:
            continue
        seen = df[col].dropna().astype(str)
        unknown = sorted(set(seen) - set(allowed))
        if unknown:
            problems.append("{} not allowed: {}".format(col, ", ".join(unknown)))

    if problems:
        raise BadInputError("expectations failed: " + "; ".join(problems))
