"""Build the same columns notebook 5 built, before the saved imputers.

These are known at prediction time:
purchase time, estimate, approval, price, freight, states, weight.
I do not use delivery date or review score. Those happen later.
"""

import numpy as np
import pandas as pd

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in DATE_COLS:
        d[c] = pd.to_datetime(d[c], errors="coerce")

    d["est_lead_days"] = (
        d["order_estimated_delivery_date"] - d["order_purchase_timestamp"]
    ).dt.days
    d["approve_hours"] = (
        d["order_approved_at"] - d["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600
    d["purchase_dow"] = d["order_purchase_timestamp"].dt.dayofweek
    d["purchase_month"] = d["order_purchase_timestamp"].dt.month
    d["same_state"] = (d["customer_state"] == d["seller_state"]).astype(float)

    for col in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    R = 6371.0
    lat1 = np.radians(d["customer_lat"])
    lon1 = np.radians(d["customer_lng"])
    lat2 = np.radians(d["seller_lat"])
    lon2 = np.radians(d["seller_lng"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d["dist_km"] = 2 * R * np.arcsin(np.sqrt(a))

    d["weight_missing"] = d["product_weight_g"].isna().astype(int)
    d["dist_missing"] = d["dist_km"].isna().astype(int)
    return d


def apply_rare_maps(df: pd.DataFrame, rare_maps: dict) -> pd.DataFrame:
    """Use the maps fitted on train. Do not recompute them."""
    d = df.copy()
    for col, keep in rare_maps.items():
        s = d[col].fillna("missing")
        d[col] = s.where(s.isin(keep), other="other")
    return d
