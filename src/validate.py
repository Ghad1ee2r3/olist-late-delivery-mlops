"""Stop bad rows before they reach the model.

Missing values are ok. The saved imputer fills those.
A missing column is not ok. I raise a clear error instead of crashing later.
"""

REQUIRED_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "customer_state",
    "seller_state",
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
    "product_weight_g",
    "product_photos_qty",
    "n_items",
    "items_price_sum",
    "freight_sum",
    "payment_value_sum",
    "max_installments",
    "payment_type_main",
    "product_category_name_english",
]

# these must never be used as model input
LEAKAGE_COLUMNS = [
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "review_score",
    "review_comment_message",
    "is_late",
]


class BadInputError(ValueError):
    """Input we can explain. Caller should catch this, not crash."""


def check_columns(df) -> None:
    if df is None or len(df) == 0:
        raise BadInputError("no rows to predict")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BadInputError("missing columns: " + ", ".join(missing))


def missing_counts(df) -> dict:
    """Nulls the imputer can fill. Logged, not rejected."""
    counts = {}
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n:
            counts[col] = n
    return counts


def drop_leakage(df):
    """If someone passes the label or a delivery date, drop it before features."""
    extra = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if not extra:
        return df
    return df.drop(columns=extra)
