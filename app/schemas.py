"""Request and response shapes for the API.

FastAPI checks these before my predict code runs.
Bad types get a 422. Bad business rules get a 400 from try_predict.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_ORDER = {
    "order_id": "b3b54427f53d13f6063ef7007bf7d371",
    "order_purchase_timestamp": "2018-06-21 08:41:07",
    "order_approved_at": "2018-06-22 02:59:29",
    "order_estimated_delivery_date": "2018-07-04 00:00:00",
    "customer_state": "SP",
    "seller_state": "SP",
    "customer_lat": -23.609430024757696,
    "customer_lng": -46.66050227039208,
    "seller_lat": None,
    "seller_lng": None,
    "product_weight_g": 200.0,
    "product_photos_qty": 5.0,
    "n_items": 1.0,
    "items_price_sum": 55.0,
    "freight_sum": 7.65,
    "payment_value_sum": 62.65,
    "max_installments": 1.0,
    "payment_type_main": "boleto",
    "product_category_name_english": "watches_gifts",
}


class OrderIn(BaseModel):
    """One order. Same columns the notebook features expect."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": SAMPLE_ORDER})

    order_id: Optional[str] = None
    order_purchase_timestamp: str
    order_approved_at: str
    order_estimated_delivery_date: str
    customer_state: str
    seller_state: str
    customer_lat: float
    customer_lng: float
    seller_lat: Optional[float] = None
    seller_lng: Optional[float] = None
    product_weight_g: Optional[float] = None
    product_photos_qty: Optional[float] = None
    n_items: float = Field(..., ge=1)
    items_price_sum: float
    freight_sum: float
    payment_value_sum: float
    max_installments: float
    payment_type_main: str
    product_category_name_english: str


class BatchIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"orders": [SAMPLE_ORDER, SAMPLE_ORDER]}}
    )

    orders: List[OrderIn] = Field(..., min_length=1)


class PredictionOut(BaseModel):
    order_id: Optional[str] = None
    prediction: int
    label: str
    probability_late: float
    model_name: str
    model_version: str


class PredictResponse(BaseModel):
    ok: bool = True
    rows: List[PredictionOut]


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str


class HealthOut(BaseModel):
    status: str
    project: str


class ModelInfoOut(BaseModel):
    model_name: str
    model_version: str
    registered_name: str
    stage: str
    metric: str
    n_features: int
