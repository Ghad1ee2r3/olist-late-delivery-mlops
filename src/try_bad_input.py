"""Show that bad input does not crash, and nulls still get a prediction.

From Task[03]:
    python -m src.try_bad_input
"""

import json

import pandas as pd

from src.config import ROOT
from src.logging_setup import get_logger, setup_logging
from src.predict import try_predict


def main() -> None:
    setup_logging()
    log = get_logger("demo")

    broken = pd.DataFrame([{"order_id": "bad-1", "n_items": 1}])
    rejected = try_predict(broken)
    log.info("demo rejected | ok=%s error=%s", rejected["ok"], rejected.get("error"))

    row = json.loads((ROOT / "data" / "sample_order.json").read_text())
    row.pop("is_late", None)
    row["product_weight_g"] = None
    with_null = try_predict(pd.DataFrame([row]))
    label = None
    if with_null.get("rows"):
        label = with_null["rows"][0]["label"]
    log.info("demo null weight | ok=%s label=%s", with_null["ok"], label)


if __name__ == "__main__":
    main()
