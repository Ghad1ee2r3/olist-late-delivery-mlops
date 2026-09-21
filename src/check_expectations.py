"""Show a good order passing checks, and a bad price being rejected.

From Task[03]:
    python -m src.check_expectations
"""

import json

import pandas as pd

from src.config import ROOT
from src.logging_setup import get_logger, setup_logging
from src.predict import try_predict


def main() -> None:
    setup_logging()
    log = get_logger("expectations")

    row = json.loads((ROOT / "data" / "sample_order.json").read_text())
    row.pop("is_late", None)
    good = try_predict(pd.DataFrame([row]))
    log.info("sample order | ok=%s", good["ok"])

    bad = dict(row)
    bad["items_price_sum"] = -10
    bad["order_id"] = "bad-price"
    rejected = try_predict(pd.DataFrame([bad]))
    log.info("negative price | ok=%s error=%s", rejected["ok"], rejected.get("error"))


if __name__ == "__main__":
    main()
