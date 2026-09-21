"""One good order, logged to the terminal and to logs/service.log.

From Task[03]:
    python -m src.run_predict
"""

import json

import pandas as pd

from src.config import ROOT
from src.logging_setup import setup_logging
from src.predict import try_predict


def main() -> None:
    setup_logging()
    row = json.loads((ROOT / "data" / "sample_order.json").read_text())
    row.pop("is_late", None)
    result = try_predict(pd.DataFrame([row]))
    # the request itself is in the log line above. this only prints the dict.
    print(result)


if __name__ == "__main__":
    main()
