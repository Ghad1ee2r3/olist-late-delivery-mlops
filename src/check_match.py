"""Prove this pipeline matches notebook 5 on the same test rows.

The saved X matrix from the notebook is the source of truth.
If this fails, inference is not the same as the notebook.

From Task[03]:
    python -m src.check_match
"""

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, resolve_path
from src.preprocess import load_bundle, to_matrix


def main() -> None:
    cfg = load_config()
    checks = cfg.get("checks", {})
    x_path = resolve_path(checks["notebook_x_test"])
    table_path = resolve_path(checks["notebook_test_table"])

    saved_x = np.load(x_path)
    test = pd.read_pickle(table_path)
    # same first rows the notebook wrote, not a new split
    n = min(20, len(test), len(saved_x))
    mine = to_matrix(test.iloc[:n], load_bundle(cfg))

    if mine.shape != saved_x[:n].shape:
        raise SystemExit(f"shape mismatch mine={mine.shape} notebook={saved_x[:n].shape}")

    max_diff = float(np.nanmax(np.abs(mine - saved_x[:n])))
    print("compared rows:", n)
    print("matrix shape:", mine.shape)
    print("max abs diff vs notebook:", max_diff)
    if max_diff > 1e-6:
        raise SystemExit("pipeline does not match notebook 5")
    print("match ok")
    print("project root name:", ROOT.name)


if __name__ == "__main__":
    main()
