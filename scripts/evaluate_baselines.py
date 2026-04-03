#!/usr/bin/env python3
"""
Print reference metrics on the held-out TEST labels only (no model training).
Use the same CSV/ticker/lookback as training so baselines match your experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import config
from src.eval_utils import baseline_metrics
from src.train import load_xy_splits


def main() -> None:
    p = argparse.ArgumentParser(description="Test-set baseline metrics (majority / always-0 / always-1)")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--ticker", type=str, default=config.DEFAULT_TICKER)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--price-column", type=str, default=None, dest="price_column")
    args = p.parse_args()

    splits = load_xy_splits(
        csv_path=args.csv,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        price_column=args.price_column,
    )
    out = baseline_metrics(splits.y_test)
    out["lookback_used"] = splits.lookback
    out["train_frac"] = config.TRAIN_FRAC
    out["val_frac"] = config.VAL_FRAC
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
