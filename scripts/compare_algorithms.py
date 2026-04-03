#!/usr/bin/env python3
"""Run several tabular backends on the same CSV/ticker and print balanced accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import config
from src.train import run


TABULAR_BACKENDS = ("sklearn", "logreg", "rf", "hist_gbm", "ensemble")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare algorithms on one dataset")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--ticker", type=str, default=config.DEFAULT_TICKER)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = p.parse_args()

    print(f"Dataset: csv={args.csv!r} ticker={args.ticker!r} start={args.start!r}")
    print(f"epochs={args.epochs} lookback={config.LOOKBACK_DAYS}")
    print()
    print(f"{'Backend':<14} {'Bal.Acc':>10} {'Thr':>8} {'Majority':>10}")
    print("-" * 48)

    for b in TABULAR_BACKENDS:
        try:
            out = run(
                csv_path=args.csv,
                ticker=args.ticker,
                start=args.start,
                epochs=args.epochs,
                model_out=None,
                backend=b,
            )
            m = out["metrics"]
            bal = m.get("balanced_accuracy", float("nan"))
            thr = m.get("threshold", float("nan"))
            maj = m.get("baselines", {}).get("majority_class_accuracy", float("nan"))
            print(f"{b:<14} {bal:10.4f} {thr:8.3f} {maj:10.4f}")
        except Exception as e:
            print(f"{b:<14} FAILED: {e}")


if __name__ == "__main__":
    main()
