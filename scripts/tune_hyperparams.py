#!/usr/bin/env python3
"""Random search with time-series CV on train split; final metric on held-out test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import config
from src.tuning import run_random_search


def main() -> None:
    p = argparse.ArgumentParser(description="Time-series hyperparameter search (logreg / rf / hist_gbm)")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--ticker", type=str, default=config.DEFAULT_TICKER)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--model", type=str, default="rf", choices=("logreg", "rf", "hist_gbm"))
    p.add_argument("--n-iter", type=int, default=35, dest="n_iter")
    p.add_argument("--cv-splits", type=int, default=5, dest="cv_splits")
    args = p.parse_args()

    out = run_random_search(
        csv_path=args.csv,
        ticker=args.ticker,
        start=args.start,
        lookback=args.lookback,
        model=args.model,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
    )
    print("Model:", out["model"])
    print("Best CV balanced accuracy:", round(out["best_cv_score"], 4))
    print("Best params:", out["best_params"])
    print("Threshold (val):", round(out["threshold"], 4))
    print("Balanced accuracy (test):", round(out["balanced_accuracy_test"], 4))
    print("Baselines:", out["baselines"])
    print(out["classification_report"])
    print("Confusion matrix [tn fp; fn tp]:")
    for row in out["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()
