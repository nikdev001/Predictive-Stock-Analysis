#!/usr/bin/env python3
"""Run trend training on several CSV paths and/or Yahoo tickers; print a compact summary."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src import config
from src.train import run


def main() -> None:
    # Edit this list to add your own CSV paths or tickers
    jobs: list[dict] = [
        {"name": "ADANIPORTS (CSV)", "csv": r"c:\Users\hp\Downloads\ADANIPORTS.csv"},
        {"name": "NIFTY 50", "ticker": "^NSEI", "start": "2015-01-01"},
        {"name": "Reliance", "ticker": "RELIANCE.NS", "start": "2015-01-01"},
        {"name": "TCS", "ticker": "TCS.NS", "start": "2015-01-01"},
        {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "start": "2015-01-01"},
    ]

    rows: list[tuple[str, float, float, float, int]] = []
    print("Backend: sklearn | epochs:", config.EPOCHS, "| lookback:", config.LOOKBACK_DAYS)
    print()

    for j in jobs:
        name = j["name"]
        try:
            kwargs = {
                "csv_path": j.get("csv"),
                "ticker": j.get("ticker"),
                "start": j.get("start"),
                "end": j.get("end"),
                "epochs": config.EPOCHS,
                "model_out": None,
                "backend": "sklearn",
            }
            out = run(**kwargs)
            m = out["metrics"]
            bal = m.get("balanced_accuracy", float("nan"))
            thr = m.get("threshold", float("nan"))
            base = m.get("baselines", {}).get("majority_class_accuracy", float("nan"))
            n = m.get("n_test", 0)
            rows.append((name, bal, base, thr, n))
            print(f"OK  {name}: balanced_acc={bal:.4f} majority_baseline={base:.4f} n_test={n}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            rows.append((name, float("nan"), float("nan"), float("nan"), 0))

    print()
    print(f"{'Dataset':<22} {'Bal.Acc':>8} {'Maj.':>8} {'Thr':>7} {'n_test':>6}")
    print("-" * 60)
    for name, bal, base, thr, n in rows:
        if bal != bal:
            print(f"{name:<22} {'(failed)':>8} {'':>8} {'':>7} {n:>6}")
        else:
            print(f"{name:<22} {bal:>8.4f} {base:>8.4f} {thr:>7.3f} {n:>6}")


if __name__ == "__main__":
    main()
