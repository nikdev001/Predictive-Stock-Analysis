#!/usr/bin/env python3
"""Compare LSTM vs GRU vs CNN+LSTM on one CSV (balanced accuracy vs majority baseline).

Tries TensorFlow first (same as run_training.py), then PyTorch with equivalent architectures.
If both fail to load (common on Windows without VC++ runtime), install dependencies or use WSL/Linux."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def _tensorflow_ok() -> bool:
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def _pytorch_ok() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(PROJECT / "data" / "small_sample.csv"))
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    use_tf = _tensorflow_ok()
    use_torch = not use_tf and _pytorch_ok()

    if not use_tf and not use_torch:
        print(
            "ERROR: Neither TensorFlow nor PyTorch could be imported.\n"
            "  - Windows: install Microsoft Visual C++ Redistributable (x64), then:\n"
            "      pip install tensorflow\n"
            "    or: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  - Or run this script in WSL2 / Linux / a conda env where TF or torch loads.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows: list[tuple[str, float, float, str]] = []

    if use_tf:
        from src.train import run

        backend_note = "tensorflow"
    else:
        from src.torch_sequence import run_torch_sequence
        from src.train import load_xy_splits

        backend_note = "pytorch"
        splits = load_xy_splits(
            csv_path=args.csv,
            lookback=args.lookback,
        )

    print(f"Backend: {backend_note}")
    print(f"CSV: {args.csv}  lookback={args.lookback}  epochs={args.epochs}\n")

    for arch in ("lstm", "gru", "cnn_lstm"):
        print(f"--- Training {arch} ---", flush=True)
        if use_tf:
            out = run(
                csv_path=args.csv,
                backend="tensorflow",
                tf_architecture=arch,
                lookback=args.lookback,
                epochs=args.epochs,
                model_out=None,
            )
            m = out["metrics"]
        else:
            m = run_torch_sequence(
                splits.X_train,
                splits.y_train,
                splits.X_val,
                splits.y_val,
                splits.X_test,
                splits.y_test,
                architecture=arch,
                epochs=args.epochs,
            )
        bal = float(m["balanced_accuracy"])
        maj = float(m["baselines"]["majority_class_accuracy"])
        rows.append((arch, bal, maj, m.get("backend", backend_note)))
        print(f"{arch}: balanced_acc={bal:.4f}  majority_baseline={maj:.4f}\n", flush=True)

    hdr = f"{'Architecture':<12} {'Balanced acc':>14} {'Majority base':>14}  Backend"
    print(hdr)
    print("-" * (len(hdr) + 2))
    for arch, bal, maj, bk in rows:
        print(f"{arch:<12} {bal:14.4f} {maj:14.4f}  {bk}")


if __name__ == "__main__":
    main()
