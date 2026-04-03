"""Threshold tuning and sanity-check baselines for binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def majority_class_baseline(y: np.ndarray) -> tuple[float, int]:
    """Accuracy if we always predict the most frequent class."""
    y = y.astype(int)
    vals, counts = np.unique(y, return_counts=True)
    maj = int(vals[np.argmax(counts)])
    acc = float(np.mean(y == maj))
    return acc, maj


def baseline_metrics(y: np.ndarray) -> dict[str, float | int]:
    """
    Reference metrics on labels only (no model). Use the same y as the held-out test set.
    Aligns with common paper baselines: majority class, always-0, always-1.
    """
    y = y.astype(int)
    n = len(y)
    maj_acc, maj = majority_class_baseline(y)
    return {
        "n": int(n),
        "majority_class": int(maj),
        "accuracy_always_majority": float(maj_acc),
        "accuracy_always_0": float(np.mean(y == 0)),
        "accuracy_always_1": float(np.mean(y == 1)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: str = "balanced_accuracy",
) -> float:
    """
    Pick decision threshold on validation probabilities (not test).
    metric: 'balanced_accuracy' | 'f1'
    """
    y_true = y_true.astype(int)
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (proba >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, pred, zero_division=0)
        else:
            s = balanced_accuracy_score(y_true, pred)
        if s > best_score:
            best_score = s
            best_t = float(t)
    return best_t


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)
