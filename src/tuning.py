"""Time-series RandomizedSearchCV on the training split only (no test leakage)."""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import config
from .eval_utils import apply_threshold, find_best_threshold
from .train import _baselines, _oversample_balance, load_xy_splits, scale_sequence_data


def _base_estimator(model: str):
    model = model.lower()
    if model == "logreg":
        return LogisticRegression(max_iter=5000, solver="lbfgs", random_state=config.RANDOM_SEED)
    if model == "rf":
        return RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)
    if model in ("hist_gbm", "hgb", "gbm"):
        return HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)
    raise ValueError(f"Unknown model: {model}. Use logreg, rf, or hist_gbm.")


def _param_grid(model: str) -> dict:
    model = model.lower()
    if model == "logreg":
        return {
            "C": loguniform(1e-3, 1e2),
            "class_weight": [None, "balanced"],
        }
    if model == "rf":
        return {
            "n_estimators": randint(80, 350),
            "max_depth": randint(6, 22),
            "min_samples_leaf": randint(3, 35),
            "class_weight": [None, "balanced"],
        }
    if model in ("hist_gbm", "hgb", "gbm"):
        return {
            "learning_rate": loguniform(0.02, 0.25),
            "max_depth": randint(4, 14),
            "max_iter": randint(80, 350),
            "l2_regularization": loguniform(1e-4, 1.0),
        }
    raise ValueError(model)


def run_random_search(
    csv_path: str | None = None,
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    lookback: int | None = None,
    model: str = "rf",
    n_iter: int = 35,
    cv_splits: int = 5,
    price_column: str | None = None,
) -> dict:
    """
    Random search with TimeSeriesSplit on scaled train features only.
    Retrains best params on (optionally oversampled) train, tunes threshold on val, reports test.
    """
    splits = load_xy_splits(
        csv_path=csv_path,
        ticker=ticker,
        start=start,
        end=end,
        lookback=lookback,
        price_column=price_column,
    )
    X_train_s, X_val_s, X_test_s, _ = scale_sequence_data(
        splits.X_train, splits.X_val, splits.X_test
    )
    n_tr = X_train_s.shape[0]
    flat_dim = splits.lookback * splits.n_features
    X_tr_flat = X_train_s.reshape(n_tr, flat_dim)
    X_val_flat = X_val_s.reshape(len(X_val_s), flat_dim)
    X_test_flat = X_test_s.reshape(len(X_test_s), flat_dim)

    y_tr = splits.y_train.astype(int)
    est = _base_estimator(model)
    grid = _param_grid(model)
    tscv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, n_tr // 200)))
    search = RandomizedSearchCV(
        est,
        grid,
        n_iter=n_iter,
        cv=tscv,
        scoring=make_scorer(balanced_accuracy_score),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tr_flat, y_tr)

    best_params = search.best_params_
    clf = clone(search.best_estimator_)
    if config.USE_BALANCED_SAMPLE_WEIGHT:
        X_fit, y_fit = _oversample_balance(X_tr_flat, splits.y_train, config.RANDOM_SEED)
        clf.fit(X_fit, y_fit.astype(int))
    else:
        clf.fit(X_tr_flat, y_tr)

    proba_val = clf.predict_proba(X_val_flat)[:, 1]
    thr = find_best_threshold(splits.y_val, proba_val, metric=config.THRESHOLD_METRIC)
    proba = clf.predict_proba(X_test_flat)[:, 1]
    y_pred = apply_threshold(proba, thr)
    y_true = splits.y_test.astype(int)

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    return {
        "best_params": best_params,
        "best_cv_score": float(search.best_score_),
        "threshold": thr,
        "balanced_accuracy_test": float(bal),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "baselines": _baselines(splits.y_test),
        "model": model,
    }
