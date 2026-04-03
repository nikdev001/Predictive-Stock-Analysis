"""Train / evaluate next-day trend model on temporal splits (LSTM or sklearn MLP)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from . import config
from .data_loader import inspect_csv_path, load_from_csv, load_from_yahoo
from .eval_utils import apply_threshold, find_best_threshold, majority_class_baseline
from .features import build_feature_frame, feature_columns, make_sequences


@dataclass
class SequenceSplits:
    """Train/val/test tensors plus test-period dates/closes for prediction export."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    lookback: int
    n_features: int
    dates_test: np.ndarray
    close_test: np.ndarray
    symbol: str
    wide_columns: list[str] | None = None  # if wide portfolio CSV: all stock/price columns


def _symbol_for_dataset(
    csv_path: str | None,
    ticker: str | None,
    price_column: str | None,
) -> str:
    if price_column:
        return str(price_column)
    if ticker:
        t = ticker.replace("^", "").strip()
        return t or "ticker"
    if csv_path:
        return Path(csv_path).stem
    return "series"


def load_xy_splits(
    csv_path: str | None = None,
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    lookback: int | None = None,
    price_column: str | None = None,
) -> SequenceSplits:
    """Load OHLCV, build sequences, return train/val/test arrays and test metadata."""
    lookback = lookback or config.LOOKBACK_DAYS
    wide_columns: list[str] | None = None
    if csv_path:
        info = inspect_csv_path(csv_path)
        if info.get("format") == "wide":
            wide_columns = info.get("price_columns") or None
        raw = load_from_csv(csv_path, price_column=price_column)
    else:
        raw = load_from_yahoo(ticker or config.DEFAULT_TICKER, start=start, end=end)
    enriched = build_feature_frame(raw)
    cols = feature_columns()
    X, y, dates, closes = make_sequences(enriched, lookback, cols)
    train_sl, val_sl, test_sl = temporal_split(len(X), config.TRAIN_FRAC, config.VAL_FRAC)
    symbol = _symbol_for_dataset(csv_path, ticker, price_column)
    return SequenceSplits(
        X_train=X[train_sl],
        y_train=y[train_sl],
        X_val=X[val_sl],
        y_val=y[val_sl],
        X_test=X[test_sl],
        y_test=y[test_sl],
        lookback=lookback,
        n_features=X.shape[2],
        dates_test=dates[test_sl],
        close_test=closes[test_sl],
        symbol=symbol,
        wide_columns=wide_columns,
    )


def write_test_predictions_csv(
    path: str | Path,
    *,
    symbol: str,
    dates_test: np.ndarray,
    close_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
) -> None:
    """Save one row per test day: symbol, as-of date, close, actual/predicted next-day-up, probability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "symbol": symbol,
            "asof_date": pd.to_datetime(dates_test),
            "close": close_test.astype(float),
            "actual_next_day_up": y_true.astype(int),
            "predicted_next_day_up": y_pred.astype(int),
            "prob_next_day_up": proba.astype(float),
        }
    )
    df.to_csv(path, index=False)


def temporal_split(n: int, train_frac: float, val_frac: float) -> tuple[slice, slice, slice]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_sl = slice(0, n_train)
    val_sl = slice(n_train, n_train + n_val)
    test_sl = slice(n_train + n_val, n)
    return train_sl, val_sl, test_sl


def scale_sequence_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on training rows only (all timesteps), then transform all splits."""
    _, lookback, nfeat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, nfeat))

    def _tf(X: np.ndarray) -> np.ndarray:
        return scaler.transform(X.reshape(-1, nfeat)).reshape(X.shape[0], lookback, nfeat).astype(np.float32)

    return _tf(X_train), _tf(X_val), _tf(X_test), scaler


def run(
    csv_path: str | None = None,
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    lookback: int | None = None,
    epochs: int | None = None,
    model_out: str | None = None,
    backend: str = "tensorflow",
    price_column: str | None = None,
    tf_architecture: str | None = None,
    predictions_csv: str | None = None,
) -> dict:
    lookback = lookback or config.LOOKBACK_DAYS
    epochs = epochs or config.EPOCHS

    splits = load_xy_splits(
        csv_path=csv_path,
        ticker=ticker,
        start=start,
        end=end,
        lookback=lookback,
        price_column=price_column,
    )

    if backend in ("sklearn", "mlp", "logreg", "rf", "hist_gbm"):
        est = _make_sklearn_estimator(backend, epochs)
        label = "sklearn" if backend in ("sklearn", "mlp") else backend
        result = _run_sklearn_tabular(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            splits.X_test,
            splits.y_test,
            est,
            label,
            model_out,
        )
    elif backend == "ensemble":
        result = _run_ensemble(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            splits.X_test,
            splits.y_test,
            epochs,
            model_out,
        )
    else:
        result = _run_tensorflow(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            splits.X_test,
            splits.y_test,
            splits.lookback,
            splits.n_features,
            epochs,
            model_out,
            tf_architecture or config.DEFAULT_TF_ARCHITECTURE,
        )

    if predictions_csv:
        write_test_predictions_csv(
            predictions_csv,
            symbol=splits.symbol,
            dates_test=splits.dates_test,
            close_test=splits.close_test,
            y_true=result["y_test"],
            y_pred=result["y_pred"],
            proba=result["test_proba"],
        )
        result["metrics"]["predictions_csv"] = str(Path(predictions_csv).resolve())

    result["metrics"]["model_symbol"] = splits.symbol
    if splits.wide_columns:
        result["metrics"]["stocks_in_csv"] = ", ".join(splits.wide_columns)
    elif not csv_path and ticker:
        result["metrics"]["series"] = ticker

    return result


def _oversample_balance(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Match sklearn's balanced effect without sample_weight (older MLP lacks it)."""
    y = y.astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y
    n = max(len(idx0), len(idx1))
    rng = np.random.default_rng(seed)
    idx0r = rng.choice(idx0, size=n, replace=True)
    idx1r = rng.choice(idx1, size=n, replace=True)
    idx = np.concatenate([idx0r, idx1r])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _baselines(y_test: np.ndarray) -> dict[str, float]:
    maj_acc, _ = majority_class_baseline(y_test)
    return {
        "majority_class_accuracy": maj_acc,
        "random_guess_expected": 0.5,
    }


def _make_sklearn_estimator(kind: str, epochs: int):
    """Build a classifier for flattened sequence features."""
    if kind in ("sklearn", "mlp"):
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=max(epochs, 200),
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=max(config.PATIENCE, 15),
            random_state=config.RANDOM_SEED,
            learning_rate_init=config.LEARNING_RATE,
            alpha=1e-4,
        )
    if kind == "logreg":
        return LogisticRegression(
            max_iter=4000,
            random_state=config.RANDOM_SEED,
            C=0.3,
            solver="lbfgs",
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=12,
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        )
    if kind == "hist_gbm":
        return HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.07,
            l2_regularization=0.15,
            random_state=config.RANDOM_SEED,
        )
    raise ValueError(f"Unknown sklearn backend: {kind}")


def _run_sklearn_tabular(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator,
    backend_label: str,
    model_out: str | None,
) -> dict:
    """Scale on train only; fit estimator; tune threshold on val; evaluate on test."""
    X_train_s, X_val_s, X_test_s, scaler = scale_sequence_data(X_train, X_val, X_test)
    n_train, lookback, nfeat = X_train_s.shape
    X_train_flat = X_train_s.reshape(n_train, lookback * nfeat)
    y_tr = y_train
    if config.USE_BALANCED_SAMPLE_WEIGHT:
        X_train_flat, y_tr = _oversample_balance(X_train_flat, y_train, config.RANDOM_SEED)

    estimator.fit(X_train_flat, y_tr)

    X_val_flat = X_val_s.reshape(len(X_val_s), lookback * nfeat)
    X_test_flat = X_test_s.reshape(len(X_test_s), lookback * nfeat)
    proba_val = estimator.predict_proba(X_val_flat)[:, 1]
    thr = find_best_threshold(y_val, proba_val, metric=config.THRESHOLD_METRIC)

    proba = estimator.predict_proba(X_test_flat)[:, 1]
    y_pred = apply_threshold(proba, thr)
    y_true = y_test.astype(int)

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    metrics = {
        "n_test": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "balanced_accuracy": float(bal_acc),
        "threshold": thr,
        "baselines": _baselines(y_test),
        "backend": backend_label,
    }

    if model_out:
        import joblib

        out = Path(model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": estimator, "scaler": scaler, "threshold": thr}, out)

    return {
        "model": estimator,
        "metrics": metrics,
        "test_proba": proba,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def _run_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    model_out: str | None,
) -> dict:
    """
    Average predicted P(up) from three models (linear, random forest, MLP) on flattened windows.
    May help slightly when errors are uncorrelated; still bounded by market noise.
    """
    X_train_s, X_val_s, X_test_s, scaler = scale_sequence_data(X_train, X_val, X_test)
    n_train, lookback, nfeat = X_train_s.shape
    flat_dim = lookback * nfeat
    X_train_flat = X_train_s.reshape(n_train, flat_dim)
    X_val_flat = X_val_s.reshape(len(X_val_s), flat_dim)
    X_test_flat = X_test_s.reshape(len(X_test_s), flat_dim)

    y_tr = y_train
    if config.USE_BALANCED_SAMPLE_WEIGHT:
        X_train_flat, y_tr = _oversample_balance(X_train_flat, y_train, config.RANDOM_SEED)

    lr = LogisticRegression(
        max_iter=3000,
        random_state=config.RANDOM_SEED,
        C=0.5,
        solver="lbfgs",
    )
    rf = RandomForestClassifier(
        random_state=config.RANDOM_SEED,
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=15,
        n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        max_iter=max(epochs, 200),
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=max(config.PATIENCE, 15),
        random_state=config.RANDOM_SEED,
        learning_rate_init=config.LEARNING_RATE,
        alpha=1e-4,
    )

    lr.fit(X_train_flat, y_tr)
    rf.fit(X_train_flat, y_tr)
    mlp.fit(X_train_flat, y_tr)

    def _mean_proba(Xf: np.ndarray) -> np.ndarray:
        p1 = lr.predict_proba(Xf)[:, 1]
        p2 = rf.predict_proba(Xf)[:, 1]
        p3 = mlp.predict_proba(Xf)[:, 1]
        return (p1 + p2 + p3) / 3.0

    proba_val = _mean_proba(X_val_flat)
    thr = find_best_threshold(y_val, proba_val, metric=config.THRESHOLD_METRIC)

    proba = _mean_proba(X_test_flat)
    y_pred = apply_threshold(proba, thr)
    y_true = y_test.astype(int)

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    metrics = {
        "n_test": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "balanced_accuracy": float(bal_acc),
        "threshold": thr,
        "baselines": _baselines(y_test),
        "backend": "ensemble",
        "ensemble_models": ["logistic_regression", "random_forest", "mlp"],
    }

    if model_out:
        import joblib

        out = Path(model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "logistic_regression": lr,
                "random_forest": rf,
                "mlp": mlp,
                "scaler": scaler,
                "threshold": thr,
            },
            out,
        )

    return {
        "model": {"lr": lr, "rf": rf, "mlp": mlp},
        "metrics": metrics,
        "test_proba": proba,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def _run_tensorflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lookback: int,
    n_features: int,
    epochs: int,
    model_out: str | None,
    architecture: str,
) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    from .model_lstm import build_tf_sequence_model, set_seed

    X_train_s, X_val_s, X_test_s, _ = scale_sequence_data(X_train, X_val, X_test)

    set_seed(config.RANDOM_SEED)
    model = build_tf_sequence_model(
        architecture,
        lookback,
        n_features,
        learning_rate=config.LEARNING_RATE,
    )

    class_weight = None
    if config.USE_BALANCED_SAMPLE_WEIGHT:
        y_int = y_train.astype(int)
        classes = np.unique(y_int)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_int)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, config.PATIENCE // 3),
            min_lr=1e-6,
        ),
    ]

    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    proba_val = model.predict(X_val_s, verbose=0).ravel()
    thr = find_best_threshold(y_val, proba_val, metric=config.THRESHOLD_METRIC)

    proba = model.predict(X_test_s, verbose=0).ravel()
    y_pred = apply_threshold(proba, thr)
    y_true = y_test.astype(int)

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    metrics = {
        "n_test": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "balanced_accuracy": float(bal_acc),
        "threshold": thr,
        "baselines": _baselines(y_test),
        "backend": "tensorflow",
        "tf_architecture": architecture,
    }

    if model_out:
        out = Path(model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(out)

    return {
        "model": model,
        "metrics": metrics,
        "test_proba": proba,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def main() -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Train next-day trend model for NIFTY-50")
    p.add_argument(
        "--list-stocks",
        action="store_true",
        help="With --csv: print stock/price column names (wide portfolio files) and exit",
    )
    p.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV (else Yahoo)")
    p.add_argument("--ticker", type=str, default=config.DEFAULT_TICKER, help="Yahoo symbol, e.g. ^NSEI")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--lookback", type=int, default=config.LOOKBACK_DAYS)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--save", type=str, default=None, help="Path to save model (.keras or .joblib)")
    p.add_argument(
        "--price-column",
        type=str,
        default=None,
        dest="price_column",
        help="For wide portfolio CSV (Date + asset columns): which column to model as Close",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=(
            "tensorflow",
            "sklearn",
            "mlp",
            "logreg",
            "rf",
            "hist_gbm",
            "ensemble",
        ),
        default="tensorflow",
        help="tensorflow=LSTM; sklearn|mlp=MLP; logreg|rf|hist_gbm=tabular; ensemble=LR+RF+MLP avg",
    )
    p.add_argument(
        "--tf-arch",
        type=str,
        default=None,
        dest="tf_arch",
        choices=("lstm", "gru", "cnn_lstm"),
        help="TensorFlow only: stacked LSTM (default), stacked GRU, or CNN+Conv1D then LSTM",
    )
    p.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        dest="predictions_csv",
        help="Write test-set predictions (symbol, date, close, actual/predicted/proba) to this CSV path",
    )
    args = p.parse_args()

    if args.list_stocks:
        if not args.csv:
            print("Error: --list-stocks requires --csv", file=sys.stderr)
            sys.exit(1)
        info = inspect_csv_path(args.csv)
        if info["format"] == "ohlcv":
            print("This file is standard OHLCV (one instrument). No separate stock columns to list.")
        elif info["format"] == "wide":
            cols = info["price_columns"]
            print("Stock / series columns in this CSV:")
            for i, c in enumerate(cols, 1):
                print(f"  {i}. {c}")
            print(f"\nTrain one at a time with: --price-column <name>   (default first column: {cols[0]!r})")
        else:
            print("Could not detect format (need Date + numeric columns for wide portfolio).")
        sys.exit(0)

    result = run(
        csv_path=args.csv,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        epochs=args.epochs,
        model_out=args.save,
        backend=args.backend,
        price_column=args.price_column,
        tf_architecture=args.tf_arch,
        predictions_csv=args.predictions_csv,
    )
    m = result["metrics"]
    print(f"Backend: {m.get('backend', 'n/a')}")
    if m.get("stocks_in_csv"):
        print(f"Stocks in CSV: {m['stocks_in_csv']}")
    if m.get("series"):
        print(f"Yahoo ticker: {m['series']}")
    print(f"Series modeled this run: {m.get('model_symbol', 'n/a')}")
    if m.get("tf_architecture"):
        print(f"TF architecture: {m['tf_architecture']}")
    print(f"Decision threshold (from validation): {m.get('threshold', 0.5):.4f}")
    print(f"Balanced accuracy (test): {m.get('balanced_accuracy', float('nan')):.4f}")
    if "baselines" in m:
        b = m["baselines"]
        print(
            f"Baselines - majority class: {b['majority_class_accuracy']:.4f}, "
            f"random ~ {b['random_guess_expected']:.2f}"
        )
    if m.get("predictions_csv"):
        print(f"Predictions saved: {m['predictions_csv']}")
    print(m["classification_report"])
    print("Confusion matrix [tn fp; fn tp]:\n", np.array(m["confusion_matrix"]))


if __name__ == "__main__":
    main()
