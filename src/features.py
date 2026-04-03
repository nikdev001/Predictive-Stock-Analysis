"""Technical features and labels for trend prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, low, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - low), (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, volatility, RSI, MACD, ATR, Bollinger, volume ratios."""
    out = df.copy()
    c = out["Close"]
    vol = out["Volume"].astype(float)
    out["ret_1"] = c.pct_change()
    out["ret_5"] = c.pct_change(5)
    out["ret_10"] = c.pct_change(10)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["rsi_14"] = _rsi(c, 14)
    macd, macd_sig, macd_hist = _macd(c)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    atr = _atr(out, 14)
    out["atr_14"] = atr
    out["atr_pct"] = atr / (c + 1e-12)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_pctb"] = (c - bb_mid) / (2 * bb_std + 1e-12)
    v_ma20 = vol.rolling(20).mean()
    out["vol_rel"] = vol / (v_ma20 + 1e-12)
    out["log_vol"] = np.log1p(vol)
    for w in [5, 10, 20]:
        ma = c.rolling(w).mean()
        out[f"close_ma{w}_ratio"] = c / (ma + 1e-12) - 1.0
    out["hl_range"] = (out["High"] - out["Low"]) / (c + 1e-12)
    out["co_ratio"] = (c - out["Open"]) / (out["Open"] + 1e-12)
    # Next-day trend: 1 if next close > today's close, else 0
    out["target"] = (c.shift(-1) > c).astype(float)
    return out


def feature_columns() -> list[str]:
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ret_1",
        "ret_5",
        "ret_10",
        "vol_10",
        "vol_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "atr_pct",
        "bb_pctb",
        "vol_rel",
        "log_vol",
        "close_ma5_ratio",
        "close_ma10_ratio",
        "close_ma20_ratio",
        "hl_range",
        "co_ratio",
    ]


def make_sequences(
    df: pd.DataFrame,
    lookback: int,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack sliding windows [samples, lookback, n_features] and binary targets.
    Window ends on day i (inclusive): features through today's close; target is next-day up.

    Also returns per-sample **as-of date** and **close** at end of window (for prediction tables).
    """
    data = df[feature_cols + ["target"]].dropna()
    feats = data[feature_cols].values
    targets = data["target"].values
    close_series = data["Close"].values
    idx = data.index
    n = len(data)
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    date_list: list[pd.Timestamp] = []
    close_list: list[float] = []
    for i in range(lookback - 1, n):
        if np.isnan(targets[i]):
            continue
        X_list.append(feats[i - lookback + 1 : i + 1])
        y_list.append(targets[i])
        t = idx[i]
        date_list.append(pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t)
        close_list.append(float(close_series[i]))
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    dates = np.array(date_list, dtype="datetime64[ns]")
    closes = np.asarray(close_list, dtype=np.float64)
    return X, y, dates, closes
