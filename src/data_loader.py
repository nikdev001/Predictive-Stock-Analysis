"""Load OHLCV from CSV, Yahoo Finance, or wide portfolio price columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def _wide_prices_to_ohlcv(df_raw: pd.DataFrame, price_column: str | None) -> pd.DataFrame:
    """
    Wide CSV: Date + one column per asset (close-only prices).
    Builds synthetic OHLCV (O=H=L=C=price, Volume=1) for feature pipeline compatibility.
    """
    date_col = None
    for c in df_raw.columns:
        if str(c).lower() == "date":
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV must include a Date column.")

    dfn = df_raw.copy()
    dfn["_dt"] = pd.to_datetime(dfn[date_col])
    numeric_cols: list[str] = []
    for c in dfn.columns:
        if c == date_col or c == "_dt":
            continue
        if pd.api.types.is_numeric_dtype(dfn[c]):
            numeric_cols.append(str(c))

    if not numeric_cols:
        raise ValueError("No numeric price columns found after Date.")

    if price_column is None:
        price_column = numeric_cols[0]
    elif price_column not in dfn.columns:
        raise ValueError(f"Column {price_column!r} not found. Available: {numeric_cols}")

    dfn = dfn.sort_values("_dt").drop_duplicates(subset=["_dt"]).set_index("_dt")
    close = dfn[price_column].astype(float)
    out = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1.0,
        },
        index=dfn.index,
    )
    out.index.name = "Date"
    return out


def inspect_csv_path(path: str | Path) -> dict:
    """
    Detect CSV shape: standard OHLCV vs wide portfolio (Date + multiple price columns).
    Returns {"format": "ohlcv"|"wide", "price_columns": list[str]}.
    """
    path = Path(path)
    df_raw = pd.read_csv(path, nrows=3)
    colmap = {c.lower(): c for c in df_raw.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    if all(k in colmap for k in required):
        return {"format": "ohlcv", "price_columns": []}

    date_col = None
    for c in df_raw.columns:
        if str(c).lower() == "date":
            date_col = c
            break
    if date_col is None:
        return {"format": "unknown", "price_columns": []}

    numeric_cols: list[str] = []
    for c in df_raw.columns:
        if c == date_col:
            continue
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            numeric_cols.append(str(c))

    return {"format": "wide", "price_columns": numeric_cols}


def load_from_csv(path: str | Path, price_column: str | None = None) -> pd.DataFrame:
    """
    Standard OHLCV: Date, Open, High, Low, Close, Volume (case-insensitive).

    Wide portfolio: Date + numeric columns (asset prices). Use price_column to pick one
    (default: first numeric column). Synthetic OHLCV is built from that series.
    """
    path = Path(path)
    df_raw = pd.read_csv(path)
    colmap = {c.lower(): c for c in df_raw.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    if all(k in colmap for k in required):
        rename = {
            colmap["date"]: "Date",
            colmap["open"]: "Open",
            colmap["high"]: "High",
            colmap["low"]: "Low",
            colmap["close"]: "Close",
            colmap["volume"]: "Volume",
        }
        df = df_raw.rename(columns=rename)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"])
        df = df.set_index("Date")
        return df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})

    return _wide_prices_to_ohlcv(df_raw, price_column)


def load_from_yahoo(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily OHLCV for a ticker (default NIFTY 50: ^NSEI)."""
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker or date range.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
