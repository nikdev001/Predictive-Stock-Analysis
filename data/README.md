# Data for this project

Document **what you used** in your report: source, symbol(s), date range, and format.

## Standard OHLCV CSV

- **Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume` (names can vary in case).
- **One row per trading day**, chronological order.
- **Example:** `small_sample.csv` (NIFTY daily range downloaded via Yahoo).

## Wide portfolio CSV

- **Columns:** `Date` plus **one numeric column per asset** (close or last price).
- The loader builds **synthetic** OHLCV: Open = High = Low = Close = that price, Volume = 1.0.
- **Choose the series** with `--price-column SYMBOL` when training.
- **Limitation:** Volume and true high/low ranges are not real; prefer Yahoo OHLCV per ticker for work comparable to many papers.

## Yahoo Finance (no file)

- Use `--ticker` (e.g. `^NSEI`, `RELIANCE.NS`) and optional `--start` / `--end`.
- Full daily OHLCV is used.

## Label used in code (clear definition)

- **Target:** binary **next trading day up** = 1 if next close **>** today’s close, else 0.
- Sequences use **past** days only; the split is **temporal** (see main README methodology).
