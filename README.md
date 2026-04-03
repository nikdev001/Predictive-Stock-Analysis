# Predictive Stock Analysis

Next-day **up / down** trend experiments on daily OHLCV (or synthetic OHLCV from wide price columns). Uses a time-based train / validation / test split, optional probability threshold tuning on validation, and several model backends.

## Methodology (aligned with common research practice)

This repo is **not** tied to one paper’s market or sentiment pipeline. It follows the **same type** of setup good empirical work uses:

| Element | What this project does |
|--------|-------------------------|
| **Temporal split** | Rows are **ordered by date**. First **70%** of sequence samples → train, next **15%** → validation, last **15%** → test. **No random shuffling** of days. Adjust `TRAIN_FRAC` / `VAL_FRAC` in `src/config.py` if needed. |
| **Clear label** | **Next-day direction:** class **1** if next close **>** today’s close, **0** otherwise (binary). |
| **Baselines** | Training prints **majority-class accuracy** on the **test** labels (always predict the more frequent class). Run `python scripts/evaluate_baselines.py` for **additional** reference numbers on the same test set: always-0 and always-1 accuracy (no model). |
| **Multiple metrics** | **Standard accuracy**, **balanced accuracy**, precision/recall/F1 (per class), **confusion matrix**, **AUC** (TensorFlow). Threshold on probabilities is tuned on **validation** only (`THRESHOLD_METRIC` in config). |
| **Document data** | Describe your source in writing: **ticker or CSV path**, **date range**, **OHLCV vs wide portfolio**. See **`data/README.md`**. |

You can still use **any** market (Yahoo `--ticker`) or **your CSV**; you do **not** need to replicate a specific paper’s horizon or sentiment stack unless you add that data yourself.

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

**TensorFlow** is only needed for `--backend tensorflow` (LSTM). For `sklearn`, `logreg`, `rf`, `hist_gbm`, and `ensemble`, you can run without TensorFlow if you omit it from the install.

## Run training (main entry)

```bash
python run_training.py [options]
```

### Data source (pick one)

| Source | Example |
|--------|---------|
| **Yahoo Finance** (default NIFTY 50 `^NSEI`) | `python run_training.py --backend sklearn --start 2015-01-01` |
| **CSV with OHLCV** | `python run_training.py --backend sklearn --csv data/your_file.csv` |
| **Wide portfolio CSV** (`Date` + one column per asset price) | `python run_training.py --backend sklearn --csv path/to/portfolio.csv --price-column AMZN` |

If `--csv` is omitted, data is downloaded for `--ticker` (default `^NSEI`).

### Useful flags

| Flag | Meaning |
|------|---------|
| `--backend` | Model: `tensorflow` (LSTM), `sklearn` / `mlp` (neural net), `logreg`, `rf`, `hist_gbm`, `ensemble` (average of linear + RF + MLP). Default: `tensorflow`. |
| `--lookback` | Days per sequence window (default from `src/config.py`, usually `60`). |
| `--epochs` | Max training epochs (LSTM / MLP). |
| `--save` | Save model: `.keras` for LSTM, `.joblib` for sklearn/ensemble. |
| `--start` / `--end` | Yahoo download range (`YYYY-MM-DD`). |
| `--price-column` | For wide CSV only: which column to treat as the close series. |
| `--tf-arch` | **TensorFlow only:** `lstm` (default), `gru`, or `cnn_lstm` (Conv1D + LSTM). Compare on the same data to see if accuracy moves; it may not. |
| `--predictions-csv` | **Path to save test predictions** as CSV: `symbol`, `asof_date`, `close`, `actual_next_day_up`, `predicted_next_day_up`, `prob_next_day_up` (one row per test day). |
| `--list-stocks` | **With `--csv` only:** print detected **stock/price column names** in a wide portfolio file, then exit (no training). |

Training output also prints **`Stocks in CSV:`** (comma-separated) and **`Series modeled this run:`** when applicable.

### Examples

```bash
# Sklearn MLP on a full OHLCV CSV (no TensorFlow)
python run_training.py --backend sklearn --csv data/small_sample.csv --lookback 30 --epochs 80 --save models/out.joblib

# Same run, plus export test-set predictions (dates & probabilities) for charts or reports
python run_training.py --backend sklearn --csv data/small_sample.csv --lookback 30 --predictions-csv outputs/test_predictions.csv

# List columns (stocks) in a wide portfolio CSV, then train one column
python run_training.py --csv path/to/portfolio.csv --list-stocks
python run_training.py --backend sklearn --csv path/to/portfolio.csv --price-column NFLX

# LSTM on NIFTY from Yahoo (needs TensorFlow)
python run_training.py --backend tensorflow --ticker "^NSEI" --start 2015-01-01 --save models/lstm.keras

# Same, but GRU or CNN→LSTM (common in papers)
python run_training.py --backend tensorflow --tf-arch gru --ticker "^NSEI" --start 2015-01-01 --save models/gru.keras
python run_training.py --backend tensorflow --tf-arch cnn_lstm --csv data/small_sample.csv --save models/cnn_lstm.keras

# Random forest on Indian stock symbol
python run_training.py --backend rf --ticker "RELIANCE.NS" --start 2015-01-01
```

## Helper scripts

From the project root:

```bash
# Compare several tabular backends on one dataset
python scripts/compare_algorithms.py --csv data/small_sample.csv

# Random hyperparameter search (time-series CV on train only)
python scripts/tune_hyperparams.py --csv data/small_sample.csv --model rf --n-iter 25

# Multiple Yahoo/CSV jobs (edit the job list inside the script)
python scripts/run_multiple_datasets.py

# Same CSV: LSTM vs GRU vs CNN+LSTM (TensorFlow, or PyTorch if TF unavailable)
python scripts/compare_tf_architectures.py --csv data/small_sample.csv --lookback 30 --epochs 50

# Test-set baseline metrics only (no training): majority / always-0 / always-1
python scripts/evaluate_baselines.py --csv data/small_sample.csv --lookback 30
```

## Configuration

Edit **`src/config.py`** for defaults: `LOOKBACK_DAYS`, train/val fractions, `EPOCHS`, `THRESHOLD_METRIC` (`balanced_accuracy` or `f1`), etc.

## Data expectations

- **Standard CSV:** columns include `Date`, `Open`, `High`, `Low`, `Close`, `Volume` (names are matched case-insensitively).
- **Wide portfolio CSV:** `Date` plus numeric columns; use `--price-column` to choose one series. The loader builds synthetic OHLCV (volume is constant 1.0).

## Notes

- Reported metrics include **standard accuracy** and **balanced accuracy**; for imbalanced days, balanced accuracy is often more informative.
- Financial daily direction is hard; test metrics near 50% are common for honest out-of-sample setups.
