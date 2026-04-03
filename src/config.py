"""Hyperparameters and paths — adjust to match your proposal."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# NIFTY 50 index on Yahoo Finance
DEFAULT_TICKER = "^NSEI"

LOOKBACK_DAYS = 60
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder is test

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3

# TensorFlow only: lstm | gru | cnn_lstm (see model_lstm.build_tf_sequence_model)
DEFAULT_TF_ARCHITECTURE = "lstm"

# Training / evaluation
USE_BALANCED_SAMPLE_WEIGHT = True
THRESHOLD_METRIC = "balanced_accuracy"  # "balanced_accuracy" | "f1"

# --- Tuning ideas (edit or use scripts/tune_hyperparams.py) ---
# - LOOKBACK_DAYS: try 30 / 60 / 90 (longer memory vs more samples lost).
# - TRAIN_FRAC / VAL_FRAC: more test years = harder but more stable metrics.
# - For LSTM (TensorFlow): batch size, LSTM units, dropout, learning rate in model_lstm.py.
# - Walk-forward: retrain on rolling windows (not implemented; use multiple date ranges).
# - Target: predict 3–5 day direction or "big move" instead of noisy daily sign.
