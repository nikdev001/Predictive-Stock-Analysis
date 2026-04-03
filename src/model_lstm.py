"""TensorFlow sequence classifiers: stacked LSTM, stacked GRU, CNN+LSTM (common in papers)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _compile_binary_head(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def build_lstm_model(
    lookback: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.Masking()(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, out)
    _compile_binary_head(model, learning_rate)
    return model


def build_gru_model(
    lookback: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Often used as an LSTM alternative; fewer parameters, similar behavior."""
    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.Masking()(inputs)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GRU(32, return_sequences=False)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, out)
    _compile_binary_head(model, learning_rate)
    return model


def build_cnn_lstm_model(
    lookback: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Local pattern extraction (Conv1D) then recurrent aggregation — common hybrid in literature."""
    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.Masking()(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, out)
    _compile_binary_head(model, learning_rate)
    return model


def build_tf_sequence_model(
    architecture: str,
    lookback: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    arch = architecture.lower().strip()
    if arch in ("lstm", "stacked_lstm"):
        return build_lstm_model(lookback, n_features, learning_rate)
    if arch in ("gru", "stacked_gru"):
        return build_gru_model(lookback, n_features, learning_rate)
    if arch in ("cnn_lstm", "cnn-lstm", "cnnlstm"):
        return build_cnn_lstm_model(lookback, n_features, learning_rate)
    raise ValueError(f"Unknown TensorFlow architecture: {architecture!r}. Use lstm, gru, or cnn_lstm.")


def set_seed(seed: int) -> None:
    tf.random.set_seed(seed)
