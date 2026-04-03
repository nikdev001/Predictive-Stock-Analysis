"""PyTorch LSTM / GRU / CNN+LSTM — same task as model_lstm.py when TensorFlow is unavailable."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .eval_utils import apply_threshold, find_best_threshold
from .train import _baselines, scale_sequence_data


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class LSTMNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.l1 = nn.LSTM(n_features, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 32, batch_first=True)
        self.do = nn.Dropout(0.25)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.l1(x)
        x = self.do(x)
        x, _ = self.l2(x)
        x = x[:, -1, :]
        x = self.do(x)
        return self.fc(x).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.l1 = nn.GRU(n_features, 64, batch_first=True)
        self.l2 = nn.GRU(64, 32, batch_first=True)
        self.do = nn.Dropout(0.25)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.l1(x)
        x = self.do(x)
        x, _ = self.l2(x)
        x = x[:, -1, :]
        x = self.do(x)
        return self.fc(x).squeeze(-1)


class CNNLSTMNet(nn.Module):
    def __init__(self, n_features: int, lookback: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.do = nn.Dropout(0.25)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.do(x)
        return self.fc(x).squeeze(-1)


def _build_net(arch: str, n_features: int, lookback: int) -> nn.Module:
    a = arch.lower()
    if a == "lstm":
        return LSTMNet(n_features)
    if a == "gru":
        return GRUNet(n_features)
    if a == "cnn_lstm":
        return CNNLSTMNet(n_features, lookback)
    raise ValueError(arch)


@torch.no_grad()
def _proba(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    t = torch.from_numpy(X).float().to(device)
    logits = model(t)
    return torch.sigmoid(logits).cpu().numpy()


def run_torch_sequence(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    architecture: str,
    epochs: int,
) -> dict:
    X_train_s, X_val_s, X_test_s, _ = scale_sequence_data(X_train, X_val, X_test)
    _, lookback, nfeat = X_train_s.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(config.RANDOM_SEED)
    model = _build_net(architecture, nfeat, lookback).to(device)

    Xt = torch.from_numpy(X_train_s).float()
    yt = torch.from_numpy(y_train).float()
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    crit = nn.BCEWithLogitsLoss()

    X_val_t = torch.from_numpy(X_val_s).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_state = None
    best_val = float("inf")
    patience = config.PATIENCE
    bad = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            lv = model(X_val_t)
            vloss = crit(lv, y_val_t).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    proba_val = _proba(model, X_val_s, device)
    thr = find_best_threshold(y_val, proba_val, metric=config.THRESHOLD_METRIC)
    proba = _proba(model, X_test_s, device)
    y_pred = apply_threshold(proba, thr)
    y_true = y_test.astype(int)

    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

    bal = balanced_accuracy_score(y_true, y_pred)
    return {
        "balanced_accuracy": float(bal),
        "baselines": _baselines(y_test),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": thr,
        "tf_architecture": architecture,
        "backend": "pytorch",
    }
