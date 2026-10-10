"""
dl_models.py
------------
Four additional deep-learning sequence models for the Review 2 comparison,
all consuming the SAME input as the existing LSTM: windows of shape
(batch, window_size=20, n_features=9).

  1. GRUClassifier          - gated recurrent unit (lighter recurrent alternative to LSTM)
  2. CNN1DClassifier        - 1D temporal convolutions (non-recurrent, local-trend detector)
  3. TransformerClassifier  - small self-attention encoder over the window
  4. LSTMAutoencoder        - UNSUPERVISED: trained on healthy windows only,
                              degradation = high reconstruction error

The three classifiers expose a logit, trained with the same class-weighted
BCE as lstm_model.train_lstm. The autoencoder is trained differently (no labels).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------
# 1. GRU
# ----------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


# ----------------------------------------------------------------------------
# 2. 1D-CNN
# ----------------------------------------------------------------------------
class CNN1DClassifier(nn.Module):
    def __init__(self, n_features, channels=32, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * channels, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):                       # x: (B, T, F)
        z = self.conv(x.transpose(1, 2))        # (B, C, T)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)   # avg + max pooling over time
        return self.head(pooled).squeeze(-1)


# ----------------------------------------------------------------------------
# 3. Transformer encoder
# ----------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=2, window_size=20, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.encoder(self.proj(x) + self.pos[:, : x.size(1)])
        return self.head(z.mean(dim=1)).squeeze(-1)


# ----------------------------------------------------------------------------
# 4. LSTM autoencoder (unsupervised anomaly detector)
# ----------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)                         # bottleneck = final hidden state
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)      # repeat across time
        dec, _ = self.decoder(z)
        return self.out(dec)


# ----------------------------------------------------------------------------
# Training / inference helpers
# ----------------------------------------------------------------------------
def train_classifier(model, X_train, y_train, epochs=15, batch_size=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    n_pos = max(1, int(y_train.sum()))
    pos_weight = torch.tensor([max(1, len(y_train) - n_pos) / n_pos], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(ds))
    return history


def predict_classifier_proba(model, X, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def train_autoencoder(model, X_normal, epochs=30, batch_size=64, lr=2e-3, seed=42):
    """Trains ONLY on healthy windows; no labels used."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(torch.tensor(X_normal, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(X_normal))
    return history


def reconstruction_error(model, X, batch_size=512):
    """Per-window mean squared reconstruction error."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).numpy())
    return np.concatenate(out)


def error_to_probability(err, ref_errors, quantile=0.99, steepness=3.0):
    """
    Maps reconstruction error to a 0-1 'degradation score' so the autoencoder
    plugs into the same threshold / FeedbackController machinery as the
    classifiers. The score is exactly 0.5 when the error equals the
    `quantile` of healthy TRAINING errors (i.e. ~1% false-alarm level), and
    changes smoothly (logistic in log-error) on either side, so it is NOT
    saturated at 0/1 -> gives the adaptive threshold borderline values to act on.
    """
    thr = np.quantile(ref_errors, quantile)
    z = steepness * (np.log(err + 1e-12) - np.log(thr + 1e-12))
    return 1.0 / (1.0 + np.exp(-z))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
