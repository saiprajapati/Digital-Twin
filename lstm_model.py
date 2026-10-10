"""
lstm_model.py
-------------
The Review 2 upgrade from single-snapshot Gradient Boosting to a sequence
model. Takes a window of the last N files' feature vectors and predicts
the condition (normal/degrading) at the end of that window, so the model
can use the TREND leading up to the current reading, not just the reading
itself.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DegradationLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) -- final timestep's hidden state
        logit = self.head(last_hidden).squeeze(-1)
        return logit


def standardize(X_train, X_test):
    """Per-feature standardization, fit on train only (no test leakage)."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1e-9
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_test_s, mean, std


def train_lstm(X_train, y_train, n_features, epochs=15, batch_size=64, lr=1e-3, device="cpu", seed=42):
    torch.manual_seed(seed)
    model = DegradationLSTM(n_features=n_features).to(device)

    # class-balanced loss, since training data is normal-heavy (same imbalance
    # Review 1 noted for Gradient Boosting)
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(loader.dataset)
        history.append(avg_loss)
        print(f"  epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}")
    return model, history


def predict_lstm(model, X_test, device="cpu", batch_size=256):
    model.eval()
    loader = DataLoader(SequenceDataset(X_test, np.zeros(len(X_test))), batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)
