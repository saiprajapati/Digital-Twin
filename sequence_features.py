"""
sequence_features.py
---------------------
Converts a chronologically-ordered, per-file feature DataFrame (as produced
by feature_engineering.py) into overlapping sliding-window sequences
suitable for an LSTM, instead of Gradient Boosting's single-snapshot rows.

Each sequence of `window_size` consecutive files becomes one training
example; the label is the condition label of the LAST file in the window
(i.e. "given the last N readings, what's the current state right now").
"""

import numpy as np
import pandas as pd


def make_sequences(df: pd.DataFrame, feature_columns: list, label_column: str, window_size: int = 20):
    """
    df must be chronologically sorted ascending already (one run's worth of data).

    Returns
    -------
    X : np.ndarray, shape (n_sequences, window_size, n_features)
    y : np.ndarray, shape (n_sequences,)
    end_indices : np.ndarray -- the row index (within df) that each sequence's
                  label corresponds to, useful for tracing predictions back
                  to the original file.
    """
    values = df[feature_columns].values
    labels = df[label_column].values
    n = len(df)

    if n < window_size:
        return (np.empty((0, window_size, len(feature_columns))),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int))

    X, y, end_indices = [], [], []
    for end in range(window_size - 1, n):
        start = end - window_size + 1
        X.append(values[start:end + 1])
        y.append(labels[end])
        end_indices.append(end)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(end_indices)


def make_sequences_multi_run(runs: dict, feature_columns: list, label_column: str, window_size: int = 20):
    """
    Same as make_sequences(), but applied independently per run and
    concatenated. Sequences are NEVER built across a run boundary, since
    that would splice together two physically unrelated timelines.
    """
    all_X, all_y, all_meta = [], [], []
    for run_id, df in runs.items():
        X, y, end_idx = make_sequences(df, feature_columns, label_column, window_size)
        all_X.append(X)
        all_y.append(y)
        all_meta.extend([(run_id, int(i)) for i in end_idx])
    X = np.concatenate(all_X, axis=0) if all_X else np.empty((0, window_size, len(feature_columns)))
    y = np.concatenate(all_y, axis=0) if all_y else np.empty((0,), dtype=int)
    return X, y, all_meta
