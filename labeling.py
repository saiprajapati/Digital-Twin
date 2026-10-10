"""
labeling.py
-----------
old_fixed_percentage_labels(): Review 1's flawed rule, kept for comparison.
zscore_health_indicator_labels(): the Review 1 fix -- per-run, data-driven
degradation boundary based on a Z-score threshold on a health indicator
(RMS by default).
"""

import numpy as np
import pandas as pd


def old_fixed_percentage_labels(df: pd.DataFrame, degrading_frac: float = 0.30) -> pd.Series:
    n = len(df)
    boundary_idx = int(np.floor(n * (1 - degrading_frac)))
    labels = np.zeros(n, dtype=int)
    labels[boundary_idx:] = 1
    return pd.Series(labels, index=df.index, name="label_old")


def zscore_health_indicator_labels(
    df: pd.DataFrame,
    indicator_col: str = "rms",
    baseline_frac: float = 0.10,
    z_threshold: float = 3.0,
    min_consecutive: int = 3,
):
    n = len(df)
    baseline_n = max(5, int(np.floor(n * baseline_frac)))
    baseline_values = df[indicator_col].iloc[:baseline_n]

    baseline_mean = baseline_values.mean()
    baseline_std = baseline_values.std(ddof=0)
    if baseline_std == 0 or np.isnan(baseline_std):
        baseline_std = 1e-9

    z_scores = (df[indicator_col].values - baseline_mean) / baseline_std
    above = z_scores > z_threshold
    boundary_idx = n

    run_length = 0
    for i, flag in enumerate(above):
        run_length = run_length + 1 if flag else 0
        if run_length >= min_consecutive:
            boundary_idx = i - min_consecutive + 1
            break

    labels = np.zeros(n, dtype=int)
    labels[boundary_idx:] = 1
    return pd.Series(labels, index=df.index, name="label_new"), boundary_idx
