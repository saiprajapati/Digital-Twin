"""
synthetic_data.py
------------------
Generates synthetic run-to-failure vibration data mirroring the NASA IMS
dataset's structure, so this pipeline is runnable/testable without the
real dataset. See the Review 1 Fix report for the real-data loader.
"""

import numpy as np
import pandas as pd
from feature_engineering import compute_features


def generate_synthetic_run(
    n_files: int = 2000,
    samples_per_file: int = 4096,
    sampling_rate_hz: float = 20000.0,
    degradation_onset_frac: float = 0.75,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    onset_idx = int(n_files * degradation_onset_frac)

    rows = []
    for file_idx in range(n_files):
        is_degrading = file_idx >= onset_idx
        if is_degrading:
            progress = (file_idx - onset_idx) / max(1, (n_files - onset_idx))
            amplitude_scale = 1.0 + 4.0 * progress
            impulsiveness = 1.0 + 8.0 * progress
        else:
            amplitude_scale = 1.0
            impulsiveness = 0.0

        base_signal = rng.normal(0, 1.0 * amplitude_scale, samples_per_file)
        if impulsiveness > 0:
            n_impulses = int(impulsiveness)
            impulse_positions = rng.integers(0, samples_per_file, n_impulses)
            base_signal[impulse_positions] += rng.normal(0, 5.0 * amplitude_scale, n_impulses)

        feats = compute_features(base_signal, sampling_rate_hz=sampling_rate_hz)
        feats["file_index"] = file_idx
        feats["true_degrading"] = int(is_degrading)
        rows.append(feats)

    return pd.DataFrame(rows).sort_values("file_index").reset_index(drop=True)


def generate_multiple_runs(
    n_runs: int = 3,
    n_files_per_run: tuple = (2156, 984, 2000),
    degradation_onset_fracs: tuple = (0.70, 0.55, 0.80),
    samples_per_file: int = 4096,
    sampling_rate_hz: float = 20000.0,
) -> dict:
    runs = {}
    for i in range(n_runs):
        n_files = n_files_per_run[i] if i < len(n_files_per_run) else 2000
        onset = degradation_onset_fracs[i] if i < len(degradation_onset_fracs) else 0.75
        runs[f"run_{i+1}"] = generate_synthetic_run(
            n_files=n_files, samples_per_file=samples_per_file,
            sampling_rate_hz=sampling_rate_hz, degradation_onset_frac=onset,
            seed=100 + i,
        )
    return runs
