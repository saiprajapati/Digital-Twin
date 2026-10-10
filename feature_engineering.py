"""
feature_engineering.py
-----------------------
Computes the same 9 features used in the Review 1 baseline, for a single
vibration snapshot (one channel, one file). Kept unchanged from Review 1
so results stay comparable across reviews.
"""

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


def compute_features(signal: np.ndarray, sampling_rate_hz: float = 20000.0) -> dict:
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    max_val = float(np.max(signal))
    min_val = float(np.min(signal))
    peak_to_peak = float(max_val - min_val)
    kurt_val = float(scipy_kurtosis(signal, fisher=True, bias=False))

    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)

    if len(fft_mag) > 1:
        dom_idx = int(np.argmax(fft_mag[1:]) + 1)
        dominant_frequency_hz = float(freqs[dom_idx])
    else:
        dominant_frequency_hz = 0.0

    spectral_energy = float(np.sum(fft_mag ** 2))

    return {
        "mean": mean_val, "std": std_val, "rms": rms_val,
        "max": max_val, "min": min_val, "peak_to_peak": peak_to_peak,
        "kurtosis": kurt_val, "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_energy": spectral_energy,
    }


FEATURE_COLUMNS = [
    "mean", "std", "rms", "max", "min",
    "peak_to_peak", "kurtosis", "dominant_frequency_hz", "spectral_energy",
]
