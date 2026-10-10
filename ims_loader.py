"""
ims_loader.py
-------------
Loads the NASA IMS bearing run-to-failure data (Kaggle: "vinayak123tyagi/bearing-dataset")
and turns every 1-second vibration file into the SAME 9 engineered features used throughout
the project (feature_engineering.compute_features).

Expected layout (any nesting depth works; folders are found automatically):
    <data_dir>/.../1st_test/2003.10.22.12.06.24   (tab/space separated, 20480 rows x 8 columns)
    <data_dir>/.../2nd_test/2004.02.12.10.32.39   (20480 rows x 4 columns)
    <data_dir>/.../3rd_test/ (or 4th_test/txt)    (20480 rows x 4 columns)

Each file name is a timestamp; sorting by name = chronological order. A folder with at
least `min_files` such files is treated as one run-to-failure run.

Monitored channel (the bearing that actually failed, per the IMS readme):
    test set 1 (Oct 2003): bearing 3 -> column 4      test set 2 (Feb 2004): bearing 1 -> column 0
    test set 3 (Mar 2004): bearing 3 -> column 2
The set is recognised from the first file's date. For an unrecognised run the channel whose
RMS grows the most from start to end is used instead.

Features are cached to <cache_dir>/<run>.csv so the slow text parsing happens only once.
"""

import os, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from feature_engineering import compute_features, FEATURE_COLUMNS

NAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}(\.\w+)?$")
KNOWN_FAILED_CHANNEL = {"2003.10": 4, "2004.02": 0, "2004.03": 2}   # keyed by YYYY.MM of first file


def discover_runs(data_dir, min_files=500):
    """Return [(folder, [sorted file names])] for every folder that looks like an IMS run, oldest first."""
    found, seen = [], set()
    for root, _, files in os.walk(data_dir):
        names = sorted(f for f in files if NAME_RE.match(f))
        if len(names) >= min_files:
            key = (names[0], names[-1], len(names))
            if key in seen:            # same run unpacked twice
                continue
            seen.add(key); found.append((root, names))
    found.sort(key=lambda t: t[1][0])
    return found


def _process_file(args):
    path, fs = args
    arr = pd.read_csv(path, sep=r"\s+", header=None, engine="c").values.astype(np.float64)
    out = {}
    for c in range(arr.shape[1]):
        for k, v in compute_features(arr[:, c], sampling_rate_hz=fs).items():
            out[f"ch{c}_{k}"] = v
    return out


def _load_run_features(folder, names, fs, n_jobs, cache_path):
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    paths = [(os.path.join(folder, n), fs) for n in names]
    if n_jobs == 1:
        rows = [_process_file(a) for a in paths]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_process_file, paths, chunksize=16))
    df = pd.DataFrame(rows)
    df.insert(0, "file_index", np.arange(len(df)))
    df.insert(1, "file_name", names)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True); df.to_csv(cache_path, index=False)
    return df


def pick_channel(df, first_name):
    key = first_name[:7]
    n_ch = len({c.split("_")[0] for c in df.columns if c.startswith("ch")})
    if key in KNOWN_FAILED_CHANNEL and KNOWN_FAILED_CHANNEL[key] < n_ch:
        return KNOWN_FAILED_CHANNEL[key], "known failed bearing"
    growth = [df[f"ch{c}_rms"].iloc[-max(5, len(df)//20):].mean() / max(df[f"ch{c}_rms"].iloc[:max(5, len(df)//20)].mean(), 1e-12)
              for c in range(n_ch)]
    return int(np.argmax(growth)), "auto (largest RMS growth)"


def load_ims_runs(data_dir, fs=20000.0, min_files=500, cache_dir="ims_cache", n_jobs=None, verbose=True):
    """Returns {run_id: DataFrame[file_index + FEATURE_COLUMNS]} ready for the pipeline."""
    runs = discover_runs(data_dir, min_files)
    if not runs:
        sample = [os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs][:8]
        raise FileNotFoundError(
            f"No IMS run folders (>= {min_files} files named like 2003.10.22.12.06.24) found under {data_dir}.\n"
            f"First files seen: {sample}\nIf the names differ, adjust NAME_RE in ims_loader.py.")
    n_jobs = n_jobs or os.cpu_count() or 1
    out = {}
    for i, (folder, names) in enumerate(runs, 1):
        rid = f"run_{i}"
        if verbose: print(f"[{rid}] {folder}  ({len(names)} files, {names[0]} -> {names[-1]})")
        feats = _load_run_features(folder, names, fs, n_jobs, os.path.join(cache_dir, f"{rid}_{len(names)}.csv") if cache_dir else None)
        ch, how = pick_channel(feats, names[0])
        if verbose: print(f"        monitoring channel {ch} ({how})")
        d = feats[["file_index"]].copy()
        for k in FEATURE_COLUMNS:
            d[k] = feats[f"ch{ch}_{k}"].values
        d.attrs["channel"] = ch
        out[rid] = d
    return out
