"""
run_streaming_demo.py
-----------------------
Runs the full Review 2 streaming stack LIVE against a real local MQTT
broker:

  StreamSimulator --(MQTT, with delay+dropout)--> IngestionService
      --(gap handling, writes sensor_readings)--> SQLite
      --(sliding windows, skipping uncertain gaps)--> LSTM inference
      --(degradation probability stream)--> FeedbackController
      (run twice: FIXED threshold vs ADAPTIVE threshold)
      --(alerts, threshold log)--> SQLite (for the dashboard)

This is not a simulation of a simulation -- the MQTT publish/subscribe,
the dropped messages, and the gap handling all actually happen over a
real broker running in this environment.
"""

import os
import json
import time
import threading
import sqlite3
import numpy as np
import pandas as pd
import torch

from feature_engineering import FEATURE_COLUMNS
from lstm_model import DegradationLSTM
from sequence_features import make_sequences
from stream_simulator import StreamSimulator
from ingestion_service import IngestionService
from feedback_controller import FeedbackController

DB_PATH = "/home/claude/review2/digital_twin.db"
WINDOW_SIZE = 20
TOPIC = "sensors/bearing/run_3"


def load_model():
    model = DegradationLSTM(n_features=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load("/home/claude/review2/lstm_model_state.pt"))
    model.eval()
    norm = np.load("/home/claude/review2/lstm_norm_stats.npz")
    return model, norm["mean"], norm["std"]


def run_live_stream():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    raw_df = pd.read_csv("/home/claude/review2/held_out_run_raw.csv")
    print(f"Replaying held-out run: {len(raw_df)} files, topic={TOPIC}")

    ingestion = IngestionService(DB_PATH, FEATURE_COLUMNS, topic=TOPIC)
    ingestion_thread = threading.Thread(target=ingestion.run, kwargs={"timeout": 180})
    ingestion_thread.start()
    time.sleep(0.5)  # let the subscriber connect before we start publishing

    simulator = StreamSimulator(topic=TOPIC, delay_seconds=0.01, dropout_prob=0.04)
    simulator.publish_run(raw_df, FEATURE_COLUMNS)

    ingestion_thread.join()
    return raw_df


def reconstruct_and_infer():
    """
    Reads back what was ACTUALLY received (including interpolated short
    gaps and NULLed long gaps) and runs the LSTM only on windows that
    contain no sensor_uncertain readings -- it would be dishonest to
    infer across a gap the system has explicitly flagged as unobserved.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sensor_readings ORDER BY file_index", conn)
    conn.close()

    model, mean, std = load_model()

    probs = {}
    skipped_windows = 0
    values = df[FEATURE_COLUMNS].values.astype(float)
    uncertain = df["sensor_uncertain"].values
    file_indices = df["file_index"].values

    for end in range(WINDOW_SIZE - 1, len(df)):
        start = end - WINDOW_SIZE + 1
        if uncertain[start:end + 1].any():
            skipped_windows += 1
            continue
        window = values[start:end + 1]
        window_std = (window - mean) / std
        x = torch.tensor(window_std, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        probs[int(file_indices[end])] = prob

    print(f"[inference] computed {len(probs)} predictions, skipped {skipped_windows} "
          f"windows that overlapped a flagged gap")
    return probs, df


def run_feedback_comparison(probs: dict, raw_df: pd.DataFrame):
    file_indices_sorted = sorted(probs.keys())
    true_labels = dict(zip(raw_df["file_index"], raw_df["label"]))

    fixed_ctrl = FeedbackController(
        DB_PATH, base_threshold=0.5, adaptive=False,
        predictions_table="predictions_fixed", adjustments_table="threshold_adjustments_fixed",
    )
    adaptive_ctrl = FeedbackController(
        DB_PATH, base_threshold=0.5, adaptive=True,
        predictions_table="predictions_adaptive", adjustments_table="threshold_adjustments_adaptive",
    )

    fixed_results, adaptive_results = [], []
    for fi in file_indices_sorted:
        p = probs[fi]
        fixed_results.append(fixed_ctrl.process(fi, p))
        adaptive_results.append(adaptive_ctrl.process(fi, p))

    def score(results):
        false_alerts, missed, lead_times = 0, 0, []
        first_true_degrading = None
        first_alert_on_true_run = None
        for r in results:
            true_label = true_labels.get(r["file_index"], 0)
            if true_label == 1 and first_true_degrading is None:
                first_true_degrading = r["file_index"]
            if r["alert_fired"] and true_label == 0:
                false_alerts += 1
            if r["alert_fired"] and true_label == 1 and first_alert_on_true_run is None:
                first_alert_on_true_run = r["file_index"]
        n_degrading_points = sum(1 for r in results if true_labels.get(r["file_index"], 0) == 1)
        n_caught = sum(1 for r in results if r["alert_fired"] and true_labels.get(r["file_index"], 0) == 1)
        false_alert_rate = false_alerts / max(1, sum(1 for r in results if true_labels.get(r["file_index"], 0) == 0))
        lead_time_files = None
        if first_true_degrading is not None and first_alert_on_true_run is not None:
            lead_time_files = first_alert_on_true_run - first_true_degrading
        return {
            "false_alerts": false_alerts,
            "false_alert_rate": false_alert_rate,
            "degrading_points_caught": n_caught,
            "degrading_points_total": n_degrading_points,
            "recall": n_caught / max(1, n_degrading_points),
            "first_true_degrading_file": first_true_degrading,
            "first_alert_file": first_alert_on_true_run,
            "lead_time_files": lead_time_files,
        }

    fixed_score = score(fixed_results)
    adaptive_score = score(adaptive_results)

    conn = sqlite3.connect(DB_PATH)
    n_adjustments = conn.execute("SELECT COUNT(*) FROM threshold_adjustments_adaptive").fetchone()[0]
    conn.close()

    return fixed_score, adaptive_score, n_adjustments


def main():
    print("=" * 72)
    print("LIVE STREAMING DEMO -- real MQTT broker, real dropout, real gap handling")
    print("=" * 72)
    raw_df = run_live_stream()

    print("\n" + "=" * 72)
    print("Reconstructing stream from what was actually received + running LSTM inference")
    print("=" * 72)
    probs, received_df = reconstruct_and_infer()

    n_total = len(raw_df)
    n_received = received_df["sensor_uncertain"].eq(0).sum()
    n_interpolated = received_df["interpolated"].sum()
    n_uncertain = received_df["sensor_uncertain"].sum()
    print(f"\nStream integrity: {n_total} files sent by simulator's source data, "
          f"{len(received_df)} rows reconstructed at ingestion "
          f"({n_interpolated} interpolated, {n_uncertain} flagged sensor_uncertain)")

    print("\n" + "=" * 72)
    print("FIXED vs ADAPTIVE feedback loop comparison (same probability stream)")
    print("=" * 72)
    fixed_score, adaptive_score, n_adjustments = run_feedback_comparison(probs, raw_df)

    print(f"\nNumber of threshold adjustments made by the adaptive controller: {n_adjustments}")
    print(f"\n{'metric':30s} {'FIXED (0.5)':>14s} {'ADAPTIVE':>14s}")
    print("-" * 60)
    for key in ["false_alerts", "false_alert_rate", "degrading_points_caught",
                "degrading_points_total", "recall", "lead_time_files"]:
        fv, av = fixed_score[key], adaptive_score[key]
        fv_s = f"{fv:.4f}" if isinstance(fv, float) else str(fv)
        av_s = f"{av:.4f}" if isinstance(av, float) else str(av)
        print(f"{key:30s} {fv_s:>14s} {av_s:>14s}")

    with open("/home/claude/review2/streaming_demo_results.json", "w") as f:
        json.dump({
            "n_total_files": int(n_total),
            "n_received_rows": int(len(received_df)),
            "n_interpolated": int(n_interpolated),
            "n_flagged_uncertain": int(n_uncertain),
            "n_threshold_adjustments": int(n_adjustments),
            "fixed": fixed_score,
            "adaptive": adaptive_score,
        }, f, indent=2)
    print("\nSaved: streaming_demo_results.json, digital_twin.db")


if __name__ == "__main__":
    main()
