"""
run_lstm_experiment.py
------------------------
Trains the Review 2 LSTM sequence model using the SAME corrected data
preparation validated in the Review 1 fix (z-score labeling, leave-one-
run-out split), (Gradient Boosting has been removed from the comparison; see run_dl_comparison.py
for the LSTM vs GRU / 1D-CNN / Transformer / LSTM-Autoencoder comparison.)
"""

import json
import numpy as np

from feature_engineering import FEATURE_COLUMNS
from synthetic_data import generate_multiple_runs
from labeling import zscore_health_indicator_labels
from evaluation import evaluate_predictions, print_metrics
from sequence_features import make_sequences_multi_run
from lstm_model import train_lstm, predict_lstm, standardize

WINDOW_SIZE = 20
HELD_OUT_RUN = "run_3"


def main():
    print("=" * 72)
    print("Loading synthetic data (3 runs, same generator as Review 1 fix)")
    print("=" * 72)
    runs = generate_multiple_runs(
        n_runs=3, n_files_per_run=(2156, 984, 2000),
        degradation_onset_fracs=(0.70, 0.55, 0.80),
    )

    labeled_runs = {}
    for run_id, df in runs.items():
        labels, boundary_idx = zscore_health_indicator_labels(
            df, indicator_col="rms", baseline_frac=0.10, z_threshold=3.0, min_consecutive=3
        )
        df2 = df.copy()
        df2["label"] = labels.values
        labeled_runs[run_id] = df2
        print(f"  [{run_id}] onset at file {boundary_idx} of {len(df2)} ({boundary_idx/len(df2):.1%})")

    train_runs = {rid: df for rid, df in labeled_runs.items() if rid != HELD_OUT_RUN}
    test_runs = {HELD_OUT_RUN: labeled_runs[HELD_OUT_RUN]}

    # ------------------------------------------------------------------
    # LSTM sequence model
    # ------------------------------------------------------------------
    print(f"\nBuilding sliding-window sequences (window_size={WINDOW_SIZE}) ...")
    X_train, y_train, _ = make_sequences_multi_run(train_runs, FEATURE_COLUMNS, "label", WINDOW_SIZE)
    X_test, y_test, test_meta = make_sequences_multi_run(test_runs, FEATURE_COLUMNS, "label", WINDOW_SIZE)
    print(f"  train sequences: {X_train.shape}, test sequences: {X_test.shape}")

    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)

    print("\nTraining LSTM ...")
    model, history = train_lstm(X_train_s, y_train, n_features=len(FEATURE_COLUMNS), epochs=15)

    lstm_proba = predict_lstm(model, X_test_s)
    lstm_pred = (lstm_proba >= 0.5).astype(int)
    lstm_metrics = evaluate_predictions(y_test, lstm_pred, lstm_proba)
    print_metrics("LSTM sequence model", lstm_metrics)

    # ------------------------------------------------------------------
    # Save everything needed for the feedback-loop demo and the report
    # ------------------------------------------------------------------
    out = {
        "window_size": WINDOW_SIZE,
        "held_out_run": HELD_OUT_RUN,
        "lstm_metrics": lstm_metrics,
        "lstm_train_loss_history": history,
    }
    with open("lstm_experiment_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Save per-file LSTM probabilities for the held-out run, aligned to
    # file index, so the FeedbackController demo can replay them in order
    np.save("lstm_test_proba.npy", lstm_proba)
    np.save("lstm_test_labels.npy", y_test)
    with open("lstm_test_meta.json", "w") as f:
        json.dump(test_meta, f)

    # Save the trained model + normalization stats so the live streaming
    # demo (run_streaming_demo.py) can run real inference on incoming
    # windows, instead of replaying pre-computed probabilities.
    import torch
    torch.save(model.state_dict(), "lstm_model_state.pt")
    np.savez("lstm_norm_stats.npz", mean=mean, std=std)

    # Also save the raw held-out run's feature data (unlabeled, as the
    # StreamSimulator would see it) so the streaming demo has something
    # realistic to replay file-by-file.
    test_runs[HELD_OUT_RUN][FEATURE_COLUMNS + ["file_index", "label"]].to_csv(
        "held_out_run_raw.csv", index=False
    )

    print_metrics("LSTM final", lstm_metrics)


if __name__ == "__main__":
    main()
