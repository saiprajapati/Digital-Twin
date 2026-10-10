"""
evaluation.py
-------------
Splitting strategies and the full metric set, carried forward from the
Review 1 fix. leave_one_run_out_split() is the validated correct approach.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score, confusion_matrix,
)


def old_last_n_percent_split(n_rows: int, test_frac: float = 0.20):
    split_at = int(np.floor(n_rows * (1 - test_frac)))
    return np.arange(0, split_at), np.arange(split_at, n_rows)


def leave_one_run_out_split(run_ids: list, held_out_run: str):
    train_run_ids = [r for r in run_ids if r != held_out_run]
    return train_run_ids, [held_out_run]


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    result["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    if len(np.unique(y_true)) > 1:
        result["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        result["pr_auc"] = float("nan")

    result["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    result["n_test"] = len(y_true)
    result["n_test_normal"] = int((y_true == 0).sum())
    result["n_test_degrading"] = int((y_true == 1).sum())
    return result


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"Test set: {metrics['n_test']} rows "
          f"({metrics['n_test_normal']} normal, {metrics['n_test_degrading']} degrading)")
    for key in ["accuracy", "precision", "recall", "f1",
                "specificity", "false_positive_rate", "balanced_accuracy", "pr_auc"]:
        val = metrics[key]
        val_str = f"{val:.4f}" if not (isinstance(val, float) and np.isnan(val)) else "N/A"
        print(f"  {key:22s}: {val_str}")
