"""
run_dl_comparison.py
--------------------
Compares five deep sequence models on the SAME corrected leave-one-run-out protocol:

    LSTM (Review 2), GRU, 1D-CNN, Transformer, LSTM-Autoencoder (unsupervised)

Each model is trained with several seeds. With --holdout all (default for real data) every
run is held out in turn (3-fold leave-one-run-out) and results are pooled over folds and seeds.

Examples
    python run_dl_comparison.py                                   # synthetic data, hold out run_3
    python run_dl_comparison.py --data-dir /path/to/bearing-dataset   # REAL NASA IMS, all 3 folds
    python run_dl_comparison.py --data-dir ... --holdout run_1 --seeds 5 --epochs 20
Outputs (./results by default): comparison_table.csv, per_fold_table.csv, comparison_results.json, *.png
"""

import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from feature_engineering import FEATURE_COLUMNS
from synthetic_data import generate_multiple_runs
from labeling import zscore_health_indicator_labels
from evaluation import evaluate_predictions
from sequence_features import make_sequences_multi_run
from lstm_model import DegradationLSTM, train_lstm, predict_lstm, standardize
from dl_models import (GRUClassifier, CNN1DClassifier, TransformerClassifier, LSTMAutoencoder,
                       train_classifier, predict_classifier_proba, train_autoencoder,
                       reconstruction_error, error_to_probability, count_parameters)

WINDOW_SIZE = 20
ADAPT_LO, ADAPT_HI = 0.20, 0.70      # FeedbackController's adjustable range
MODELS = ["LSTM", "GRU", "1D-CNN", "Transformer", "LSTM-Autoencoder"]
COLORS = {"LSTM": "#2563eb", "GRU": "#16a34a", "1D-CNN": "#d97706",
          "Transformer": "#9333ea", "LSTM-Autoencoder": "#dc2626"}


def get_labeled_runs(data_dir):
    if data_dir:
        from ims_loader import load_ims_runs
        runs = load_ims_runs(data_dir)
    else:
        runs = generate_multiple_runs(n_runs=3, n_files_per_run=(2156, 984, 2000),
                                      degradation_onset_fracs=(0.70, 0.55, 0.80))
    labeled = {}
    for rid, df in runs.items():
        labels, onset = zscore_health_indicator_labels(df, "rms", 0.10, 3.0, 3)
        d = df.copy(); d["label"] = labels.values; labeled[rid] = d
        b = max(20, int(len(d) * 0.10))
        print(f"  [{rid}] raw RMS: healthy-start mean {d['rms'].iloc[:b].mean():.4g}, last-10% mean {d['rms'].iloc[-b:].mean():.4g}")
        flag = "" if 0.2 < onset / len(d) < 0.98 else "   <-- check: onset very early/absent"
        print(f"  [{rid}] {len(d)} files, label onset at file {onset} ({onset/len(d):.1%}){flag}")
    return labeled


def baseline_normalize(df, baseline_frac=0.10):
    """
    Per-run normalisation against the run's OWN healthy start (first `baseline_frac` of files, the same
    baseline the labeller uses): z-score each feature, then compress with sign(z)*log1p(|z|) so the huge
    values reached during failure stay on a usable scale. Different IMS tests have different absolute
    vibration levels (sensors, load), so un-normalised models do not transfer between runs.
    """
    n = len(df); b = max(20, int(n * baseline_frac)); out = df.copy()
    for c in FEATURE_COLUMNS:
        base = df[c].iloc[:b]
        sd = max(float(base.std(ddof=0)), 1e-9)
        z = (df[c] - float(base.mean())) / sd
        out[c] = np.sign(z) * np.log1p(np.abs(z))
    return out


def sustained_first(pred, k=3):
    run = 0
    for i, p in enumerate(pred):
        run = run + 1 if p else 0
        if run >= k: return i - k + 1
    return None


def extra_metrics(y, pred, proba, file_idx):
    onset = int(np.argmax(y == 1)) if y.max() == 1 else None
    first = None
    if onset is not None:                       # same convention as run_streaming_demo.py
        f = sustained_first(pred[onset:]); first = None if f is None else f + onset
    lead = None if first is None else int(file_idx[first] - file_idx[onset])
    return {"detection_delay_files": lead, "false_alerts": int(((pred == 1) & (y == 0)).sum()),
            "borderline_frac": float(((proba > ADAPT_LO) & (proba < ADAPT_HI)).mean())}


def build(name, nf):
    return {"LSTM": lambda: DegradationLSTM(nf), "GRU": lambda: GRUClassifier(nf),
            "1D-CNN": lambda: CNN1DClassifier(nf),
            "Transformer": lambda: TransformerClassifier(nf, window_size=WINDOW_SIZE)}[name]()


def run_fold(labeled, held, seeds, epochs):
    nf = len(FEATURE_COLUMNS)
    train_runs = {k: v for k, v in labeled.items() if k != held}
    X_tr, y_tr, _ = make_sequences_multi_run(train_runs, FEATURE_COLUMNS, "label", WINDOW_SIZE)
    X_te, y_te, meta = make_sequences_multi_run({held: labeled[held]}, FEATURE_COLUMNS, "label", WINDOW_SIZE)
    X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)
    file_idx = np.array([m[1] for m in meta])
    print(f"\n=== fold: hold out {held} | train windows {X_tr.shape[0]} | test windows {X_te.shape[0]} "
          f"({int(y_te.sum())} degrading) ===")
    res = {m: [] for m in MODELS}; probas, hists = {}, {}
    for name in MODELS:
        for s in range(seeds):
            seed = 42 + s; torch.manual_seed(seed); t0 = time.time()
            if name == "LSTM-Autoencoder":
                model = LSTMAutoencoder(nf)
                hist = train_autoencoder(model, X_tr_s[y_tr == 0], epochs=max(30, 2 * epochs), seed=seed)
                train_t = time.time() - t0
                ref = reconstruction_error(model, X_tr_s[y_tr == 0])
                t1 = time.time(); proba = error_to_probability(reconstruction_error(model, X_te_s), ref); inf_t = time.time() - t1
            else:
                model = build(name, nf)
                if name == "LSTM":
                    import io, contextlib
                    with contextlib.redirect_stdout(io.StringIO()):
                        model, hist = train_lstm(X_tr_s, y_tr, nf, epochs=epochs, seed=seed)
                else:
                    hist = train_classifier(model, X_tr_s, y_tr, epochs=epochs, seed=seed)
                train_t = time.time() - t0
                t1 = time.time()
                proba = predict_lstm(model, X_te_s) if name == "LSTM" else predict_classifier_proba(model, X_te_s)
                inf_t = time.time() - t1
            pred = (proba >= 0.5).astype(int)
            m = {**evaluate_predictions(y_te, pred, proba), **extra_metrics(y_te, pred, proba, file_idx),
                 "train_time_s": train_t, "params": count_parameters(model), "fold": held, "seed": seed}
            res[name].append(m)
            if s == 0: probas[name] = proba; hists[name] = hist
            print(f"  {name:17s} seed {seed}: F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"PR-AUC={m['pr_auc']:.4f} delay={m['detection_delay_files']} ({train_t:.1f}s)")
    return res, probas, hists, y_te, file_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="NASA IMS folder; omit for synthetic data")
    ap.add_argument("--holdout", default=None, help="run_1/run_2/run_3 or 'all' (default: all for real data, run_3 for synthetic)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="results")
    ap.add_argument("--normalize", choices=["auto", "none", "baseline"], default="auto",
                    help="per-run baseline normalisation; auto = on for real data (--data-dir), off for synthetic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("Loading data:", a.data_dir or "SYNTHETIC")
    labeled = get_labeled_runs(a.data_dir)
    if a.normalize == "baseline" or (a.normalize == "auto" and a.data_dir):
        print("Applying per-run baseline normalisation (first 10% of each run)")
        labeled = {k: baseline_normalize(v) for k, v in labeled.items()}
    hold = a.holdout or ("all" if a.data_dir else "run_3")
    folds = list(labeled) if hold == "all" else [hold]

    allres = {m: [] for m in MODELS}; fold_plots = {}
    for h in folds:
        res, probas, hists, y_te, fidx = run_fold(labeled, h, a.seeds, a.epochs)
        for m in MODELS: allres[m] += res[m]
        fold_plots[h] = (probas, hists, y_te, fidx)

    keys = ["accuracy", "precision", "recall", "f1", "pr_auc", "balanced_accuracy", "false_alerts",
            "detection_delay_files", "borderline_frac", "train_time_s", "params"]
    def agg(runs, k):
        v = [r[k] for r in runs if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None
    summary = {m: {k: agg(allres[m], k) for k in keys} for m in MODELS}
    json.dump({"folds": folds, "seeds": a.seeds, "data": a.data_dir or "synthetic", "summary": summary,
               "per_run": {m: [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in allres[m]] for m in MODELS}},
              open(f"{a.out}/comparison_results.json", "w"), indent=2, default=float)

    def fmt(m, k, d=4):
        v = summary[m][k]
        if v is None: return "-"
        return f"{v['mean']:.{d}f}±{v['std']:.{d}f}" if v["std"] > 0 else f"{v['mean']:.{d}f}"
    table = pd.DataFrame([{"Model": m, "Accuracy": fmt(m, "accuracy"), "Precision": fmt(m, "precision"),
                           "Recall": fmt(m, "recall"), "F1": fmt(m, "f1"), "PR-AUC": fmt(m, "pr_auc"),
                           "False alerts": fmt(m, "false_alerts", 1), "Detection delay (files)": fmt(m, "detection_delay_files", 1),
                           "Scores in 0.2-0.7": f"{summary[m]['borderline_frac']['mean']*100:.1f}%",
                           "Params": fmt(m, "params", 0), "Train time (s)": fmt(m, "train_time_s", 1)} for m in MODELS])
    table.to_csv(f"{a.out}/comparison_table.csv", index=False)
    rows = []
    for m in MODELS:
        for h in folds:
            rr = [r for r in allres[m] if r["fold"] == h]
            rows.append({"Model": m, "Held-out run": h, **{k: np.mean([r[k] for r in rr if r[k] is not None] or [np.nan])
                         for k in ("precision", "recall", "f1", "pr_auc", "false_alerts", "detection_delay_files")}})
    pd.DataFrame(rows).round(4).to_csv(f"{a.out}/per_fold_table.csv", index=False)
    print("\n" + table.to_string(index=False))
    make_plots(a.out, summary, fold_plots, folds, allres)
    print(f"\nSaved results + graphs to ./{a.out}/")


def make_plots(out, summary, fold_plots, folds, allres):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = MODELS

    mets = ["precision", "recall", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(10, 5)); w = 0.8 / len(names)
    for i, n in enumerate(names):
        mu = [summary[n][m]["mean"] for m in mets]; sd = [summary[n][m]["std"] for m in mets]
        ax.bar(np.arange(len(mets)) + i * w, mu, w, yerr=sd, label=n, color=COLORS[n], capsize=2)
    ax.set_xticks(np.arange(len(mets)) + w * (len(names) - 1) / 2); ax.set_xticklabels(["Precision", "Recall", "F1", "PR-AUC"])
    lo = min(summary[n][m]["mean"] - summary[n][m]["std"] for n in names for m in mets)
    lo = max(0, np.floor((lo - 0.01) * 20) / 20); ax.set_ylim(lo, 1.005)
    ax.set_ylabel(f"score (axis starts at {lo:.2f})")
    ax.set_title(f"Detection performance by model (mean ± std; folds: {', '.join(folds)})")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(f"{out}/01_metrics_comparison.png", dpi=160); plt.close(fig)

    for h, (probas, hists, y_te, fidx) in fold_plots.items():
        tag = h if len(fold_plots) > 1 else ""
        sfx = f"_{h}" if tag else ""
        fig, ax = plt.subplots(figsize=(6, 5))
        for n in names:
            p, r, _ = precision_recall_curve(y_te, probas[n]); ax.plot(r, p, label=n, color=COLORS[n], lw=1.8)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Precision–Recall curves (held-out {h})")
        ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/02_pr_curves{sfx}.png", dpi=160); plt.close(fig)

        fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.6 * len(names)), sharex=True)
        onset = fidx[int(np.argmax(y_te == 1))] if y_te.max() == 1 else None
        for ax, n in zip(axes, names):
            ax.plot(fidx, probas[n], color=COLORS[n], lw=1)
            ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.axhspan(ADAPT_LO, ADAPT_HI, color="#fde68a", alpha=0.3)
            if onset is not None: ax.axvline(onset, color="k", ls="--", lw=0.8)
            ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=8); ax.set_ylim(-0.05, 1.05)
        axes[-1].set_xlabel("file index (dashed = labelled degradation onset; shaded = adaptive-threshold range 0.2–0.7)")
        fig.suptitle(f"Degradation score over held-out {h}", y=0.995)
        fig.tight_layout(); fig.savefig(f"{out}/03_probability_timeline{sfx}.png", dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for n, hs in hists.items():
            ax.plot(range(1, len(hs) + 1), hs, label=n + (" (MSE)" if "Auto" in n else " (BCE)"), color=COLORS[n])
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Training loss (seed 0, fold {h})")
        fig.tight_layout(); fig.savefig(f"{out}/04_training_loss{sfx}.png", dpi=160); plt.close(fig)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.barh(names, [summary[n]["borderline_frac"]["mean"] * 100 for n in names], color=[COLORS[n] for n in names])
    a.set_xlabel("% of scores in 0.2–0.7"); a.set_title("Borderline predictions\n(room for adaptive threshold to help)")
    d = [summary[n]["detection_delay_files"]["mean"] if summary[n]["detection_delay_files"] else np.nan for n in names]
    b.barh(names, d, color=[COLORS[n] for n in names]); b.axvline(0, color="k", lw=0.8)
    b.set_xlabel("files from labelled onset to first sustained alert"); b.set_title("Detection delay"); b.set_yticklabels([])
    fig.tight_layout(); fig.savefig(f"{out}/05_borderline_and_delay.png", dpi=160); plt.close(fig)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.bar(names, [summary[n]["params"]["mean"] for n in names], color=[COLORS[n] for n in names]); a.set_ylabel("trainable parameters")
    b.bar(names, [summary[n]["train_time_s"]["mean"] for n in names], color=[COLORS[n] for n in names]); b.set_ylabel("train time (s, CPU)")
    for ax in (a, b): ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(f"{out}/06_model_cost.png", dpi=160); plt.close(fig)

    if len(folds) > 1:        # per-fold F1, to show consistency across the three IMS test sets
        fig, ax = plt.subplots(figsize=(8, 4)); w = 0.8 / len(names)
        for i, n in enumerate(names):
            v = [np.mean([r["f1"] for r in allres[n] if r["fold"] == h]) for h in folds]
            ax.bar(np.arange(len(folds)) + i * w, v, w, label=n, color=COLORS[n])
        ax.set_xticks(np.arange(len(folds)) + w * (len(names) - 1) / 2); ax.set_xticklabels([f"held-out {h}" for h in folds])
        ax.set_ylabel("F1"); ax.set_title("F1 per held-out run"); ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout(); fig.savefig(f"{out}/07_f1_per_fold.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
