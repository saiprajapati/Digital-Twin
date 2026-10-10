# Review 2 — Real-Time + Feedback Loop

This implements items 2–7 of the Review 2 deliverables list, on top of the
corrected Review 1 pipeline (z-score labeling, leave-one-run-out split).

**Everything below was actually run, not just written.** A real local MQTT
broker (Mosquitto) was used for the streaming components; the LSTM was
actually trained; the feedback loop comparison used a real, honest
probability stream. All numbers in this README come from those real runs.

## Files

| File | Deliverable | Status |
|---|---|---|
| `sequence_features.py` | Sliding-window builder for LSTM input | Used in training |
| `lstm_model.py` | #2 LSTM sequence model | Trained, verified |
| `run_lstm_experiment.py` | Trains LSTM on the corrected split, compares to Gradient Boosting | **Run — see results below** |
| `stream_simulator.py` | #3 Realistic StreamSimulator (delay + dropout over real MQTT) | **Run — see results below** |
| `ingestion_service.py` | #4 IngestionService with gap interpolation/flagging | **Run — see results below** |
| `feedback_controller.py` | #5 FeedbackController (the centerpiece) | **Run — see results below** |
| `run_streaming_demo.py` | #6 Full live demo + fixed vs. adaptive comparison | **Run — see results below** |
| `dashboard.py` | #7 Dashboard v2 (auto-refresh, threshold log) | **Started and confirmed serving (HTTP 200)** |
| `feature_engineering.py`, `labeling.py`, `evaluation.py`, `synthetic_data.py` | Carried forward from the Review 1 fix, unchanged | — |

## How to run it yourself

```bash
pip install torch paho-mqtt streamlit pandas numpy scipy scikit-learn
# Start a local MQTT broker (or point stream_simulator.py / ingestion_service.py at a real one)
mosquitto -d -p 1883

python run_lstm_experiment.py     # trains + saves the LSTM, needed by the next step
python run_streaming_demo.py      # full live streaming + feedback loop comparison
streamlit run dashboard.py        # view the results (reads digital_twin.db)
```

## #2 — LSTM vs. Gradient Boosting (same held-out run, same corrected split)

| Metric | Gradient Boosting | LSTM |
|---|---|---|
| Accuracy | 0.9995 | 0.9939 |
| Precision | 0.9975 | 1.0000 |
| Recall | 1.0000 | 0.9698 |
| F1 | 0.9987 | 0.9847 |
| Specificity | 0.9994 | 1.0000 |
| PR-AUC | 1.0000 | 0.9864 |

**Honest finding:** on this synthetic data, Gradient Boosting slightly
*outperforms* the LSTM. This is expected and worth saying plainly to your
supervisor rather than hiding: the synthetic generator produces a very
clean, easily-separable signal, which doesn't showcase the LSTM's actual
advantage — using sequence *trend* information — because a single snapshot
is already enough to classify almost perfectly. The real NASA IMS data,
being noisier and more gradual, is where the LSTM's sequence-awareness is
expected to matter more; this comparison should be re-run on the real data
once it's available.

## #3 + #4 — Streaming realism and gap handling (real MQTT run)

- StreamSimulator published 1,933 messages and deliberately dropped 67
  (**3.4% observed dropout**, close to the configured 4% target) while
  replaying the held-out run.
- IngestionService detected every resulting gap in the `file_index`
  sequence and **linearly interpolated 67 points** (all gaps were 1–2
  files, i.e. "short" by this project's definition of ≤3).
- Zero gaps were long enough to require the "sensor_uncertain" flag in
  this run — worth testing again with a higher dropout rate if you want to
  see that path exercised too (try `dropout_prob=0.15` or higher in
  `stream_simulator.py`).
- Final reconstructed stream: all 2,000 original files accounted for
  (interpolated or directly received), zero fabricated multi-point runs.

## #5 + #6 — FeedbackController and the fixed-vs-adaptive comparison

The adaptive controller made **7 threshold adjustments** over the run
(tightening from 0.50 down toward its floor of 0.20 as the health score
trended toward degradation, confirmed in `threshold_adjustments_adaptive`).

| Metric | FIXED (0.5) | ADAPTIVE |
|---|---|---|
| False alerts | 0 | 0 |
| False alert rate | 0.0000 | 0.0000 |
| Degrading points caught | 384 / 397 | 384 / 397 |
| Recall | 0.9673 | 0.9673 |
| Lead time (files) | 13 | 13 |

**Honest finding, worth leading with rather than hiding:** the two
strategies produced *identical* results here. Investigating why (see
`streaming_demo_results.json` and the check below) showed that only
**0.1% of the LSTM's predicted probabilities fall between 0.2 and 0.7** —
the entire adjustable range of the adaptive threshold. The model is so
confident on nearly every file (probabilities cluster near 0.0 or 1.0)
that moving the threshold within its allowed range essentially never
changes which side of the line a prediction falls on.

This is a genuinely useful finding, not a dead end: it means the feedback
loop's real value can only be demonstrated where the model is *genuinely
uncertain* — which is exactly the kind of borderline case the real NASA
IMS data is expected to produce much more of (Review 1's real baseline had
a mean predicted probability of 0.337, with meaningful mass sitting near
the 0.5 decision boundary, unlike this clean synthetic run). **The
feedback loop mechanism itself is implemented, tested, and logging
correctly — what's still needed is a dataset with enough borderline
predictions to show its benefit, which the real data should provide.**

## #7 — Dashboard v2

Started via `streamlit run dashboard.py` and confirmed serving (HTTP 200,
no exceptions in the server log). It reads live from `digital_twin.db`:
sensor trends, the twin's health score with both fixed and adaptive
threshold lines overlaid, the full timestamped threshold-adjustment log,
and recent alerts — auto-refreshing every 3 seconds rather than Review 1's
static, manually-reloaded view.

## What's still needed before this is "done" for real

1. **Re-run `run_lstm_experiment.py` and `run_streaming_demo.py` against
   the real NASA IMS data** (via the Kaggle notebook's real-data loader
   from the Review 1 fix), not just synthetic verification data.
2. **Re-test the feedback loop comparison on real data specifically**,
   since the real data's lower-confidence predictions are exactly where
   the adaptive threshold is expected to actually diverge from the fixed
   one — this synthetic run couldn't demonstrate that by construction.
3. Consider testing the IngestionService's long-gap ("sensor_uncertain")
   path directly with a higher dropout rate, since this run only exercised
   short-gap interpolation.
