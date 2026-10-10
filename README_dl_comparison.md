# DL model comparison (Review 2 extension) - Gradient Boosting removed

Models compared: LSTM, GRU, 1D-CNN, Transformer encoder, LSTM-Autoencoder (unsupervised).
All use windows of the last 20 files x the same 9 features and the leave-one-run-out protocol.

Files
- `dl_models.py`          GRU, 1D-CNN, Transformer, LSTM-Autoencoder + training helpers
- `ims_loader.py`         loader for the NASA IMS bearing data (Kaggle: vinayak123tyagi/bearing-dataset)
- `run_dl_comparison.py`  trains/evaluates everything, writes tables + graphs to `results/`
- `make_mock_ims.py`      writes a tiny MOCK dataset in IMS file format (only to test the loader)

## Run on synthetic data (what the report's Section 4.2 used)
    pip install torch scikit-learn pandas scipy matplotlib
    python run_dl_comparison.py --seeds 3

## Run on the REAL NASA IMS data
1. Download the dataset from Kaggle and unzip it anywhere (or, in a Kaggle notebook, "Add data" -> this dataset;
   it is then under /kaggle/input/bearing-dataset).
2. Put this folder's .py files next to each other and run:
       python run_dl_comparison.py --data-dir <path-to-dataset> --seeds 3
   Kaggle notebook:  !python run_dl_comparison.py --data-dir /kaggle/input/bearing-dataset --seeds 3
- Runs are found automatically (any folder with >= 500 timestamp-named files: 1st_test = 2156 files / 8 channels,
  2nd_test = 984 / 4, 3rd_test (4th_test/txt) = 4448 / 4).
- The monitored channel is the bearing that actually failed (set 1: bearing 3, set 2: bearing 1, set 3: bearing 3);
  an unrecognised run falls back to the channel whose RMS grows most.
- Feature extraction (first run only) takes several minutes and is cached in `ims_cache/`.
- All three leave-one-run-out folds are run and pooled (`--holdout run_2` for a single fold).
- Check the printed "label onset" lines: a flagged run means the z-score labeller found no/very early degradation.

Outputs in `results/`: comparison_table.csv, per_fold_table.csv, comparison_results.json, 01_metrics_comparison.png,
02_pr_curves*.png, 03_probability_timeline*.png, 04_training_loss*.png, 05_borderline_and_delay.png, 06_model_cost.png,
07_f1_per_fold.png (real data).

Definitions: detection delay = files from labelled onset to first alert that persists 3 files; false alerts = alerts on
healthy files; "scores in 0.2-0.7" = share of scores inside the FeedbackController's adjustable range.
The autoencoder score = logistic(3 * (log err - log q99 of healthy training error)); steepness 3 is a design choice.
