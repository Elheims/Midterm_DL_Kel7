# Task 2: Remaining Useful Life (RUL)

Predict RUL (capped at 150) at the last recorded cycle of each test machine.
Output: `prediksi_rul.csv` with columns `id_unit,rul`, rul in [0, 150].

## Parts (done one at a time)

- [x] a. Exploration and preprocessing: operating settings vs sensors, uninformative sensors, missing values, spikes, normalization
  - `data.py`: `Preprocessor` (regime KMeans, per-regime z-score, causal Hampel spike filter, forward fill, sensor selection)
  - `eda.py`: numbers in `results/task2/eda.json`, figures `laporan/figures/t2_*.pdf`
  - `check_preprocess.py`: causality / no-NaN / stability checks
- [x] b. RUL labels from run-to-failure histories, 150-cycle cap
  - `labels.py`: `add_rul()` (RUL = min(150, failure cycle − cycle)), `check_labels()`, figure `t2_labels`, numbers in `results/task2/labels.json`
- [x] c. Sliding windows, variable-length histories, no future leakage
  - `windows.py`: `make_windows()` (window = last W cycles ending at t, target RUL(t); front padding for short histories), `check_windows()` (leakage checks, plus a deliberately leaky version that must be rejected), figure `t2_windows`, numbers in `results/task2/windows.json`
- [x] d. Models: >= 2 sequence architectures + non-deep-learning baseline
  - [x] `evaluate.py`: the ruler (80/20 machine split, 5 test-like cuts per held-out machine, RMSE + asymmetric score)
  - [x] `baseline.py`: 42 window features (last-5 mean, mean, slope per sensor) → Ridge, boosted trees
  - [x] 1D-CNN: `models.py` (CNN1D), `train_nn.py` (training loop, early stopping, overfit-one-batch), `deep.py` (`cnn_ablation()`: original + 3 fixes × 3 seeds → `results/task2/cnn_ablation.json`, figures `t2_cnn_*`). Chosen: W=50 (best on the early-stop machines)
  - [x] error analysis of the CNN: `error_analysis.py` (wear onset vs prediction error, 3 seeds × 3 thresholds) → `results/task2/error_analysis.json`, figure `t2_error_analysis`
  - [x] GRU: `models.py` (GRUNet, 2 layers × 64, 44.5k params), `deep.gru_runs()` (W = 30 and 50, 3 seeds each → `results/task2/gru_runs.json`, figures `t2_gru_*`). Chosen: W=50 (best on the early-stop machines). In `run_all` it runs as a separate process (`gru_runs_isolated`), see "Windows note" below
  - [x] ablations (W = 30 vs 50 for CNN and GRU; CNN pooled head vs flatten; CNN stronger regularisation)
- [x] e. Validation split by machine; RMSE and asymmetric score per RUL range
  - `validation.py`: per-range table (0-25, 26-50, 51-100, 101-149, 150) for Ridge, trees, CNN W50, GRU W50 → `results/task2/validation_by_range.json`; 5-fold CV by machine, every model retrained per fold → `validation_cv.json`; figure `t2_validation`
- [x] f. Uncertainty: prediction intervals, when to service a machine
  - [x] `uncertainty.py`: 90% intervals for GRU W50 from MC dropout (50 passes), a 5-seed ensemble (seeds 3-4 trained here) and both (5 × 20); raw vs calibrated (split-conformal on the 72 early-stop machines) vs constant width; coverage per RUL range → `results/task2/uncertainty.json`, figure `t2_uncertainty`
  - [x] `maintenance.py`: replay the 180 held-out machines cycle by cycle; point rule (prediction ≤ L) vs cautious rule (lower end of 90% interval ≤ L), lead time 10 cycles → `results/task2/maintenance.json`, figure `t2_maintenance`
- [x] final model + `prediksi_rul.csv`
  - `final.py`: shift curve (`final_shift.json`), shift decision by simulating 2,000 test sets (`final_shift_simulation.json`, chosen shift = 3), 5 GRUs W=50 retrained on all 900 machines (810 train / 90 early stop) → `final_weights/gru_W50_seed{0-4}.pt` (in git)
  - `predict.py`: **no training**; loads the final weights, writes `../prediksi_rul.csv` with format checks (~15 s): `python -m task2_rul.predict`
  - `demo.py`: **live reproduction for the oral exam**; no training, recomputes 11 report numbers from the saved validation models and checks the 300 test predictions against `prediksi_rul.csv` (under a minute): `python -m task2_rul.demo`

Scoring: RMSE and s = sum(exp(-d/13) - 1 if d < 0 else exp(d/10) - 1), d = predicted - actual.

## Windows note: GRU training crashes the process *at exit*

On Windows with PyTorch 2.11 (CUDA 12.8), a Python process that has **trained** a GRU or LSTM with
cuDNN crashes while closing (exit code 0xC0000409 = 3221226505), after all work is done and every
file is written. Tested: prediction-only use does not crash; training with cuDNN disabled does not
crash but is ~18x slower (27.7 s vs 1.5 s per epoch); `os._exit` and `TerminateProcess` do not avoid
it. Therefore `run_all.py` runs the GRU training as a separate process (`deep.gru_runs_isolated`) and
accepts that one exit code only if the process wrote a fresh `gru_runs.json` and all 6 checkpoints.
Results are unaffected: three consecutive runs gave identical numbers. On Linux/Colab the child
process simply exits with 0.
