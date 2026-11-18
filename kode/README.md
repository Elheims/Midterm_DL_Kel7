# kode/

## Environment

- Python 3.11, PyTorch (CUDA 12.8 wheels), tested on an RTX 3060 Laptop GPU (6 GB).
- Main versions: Python 3.11.15, PyTorch 2.11.0, NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2.
- All exact versions: `requirements.txt` (from `pip freeze`).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

## Data

Put the extracted `data_uts_dl.zip` into `../data/raw/` (git-ignored). `kamus_data.md` in the zip describes every file.

## Reproduce everything

```bash
python run_all.py            # all tasks
python run_all.py --task 2   # only Task 2 (RUL)
```

Task 2 takes about 14 minutes on an RTX 3060 Laptop GPU (verified 18 Nov 2026: exit code 0; a second
full run reproduced every result value and an identical `prediksi_rul.csv`). Numbers may differ
slightly on a different GPU or driver.

To reproduce only the Task 2 predictions from the saved weights (no training, about 15 seconds):

```bash
python -m task2_rul.predict
```

Live reproduction for the oral exam (no training, under a minute, needs only the dataset in `../data/raw/`).
It recomputes numbers from the report and compares them with the report and with `prediksi_rul.csv`:

```bash
python -m task2_rul.demo
```

Each task writes numbers to `results/taskN/*.json` and figures to `../laporan/figures/`.
The report only quotes numbers that come from these files.

## Structure

| Folder | Content |
|---|---|
| `common/` | Shared helpers (seeding, paths) |
| `task1_klasifikasi/` | Task 1: multimodal defect classification |
| `task2_rul/` | Task 2: remaining useful life regression |
| `task3_clustering/` | Task 3: multimodal clustering |
| `results/` | Metrics (JSON), predictions, checkpoints (checkpoints git-ignored) |

## Seeds

Task 2 uses seeds 0, 1, 2 for every model comparison (`common/seed.py`) and seeds 0-4 for the five-GRU
ensemble (`task2_rul/uncertainty.py`, `task2_rul/final.py`). Data splits and test-like cuts use seed 0.
