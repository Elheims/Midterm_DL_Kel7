"""Task 2: live reproduction for the oral exam. No training, about one minute.

What it does:
1. Rebuilds the validation split (720 training / 180 held-out machines, 900 test-like cuts).
2. Loads the saved validation models (results/task2/checkpoints/: 3 CNNs and 3 GRUs, W=50).
3. Recomputes the numbers printed in the report and compares each one with the report.
4. Recomputes the 300 test predictions from the final weights and compares them with prediksi_rul.csv
   (the file is only read, never overwritten).

Needs only the dataset in ../data/raw/. Run from kode/:  python -m task2_rul.demo
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from common.seed import SEEDS, set_seed
from task2_rul.deep import CKPT, prepare
from task2_rul.evaluate import load_split, metrics, metrics_by_range
from task2_rul.models import build
from task2_rul.predict import CSV, make_predictions
from task2_rul.train_nn import DEVICE, GPUWindows, predict

W = 50

# (label, value printed in the report, decimals printed in the report)
REPORT = {
    "gru_rmse": ("GRU W=50, held-out RMSE (mean of 3 seeds)", 20.70, 2),
    "gru_rmse_std": ("GRU W=50, held-out RMSE (std of 3 seeds)", 0.10, 2),
    "gru_score": ("GRU W=50, asymmetric score per prediction", 34, 0),
    "cnn_rmse": ("1D-CNN W=50, held-out RMSE (mean of 3 seeds)", 21.45, 2),
    "cnn_rmse_std": ("1D-CNN W=50, held-out RMSE (std of 3 seeds)", 0.63, 2),
    "cnn_score": ("1D-CNN W=50, asymmetric score per prediction", 32, 0),
    "gru_rmse_0-25": ("GRU RMSE at true RUL 0-25", 8.5, 1),
    "gru_rmse_26-50": ("GRU RMSE at true RUL 26-50", 17.8, 1),
    "gru_rmse_51-100": ("GRU RMSE at true RUL 51-100", 29.6, 1),
    "gru_rmse_101-149": ("GRU RMSE at true RUL 101-149", 20.7, 1),
    "gru_rmse_150": ("GRU RMSE at true RUL 150", 15.9, 1),
}


def seed_predictions(arch: str, n_feat: int, g: GPUWindows) -> list[np.ndarray]:
    preds = []
    for s in SEEDS:
        m = build(arch, n_feat, W).to(DEVICE)
        m.load_state_dict(torch.load(CKPT / f"{arch}_W{W}_seed{s}.pt", map_location=DEVICE))
        preds.append(predict(m, g))
    return preds


def main() -> dict:
    t0 = time.time()
    set_seed(0)
    print("[demo] rebuilding the validation split (180 held-out machines) ...", flush=True)
    sp = load_split(seed=0)
    _, data, _ = prepare(W, sp=sp)
    g, y = GPUWindows(data["ruler"]), data["ruler"].y

    got = {}
    for arch in ("gru", "cnn"):
        print(f"[demo] loading 3 saved {arch.upper()} models and predicting {len(y)} cuts ...", flush=True)
        preds = seed_predictions(arch, len(sp.feats), g)
        rmse = [metrics(p, y)["rmse"] for p in preds]
        got[f"{arch}_rmse"] = float(np.mean(rmse))
        got[f"{arch}_rmse_std"] = float(np.std(rmse, ddof=1))
        got[f"{arch}_score"] = float(np.mean([metrics(p, y)["score_mean"] for p in preds]))
        if arch == "gru":
            for name in ("0-25", "26-50", "51-100", "101-149", "150"):
                got[f"gru_rmse_{name}"] = float(np.mean([metrics_by_range(p, y)[name]["rmse"] for p in preds]))

    print("\n  Number from the report" + " " * 28 + "Report   Reproduced   Match")
    all_ok = True
    for key, (label, value, dec) in REPORT.items():
        ok = abs(got[key] - value) <= 0.5 * 10 ** (-dec) + 0.002     # equal after rounding as printed
        all_ok &= ok
        print(f"  {label:48s} {value:8.{dec}f} {got[key]:12.3f}   {'yes' if ok else 'NO'}")

    print("\n[demo] recomputing the 300 test predictions from the final weights ...", flush=True)
    df, _, _, delta = make_predictions()
    saved = pd.read_csv(CSV)
    same = saved.id_unit.equals(df.id_unit) and bool(np.allclose(saved.rul, df.rul))
    all_ok &= same
    print(f"[demo] prediksi_rul.csv: {len(df)} rows, shift {delta}, identical to the saved file: {'yes' if same else 'NO'}")
    print(f"[demo] finished in {time.time() - t0:.0f} s; all numbers match the report: {'yes' if all_ok else 'NO'}")
    if not all_ok:
        raise RuntimeError("a reproduced number differs from the report")
    return got


if __name__ == "__main__":
    main()
