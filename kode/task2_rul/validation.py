"""Task 2, part (e): validation.

1. Per RUL range (on the ruler split: 180 held-out machines, 900 test-like cuts).
   One overall average hides where a model is good or bad, so RMSE and the asymmetric score are
   reported per range of true remaining life: 0-25 (about to fail), 26-50, 51-100, 101-149 and 150
   (capped = still healthy). Models: the two baselines and the chosen deep models (CNN and GRU,
   W=50; 3 seeds each, loaded from their checkpoints, plus the average of the 3 seeds).

2. 5-fold cross-validation by machine.
   All results so far come from one particular set of 180 held-out machines. Here the 900 machines
   are split into 5 groups; each group is held out once while everything (preprocessing, sensor
   selection, model) is fitted on the other 4. Every model is retrained in every fold (deep models:
   seed 0). If the ranking of the models is the same in every fold, it is not luck of one split.
   The deep models train a GRU, so this part runs as a separate process in run_all (see deep.py).

Run from kode/:  python -m task2_rul.validation          (everything)
                 python -m task2_rul.validation --cv     (only the cross-validation)
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import plotstyle
from common.plotstyle import SERIES
from common.seed import FIG_DIR, RESULTS_DIR, SEEDS, set_seed
from task2_rul import baseline
from task2_rul.deep import CKPT, prepare, run_isolated, train_predict
from task2_rul.evaluate import RUL_RANGES, load_split, metrics, metrics_by_range
from task2_rul.models import build
from task2_rul.train_nn import DEVICE, GPUWindows, predict

OUT = RESULTS_DIR / "task2"
DEEP = {"cnn": "cnn_W50", "gru": "gru_W50"}       # chosen deep models (part d)
W = 50
N_FOLDS = 5
NAMES = {"ridge": "Ridge", "boosted_trees": "Boosted trees", "cnn": "1D-CNN", "gru": "GRU"}


def by_range() -> dict:
    set_seed(0)
    sp = load_split(seed=0)
    preds, y = baseline.fit_predict(sp)
    R = {name: {"overall": metrics(p, y), "ranges": metrics_by_range(p, y)} for name, p in preds.items()}

    _, data, _ = prepare(W, sp=sp)
    g = GPUWindows(data["ruler"])
    y_deep = data["ruler"].y
    assert np.array_equal(y_deep, y), "deep and baseline models must be scored on the same cuts"
    for arch, tag in DEEP.items():
        per_seed = []
        for s in SEEDS:
            m = build(arch, len(sp.feats), W).to(DEVICE)
            m.load_state_dict(torch.load(CKPT / f"{tag}_seed{s}.pt", map_location=DEVICE))
            per_seed.append(predict(m, g))
        rng = {}
        for name, _, _ in RUL_RANGES:
            vals = [metrics_by_range(p, y)[name] for p in per_seed]
            rng[name] = {k: {"mean": round(float(np.mean([v[k] for v in vals])), 3),
                             "std": round(float(np.std([v[k] for v in vals], ddof=1)), 3)}
                         for k in ("rmse", "score_mean", "bias")}
            rng[name]["n"] = vals[0]["n"]
        ens = np.mean(per_seed, axis=0)
        R[arch] = {"overall_seed_mean_rmse": round(float(np.mean([metrics(p, y)["rmse"] for p in per_seed])), 3),
                   "ranges": rng, "ensemble": {"overall": metrics(ens, y), "ranges": metrics_by_range(ens, y)}}
    with open(OUT / "validation_by_range.json", "w") as f:
        json.dump(R, f, indent=2)

    print("[validation] RMSE per true-RUL range (deep: mean of 3 seeds)")
    print("  " + "model".ljust(14) + "".join(n.rjust(9) for n, _, _ in RUL_RANGES))
    for k in ("ridge", "boosted_trees"):
        print("  " + NAMES[k].ljust(14) + "".join(f"{R[k]['ranges'][n]['rmse']:9.1f}" for n, _, _ in RUL_RANGES))
    for k in DEEP:
        print("  " + NAMES[k].ljust(14) + "".join(f"{R[k]['ranges'][n]['rmse']['mean']:9.1f}" for n, _, _ in RUL_RANGES))
    print("  n cuts".ljust(16) + "".join(f"{R['ridge']['ranges'][n]['n']:9d}" for n, _, _ in RUL_RANGES))
    return R


def cross_validation() -> dict:
    R = {"n_folds": N_FOLDS, "folds": {}}
    for k in range(N_FOLDS):
        t0 = time.time()
        set_seed(0)
        sp = load_split(seed=0, fold=k, n_folds=N_FOLDS)
        preds, y = baseline.fit_predict(sp)
        fold = {name: metrics(p, y) for name, p in preds.items()}
        for arch in DEEP:
            p, y2 = train_predict(arch, W, sp, seed=0)
            assert np.array_equal(y2, y)
            fold[arch] = metrics(p, y)
        fold["held_out_machines"] = int(len(sp.val_units))
        fold["kept_sensors"] = sp.feats
        R["folds"][k] = fold
        print(f"[cv] fold {k}: " + "  ".join(f"{NAMES[m]} {fold[m]['rmse']:.2f}" for m in NAMES)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    R["summary"] = {m: {key: {"mean": round(float(np.mean([R["folds"][k][m][key] for k in range(N_FOLDS)])), 3),
                              "std": round(float(np.std([R["folds"][k][m][key] for k in range(N_FOLDS)], ddof=1)), 3)}
                        for key in ("rmse", "score_mean")}
                    for m in NAMES}
    R["best_model_per_fold"] = [min(NAMES, key=lambda m: R["folds"][k][m]["rmse"]) for k in range(N_FOLDS)]
    R["same_sensors_in_every_fold"] = all(R["folds"][k]["kept_sensors"] == R["folds"][0]["kept_sensors"]
                                          for k in range(N_FOLDS))
    with open(OUT / "validation_cv.json", "w") as f:
        json.dump(R, f, indent=2)
    print("[cv] mean ± std over folds: " + "  ".join(
        f"{NAMES[m]} {R['summary'][m]['rmse']['mean']:.2f}±{R['summary'][m]['rmse']['std']:.2f}" for m in NAMES))
    print(f"[cv] best model per fold: {R['best_model_per_fold']}; same sensors kept in every fold: "
          f"{R['same_sensors_in_every_fold']}")
    return R


def figure() -> None:
    plotstyle.apply()
    rng = json.load(open(OUT / "validation_by_range.json"))
    cv = json.load(open(OUT / "validation_cv.json"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.28)
    names = [n for n, _, _ in RUL_RANGES]
    x = np.arange(len(names))
    w = 0.2
    for i, m in enumerate(NAMES):
        r = rng[m]["ranges"]
        vals = [r[n]["rmse"]["mean"] if isinstance(r[n]["rmse"], dict) else r[n]["rmse"] for n in names]
        ax[0].bar(x + (i - 1.5) * w, vals, w * 0.92, color=SERIES[i], label=NAMES[m])
    ax[0].set(xticks=x, xlabel="true remaining life (cycles)", ylabel="RMSE (cycles)",
              title="(a) Error per range of remaining life")
    ax[0].set_xticklabels(names)
    ax[0].set_ylim(top=max(ax[0].get_ylim()[1], 40))  # leave room for the legend above the bars
    ax[0].legend(ncol=4, loc="upper center", fontsize=7.5)
    folds = np.arange(cv["n_folds"])
    for i, m in enumerate(NAMES):
        ax[1].plot(folds + 1, [cv["folds"][str(k)][m]["rmse"] for k in folds], marker="o", ms=5,
                   color=SERIES[i], label=NAMES[m])
    ax[1].set(xticks=folds + 1, xlabel="fold (held-out group of 180 machines)", ylabel="RMSE (cycles)",
              title="(b) 5-fold cross-validation")
    plotstyle.save(fig, FIG_DIR / "t2_validation")


def main() -> None:
    by_range()
    run_isolated("task2_rul.validation", "--cv", [OUT / "validation_cv.json"], "validation_cv")
    figure()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()
    if a.cv:
        cross_validation()
    else:
        main()
