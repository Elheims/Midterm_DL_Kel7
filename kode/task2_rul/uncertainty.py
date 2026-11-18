"""Task 2, part (f), step 1: prediction intervals for the chosen model (GRU, W=50).

A point prediction ("45 cycles left") hides how sure the model is. We build 90% intervals in three
standard ways and check whether they are honest (coverage: does the interval contain the true RUL
in 90% of the cases?).

Samples per prediction:
  - MC dropout (Gal & Ghahramani 2016): one model (seed 0), dropout kept ON at prediction time,
    50 forward passes -> 50 slightly different answers.
  - Deep ensemble (Lakshminarayanan et al. 2017): 5 GRUs trained from different random starts
    (seeds 0-4; 3 and 4 are trained here with exactly the same protocol), dropout off -> 5 answers.
  - Both: 5 models x 20 MC-dropout passes = 100 answers.
Raw interval: centre = mean of the samples; 90% interval = centre +- 1.645 * std of the samples.

These samples only measure how unsure the *model* is, not how unpredictable the *machine* is
(e.g. a machine whose wear starts late), so raw intervals are expected to be too narrow.
Calibration (normalised split-conformal prediction): on the 72 held-aside training machines
(10 cuts each, never used to fit weights) compute s = |true - centre| / std and take its 90%
quantile q (finite-sample corrected). Calibrated interval = centre +- q * std. A constant-width
version (centre +- 90% quantile of |true - centre|) is reported for comparison.
Coverage and width are measured on the 180 held-out machines, overall and per RUL range.

Run from kode/:  python -m task2_rul.uncertainty --work    (trains seeds 3-4, computes everything)
run_all uses main(), which runs --work as a separate process (GRU training, see deep.py).
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import plotstyle
from common.plotstyle import INK_2, MUTED, SERIES
from common.seed import FIG_DIR, RESULTS_DIR, set_seed
from task2_rul.deep import CKPT, prepare, run_isolated
from task2_rul.evaluate import RUL_RANGES, load_split, metrics
from task2_rul.labels import RUL_CAP
from task2_rul.models import build
from task2_rul.train_nn import DEVICE, GPUWindows, TrainConfig, fit, predict

OUT = RESULTS_DIR / "task2"
TAG, ARCH, W = "gru_W50", "gru", 50
ENSEMBLE_SEEDS = (0, 1, 2, 3, 4)
EXTRA_SEEDS = (3, 4)
N_MC, N_MC_PER_MODEL = 50, 20
Z90 = 1.645
TARGET = 0.90


def train_extra_seeds(sp, data) -> None:
    g = {k: GPUWindows(v) for k, v in data.items()}
    for s in EXTRA_SEEDS:
        set_seed(s)
        model = build(ARCH, len(sp.feats), W)
        model, hist = fit(model, g["train"], g["stop"], TrainConfig(), seed=s, log=lambda *a: None)
        torch.save(model.state_dict(), CKPT / f"{TAG}_seed{s}.pt")
        print(f"[uncertainty] trained extra seed {s}: best early-stop RMSE "
              f"{min(h['stop_rmse'] for h in hist):.2f}", flush=True)


def load_model(sp, s):
    m = build(ARCH, len(sp.feats), W).to(DEVICE)
    m.load_state_dict(torch.load(CKPT / f"{TAG}_seed{s}.pt", map_location=DEVICE))
    return m


def samples(sp, gw: GPUWindows) -> dict:
    """(n_samples, n_windows) arrays of RUL answers for the three methods."""
    models = {s: load_model(sp, s) for s in ENSEMBLE_SEEDS}
    torch.manual_seed(0)                                          # MC-dropout masks reproducible
    mc = np.stack([predict(models[0], gw, mc=True) for _ in range(N_MC)])
    ens = np.stack([predict(models[s], gw) for s in ENSEMBLE_SEEDS])
    both = np.stack([predict(models[s], gw, mc=True) for s in ENSEMBLE_SEEDS for _ in range(N_MC_PER_MODEL)])
    return {"mc_dropout": mc, "ensemble": ens, "ensemble+mc": both}


def conformal_q(scores: np.ndarray, level: float = TARGET) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n), method="higher"))


def interval_stats(lo, hi, y) -> dict:
    inside = (y >= lo) & (y <= hi)
    out = {"coverage": round(float(inside.mean()), 3), "mean_width": round(float(np.mean(hi - lo)), 1)}
    out["per_range"] = {}
    for name, a, b in RUL_RANGES:
        k = (y >= a) & (y <= b)
        out["per_range"][name] = {"coverage": round(float(inside[k].mean()), 3),
                                  "mean_width": round(float(np.mean(hi[k] - lo[k])), 1)}
    return out


def work() -> dict:
    plotstyle.apply()
    sp = load_split(seed=0)
    _, data, _ = prepare(W, sp=sp)
    train_extra_seeds(sp, data)
    g_cal, g_test = GPUWindows(data["stop"]), GPUWindows(data["ruler"])
    y_cal, y_test = data["stop"].y.astype(float), data["ruler"].y.astype(float)
    S_cal, S_test = samples(sp, g_cal), samples(sp, g_test)

    R = {"model": TAG, "ensemble_seeds": list(ENSEMBLE_SEEDS), "n_calibration_cuts": len(y_cal),
         "n_test_cuts": len(y_test), "methods": {}}
    keep = {}
    for name in S_test:
        c_cal, sd_cal = S_cal[name].mean(0), S_cal[name].std(0) + 1e-6
        c, sd = S_test[name].mean(0), S_test[name].std(0) + 1e-6
        q_norm = conformal_q(np.abs(y_cal - c_cal) / sd_cal)
        q_const = conformal_q(np.abs(y_cal - c_cal))
        clip = lambda a: np.clip(a, 0, RUL_CAP)
        raw = (clip(c - Z90 * sd), clip(c + Z90 * sd))
        cal = (clip(c - q_norm * sd), clip(c + q_norm * sd))
        const = (clip(c - q_const), clip(c + q_const))
        err = np.abs(y_test - c)
        R["methods"][name] = {
            "point_metrics_of_centre": metrics(c, y_test),
            "mean_std": round(float(sd.mean()), 2),
            "spearman_std_vs_abs_error": round(float(np.corrcoef(np.argsort(np.argsort(sd)),
                                                                 np.argsort(np.argsort(err)))[0, 1]), 3),
            "raw_90": interval_stats(*raw, y_test),
            "calibrated_90": {"scale_q": round(q_norm, 2), **interval_stats(*cal, y_test)},
            "constant_width_90": {"half_width": round(q_const, 1), **interval_stats(*const, y_test)},
        }
        keep[name] = (c, sd, q_norm)
        m = R["methods"][name]
        print(f"[uncertainty] {name:12s} centre RMSE {m['point_metrics_of_centre']['rmse']:.2f} | mean std "
              f"{m['mean_std']:.1f} | corr(std,|err|) {m['spearman_std_vs_abs_error']:.2f} | coverage raw "
              f"{m['raw_90']['coverage']:.2f} (width {m['raw_90']['mean_width']}) -> calibrated "
              f"{m['calibrated_90']['coverage']:.2f} (x{m['calibrated_90']['scale_q']:.2f}, width "
              f"{m['calibrated_90']['mean_width']}) | constant {m['constant_width_90']['coverage']:.2f} "
              f"(width {m['constant_width_90']['mean_width']})", flush=True)
    with open(OUT / "uncertainty.json", "w") as f:
        json.dump(R, f, indent=2)

    # ---------- figure: coverage per range (raw vs calibrated), and example machines over their life
    best = "ensemble+mc"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(wspace=0.28)
    names = [n for n, _, _ in RUL_RANGES]
    x = np.arange(len(names))
    for i, (key, lab, col) in enumerate([("raw_90", "raw", MUTED), ("calibrated_90", "calibrated", SERIES[0]),
                                         ("constant_width_90", "constant width", SERIES[1])]):
        vals = [R["methods"][best][key]["per_range"][n]["coverage"] for n in names]
        ax[0].bar(x + (i - 1) * 0.27, vals, 0.25, color=col, label=lab)
    ax[0].axhline(TARGET, color=INK_2, lw=1, ls="--")
    ax[0].set(xticks=x, ylim=(0, 1.25), yticks=np.arange(0, 1.01, 0.2), xlabel="true remaining life (cycles)",
              ylabel="share of true RUL inside", title="(a) Coverage of the 90% interval")
    ax[0].set_xticklabels(names)
    ax[0].legend(loc="upper center", ncol=3, fontsize=7.5)

    # whole-life trajectories of 3 held-out machines with calibrated intervals
    from task2_rul.windows import make_windows
    units = sp.val_units[[0, 60, 120]]
    q = keep[best][2]
    models = {s: load_model(sp, s) for s in ENSEMBLE_SEEDS}
    torch.manual_seed(1)
    for k, u in enumerate(units):
        dfu = sp.val[sp.val.id_unit == u].reset_index(drop=True)
        wu = make_windows(dfu, sp.feats, W, first_end=W)
        gu = GPUWindows(wu)
        smp = np.stack([predict(models[s], gu, mc=True) for s in ENSEMBLE_SEEDS for _ in range(N_MC_PER_MODEL)])
        c, sd = smp.mean(0), smp.std(0)
        ax[1].fill_between(wu.cycle, np.clip(c - q * sd, 0, RUL_CAP), np.clip(c + q * sd, 0, RUL_CAP),
                           color=SERIES[k], alpha=0.18, lw=0)
        ax[1].plot(wu.cycle, c, color=SERIES[k], lw=1.4, label=f"machine {u}")
        ax[1].plot(wu.cycle, wu.y, color=SERIES[k], lw=1, ls="--")
    ax[1].set(xlabel="cycle", ylabel="RUL (cycles)", title="(b) Prediction (line), 90% interval (band), truth (dashed)")
    ax[1].legend(loc="upper right", fontsize=7.5)
    plotstyle.save(fig, FIG_DIR / "t2_uncertainty")
    return R


def main() -> None:
    files = [OUT / "uncertainty.json"] + [CKPT / f"{TAG}_seed{s}.pt" for s in EXTRA_SEEDS]
    run_isolated("task2_rul.uncertainty", "--work", files, "uncertainty")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", action="store_true")
    a = ap.parse_args()
    if a.work:
        work()
    else:
        main()
