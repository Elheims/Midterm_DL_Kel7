"""Task 2, part (d): why are some predictions far off? (error analysis of the chosen CNN, W=50)

Idea: the sensors show how worn a machine is *now*, not how fast it will keep wearing. If machines
start wearing at very different times before failure, two machines that look equally worn can have
very different remaining lives, and no window-based model can tell them apart.

How we measure it (analysis only - this uses the full history up to failure, so it can never be a
model input):
  - wear level: mean over the 14 kept sensors of |z - healthy start level|, averaged over the
    previous 15 cycles, minus the machine's own mean wear level in its first 40 cycles
    (so every machine starts at about 0).
  - wear onset: how many cycles before failure the wear level first exceeds a threshold
    (0.25 by default; 0.15 and 0.35 are checked for robustness).
Then, for test-like cuts with true RUL 40-120 (where the big errors are), we correlate the error
(prediction - truth) with the wear onset, for each of the 3 seeds.

Run from kode/:  python -m task2_rul.error_analysis      (needs the cnn_W50 checkpoints)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from common import plotstyle
from common.plotstyle import INK_2, MUTED, SERIES
from common.seed import FIG_DIR, RESULTS_DIR, SEEDS, set_seed
from task2_rul.data import unit_trend
from task2_rul.deep import CKPT, prepare
from task2_rul.models import build
from task2_rul.train_nn import DEVICE, GPUWindows, predict

TAG, W = "cnn_W50", 50
THRESHOLDS = (0.15, 0.25, 0.35)


def wear_level(sp) -> pd.Series:
    v = sp.val
    z = v[sp.feats] - sp.prep.start_level_[sp.feats]
    hi = z.abs().mean(axis=1).groupby(v.id_unit).transform(lambda x: x.rolling(15, min_periods=1).mean())
    first = v.groupby("id_unit").cumcount() < 40
    return hi - v.id_unit.map(hi[first].groupby(v.id_unit[first]).mean())


def wear_onset(sp, wear: pd.Series, thr: float) -> pd.Series:
    v = sp.val
    left = v.groupby("id_unit").siklus.transform("max") - v.siklus
    above = wear > thr
    return left[above].groupby(v.id_unit[above]).max()   # earliest exceedance, in cycles before failure


def main() -> dict:
    set_seed(0)
    plotstyle.apply()
    sp, data, _ = prepare(W)
    r = data["ruler"]
    gr = GPUWindows(r)
    wear = wear_level(sp)
    onsets = {t: wear_onset(sp, wear, t) for t in THRESHOLDS}

    # failure mode per held-out machine (same idea as part a)
    left = sp.val.groupby("id_unit").siklus.transform("max") - sp.val.siklus
    delta = unit_trend(sp.val[sp.feats], sp.val.id_unit, left)
    mode = pd.Series(KMeans(2, n_init=10, random_state=0).fit_predict((delta / delta.std()).to_numpy()),
                     index=delta.index)

    R = {"model": TAG, "definition": __doc__.split("How we measure it")[1].split("Run from")[0].strip(),
         "per_seed": {}}
    tables = {}
    for s in SEEDS:
        m = build("cnn", len(sp.feats), W).to(DEVICE)
        m.load_state_dict(torch.load(CKPT / f"{TAG}_seed{s}.pt", map_location=DEVICE))
        df = pd.DataFrame({"unit": r.unit, "cut": r.cycle, "true": r.y, "pred": predict(m, gr)})
        df["err"] = df.pred - df.true
        df["mode"] = df.unit.map(mode)
        for t in THRESHOLDS:
            df[f"onset_{t}"] = df.unit.map(onsets[t])
        mid = df[(df.true >= 40) & (df.true <= 120)]
        R["per_seed"][s] = {
            "n_mid_cuts": len(mid),
            "corr_error_vs_onset": {str(t): round(float(mid.err.corr(mid[f"onset_{t}"])), 3) for t in THRESHOLDS},
            "mean_abs_error_by_mode": df.groupby("mode").err.apply(lambda e: round(float(e.abs().mean()), 1)).to_dict(),
            "median_onset_late_misses_gt40": float(mid[mid.err > 40]["onset_0.25"].median()),
            "median_onset_early_misses_lt_minus40": float(mid[mid.err < -40]["onset_0.25"].median()),
            "median_onset_all_mid": float(mid["onset_0.25"].median()),
        }
        tables[s] = df
    on = onsets[0.25]
    R["onset_0.25_over_machines"] = on.describe().round(0).to_dict()
    cors = [R["per_seed"][s]["corr_error_vs_onset"][str(t)] for s in SEEDS for t in THRESHOLDS]
    R["corr_range_all_seeds_thresholds"] = [min(cors), max(cors)]
    with open(RESULTS_DIR / "task2" / "error_analysis.json", "w") as f:
        json.dump(R, f, indent=2, default=str)

    # ---------- figure (seed 0, threshold 0.25): worst late miss, worst early miss, a typical machine
    df = tables[SEEDS[0]]
    late = df.loc[df.err.idxmax()]
    early = df.loc[df.err.idxmin()]
    typical = int((on - on.median()).abs().idxmin())
    v = sp.val
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.28)
    cases = [(int(late.unit), int(late.cut), f"machine {int(late.unit)}: guessed {late.pred:.0f}, truth {late.true:.0f}", SERIES[1]),
             (int(early.unit), int(early.cut), f"machine {int(early.unit)}: guessed {early.pred:.0f}, truth {early.true:.0f}", SERIES[6]),
             (typical, None, f"machine {typical}: typical wear onset", MUTED)]
    for u, cut, lab, col in cases:
        k = v.id_unit == u
        ax[0].plot(left[k], wear[k], color=col, lw=1.6, label=lab)
        if cut is not None:
            kk = k & (v.siklus == cut)
            ax[0].scatter(left[kk], wear[kk], s=60, color=col, zorder=4, edgecolor="white")
    ax[0].set_xlim(250, -5)
    ax[0].set(xlabel="cycles before failure", ylabel="wear level",
              title="(a) Machines wear at different speeds (dot = prediction point)")
    ax[0].legend(loc="upper left")
    mid = df[(df.true >= 40) & (df.true <= 120)]
    ax[1].scatter(mid["onset_0.25"], mid.err, s=10, alpha=0.5, color=SERIES[0], lw=0)
    ax[1].axhline(0, color=INK_2, lw=1, ls="--")
    ax[1].set(xlabel="wear onset (cycles before failure)", ylabel="prediction − truth (cycles)",
              title=f"(b) Error vs wear onset, r = {R['per_seed'][SEEDS[0]]['corr_error_vs_onset']['0.25']:.2f}")
    plotstyle.save(fig, FIG_DIR / "t2_error_analysis")

    R["figure_cases"] = {"late": cases[0][2], "early": cases[1][2], "typical": typical}
    with open(RESULTS_DIR / "task2" / "error_analysis.json", "w") as f:
        json.dump(R, f, indent=2, default=str)
    for s in SEEDS:
        p = R["per_seed"][s]
        print(f"[error analysis] seed {s}: corr(error, onset) {p['corr_error_vs_onset']}  "
              f"|error| by mode {p['mean_abs_error_by_mode']}  onset median: late misses "
              f"{p['median_onset_late_misses_gt40']}, early misses {p['median_onset_early_misses_lt_minus40']}, "
              f"all {p['median_onset_all_mid']}")
    print(f"[error analysis] figure cases: {R['figure_cases']}")
    return R


if __name__ == "__main__":
    main()
