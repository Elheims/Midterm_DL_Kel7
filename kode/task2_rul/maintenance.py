"""Task 2, part (f), step 2: when should a machine be serviced?

Replay: every one of the 180 held-out machines is replayed cycle by cycle, from cycle 50 (shortest
test history) until it failed. At every cycle the final model (5 GRUs x 20 MC-dropout passes, W=50)
gives a prediction and a calibrated 90% interval (from uncertainty.py). A rule decides at which
cycle to service; we record the true remaining life at that moment.

Rules (threshold L in cycles):
  - point rule:     service at the first cycle where the predicted RUL <= L
  - cautious rule:  service at the first cycle where the LOWER end of the 90% interval <= L
Outcome per machine, assuming maintenance needs D = 10 cycles to arrange (our assumption):
  - too late:  true RUL at the service decision < D, or the rule never fired before failure
  - wasted life: true RUL at the service decision (cycles of useful life thrown away)
A good rule has few machines too late and little wasted life; we report the trade-off for several L.

Run from kode/:  python -m task2_rul.maintenance --work
run_all uses main(), which runs --work as a separate process (cuDNN GRU, see deep.py).
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import plotstyle
from common.plotstyle import INK_2, SERIES
from common.seed import FIG_DIR, RESULTS_DIR, set_seed
from task2_rul.deep import run_isolated
from task2_rul.evaluate import MIN_CUT, load_split
from task2_rul.labels import RUL_CAP
from task2_rul.train_nn import GPUWindows, predict
from task2_rul.uncertainty import ENSEMBLE_SEEDS, N_MC_PER_MODEL, W, load_model
from task2_rul.windows import make_windows

OUT = RESULTS_DIR / "task2"
LEAD = 10                                 # cycles needed to arrange maintenance (assumption)
THRESHOLDS = tuple(range(0, 81, 2))      # fine grid, so the two rules can be compared at equal risk
LABELLED = (10, 20, 30, 40, 50)


def replay() -> pd.DataFrame:
    """Prediction, std and true RUL at every cycle >= 50 of every held-out machine."""
    set_seed(0)
    sp = load_split(seed=0)
    win = make_windows(sp.val, sp.feats, W, first_end=MIN_CUT)
    g = GPUWindows(win)
    models = [load_model(sp, s) for s in ENSEMBLE_SEEDS]
    torch.manual_seed(0)
    smp = np.stack([predict(m, g, mc=True) for m in models for _ in range(N_MC_PER_MODEL)])
    life = sp.val.groupby("id_unit").siklus.max()
    df = pd.DataFrame({"unit": win.unit, "cycle": win.cycle, "pred": smp.mean(0), "std": smp.std(0)})
    df["true_left"] = df.unit.map(life) - df.cycle          # uncapped: real cycles until failure
    return df


def apply_rule(df: pd.DataFrame, value: str, L: float) -> pd.DataFrame:
    """First cycle per machine where df[value] <= L; machines where it never fires get true_left = 0."""
    hit = df[df[value] <= L].groupby("unit").first()
    out = pd.DataFrame(index=df.unit.unique())
    out["true_left_at_service"] = hit["true_left"]
    out["fired"] = out.true_left_at_service.notna()
    out["true_left_at_service"] = out.true_left_at_service.fillna(0)
    return out


def work() -> dict:
    plotstyle.apply()
    q = json.load(open(OUT / "uncertainty.json"))["methods"]["ensemble+mc"]["calibrated_90"]["scale_q"]
    df = replay()
    df["lower"] = np.clip(df.pred - q * df["std"], 0, RUL_CAP)
    R = {"lead_time": LEAD, "interval_scale_q": q, "machines": int(df.unit.nunique()),
         "replayed_cycles": len(df), "rules": {}}
    # how much does the interval width vary near the decision point (prediction 25-45 cycles)?
    near = df[(df.pred >= 25) & (df.pred <= 45)]
    width = 2 * q * near["std"]
    R["interval_width_near_decision"] = {"mean": round(float(width.mean()), 1), "sd": round(float(width.std()), 1),
                                         "p10": round(float(width.quantile(0.1)), 1),
                                         "p90": round(float(width.quantile(0.9)), 1),
                                         "corr_with_abs_error": round(float(np.corrcoef(
                                             near["std"], np.abs(near.pred - near.true_left.clip(upper=RUL_CAP)))[0, 1]), 3)}
    print(f"[maintenance] interval width when prediction is 25-45: {R['interval_width_near_decision']}", flush=True)
    for rule, col in (("point", "pred"), ("cautious", "lower")):
        R["rules"][rule] = {}
        for L in THRESHOLDS:
            o = apply_rule(df, col, L)
            late = o.true_left_at_service < LEAD
            R["rules"][rule][L] = {
                "share_too_late": round(float(late.mean()), 3),
                "never_fired": int((~o.fired).sum()),
                "mean_wasted_life": round(float(o.true_left_at_service.mean()), 1),
                "median_wasted_life": round(float(o.true_left_at_service.median()), 1),
            }
            r = R["rules"][rule][L]
            if L in LABELLED:
                print(f"[maintenance] {rule:8s} L={L:3d}: too late {r['share_too_late']:.1%} "
                      f"(never fired {r['never_fired']}), wasted life mean {r['mean_wasted_life']:.1f} "
                      f"median {r['median_wasted_life']:.1f}", flush=True)
    # fair comparison: for each rule, the least wasted life at a given maximum share of late machines
    R["least_waste_at_risk"] = {}
    for rule in R["rules"]:
        R["least_waste_at_risk"][rule] = {}
        for max_late in (0.0, 0.01, 0.05):
            ok = [(v["mean_wasted_life"], L) for L, v in R["rules"][rule].items() if v["share_too_late"] <= max_late]
            w, L = min(ok)
            R["least_waste_at_risk"][rule][str(max_late)] = {"L": L, "mean_wasted_life": w}
        print(f"[maintenance] {rule:8s} least waste: " + ", ".join(
            f"≤{float(k):.0%} late → L={v['L']}, waste {v['mean_wasted_life']}"
            for k, v in R["least_waste_at_risk"][rule].items()), flush=True)
    with open(OUT / "maintenance.json", "w") as f:
        json.dump(R, f, indent=2)

    # ---------- figure: trade-off curves, and the distribution for two comparable settings
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
    fig.subplots_adjust(wspace=0.3)
    for i, (rule, lab) in enumerate((("point", "service when prediction ≤ L"),
                                     ("cautious", "service when lower end of 90% interval ≤ L"))):
        pts = R["rules"][rule]
        xs = [pts[L]["mean_wasted_life"] for L in THRESHOLDS]
        ys = [100 * pts[L]["share_too_late"] for L in THRESHOLDS]
        ax[0].plot(xs, ys, marker="o", ms=3, color=SERIES[i], label=lab)
        best = R["least_waste_at_risk"][rule]["0.0"]
        ax[0].scatter([best["mean_wasted_life"]], [0], s=70, color=SERIES[i], zorder=4, edgecolor="white")
        ax[0].annotate(f"L={best['L']}: no machine late,\n{best['mean_wasted_life']:.0f} cycles wasted",
                       (best["mean_wasted_life"], 0), textcoords="offset points",
                       xytext=(8, 14 + 26 * i), fontsize=7, color=SERIES[i])
    ax[0].set(xlabel="mean wasted life (cycles left at service)", ylabel="machines too late (%)",
              title=f"(a) Trade-off (too late = < {LEAD} cycles left)", ylim=(-2, 40), xlim=(0, 100))
    ax[0].legend(fontsize=7.5)
    bins = np.arange(0, 121, 5)
    for i, (rule, col) in enumerate((("point", "pred"), ("cautious", "lower"))):
        L = R["least_waste_at_risk"][rule]["0.0"]["L"]          # cheapest setting with no machine too late
        o = apply_rule(df, col, L)
        ax[1].hist(o.true_left_at_service, bins=bins, color=SERIES[i], alpha=0.6,
                   label=f"{rule} rule, L={L}")
    ax[1].axvline(LEAD, color=INK_2, lw=1, ls="--")
    ax[1].set(xlabel="true cycles left when service is triggered", ylabel="machines",
              title="(b) When each machine would be serviced")
    ax[1].legend(fontsize=7.5)
    plotstyle.save(fig, FIG_DIR / "t2_maintenance")
    return R


def main() -> None:
    run_isolated("task2_rul.maintenance", "--work", [OUT / "maintenance.json"], "maintenance")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", action="store_true")
    a = ap.parse_args()
    if a.work:
        work()
    else:
        main()
