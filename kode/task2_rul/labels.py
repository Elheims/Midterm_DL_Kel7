"""Task 2, part (b): RUL labels for the training machines.

Training machines run until they fail, so the last recorded cycle of each machine is its failure
cycle. At cycle t the true remaining life is (failure cycle - t). Following the piecewise-linear
target of Heimes (2008), and the exam's own definition, it is capped at 150:

    RUL(t) = min(150, failure_cycle - t)

Why the cap: early in life the sensors do not yet show wear (see t2_failure_modes), so a machine
with 300 cycles left looks like one with 200 left. An uncapped label would ask the model to tell
them apart from information that is not in the data. The hidden test labels use the same cap.

This only works for run-to-failure data (training). Test histories stop before failure, so their
labels cannot be computed; they are what we predict.

Run from kode/:  python -m task2_rul.labels
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import plotstyle
from common.plotstyle import INK_2, MUTED, SERIES
from common.seed import FIG_DIR, RESULTS_DIR
from task2_rul.data import Preprocessor, load_raw

RUL_CAP = 150


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add `rul_raw` (uncapped) and `rul` (capped) columns. Training (run-to-failure) data only."""
    out = df.copy()
    failure_cycle = out.groupby("id_unit").siklus.transform("max")
    out["rul_raw"] = failure_cycle - out.siklus
    out["rul"] = out.rul_raw.clip(upper=cap)
    return out


def check_labels(df: pd.DataFrame, cap: int = RUL_CAP) -> None:
    """Assertions that the labels mean what we say they mean."""
    last = df.groupby("id_unit").tail(1)
    assert (last.rul == 0).all(), "RUL at the failure cycle must be 0"
    assert df.rul.between(0, cap).all(), "RUL must lie in [0, cap]"
    step = df.groupby("id_unit").rul_raw.diff().dropna()
    assert (step == -1).all(), "uncapped RUL must drop by exactly 1 per cycle"
    life = df.groupby("id_unit").siklus.transform("max")
    first = df.groupby("id_unit").head(1)
    assert (first.siklus == 1).all(), "histories must start at cycle 1"
    assert (first.rul_raw == life.loc[first.index] - 1).all(), "first-cycle RUL must equal life length - 1"


def main() -> dict:
    plotstyle.apply()
    out_dir = RESULTS_DIR / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = add_rul(load_raw("train"))
    check_labels(tr)
    print("[labels] checks passed: last cycle = 0, range [0, 150], drops by 1 per cycle")

    life = tr.groupby("id_unit").siklus.max()
    R = {
        "cap": RUL_CAP,
        "rows": len(tr),
        "share_rows_at_cap": round(float((tr.rul == RUL_CAP).mean()), 4),
        "share_rows_rul_le_50": round(float((tr.rul <= 50).mean()), 4),
        "share_machines_life_gt_cap": round(float((life > RUL_CAP).mean()), 4),
        "rows_per_rul_value_below_cap": int(((tr.rul < RUL_CAP).sum()) / RUL_CAP),
        "uncapped_rul_max": int(tr.rul_raw.max()),
    }
    with open(out_dir / "labels.json", "w") as f:
        json.dump(R, f, indent=2)

    # ---------- figure: label of one machine next to its wear signal, and the label histogram
    prep = Preprocessor().fit(load_raw("train"))
    tp = prep.transform(tr)
    unit = 11  # same example machine as in part (a)
    m = tr.id_unit == unit
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    fig.subplots_adjust(wspace=0.42)
    knee = int(tr.siklus[m].max() - RUL_CAP)  # cycle where the capped label starts to fall
    ax[0].plot(tr.siklus[m], tp.s04[m].rolling(15, min_periods=1).mean(), color=SERIES[0])
    ax[0].set(xlabel="cycle", ylabel="s04 (z), 15-cycle mean", title=f"(a) Wear signal, machine {unit}")
    for a in ax[:2]:
        a.axvline(knee, color=INK_2, lw=0.8, ls=":")
    ax[1].plot(tr.siklus[m], tr.rul_raw[m], color=MUTED, ls="--", label="uncapped")
    ax[1].plot(tr.siklus[m], tr.rul[m], color=SERIES[1], label="capped at 150 (used)")
    ax[1].set(xlabel="cycle", ylabel="RUL (cycles)", title="(b) RUL label, same machine")
    ax[1].legend(loc="upper right")
    ax[2].hist(tr.rul, bins=np.arange(0, RUL_CAP + 6, 5), color=SERIES[1])
    ax[2].set(xlabel="RUL label", ylabel="training rows", title="(c) All training labels")
    ax[2].annotate(f"{R['share_rows_at_cap']:.0%} of rows at the cap", xy=(RUL_CAP, ax[2].get_ylim()[1] * 0.9),
                   xytext=(25, ax[2].get_ylim()[1] * 0.75), fontsize=8, color=INK_2,
                   arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8))
    plotstyle.save(fig, FIG_DIR / "t2_labels")

    print(json.dumps(R, indent=1))
    return R


if __name__ == "__main__":
    main()
