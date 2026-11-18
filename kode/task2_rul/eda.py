"""Task 2, part (a): exploration of the RUL data. Writes results/task2/eda.json and figures.

Run from kode/:  python -m task2_rul.eda
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import plotstyle
from common.plotstyle import INK_2, MUTED, SERIES
from common.seed import FIG_DIR, RESULTS_DIR, set_seed
from task2_rul.data import SENSORS, SETTINGS, Preprocessor, load_raw, unit_trend

OUT = RESULTS_DIR / "task2"


def eta_squared(x: pd.Series, groups: np.ndarray) -> float:
    """Share of the variance of x explained by the group means."""
    d = pd.DataFrame({"x": x, "g": groups}).dropna()
    within = d.groupby("g").x.var(ddof=0).mul(d.groupby("g").size()).sum() / len(d)
    return float(1 - within / d.x.var(ddof=0))


def run_lengths(mask: pd.Series, ids: pd.Series) -> np.ndarray:
    """Lengths of consecutive True runs within each machine."""
    starts = mask & ~mask.groupby(ids).shift(1, fill_value=False)
    run_id = starts.cumsum().where(mask)
    return run_id.dropna().value_counts().to_numpy()


def main() -> dict:
    set_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    tr, te = load_raw("train"), load_raw("test")
    prep = Preprocessor().fit(tr)
    tr_p, info = prep.transform(tr, return_info=True)
    te_p, info_te = prep.transform(te, return_info=True)
    regime_tr, regime_te = tr_p.regime.to_numpy(), te_p.regime.to_numpy()
    z_raw = prep._zscore(tr, regime_tr)                       # per-regime z before cleaning
    cycles_left = tr.groupby("id_unit").siklus.transform("max") - tr.siklus
    rul = cycles_left.clip(upper=150)
    R = {}

    # ---------- 1. overview
    len_tr = tr.groupby("id_unit").size()
    len_te = te.groupby("id_unit").size()
    R["overview"] = {
        "train_rows": len(tr), "test_rows": len(te),
        "train_units": int(len_tr.size), "test_units": int(len_te.size),
        "train_length": len_tr.describe().round(1).to_dict(),
        "test_length": len_te.describe().round(1).to_dict(),
        "unit_overlap": len(set(tr.id_unit) & set(te.id_unit)),
    }

    # ---------- 2. operating regimes
    centres = pd.DataFrame(prep.kmeans_.cluster_centers_ * prep.set_scale_.to_numpy(), columns=SETTINGS)
    dist = prep.kmeans_.transform((tr[SETTINGS] / prep.set_scale_).to_numpy()).min(axis=1)
    switch = pd.Series(regime_tr).groupby(tr.id_unit.to_numpy()).apply(lambda r: (r.diff().iloc[1:] != 0).mean())
    eta = {s: round(eta_squared(tr[s], regime_tr), 3) for s in SENSORS}
    corr_raw = tr[SENSORS].corrwith(rul, method="spearman")
    corr_z = z_raw.corrwith(rul, method="spearman")
    R["regimes"] = {
        "centres": centres.round(3).to_dict(orient="index"),
        "max_dist_to_centre_std_units": round(float(dist.max()), 4),
        "share_train": pd.Series(regime_tr).value_counts(normalize=True).sort_index().round(3).to_dict(),
        "share_test": pd.Series(regime_te).value_counts(normalize=True).sort_index().round(3).to_dict(),
        "mean_regime_switch_rate": round(float(switch.mean()), 3),
        "eta2_regime": eta,
        "spearman_rul_raw": corr_raw.round(3).to_dict(),
        "spearman_rul_regime_z": corr_z.round(3).to_dict(),
        "within_regime_sd_ratio_max_min": (prep.sd_.max() / prep.sd_.min()).round(3).to_dict(),
    }

    # ---------- 3. uninformative sensors
    R["sensors"] = {
        "trend_mean_abs_change_z": prep.trend_.round(3).to_dict(),
        "threshold": prep.trend_threshold,
        "kept": prep.keep_,
        "dropped": [s for s in SENSORS if s not in prep.keep_],
    }

    # ---------- 4. missing values
    miss = info["missing"]
    runs = run_lengths(miss["s04"], tr.id_unit)
    by_regime = miss.groupby(regime_tr).mean().mean(axis=1)
    near = miss[cycles_left < 25].mean().mean()
    far = miss[cycles_left >= 150].mean().mean()
    p = float(miss.to_numpy().mean())
    n_per_row = miss.sum(axis=1)
    R["missing"] = {
        "rate_per_sensor_train": miss.mean().round(4).to_dict(),
        "rate_overall_train": round(p, 4),
        "rate_overall_test": round(float(info_te["missing"].to_numpy().mean()), 4),
        "rate_test_last_cycle": round(float(info_te["missing"].groupby(te.id_unit).tail(1).to_numpy().mean()), 4),
        "rows_with_any_nan": round(float((n_per_row > 0).mean()), 4),
        "rows_with_2plus_nan": round(float((n_per_row >= 2).mean()), 4),
        "rows_with_2plus_nan_if_independent": round(1 - (1 - p) ** 21 - 21 * p * (1 - p) ** 20, 4),
        "s04_run_length_share_1": round(float((runs == 1).mean()), 4),
        "s04_max_run_length": int(runs.max()),
        "rate_by_regime_min_max": [round(float(by_regime.min()), 4), round(float(by_regime.max()), 4)],
        "rate_last25_cycles": round(float(near), 4),
        "rate_150plus_cycles_left": round(float(far), 4),
    }

    # ---------- 5. spikes
    spike = info["spike"]
    resid_sd = info["residual"] / prep.noise_sd_
    sp_vals = resid_sd[spike].stack().dropna()  # pandas 3 keeps NaN in stack(); drop unflagged cells
    sp_z = info["residual"][spike].stack().dropna()
    R["spikes"] = {
        "rule": f"|x_t - median(previous {prep.window})| > {prep.k} * robust noise sd",
        "rate_per_sensor_train": spike.mean().round(5).to_dict(),
        "rate_overall_train": round(float(spike.to_numpy().mean()), 5),
        "rate_overall_test": round(float(info_te["spike"].to_numpy().mean()), 5),
        "share_positive": round(float((sp_vals > 0).mean()), 4),
        "size_noise_sd_median": round(float(sp_vals.abs().median()), 2),
        "size_z_median": round(float(sp_z.abs().median()), 2),
        "consecutive_flagged": int((spike & spike.groupby(tr.id_unit).shift(1, fill_value=False)).to_numpy().sum()),
    }

    # ---------- 6. failure modes (kept sensors, cleaned data)
    zc = tr_p[prep.keep_]
    delta = unit_trend(zc, tr.id_unit, cycles_left)
    mode = KMeans(2, n_init=10, random_state=0).fit_predict((delta / delta.std()).to_numpy())
    mode = pd.Series(mode, index=delta.index)
    R["failure_modes"] = {
        "sizes": mode.value_counts().sort_index().to_dict(),
        "mean_change_per_mode": delta.groupby(mode).mean().round(2).to_dict(orient="index"),
        "life_length_per_mode": len_tr.groupby(mode).describe()[["mean", "min", "max"]].round(0).to_dict(orient="index"),
    }
    R["init_offset_sd_between_units"] = zc[tr.groupby("id_unit").cumcount() < 20].groupby(
        tr.id_unit).mean().std().round(3).to_dict()

    with open(OUT / "eda.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # =============== figures
    # example machine: median-ish length, with spikes in s04
    cand = len_tr[(len_tr > 220) & (len_tr < 320)].index
    n_sp = spike["s04"].groupby(tr.id_unit).sum().loc[cand]
    unit = int(n_sp.idxmax())
    m = (tr.id_unit == unit).to_numpy()
    cyc = tr.siklus[m].to_numpy()

    # Fig 1: regimes and per-regime normalisation
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    sub = tr.sample(6000, random_state=0)
    reg_sub = regime_tr[sub.index]
    for r in range(prep.n_regimes):
        k = reg_sub == r
        ax[0].scatter(sub.setel_1[k], sub.setel_2[k], s=10, color=SERIES[r], label=f"R{r}")
    ax[0].set(xlabel="setel_1", ylabel="setel_2", title="(a) Six operating regimes")
    ax[0].legend(ncol=2, loc="lower right", handletextpad=0.2)
    for r in range(prep.n_regimes):
        k = regime_tr[m] == r
        ax[1].scatter(cyc[k], tr.s04[m][k], s=8, color=SERIES[r])
    ax[1].set(xlabel="cycle", ylabel="s04 (raw)", title=f"(b) Raw s04, machine {unit}")
    zz = tr_p.s04[m]
    ax[2].plot(cyc, zz, color=SERIES[0], lw=0.8, alpha=0.6, label="per-regime z")
    ax[2].plot(cyc, zz.rolling(15, min_periods=1).mean(), color=INK_2, lw=1.8, label="15-cycle mean")
    ax[2].set(xlabel="cycle", ylabel="s04 (z)", title="(c) After per-regime z-score")
    ax[2].legend(loc="upper left")
    plotstyle.save(fig, FIG_DIR / "t2_regimes")

    # Fig 2: sensor informativeness
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.0))
    x = np.arange(len(SENSORS))
    ax[0].bar(x - 0.2, corr_raw.abs(), 0.38, color=MUTED, label="raw")
    ax[0].bar(x + 0.2, corr_z.abs(), 0.38, color=SERIES[0], label="per-regime z")
    ax[0].set(xticks=x, ylabel="|Spearman ρ| with RUL", title="(a) Correlation with capped RUL")
    ax[0].set_xticklabels(SENSORS, rotation=90)
    ax[0].legend()
    colors = [SERIES[0] if s in prep.keep_ else MUTED for s in SENSORS]
    ax[1].bar(x, prep.trend_, 0.7, color=colors)
    ax[1].axhline(prep.trend_threshold, color=INK_2, lw=1, ls="--", label=f"threshold {prep.trend_threshold}")
    ax[1].legend(loc="upper right")
    ax[1].set(xticks=x, ylabel="mean |last20 − first20| (z)", title="(b) Degradation trend (grey = dropped)")
    ax[1].set_xticklabels(SENSORS, rotation=90)
    plotstyle.save(fig, FIG_DIR / "t2_sensors")

    # Fig 3: missing values and spikes
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.0))
    zr = z_raw.s04[m].to_numpy()
    ax[0].plot(cyc, zr, color=MUTED, lw=0.8, label="before cleaning")
    ax[0].plot(cyc, tr_p.s04[m], color=SERIES[0], lw=0.9, label="after cleaning")
    sk = spike.s04[m].to_numpy()
    ax[0].scatter(cyc[sk], zr[sk], s=30, color=SERIES[1], zorder=3, label="spike")
    mk = miss.s04[m].to_numpy()
    ax[0].scatter(cyc[mk], np.full(mk.sum(), np.nanmin(zr) - 0.5), marker="|", s=40, color=SERIES[6],
                  label="missing")
    ax[0].set(xlabel="cycle", ylabel="s04 (z)", title=f"(a) Machine {unit}: spikes and gaps")
    ax[0].legend(loc="upper left", ncol=2)
    allres = resid_sd.to_numpy().ravel()
    allres = allres[np.isfinite(allres)]
    bins = np.linspace(-25, 10, 141)
    ax[1].hist(allres, bins=bins, color=SERIES[0])
    ax[1].axvline(-prep.k, color=INK_2, lw=1, ls="--")
    ax[1].axvline(prep.k, color=INK_2, lw=1, ls="--")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="residual vs trailing median (noise sd)", ylabel="count (log)",
              title="(b) Residuals, all sensors: spikes are the side bumps")
    plotstyle.save(fig, FIG_DIR / "t2_missing_spikes")

    # Fig 4: failure modes
    show = ["s11", "s12", "s13"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=True)
    mode_row = tr.id_unit.map(mode)
    for j, s in enumerate(show):
        for k in range(2):
            sel = (mode_row == k) & (cycles_left <= 250)
            curve = zc[s][sel].groupby(cycles_left[sel]).mean()
            ax[j].plot(curve.index, curve.values, color=SERIES[k],
                       label=f"mode {k} (n={int((mode == k).sum())})")
        ax[j].invert_xaxis()
        ax[j].set(xlabel="cycles before failure", title=s)
    ax[0].set_ylabel("mean z")
    ax[0].legend(loc="lower left")
    plotstyle.save(fig, FIG_DIR / "t2_failure_modes")

    print(json.dumps({k: R[k] for k in ["sensors", "failure_modes"]}, indent=1, default=str)[:1500])
    return R


if __name__ == "__main__":
    main()
