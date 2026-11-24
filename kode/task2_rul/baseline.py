"""Task 2, part (d0): non-deep-learning baseline.

A classical regressor cannot read a W x F window directly, so each window is summarised by three
numbers per sensor (standard hand-made features for RUL):
  - recent level: mean of the last 5 cycles   -> how worn the machine is now
  - window mean                               -> a steadier version of the level
  - slope (least squares over the window)     -> how fast the sensor is changing
14 sensors x 3 = 42 features.

Models:
  - Ridge regression: linear, the simplest sensible yardstick.
  - Gradient-boosted trees (HistGradientBoosting): learns if-then rules, which suits the two
    failure modes (different sensors degrade in each). Fixed number of iterations, no internal
    early stopping: its internal split is by row, which would leak between rows of one machine.

Evaluated with the ruler in evaluate.py (held-out machines, test-like cuts).
Run from kode/:  python -m task2_rul.baseline
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import plotstyle
from common.plotstyle import INK_2, SERIES
from common.seed import FIG_DIR, RESULTS_DIR, set_seed
from task2_rul.evaluate import cut_windows, load_split, metrics
from task2_rul.labels import RUL_CAP
from task2_rul.windows import Windows, make_windows

W = 50  # same window length as the chosen deep models, so the comparison is fair
N_LAST = 5


def window_features(win: Windows, chunk: int = 20000) -> np.ndarray:
    """(n_windows, 3F): recent level, window mean, slope per sensor."""
    t = np.arange(win.W, dtype=np.float32) - (win.W - 1) / 2
    out = []
    for s in range(0, len(win), chunk):
        X = win.X(np.arange(s, min(s + chunk, len(win))))           # (n, W, F)
        last = X[:, -N_LAST:].mean(axis=1)
        mean = X.mean(axis=1)
        slope = (X * t[None, :, None]).sum(axis=1) / (t ** 2).sum()
        out.append(np.concatenate([last, mean, slope], axis=1))
    return np.vstack(out)


def make_models() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "boosted_trees": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                       early_stopping=False, random_state=0),
    }


def fit_predict(sp) -> tuple[dict, np.ndarray]:
    """Fit both baselines on a split; return ({name: predictions at held-out cuts}, true RUL)."""
    w_tr = make_windows(sp.train, sp.feats, W)
    w_cut = cut_windows(sp.val, sp.feats, W, sp.cuts)
    F_tr, F_cut = window_features(w_tr), window_features(w_cut)
    preds = {}
    for name, model in make_models().items():
        model.fit(F_tr, w_tr.y)
        preds[name] = np.clip(model.predict(F_cut), 0, RUL_CAP)
    return preds, w_cut.y


def feature_names(feats):
    return [f"{k}_{s}" for k in ("last5", "mean", "slope") for s in feats]


def main() -> dict:
    set_seed(0)
    plotstyle.apply()
    sp = load_split(seed=0)
    w_tr = make_windows(sp.train, sp.feats, W)
    w_cut = cut_windows(sp.val, sp.feats, W, sp.cuts)
    w_val_all = make_windows(sp.val, sp.feats, W)          # every validation window, for comparison
    F_tr, F_cut, F_all = window_features(w_tr), window_features(w_cut), window_features(w_val_all)
    print(f"[baseline] features: train {F_tr.shape}, val cuts {F_cut.shape}, val all windows {F_all.shape}")

    models = make_models()
    R = {"W": W, "features": feature_names(sp.feats), "models": {}}
    preds = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(F_tr, w_tr.y)
        fit_s = time.time() - t0
        p_cut = np.clip(model.predict(F_cut), 0, RUL_CAP)
        p_all = np.clip(model.predict(F_all), 0, RUL_CAP)
        p_tr = np.clip(model.predict(F_tr), 0, RUL_CAP)
        preds[name] = p_cut
        R["models"][name] = {
            "val_cuts": metrics(p_cut, w_cut.y),             # main number: test-like
            "val_all_windows": metrics(p_all, w_val_all.y),  # for comparison only
            "train_windows": metrics(p_tr, w_tr.y),          # to spot over/under-fitting
            "fit_seconds": round(fit_s, 1),
        }
        m = R["models"][name]
        c = m["val_cuts"]
        print(f"[baseline] {name:14s} val-cut RMSE {c['rmse']:6.2f}  score mean {c['score_mean']:6.2f} "
              f"median {c['score_median']:5.2f} worst5%-share {c['score_share_worst5pct']:.2f} "
              f"late>20 {c['share_late_over_20']:.3f} bias {c['bias']:+.1f} | all-window RMSE "
              f"{m['val_all_windows']['rmse']:6.2f} | train RMSE {m['train_windows']['rmse']:6.2f} | fit {fit_s:.1f}s")

    with open(RESULTS_DIR / "task2" / "baseline.json", "w") as f:
        json.dump(R, f, indent=2)

    # ---------- figure: predicted vs true RUL at the test-like cuts
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for a, (name, label) in zip(ax, [("ridge", "Ridge"), ("boosted_trees", "Boosted trees")]):
        a.scatter(w_cut.y, preds[name], s=8, alpha=0.5, color=SERIES[0], lw=0)
        a.plot([0, RUL_CAP], [0, RUL_CAP], color=INK_2, lw=1, ls="--")
        mm = R["models"][name]["val_cuts"]
        a.set(xlabel="true RUL", title=f"{label}: RMSE {mm['rmse']:.1f}, score/machine {mm['score_mean']:.2f}",
              xlim=(-5, 155), ylim=(-5, 155))
    ax[0].set_ylabel("predicted RUL")
    plotstyle.save(fig, FIG_DIR / "t2_baseline")
    return R


if __name__ == "__main__":
    main()
