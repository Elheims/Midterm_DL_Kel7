"""Task 2: the "ruler" every model is measured with (machine split, test-like cuts, metrics).

Built before the models so that the baseline and the deep models are judged identically.
The full validation analysis (5-fold, per-RUL-range tables) is part (e).

1. Split by machine, not by row: 80% of the training machines train the model, 20% are held out
   completely (also from the preprocessing fit). Rows of one machine are near-duplicates, so a
   row split would let the model "recognise" a machine instead of learning wear.
2. Test-like cuts: test histories stop at a random cycle before failure. For every held-out
   machine we draw `n_cuts` cut cycles uniformly in [50, T-1] (50 = shortest test history,
   T = failure cycle) and predict RUL at each cut, from the window ending there. Because the
   preprocessing is causal (checked in part a), that window is identical to the one we would
   get by preprocessing the cut history on its own.
3. Metrics: RMSE, and the exam's asymmetric score with d = predicted - true:
       s = exp(-d/13) - 1 if d < 0 (early),  exp(d/10) - 1 if d >= 0 (late)
   summed over machines (as on the test) and averaged per prediction (comparable across sizes).

Run from kode/:  python -m task2_rul.evaluate     (checks that the cuts resemble the test)
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.seed import RESULTS_DIR
from task2_rul.data import Preprocessor, load_raw
from task2_rul.labels import RUL_CAP, add_rul
from task2_rul.windows import Windows, make_windows

MIN_CUT = 50  # shortest test history


def split_machines(units, val_frac: float = 0.2, seed: int = 0):
    """Random machine-level split. Returns (train_units, val_units), both sorted."""
    u = np.array(sorted(units))
    np.random.default_rng(seed).shuffle(u)
    n_val = int(round(len(u) * val_frac))
    return np.sort(u[n_val:]), np.sort(u[:n_val])


def kfold_machines(units, fold: int, n_folds: int = 5, seed: int = 0):
    """Machine-level K-fold: shuffle the machines once, cut into n_folds groups; group `fold` is held out.
    Every machine is held out in exactly one fold."""
    u = np.array(sorted(units))
    np.random.default_rng(seed).shuffle(u)
    parts = np.array_split(u, n_folds)
    va = parts[fold]
    tr = np.concatenate([p for k, p in enumerate(parts) if k != fold])
    return np.sort(tr), np.sort(va)


# true-RUL ranges for the per-range tables (150 = capped = still healthy)
RUL_RANGES = [("0-25", 0, 25), ("26-50", 26, 50), ("51-100", 51, 100), ("101-149", 101, 149), ("150", 150, 150)]


def metrics_by_range(pred, true) -> dict:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = {}
    for name, lo, hi in RUL_RANGES:
        k = (true >= lo) & (true <= hi)
        m = metrics(pred[k], true[k])
        out[name] = {key: m[key] for key in ("rmse", "score_mean", "bias", "n")}
    return out


def test_like_cuts(raw: pd.DataFrame, n_cuts: int = 5, seed: int = 0, min_cut: int = MIN_CUT) -> pd.DataFrame:
    """For each machine: n_cuts cut cycles uniform in [min_cut, T-1] and the capped RUL at each cut."""
    rng = np.random.default_rng(seed)
    rows = []
    for u, T in raw.groupby("id_unit").siklus.max().items():
        for c in rng.integers(min_cut, T, size=n_cuts):
            rows.append((u, int(c), min(RUL_CAP, int(T - c))))
    return pd.DataFrame(rows, columns=["id_unit", "cut", "rul"])


def asym_score(pred, true) -> np.ndarray:
    d = np.asarray(pred, float) - np.asarray(true, float)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def metrics(pred, true) -> dict:
    pred = np.clip(np.asarray(pred, float), 0, RUL_CAP)
    true = np.asarray(true, float)
    s = asym_score(pred, true)
    worst = np.sort(s)[::-1][: max(1, int(round(0.05 * len(s))))]
    d = pred - true
    return {"rmse": round(float(np.sqrt(np.mean((pred - true) ** 2))), 3),
            "score_sum": round(float(s.sum()), 1),
            "score_mean": round(float(s.mean()), 3),
            "score_median": round(float(np.median(s)), 3),
            "score_share_worst5pct": round(float(worst.sum() / s.sum()), 3),  # heavy tail: late misses
            "share_late_over_20": round(float((d > 20).mean()), 3),
            "bias": round(float(np.mean(d)), 3),
            "n": int(len(true))}


@dataclass
class Split:
    prep: Preprocessor
    feats: list
    train: pd.DataFrame      # preprocessed training-machine rows with `rul`
    val: pd.DataFrame        # preprocessed held-out-machine rows with `rul`
    cuts: pd.DataFrame       # test-like cut points on the held-out machines
    train_units: np.ndarray
    val_units: np.ndarray


def load_split(seed: int = 0, val_frac: float = 0.2, n_cuts: int = 5, fold: int | None = None,
               n_folds: int = 5) -> Split:
    """fold=None: the fixed 80/20 ruler split. fold=k: fold k of a 5-fold machine-level split."""
    raw = load_raw("train")
    if fold is None:
        tr_u, va_u = split_machines(raw.id_unit.unique(), val_frac, seed)
    else:
        tr_u, va_u = kfold_machines(raw.id_unit.unique(), fold, n_folds, seed)
    raw_tr = raw[raw.id_unit.isin(tr_u)].reset_index(drop=True)
    raw_va = raw[raw.id_unit.isin(va_u)].reset_index(drop=True)
    prep = Preprocessor().fit(raw_tr)                       # held-out machines never seen here
    return Split(prep=prep, feats=prep.keep_,
                 train=add_rul(prep.transform(raw_tr)), val=add_rul(prep.transform(raw_va)),
                 cuts=test_like_cuts(raw_va, n_cuts=n_cuts, seed=seed),
                 train_units=tr_u, val_units=va_u)


def cut_windows(val: pd.DataFrame, feats: list, W: int, cuts: pd.DataFrame) -> Windows:
    """Windows ending at the cut cycles; target = capped RUL at the cut."""
    w = make_windows(val, feats, W, first_end=1)
    have = pd.MultiIndex.from_arrays([w.unit, w.cycle])
    pos = have.get_indexer(pd.MultiIndex.from_arrays([cuts.id_unit.to_numpy(), cuts.cut.to_numpy()]))
    assert (pos >= 0).all()
    out = dataclasses.replace(w, ends=w.ends[pos], unit=w.unit[pos], cycle=w.cycle[pos], y=w.y[pos])
    assert np.array_equal(out.y, cuts.rul.to_numpy(np.float32)), "cut target must equal capped RUL"
    return out


def main() -> dict:
    """Check that the test-like cuts resemble the real test set."""
    sp = load_split()
    te = load_raw("test")
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    cut_len = sp.cuts.cut.quantile(q).round(0)
    test_len = te.groupby("id_unit").size().quantile(q).round(0)
    R = {
        "train_machines": len(sp.train_units), "val_machines": len(sp.val_units),
        "overlap": len(set(sp.train_units) & set(sp.val_units)),
        "val_cut_points": len(sp.cuts),
        "history_length_quantiles": {"q": q, "val_cuts": cut_len.tolist(), "test": test_len.tolist()},
        "val_cut_share_rul_at_cap": round(float((sp.cuts.rul == RUL_CAP).mean()), 3),
        "val_cut_share_rul_le_50": round(float((sp.cuts.rul <= 50).mean()), 3),
    }
    assert R["overlap"] == 0
    (RESULTS_DIR / "task2").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "task2" / "evaluate.json", "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=1))
    return R


if __name__ == "__main__":
    main()
