"""Task 2: final model and prediksi_rul.csv.

Final model = average of 5 GRUs (W=50, seeds 0-4), the best model in every test of parts (d)-(e).

1. Score-aware shift. The asymmetric score punishes late (too high) predictions more than early
   ones, so lowering every prediction by delta cycles lowers the score at some RMSE cost.
   choose_shift() reports both metrics for delta = 0..20 on the 72 held-aside machines and the 180
   held-out machines of the validation setup (5 GRUs trained on 648 machines).
   shift_simulation() makes the decision: 2,000 simulated test sets of 300 validation machines
   (one random cut each, like the real test); each metric's change is divided by its natural spread
   between test sets, so both metrics (3 points each) are in the same unit; the shift with the
   largest net benefit (score gain - RMSE cost) is used. The real test set is never used.
2. Retraining (train_final, --train). Same recipe on all 900 training machines: preprocessing and
   sensor selection fitted on all 900; 90 machines (10%) held aside for early stopping; 810 machines
   give the training windows; 5 seeds. Weights are saved in task2_rul/final_weights/ (kept in git).
3. Prediction (predict.py): load the preprocessing (refitted, deterministic) and the 5 final GRUs,
   predict at the last recorded cycle of each test machine, subtract delta, clip to [0, 150],
   write prediksi_rul.csv in the repository root and check its format.

Run from kode/:  python -m task2_rul.final            (shift + retraining as a separate process + predict)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.seed import RESULTS_DIR, set_seed
from task2_rul.data import Preprocessor, load_raw
from task2_rul.deep import CKPT, prepare, run_isolated
from task2_rul.evaluate import Split, asym_score, load_split, metrics, split_machines, test_like_cuts
from task2_rul.labels import RUL_CAP, add_rul
from task2_rul.models import build
from task2_rul.train_nn import DEVICE, GPUWindows, TrainConfig, fit, predict

OUT = RESULTS_DIR / "task2"
FINAL_DIR = Path(__file__).resolve().parent / "final_weights"
ARCH, W, SEEDS5 = "gru", 50, (0, 1, 2, 3, 4)
DELTAS = np.arange(0, 21, 1)


def ensemble_predict(paths, n_feat: int, gw: GPUWindows) -> np.ndarray:
    preds = []
    for p in paths:
        m = build(ARCH, n_feat, W).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        preds.append(predict(m, gw))
    return np.mean(preds, axis=0)


def choose_shift() -> dict:
    set_seed(0)
    sp = load_split(seed=0)
    _, data, _ = prepare(W, sp=sp)
    paths = [CKPT / f"gru_W50_seed{s}.pt" for s in SEEDS5]
    p_stop = ensemble_predict(paths, len(sp.feats), GPUWindows(data["stop"]))
    p_test = ensemble_predict(paths, len(sp.feats), GPUWindows(data["ruler"]))
    y_stop, y_test = data["stop"].y, data["ruler"].y
    curve = []
    for d in DELTAS:
        a = metrics(np.clip(p_stop - d, 0, RUL_CAP), y_stop)
        b = metrics(np.clip(p_test - d, 0, RUL_CAP), y_test)
        curve.append({"delta": int(d), "stop_score_mean": a["score_mean"], "stop_rmse": a["rmse"],
                      "heldout_score_mean": b["score_mean"], "heldout_rmse": b["rmse"]})
    best = min(curve, key=lambda c: c["stop_score_mean"])
    R = {"note": "shift curve for the report; the shift actually used is chosen by shift_simulation()",
         "curve_on": "72 held-aside machines (720 cuts) and 180 held-out machines (900 cuts)",
         "score_only_optimal_delta": best["delta"], "no_shift": curve[0], "score_only_optimum": best,
         "curve": curve}
    with open(OUT / "final_shift.json", "w") as f:
        json.dump(R, f, indent=2)
    print(f"[final] shift curve: score alone would want delta = {best['delta']} (held-out RMSE "
          f"{best['heldout_rmse']:.2f} vs {curve[0]['heldout_rmse']:.2f} unshifted) - not used; "
          f"the decision weighs both metrics (simulation)")
    return R


SIM_SHIFTS = tuple(range(0, 14))
N_SIM, N_TEST = 2000, 300


def shift_simulation() -> dict:
    """How likely does each shift help on a test set like the real one?

    Pool all validation machines (72 held-aside + 180 held-out = 252, all unseen by the 5 GRUs'
    weights) with their test-like cuts. One simulated test set = 300 machines drawn with replacement,
    one random cut each (like the real test). For each shift compute the two exam metrics; repeat
    2,000 times and compare every shift with no shift.
    """
    set_seed(0)
    sp = load_split(seed=0)
    _, data, _ = prepare(W, sp=sp)
    paths = [CKPT / f"gru_W50_seed{s}.pt" for s in SEEDS5]
    pred = np.concatenate([ensemble_predict(paths, len(sp.feats), GPUWindows(data[k])) for k in ("stop", "ruler")])
    true = np.concatenate([data[k].y for k in ("stop", "ruler")]).astype(float)
    unit = np.concatenate([data[k].unit for k in ("stop", "ruler")])
    units = np.unique(unit)
    rows_of = {u: np.flatnonzero(unit == u) for u in units}
    rng = np.random.default_rng(0)
    res = {d: {"rmse": [], "score_sum": []} for d in SIM_SHIFTS}
    for _ in range(N_SIM):
        pick = np.array([rng.choice(rows_of[u]) for u in rng.choice(units, N_TEST)])
        for d in SIM_SHIFTS:
            p = np.clip(pred[pick] - d, 0, RUL_CAP)
            res[d]["rmse"].append(float(np.sqrt(np.mean((p - true[pick]) ** 2))))
            res[d]["score_sum"].append(float(asym_score(p, true[pick]).sum()))
    base = {k: np.array(res[0][k]) for k in ("rmse", "score_sum")}
    # natural spread of each metric between test sets (no shift) -> common unit for both metrics
    sd = {k: float(np.std(base[k], ddof=1)) for k in base}
    R = {"machines_pooled": len(units), "cuts_pooled": len(true), "simulations": N_SIM, "test_size": N_TEST,
         "spread_between_test_sets_no_shift": {k: round(v, 2) for k, v in sd.items()},
         "criterion": "net benefit = score improvement / spread(score) - RMSE increase / spread(RMSE), "
                      "medians over simulated test sets; both metrics weighted equally (3 points each)",
         "shifts": {}}
    for d in SIM_SHIFTS:
        r, s = np.array(res[d]["rmse"]), np.array(res[d]["score_sum"])
        R["shifts"][d] = {
            "rmse_median": round(float(np.median(r)), 2),
            "score_sum_median": round(float(np.median(s)), 0),
            "rmse_change_vs_0": round(float(np.median(r - base["rmse"])), 2),
            "score_change_vs_0_pct": round(float(np.median(100 * (s / base["score_sum"] - 1))), 1),
            "p_rmse_better_than_0": round(float(np.mean(r < base["rmse"])), 3),
            "p_score_better_than_0": round(float(np.mean(s < base["score_sum"])), 3),
            "score_sum_p90": round(float(np.percentile(s, 90)), 0),
            "rmse_cost_in_spreads": round(float(np.median(r - base["rmse"]) / sd["rmse"]), 3),
            "score_gain_in_spreads": round(float(np.median(base["score_sum"] - s) / sd["score_sum"]), 3),
        }
        v = R["shifts"][d]
        v["net_benefit"] = round(v["score_gain_in_spreads"] - v["rmse_cost_in_spreads"], 3)
    R["best_shift_by_net_benefit"] = max(R["shifts"], key=lambda d: R["shifts"][d]["net_benefit"])
    with open(OUT / "final_shift_simulation.json", "w") as f:
        json.dump(R, f, indent=2)
    print(f"[final] simulated {N_SIM} test sets of {N_TEST} machines (pool: {len(units)} validation machines); "
          f"spread between test sets without shift: RMSE {sd['rmse']:.2f}, score sum {sd['score_sum']:.0f}")
    print("  shift | RMSE median (change) | P(RMSE better) | score sum median (change) | P(score better) | "
          "RMSE cost | score gain | net (in spreads)")
    for d, v in R["shifts"].items():
        print(f"  {d:5d} | {v['rmse_median']:6.2f} ({v['rmse_change_vs_0']:+.2f})       | {v['p_rmse_better_than_0']:13.0%} | "
              f"{v['score_sum_median']:8.0f} ({v['score_change_vs_0_pct']:+.0f}%)         | {v['p_score_better_than_0']:14.0%} | "
              f"{v['rmse_cost_in_spreads']:9.2f} | {v['score_gain_in_spreads']:10.2f} | {v['net_benefit']:+.2f}")
    print(f"[final] best shift by net benefit: {R['best_shift_by_net_benefit']}")
    return R


def final_split() -> tuple[Preprocessor, Split]:
    """All 900 machines on the training side (no ruler); preprocessing fitted on all of them."""
    raw = load_raw("train")
    prep = Preprocessor().fit(raw)
    tr = add_rul(prep.transform(raw))
    sp = Split(prep=prep, feats=prep.keep_, train=tr, val=tr.iloc[:0], cuts=None,
               train_units=np.sort(raw.id_unit.unique()), val_units=np.array([], int))
    return prep, sp


def train_final() -> dict:
    from task2_rul.evaluate import cut_windows
    from task2_rul.windows import make_windows

    FINAL_DIR.mkdir(exist_ok=True)
    prep, sp = final_split()
    fit_u, stop_u = split_machines(sp.train_units, val_frac=0.1, seed=0)
    tr_fit = sp.train[sp.train.id_unit.isin(fit_u)].reset_index(drop=True)
    tr_stop = sp.train[sp.train.id_unit.isin(stop_u)].reset_index(drop=True)
    g_train = GPUWindows(make_windows(tr_fit, sp.feats, W))
    w_stop = cut_windows(tr_stop, sp.feats, W, test_like_cuts(tr_stop, n_cuts=10, seed=0))
    g_stop = GPUWindows(w_stop)
    R = {"fit_machines": len(fit_u), "stop_machines": len(stop_u), "kept_sensors": sp.feats, "seeds": {}}
    for s in SEEDS5:
        set_seed(s)
        model, hist = fit(build(ARCH, len(sp.feats), W), g_train, g_stop, TrainConfig(), seed=s,
                          log=lambda *a: None)
        torch.save(model.state_dict(), FINAL_DIR / f"gru_W50_seed{s}.pt")
        R["seeds"][s] = {"best_stop_rmse": min(h["stop_rmse"] for h in hist),
                         "best_epoch": int(np.argmin([h["stop_rmse"] for h in hist])) + 1}
        print(f"[final] trained seed {s} on {len(fit_u)} machines: early-stop RMSE "
              f"{R['seeds'][s]['best_stop_rmse']:.2f} (epoch {R['seeds'][s]['best_epoch']})", flush=True)
    R["ensemble_stop"] = metrics(ensemble_predict([FINAL_DIR / f"gru_W50_seed{s}.pt" for s in SEEDS5],
                                                  len(sp.feats), g_stop), w_stop.y)
    with open(OUT / "final_train.json", "w") as f:
        json.dump(R, f, indent=2)
    return R


def main() -> None:
    from task2_rul import predict as t2_predict

    choose_shift()          # the shift curve on held-aside and held-out machines (for the report)
    shift_simulation()      # the decision: shift with the best net benefit over 2,000 simulated test sets
    files = [OUT / "final_train.json"] + [FINAL_DIR / f"gru_W50_seed{s}.pt" for s in SEEDS5]
    run_isolated("task2_rul.final", "--train", files, "final_train")
    t2_predict.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--shift-only", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_final()
    elif a.shift_only:
        choose_shift()
    elif a.simulate:
        shift_simulation()
    else:
        main()
