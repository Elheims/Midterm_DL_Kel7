"""Task 2, part (d): train and evaluate one deep architecture end to end.

Steps (same for every architecture):
1. Data: the ruler split (720 training / 180 ruler machines). From the 720, 72 machines are held
   aside for early stopping (10 test-like cuts each); the other 648 provide the training windows.
   The 180 ruler machines are only used for the final numbers.
2. Sanity check: overfit one batch of 64 windows (must reach ~0 error).
3. Train with 3 seeds; learning curves; evaluate each seed on the ruler cuts; mean +- std.
   The average of the 3 seeds' predictions (a small ensemble) is reported too (used in part f).

Run from kode/:  python -m task2_rul.deep --arch cnn --W 30
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import plotstyle
from common.plotstyle import INK_2, MUTED, SERIES
from common.seed import FIG_DIR, RESULTS_DIR, SEEDS, set_seed
from task2_rul.evaluate import cut_windows, load_split, metrics, split_machines, test_like_cuts
from task2_rul.labels import RUL_CAP
from task2_rul.models import build, n_params
from task2_rul.train_nn import DEVICE, GPUWindows, TrainConfig, fit, overfit_one_batch, predict
from task2_rul.windows import make_windows

OUT = RESULTS_DIR / "task2"
CKPT = OUT / "checkpoints"


def prepare(W: int, seed: int = 0, sp=None):
    sp = sp if sp is not None else load_split(seed=0)          # default: same ruler for every model
    fit_u, stop_u = split_machines(sp.train_units, val_frac=0.1, seed=seed)
    tr_fit = sp.train[sp.train.id_unit.isin(fit_u)].reset_index(drop=True)
    tr_stop = sp.train[sp.train.id_unit.isin(stop_u)].reset_index(drop=True)
    stop_cuts = test_like_cuts(tr_stop, n_cuts=10, seed=seed)
    data = {
        "train": make_windows(tr_fit, sp.feats, W),
        "stop": cut_windows(tr_stop, sp.feats, W, stop_cuts),
        "ruler": cut_windows(sp.val, sp.feats, W, sp.cuts),
    }
    return sp, data, {"fit_machines": len(fit_u), "stop_machines": len(stop_u)}


def run(arch: str = "cnn", W: int = 30, seeds=SEEDS, cfg: TrainConfig | None = None, tag: str | None = None,
        model_kw: dict | None = None) -> dict:
    cfg = cfg or TrainConfig()
    model_kw = model_kw or {}
    tag = tag or f"{arch}_W{W}"
    plotstyle.apply()
    CKPT.mkdir(parents=True, exist_ok=True)
    sp, data, sizes = prepare(W)
    F = len(sp.feats)
    g = {k: GPUWindows(v) for k, v in data.items()}
    P = n_params(build(arch, F, W, **model_kw))
    print(f"[{tag}] device {DEVICE}; windows: train {len(g['train'])}, early-stop cuts {len(g['stop'])}, "
          f"ruler cuts {len(g['ruler'])}; params {P}; model {model_kw}; train config {cfg.__dict__}")

    # ---- 1. sanity check: overfit one batch
    set_seed(0)
    curve = overfit_one_batch(build(arch, F, W, **model_kw), g["train"])
    # Adam briefly spikes once the loss is near zero, so judge the best error of the last 100 steps
    reached = min(curve[-100:])
    print(f"[{tag}] overfit one batch: RMSE {curve[0]:.1f} -> {reached:.3f} cycles (best of last 100 of "
          f"{len(curve)} steps)")
    assert reached < 2.0, "network cannot memorise 64 windows: likely a bug"

    # ---- 2. three seeds
    R = {"arch": arch, "W": W, "model_kw": model_kw, "config": cfg.__dict__, "sizes": sizes, "n_params": P,
         "overfit_one_batch": {"start_rmse": round(curve[0], 2), "best_last100_rmse": round(reached, 4),
                               "last_step_rmse": round(curve[-1], 4), "steps": len(curve)},
         "seeds": {}}
    preds, hists = [], {}
    for s in seeds:
        set_seed(s)
        model = build(arch, F, W, **model_kw)
        t0 = time.time()
        print(f"[{tag}] seed {s}")
        model, hist = fit(model, g["train"], g["stop"], cfg, seed=s)
        p = predict(model, g["ruler"])
        preds.append(p)
        hists[s] = hist
        torch.save(model.state_dict(), CKPT / f"{tag}_seed{s}.pt")
        best_ep = int(np.argmin([h["stop_rmse"] for h in hist])) + 1
        R["seeds"][s] = {"ruler": metrics(p, data["ruler"].y), "best_epoch": best_ep, "epochs_run": len(hist),
                         "train_minutes": round((time.time() - t0) / 60, 2), "history": hist}
        m = R["seeds"][s]["ruler"]
        print(f"[{tag}] seed {s}: ruler RMSE {m['rmse']:.2f}  score mean {m['score_mean']:.2f}  "
              f"median {m['score_median']:.2f}  best epoch {best_ep}/{len(hist)}")

    keys = ["rmse", "score_mean", "score_median", "score_share_worst5pct", "share_late_over_20", "bias"]
    R["summary"] = {k: {"mean": round(float(np.mean([R["seeds"][s]["ruler"][k] for s in seeds])), 3),
                        "std": round(float(np.std([R["seeds"][s]["ruler"][k] for s in seeds], ddof=1)), 3)}
                    for k in keys}
    R["ensemble_of_seeds"] = metrics(np.mean(preds, axis=0), data["ruler"].y)
    R["stop_rmse_best"] = {"mean": round(float(np.mean([min(h["stop_rmse"] for h in hists[s]) for s in seeds])), 3)}
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(R, f, indent=2)
    S = R["summary"]
    print(f"[{tag}] 3 seeds: RMSE {S['rmse']['mean']:.2f} ± {S['rmse']['std']:.2f}, score mean "
          f"{S['score_mean']['mean']:.1f} ± {S['score_mean']['std']:.1f}; ensemble RMSE {R['ensemble_of_seeds']['rmse']:.2f}")

    # ---- 3. figure: sanity check, learning curves, predicted vs true
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    fig.subplots_adjust(wspace=0.35)
    ax[0].plot(curve, color=SERIES[0])
    ax[0].set(yscale="log", xlabel="step", ylabel="RMSE on the batch (cycles)", title="(a) Overfit one batch (64 windows)")
    for k, s in enumerate(seeds):
        h = hists[s]
        ep = [x["epoch"] for x in h]
        ax[1].plot(ep, [x["train_rmse"] for x in h], color=SERIES[k], ls="--", lw=1.2)
        ax[1].plot(ep, [x["stop_rmse"] for x in h], color=SERIES[k], label=f"seed {s}")
    ax[1].plot([], [], color=MUTED, ls="--", label="train (dropout on)")
    ax[1].plot([], [], color=MUTED, label="early-stop machines")
    ax[1].set(xlabel="epoch", ylabel="RMSE (cycles)", title="(b) Learning curves")
    ax[1].legend(fontsize=7, ncol=1)
    ax[2].scatter(data["ruler"].y, preds[0], s=8, alpha=0.5, color=SERIES[0], lw=0)
    ax[2].plot([0, RUL_CAP], [0, RUL_CAP], color=INK_2, lw=1, ls="--")
    m0 = R["seeds"][seeds[0]]["ruler"]
    ax[2].set(xlabel="true RUL", ylabel="predicted RUL", xlim=(-5, 155), ylim=(-5, 155),
              title=f"(c) Ruler cuts, seed {seeds[0]}: RMSE {m0['rmse']:.1f}")
    plotstyle.save(fig, FIG_DIR / f"t2_{tag}")
    return R


def train_predict(arch: str, W: int, sp, seed: int, model_kw: dict | None = None,
                  cfg: TrainConfig | None = None):
    """Same training as run(), on any split, without figures or checkpoints (used for cross-validation).
    Returns (predictions, true RUL) at the held-out machines' test-like cuts."""
    _, data, _ = prepare(W, sp=sp)
    g = {k: GPUWindows(v) for k, v in data.items()}
    set_seed(seed)
    model = build(arch, len(sp.feats), W, **(model_kw or {}))
    model, _ = fit(model, g["train"], g["stop"], cfg or TrainConfig(), seed=seed, log=lambda *a: None)
    return predict(model, g["ruler"]), data["ruler"].y


# CNN variants: the original plus three standard fixes for the early overfitting, one change each.
CNN_VARIANTS = {
    "cnn_W30":     dict(W=30, model_kw={}, cfg=TrainConfig()),                                   # original
    "cnn_W30_gap": dict(W=30, model_kw={"head": "gap"}, cfg=TrainConfig()),                      # fix 1: smaller head
    "cnn_W30_reg": dict(W=30, model_kw={"dropout": 0.5}, cfg=TrainConfig(weight_decay=1e-4)),    # fix 2: regularisation
    "cnn_W50":     dict(W=50, model_kw={}, cfg=TrainConfig()),                                   # fix 3: longer window
}


# GRU: same protocol, both window lengths; the window is chosen on the early-stop machines.
GRU_VARIANTS = {
    "gru_W30": dict(W=30, model_kw={}, cfg=TrainConfig()),
    "gru_W50": dict(W=50, model_kw={}, cfg=TrainConfig()),
}


def run_variants(arch: str, variants: dict, name: str) -> dict:
    """Train every variant (3 seeds each) and write one comparison table `<name>.json`."""
    table = {}
    for tag, v in variants.items():
        R = run(arch, v["W"], cfg=v["cfg"], tag=tag, model_kw=v["model_kw"])
        table[tag] = {"n_params": R["n_params"], "stop_rmse_best_mean": R["stop_rmse_best"]["mean"],
                      **{k: R["summary"][k] for k in ("rmse", "score_mean", "score_median", "share_late_over_20")},
                      "ensemble_rmse": R["ensemble_of_seeds"]["rmse"],
                      "best_epochs": [R["seeds"][s]["best_epoch"] for s in R["seeds"]],
                      "minutes_per_seed": round(float(np.mean([R["seeds"][s]["train_minutes"] for s in R["seeds"]])), 2)}
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n[{name}]  (early-stop = held-aside training machines; ruler = 180 machines)")
    for tag, t in table.items():
        print(f"  {tag:12s} params {t['n_params']:7d} | early-stop best {t['stop_rmse_best_mean']:6.2f} | ruler RMSE "
              f"{t['rmse']['mean']:6.2f} ± {t['rmse']['std']:.2f} | score {t['score_mean']['mean']:6.1f} ± "
              f"{t['score_mean']['std']:5.1f} | median {t['score_median']['mean']:.2f} | best epochs {t['best_epochs']} "
              f"| {t['minutes_per_seed']} min/seed")
    return table


def cnn_ablation() -> dict:
    return run_variants("cnn", CNN_VARIANTS, "cnn_ablation")


def gru_runs() -> dict:
    return run_variants("gru", GRU_VARIANTS, "gru_runs")


# Windows + PyTorch 2.11 (CUDA 12.8): after a GRU or LSTM has been *trained* with cuDNN, the Python
# process crashes while closing (exit code 0xC0000409), after all work is done and all files are
# written. Prediction-only use of cuDNN RNNs does not crash, nor does training with cuDNN disabled
# (but that is ~18x slower: 27.7 s vs 1.5 s per epoch). os._exit and TerminateProcess do not avoid it.
# So the GRU training runs as a separate process, and we accept that one exit code only if the
# process wrote fresh results: the summary JSON and every checkpoint, all newer than its start.
WINDOWS_CUDNN_RNN_EXIT = {0xC0000409, 0xC0000409 - 2**32}


def run_isolated(module: str, flag: str | None, files: list, name: str) -> None:
    """Run `python -m <module> [flag]` as a child process; accept the Windows cuDNN-RNN exit code only
    if every file in `files` was (re)written after the child started."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    t0 = time.time()
    cmd = [sys.executable, "-m", module] + ([flag] if flag else [])
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    fresh = all(f.exists() and os.path.getmtime(f) > t0 for f in files)
    if proc.returncode == 0 or (proc.returncode in WINDOWS_CUDNN_RNN_EXIT and fresh):
        print(f"[{name}] finished (exit code {proc.returncode}); {len(files)} result files written fresh: {fresh}")
        return
    raise RuntimeError(f"{name} failed: exit code {proc.returncode}, fresh result files: {fresh}")


def gru_runs_isolated() -> None:
    files = [OUT / "gru_runs.json"] + [CKPT / f"{tag}_seed{s}.pt" for tag in GRU_VARIANTS for s in SEEDS]
    run_isolated("task2_rul.deep", "--gru", files, "gru_runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cnn")
    ap.add_argument("--W", type=int, default=30)
    ap.add_argument("--cnn-ablation", action="store_true")
    ap.add_argument("--gru", action="store_true")
    a = ap.parse_args()
    if a.cnn_ablation:
        cnn_ablation()
    elif a.gru:
        gru_runs()
    else:
        run(a.arch, a.W)


if __name__ == "__main__":
    main()
