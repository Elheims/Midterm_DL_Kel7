"""Task 2, part (c): sliding windows.

A sample is the last W cycles of one machine, ending at cycle t, and its target is RUL(t):

    input  = readings of cycles t-W+1, ..., t        (shape W x F)
    target = RUL(t) = min(150, failure cycle - t)

Why this design:
- No future leakage: a window ending at t contains only cycles <= t of the same machine.
  This is exactly the test situation (history up to the last recorded cycle -> RUL).
- Fixed shape: every sample is W x F whatever the length of the machine's history.
- Short histories: a history shorter than W is padded at the front by repeating its first
  reading ("nothing happened before cycle 1"). With W <= 50 no test machine needs padding
  (shortest test history: 50 cycles); padding matters for early training cycles and for
  validation histories cut short in part (e).
- Memory: rows are stored once per machine (with W-1 padding rows in front); a window is a
  slice of that array, built only when a batch needs it.

Run from kode/:  python -m task2_rul.windows
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import plotstyle
from common.plotstyle import INK_2, SERIES
from common.seed import FIG_DIR, RESULTS_DIR
from task2_rul.data import Preprocessor, load_raw
from task2_rul.labels import add_rul


@dataclass
class Windows:
    data: np.ndarray        # (rows incl. padding, F) float32; each machine block starts with W-1 pad rows
    cycle_data: np.ndarray  # (rows incl. padding,) cycle number of each stored row; 0 for padding rows
    unit_data: np.ndarray   # (rows incl. padding,) machine id of each stored row
    ends: np.ndarray        # (N,) row index in `data` of the last row of each window
    unit: np.ndarray        # (N,) machine id of each window
    cycle: np.ndarray       # (N,) cycle t at which each window ends
    y: np.ndarray | None    # (N,) RUL(t) target, None for test data
    W: int

    def __len__(self) -> int:
        return len(self.ends)

    def _rows(self, idx) -> np.ndarray:
        """(n, W) array of stored-row indices for the windows in idx."""
        return self.ends[idx][:, None] + np.arange(-self.W + 1, 1)

    def X(self, idx=None) -> np.ndarray:
        """Window inputs, shape (n, W, F). idx: indices or slice; None = all windows."""
        idx = np.arange(len(self)) if idx is None else idx
        return self.data[self._rows(idx)]


def make_windows(df: pd.DataFrame, feats: list[str], W: int, which: str = "all",
                 first_end: int | None = None) -> Windows:
    """Build windows from a preprocessed frame sorted by (id_unit, siklus).

    which="all":  one window ending at every cycle t >= first_end (training / validation).
                  first_end defaults to W, i.e. no padded windows; use 1 to include them.
    which="last": one window per machine, ending at its last recorded cycle (test prediction).
    If `df` has an `rul` column, it becomes the target y.
    """
    first_end = W if first_end is None else first_end
    blocks, cyc_blocks, unit_blocks, ends, units, cycles, ys = [], [], [], [], [], [], []
    offset = 0
    has_y = "rul" in df.columns
    for u, g in df.groupby("id_unit", sort=True):
        x = g[feats].to_numpy(np.float32)
        pad = np.repeat(x[:1], W - 1, axis=0)
        blocks.append(np.vstack([pad, x]))
        cyc_blocks.append(np.concatenate([np.zeros(W - 1, np.int32), g.siklus.to_numpy(np.int32)]))
        unit_blocks.append(np.full(len(x) + W - 1, u, np.int32))
        n = len(x)
        pos = np.arange(n) if which == "all" else np.array([n - 1])
        if which == "all":
            pos = pos[g.siklus.to_numpy()[pos] >= first_end]
        ends.append(offset + W - 1 + pos)
        units.append(np.full(len(pos), u, np.int32))
        cycles.append(g.siklus.to_numpy(np.int32)[pos])
        if has_y:
            ys.append(g.rul.to_numpy(np.float32)[pos])
        offset += n + W - 1
    return Windows(
        data=np.vstack(blocks), cycle_data=np.concatenate(cyc_blocks), unit_data=np.concatenate(unit_blocks),
        ends=np.concatenate(ends), unit=np.concatenate(units), cycle=np.concatenate(cycles),
        y=np.concatenate(ys) if has_y else None, W=W,
    )


def check_windows(win: Windows, df: pd.DataFrame, feats: list[str], n_check: int = 2000, seed: int = 0) -> None:
    """Leakage and correctness checks on a random sample of windows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(win), min(n_check, len(win)), replace=False)
    rows = win._rows(idx)
    cyc, unit = win.cycle_data[rows], win.unit_data[rows]
    # 1. one machine per window
    assert (unit == win.unit[idx][:, None]).all(), "window mixes machines"
    # 2. the window ends at its own cycle t and never contains a later cycle
    assert (cyc[:, -1] == win.cycle[idx]).all(), "window does not end at its cycle"
    assert (cyc.max(axis=1) == win.cycle[idx]).all(), "window contains a future cycle"
    # 3. real (non-padding) cycles are consecutive and end at t
    real = cyc > 0
    expect = win.cycle[idx][:, None] + np.arange(-win.W + 1, 1)
    assert (cyc[real] == expect[real]).all(), "cycles inside a window are not consecutive"
    # 4. inputs equal the preprocessed rows, target equals RUL at t
    key = df.set_index(["id_unit", "siklus"])
    j = rng.choice(len(idx), 50, replace=False)
    for k in j:
        i = idx[k]
        t = win.cycle[i]
        lo = max(1, t - win.W + 1)
        ref = key.loc[win.unit[i]].loc[lo:t, feats].to_numpy(np.float32)
        got = win.X(np.array([i]))[0][-len(ref):]
        assert np.array_equal(got, ref), "window values differ from preprocessed rows"
        if win.y is not None:
            assert win.y[i] == key.loc[(win.unit[i], t), "rul"], "target is not RUL(t)"


def main() -> dict:
    plotstyle.apply()
    out_dir = RESULTS_DIR / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_raw, te_raw = load_raw("train"), load_raw("test")
    prep = Preprocessor().fit(tr_raw)              # fitted on training machines only
    feats = prep.keep_
    tr = add_rul(prep.transform(tr_raw))           # preprocessed rows + RUL target
    te = prep.transform(te_raw)

    R = {"features": feats, "n_features": len(feats)}
    for W in (30, 50):
        w_all = make_windows(tr, feats, W)                  # windows ending at t >= W
        w_pad = make_windows(tr, feats, W, first_end=1)      # incl. padded early windows
        w_te = make_windows(te, feats, W, which="last")
        check_windows(w_all, tr, feats)
        check_windows(w_pad, tr, feats)
        check_windows(w_te, te, feats)
        te_last = te.groupby("id_unit").siklus.max().to_numpy()
        assert (w_te.cycle == te_last).all() and len(w_te) == te.id_unit.nunique()
        # check the checker: shift every window one cycle into the future; the check must fail
        leaky = dataclasses.replace(w_all, ends=w_all.ends + 1)
        try:
            check_windows(leaky, tr, feats)
        except AssertionError:
            pass
        else:
            raise RuntimeError("leakage check did not detect windows that include a future cycle")
        R[f"W{W}"] = {
            "train_windows": len(w_all),
            "train_windows_incl_padded": len(w_pad),
            "share_train_windows_at_cap": round(float((w_all.y == 150).mean()), 4),
            "test_windows": len(w_te),
            "test_machines_needing_padding": int((te.groupby("id_unit").size() < W).sum()),
            "window_shape": [W, len(feats)],
            "MB_if_all_copied": round(len(w_all) * W * len(feats) * 4 / 2**20, 1),
            "MB_stored": round(w_all.data.nbytes / 2**20, 1),
        }
        print(f"[windows] W={W}: checks passed (single machine, ends at t, no future cycle, "
              f"values and targets match; last test window = last recorded cycle; "
              f"a deliberately leaky version is rejected)")

    with open(out_dir / "windows.json", "w") as f:
        json.dump(R, f, indent=2)

    # ---------- figure: what a window is, and why W <= 50
    W = 30
    unit = 11
    g = tr[tr.id_unit == unit]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(wspace=0.3)
    ax[0].plot(g.siklus, g.s04, color=SERIES[0], lw=0.9)
    for k, t in enumerate((120, 220, 300)):
        ax[0].axvspan(t - W + 1, t, color=SERIES[1], alpha=0.18, lw=0)
        rul = int(g.loc[g.siklus == t, "rul"].iloc[0])
        ax[0].annotate(f"ends t={t}\nRUL={rul}", xy=(t, g.s04.max()),
                       xytext=(t - W / 2, g.s04.max() + 0.5), ha="center", fontsize=7.5, color=INK_2)
    ax[0].set(xlabel="cycle", ylabel="s04 (z)",
              title=f"(a) Three windows (W={W}) of machine {unit} and their targets")
    ax[0].set_ylim(top=g.s04.max() + 1.6)
    len_tr = tr_raw.groupby("id_unit").size()
    len_te = te_raw.groupby("id_unit").size()
    bins = np.arange(0, 800, 20)
    ax[1].hist(len_tr, bins=bins, color=SERIES[0], alpha=0.6, label="train (full life)")
    ax[1].hist(len_te, bins=bins, color=SERIES[1], alpha=0.6, label="test (cut short)")
    for Wv, ls in ((30, ":"), (50, "--")):
        ax[1].axvline(Wv, color=INK_2, lw=1, ls=ls, label=f"W={Wv}")
    ax[1].set(xlabel="history length (cycles)", ylabel="machines", title="(b) History lengths")
    ax[1].legend()
    plotstyle.save(fig, FIG_DIR / "t2_windows")

    print(json.dumps(R, indent=1))
    return R


if __name__ == "__main__":
    main()
