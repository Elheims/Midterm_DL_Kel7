"""Task 2: write prediksi_rul.csv from the saved final model (no training; for the oral-exam demo).

1. Refit the preprocessing on all 900 training machines (deterministic, same as in final.py).
2. Load the 5 final GRUs (task2_rul/final_weights/) and average their predictions at the last
   recorded cycle of each of the 300 test machines.
3. Subtract the shift chosen by the simulation in final.py (best net benefit), clip to [0, 150].
4. Check the format (300 rows, the exact test ids, columns id_unit,rul, values in [0, 150]) and write
   prediksi_rul.csv in the repository root.

Run from kode/:  python -m task2_rul.predict
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common.seed import RESULTS_DIR, ROOT, set_seed
from task2_rul.data import load_raw
from task2_rul.final import FINAL_DIR, SEEDS5, W, ensemble_predict, final_split
from task2_rul.labels import RUL_CAP
from task2_rul.train_nn import GPUWindows
from task2_rul.windows import make_windows

OUT = RESULTS_DIR / "task2"
CSV = ROOT / "prediksi_rul.csv"


def make_predictions():
    """Predictions for the 300 test machines from the final weights, with format checks (writes nothing)."""
    set_seed(0)
    delta = int(json.load(open(OUT / "final_shift_simulation.json"))["best_shift_by_net_benefit"])
    prep, sp = final_split()
    te_raw = load_raw("test")
    te = prep.transform(te_raw)
    win = make_windows(te, sp.feats, W, which="last")
    pred = ensemble_predict([FINAL_DIR / f"gru_W50_seed{s}.pt" for s in SEEDS5], len(sp.feats), GPUWindows(win))
    rul = np.round(np.clip(pred - delta, 0, RUL_CAP), 1)
    df = pd.DataFrame({"id_unit": win.unit.astype(int), "rul": rul})

    # format checks (the file is graded automatically)
    test_ids = np.sort(te_raw.id_unit.unique())
    assert list(df.columns) == ["id_unit", "rul"]
    assert len(df) == 300 == len(test_ids) and df.id_unit.is_unique
    assert np.array_equal(np.sort(df.id_unit.to_numpy()), test_ids), "ids must be exactly the test machines"
    assert df.rul.notna().all() and df.rul.between(0, RUL_CAP).all()
    assert np.array_equal(win.cycle, te_raw.groupby("id_unit").siklus.max().to_numpy()), "must predict at last cycle"
    return df, pred, win, delta


def main() -> pd.DataFrame:
    df, pred, win, delta = make_predictions()
    df.to_csv(CSV, index=False)

    summary ={"delta": delta, "rows": len(df), "rul_describe": df.rul.describe().round(1).to_dict(),
               "share_at_150_before_shift": round(float((pred >= RUL_CAP - 0.5).mean()), 3),
               "share_below_50": round(float((df.rul < 50).mean()), 3)}
    pd.DataFrame({"id_unit": df.id_unit, "rul_unshifted": np.round(pred, 2), "rul": df.rul,
                  "last_cycle": win.cycle}).to_csv(OUT / "test_predictions_detail.csv", index=False)
    with open(OUT / "test_predictions.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[predict] wrote {CSV} ({len(df)} rows, shift {delta}); format checks passed")
    print(f"[predict] predicted RUL: {summary['rul_describe']}")
    return df


if __name__ == "__main__":
    main()
