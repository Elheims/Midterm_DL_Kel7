"""Sanity checks for the Task 2 preprocessing. Run from kode/: python -m task2_rul.check_preprocess

1. Causality: preprocessing a machine's history cut at cycle t gives exactly the same values for
   cycles <= t as preprocessing the full history. So no future cycle leaks into any feature,
   and the training inputs match what the model sees at the last test cycle.
2. Output has no NaN, on train and test.
3. Train-only fitting: statistics fitted on a subset of machines differ only slightly from the
   full fit (used later for fold-wise fitting inside cross-validation).
"""
import numpy as np

from common.seed import set_seed
from task2_rul.data import Preprocessor, load_raw


def main() -> None:
    set_seed(0)
    tr, te = load_raw("train"), load_raw("test")
    prep = Preprocessor().fit(tr)
    full = prep.transform(tr)
    feats = prep.keep_

    rng = np.random.default_rng(0)
    units = rng.choice(tr.id_unit.unique(), 50, replace=False)
    worst = 0.0
    for u in units:
        h = tr[tr.id_unit == u]
        cut = int(rng.integers(30, len(h)))
        part = prep.transform(h.iloc[:cut])
        ref = full.loc[h.index[:cut], feats].to_numpy()
        worst = max(worst, float(np.abs(part[feats].to_numpy() - ref).max()))
    print(f"[1] causality: max |difference| truncated vs full over 50 machines = {worst:.2e}")
    assert worst < 1e-9

    te_p = prep.transform(te)
    n_nan = int(full[feats].isna().sum().sum() + te_p[feats].isna().sum().sum())
    print(f"[2] NaN after preprocessing (train+test): {n_nan}")
    assert n_nan == 0

    half = tr[tr.id_unit.isin(tr.id_unit.unique()[::2])]
    p2 = Preprocessor().fit(half)
    # regime ids may be permuted between fits: match each regime to the nearest full-fit centre
    match = prep.kmeans_.predict(p2.kmeans_.cluster_centers_ * p2.set_scale_.to_numpy() / prep.set_scale_.to_numpy())
    diff = (p2.mu_.to_numpy() - prep.mu_.to_numpy()[match]) / prep.sd_.to_numpy()[match]
    print(f"[3] fit on half the machines: same sensors kept = {p2.keep_ == prep.keep_}, "
          f"max |regime-mean shift| = {np.abs(diff).max():.3f} within-regime sd")
    print("all checks passed")


if __name__ == "__main__":
    main()
