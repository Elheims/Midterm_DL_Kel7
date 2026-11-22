"""Task 2: loading and preprocessing of the run-to-failure logs (part a).

Pipeline, fitted on training machines only and then applied unchanged to any split:

1. Operating regime: the three settings take 6 discrete combinations. KMeans (k=6) on the
   standardised settings assigns every cycle to a regime.
2. Per-regime normalisation: each sensor is z-scored with the mean/std of its regime.
   The regime explains up to 99.9% of a sensor's variance, so a global z-score would mostly
   encode the regime, not wear.
3. Spike removal (causal Hampel filter): a reading is a spike if it deviates from the median of
   the previous `window` readings of the same machine by more than `k` noise standard
   deviations. Only past cycles are used, so the filter behaves the same at the last test cycle.
4. Missing values (incl. removed spikes): forward fill within each machine (past values only).
   Values with no earlier reading (start of a history) get the healthy starting level: the mean
   z of the first 10 cycles of the training machines (a new machine is healthy by definition).
5. Sensor selection: drop sensors without a degradation trend (mean |change| between the first
   and last 20 cycles of a training machine below `trend_threshold`, in z units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common.seed import DATA_DIR

SETTINGS = ["setel_1", "setel_2", "setel_3"]
SENSORS = [f"s{i:02d}" for i in range(1, 22)]


def load_raw(split: str) -> pd.DataFrame:
    """split: 'train' or 'test'. Rows sorted by machine and cycle."""
    df = pd.read_csv(DATA_DIR / f"rul_{split}.csv.gz")
    return df.sort_values(["id_unit", "siklus"]).reset_index(drop=True)


def unit_trend(z: pd.DataFrame, ids: pd.Series, cycles_left: pd.Series, n: int = 20) -> pd.DataFrame:
    """Per machine: mean of the last n cycles minus mean of the first n cycles (training data only)."""
    pos = z.groupby(ids).cumcount()
    first = z[pos < n].groupby(ids[pos < n]).mean()
    last = z[cycles_left < n].groupby(ids[cycles_left < n]).mean()
    return last - first


class Preprocessor:
    def __init__(self, n_regimes: int = 6, window: int = 10, k: float = 5.0,
                 trend_threshold: float = 0.5, seed: int = 0):
        self.n_regimes = n_regimes
        self.window = window
        self.k = k
        self.trend_threshold = trend_threshold
        self.seed = seed

    # ---------- steps
    def _regime(self, df: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict((df[SETTINGS] / self.set_scale_).to_numpy())

    def _zscore(self, df: pd.DataFrame, regime: np.ndarray) -> pd.DataFrame:
        mu = self.mu_.to_numpy()[regime]
        sd = self.sd_.to_numpy()[regime]
        return pd.DataFrame((df[SENSORS].to_numpy() - mu) / sd, columns=SENSORS, index=df.index)

    def _residual(self, z: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        """Reading minus the median of the previous `window` readings of the same machine."""
        past = z.groupby(ids).shift(1)
        med = past.groupby(ids).rolling(self.window, min_periods=3).median()
        med = med.reset_index(level=0, drop=True).sort_index()
        return z - med

    # ---------- API
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.set_scale_ = df[SETTINGS].std()
        self.kmeans_ = KMeans(self.n_regimes, n_init=10, random_state=self.seed)
        self.kmeans_.fit((df[SETTINGS] / self.set_scale_).to_numpy())
        regime = self.kmeans_.labels_
        g = df[SENSORS].groupby(regime)
        self.mu_, self.sd_ = g.mean(), g.std()

        z = self._zscore(df, regime)
        res = self._residual(z, df.id_unit)
        self.noise_sd_ = 1.4826 * res.abs().median()  # robust noise scale per sensor
        self.start_level_ = z[df.groupby("id_unit").cumcount() < 10].mean()  # healthy level per sensor

        cycles_left = df.groupby("id_unit").siklus.transform("max") - df.siklus
        self.trend_ = unit_trend(z, df.id_unit, cycles_left).abs().mean()
        self.keep_ = [s for s in SENSORS if self.trend_[s] >= self.trend_threshold]
        return self

    def transform(self, df: pd.DataFrame, return_info: bool = False):
        regime = self._regime(df)
        z = self._zscore(df, regime)
        missing = z.isna()
        res = self._residual(z, df.id_unit)
        spike = (res.abs() > self.k * self.noise_sd_) & ~missing
        z = z.mask(spike)
        z = z.groupby(df.id_unit).ffill().fillna(self.start_level_)

        out = pd.concat([df[["id_unit", "siklus"]], pd.Series(regime, index=df.index, name="regime"),
                         z[self.keep_]], axis=1)
        if return_info:
            info = {"missing": missing, "spike": spike, "residual": res}
            return out, info
        return out
