"""
Stage 2 — Regime labeling.

Two separate ideas live here:

1. THE TARGET VOLATILITY SERIES (README 9.1)
   For every day t we build two continuous numbers for the label ticker:

   * ``trailing_vol``  realized vol of the H days ENDING at t   (past, known at t)
   * ``target_vol``    what the model must forecast:
        - label_mode "forward"  : realized vol of days t+1 ... t+H
                                  (no day is shared with anything known at t)
        - label_mode "backward" : the H-day vol ending at t+1
                                  (legacy; 19 of 20 days overlap the inputs)

2. THE REGIME LABELER (README 9.3 and 9.7)
   ``RegimeLabeler`` turns a continuous vol number into low / medium / high.
   It is FIT ON TRAINING ROWS ONLY and then applied forward, like any other
   fitted component (scaler, encoder...). Three methods:

   * "gmm"        Gaussian mixture on vol (each day labeled independently)
   * "percentile" quantile thresholds taken from the training rows
   * "hmm"        Gaussian hidden Markov model on log-vol: regimes are sticky and
                  the fitted transition matrix is reported (needs ``hmmlearn``)

   Labels are always ordered by severity: 0 = low, 1 = medium, 2 = high.

Regimes are therefore defined RELATIVE TO THE TRAINING WINDOW. In walk-forward
validation every fold has its own labeler, so a fold whose training years were
calm will call a smaller spike "high". That is the price of not peeking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.features import log_returns, realized_vol
from src.utils import load_config, log, resolve_path

REGIME_NAMES = {0: "low_vol", 1: "medium_vol", 2: "high_vol"}


# --------------------------------------------------------------------------- #
# 1. Continuous target series
# --------------------------------------------------------------------------- #
def compute_vol_frame(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return DataFrame[trailing_vol, target_vol] indexed by date (NaNs dropped)."""
    ticker = cfg["label_ticker"]
    horizon = int(cfg["vol_window"])
    mode = cfg.get("label_mode", "forward")

    ret = log_returns(prices[ticker])
    trailing = realized_vol(ret, horizon)  # value at t uses returns t-H+1 ... t

    if mode == "forward":
        # trailing at t+H uses returns t+1 ... t+H  ->  strictly the future
        target = trailing.shift(-horizon)
    elif mode == "backward":
        # legacy behaviour: the window ending tomorrow (mostly already seen)
        target = trailing.shift(-1)
    else:
        raise ValueError(f"Unknown label_mode: {mode!r} (use 'forward' or 'backward')")

    return pd.DataFrame({"trailing_vol": trailing, "target_vol": target}).dropna()


# --------------------------------------------------------------------------- #
# 2. Regime labeler (fit on train only)
# --------------------------------------------------------------------------- #
def _rank_of(values: np.ndarray) -> np.ndarray:
    """rank[i] = position of component i when sorted ascending."""
    rank = np.empty(len(values), dtype=int)
    rank[np.argsort(values)] = np.arange(len(values))
    return rank


class RegimeLabeler:
    """Fit on training volatility, then assign regimes anywhere."""

    def __init__(self, method: str = "gmm", n_regimes: int = 3,
                 percentile_bounds=(0.33, 0.66), seed: int = 42):
        if method not in {"gmm", "percentile", "hmm"}:
            raise ValueError(f"Unknown labeling_method: {method!r}")
        self.method = method
        self.n_regimes = n_regimes
        self.percentile_bounds = list(percentile_bounds)
        self.seed = seed
        self.fitted_ = False

    # -- fitting ----------------------------------------------------------- #
    def fit(self, vol: pd.Series) -> "RegimeLabeler":
        vol = vol.dropna()
        v = np.asarray(vol.values, dtype=float)
        if len(v) < 50:
            raise ValueError(f"Need >= 50 training observations to fit the labeler, got {len(v)}")
        n = self.n_regimes

        if self.method == "gmm":
            self.model_ = GaussianMixture(n_components=n, random_state=self.seed, n_init=5)
            self.model_.fit(v.reshape(-1, 1))
            self._rank = _rank_of(self.model_.means_.ravel())

        elif self.method == "percentile":
            if len(self.percentile_bounds) != n - 1:
                raise ValueError("percentile_bounds must have n_regimes - 1 entries")
            self.thresholds_ = np.quantile(v, self.percentile_bounds)

        else:  # hmm
            try:
                from hmmlearn.hmm import GaussianHMM
            except ImportError as e:  # pragma: no cover
                raise ImportError("labeling_method 'hmm' needs hmmlearn: pip install hmmlearn") from e
            x = np.log(np.maximum(v, 1e-6)).reshape(-1, 1)
            best, best_ll = None, -np.inf
            for i in range(5):                      # EM finds local optima: keep the best of 5 restarts
                m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=200,
                                random_state=self.seed + i, min_covar=1e-4)
                try:
                    m.fit(x)
                    ll = m.score(x)
                except Exception:  # noqa: BLE001
                    continue
                if ll > best_ll:
                    best, best_ll = m, ll
            if best is None:
                raise RuntimeError("HMM fitting failed for every restart")
            self.model_ = best
            self._rank = _rank_of(self.model_.means_.ravel())

        self.fitted_ = True

        # Average vol of each regime on the TRAINING data. The strategy layer uses
        # this to turn a predicted regime distribution into an expected volatility.
        train_regimes = self.assign(vol)
        means = pd.Series(v).groupby(train_regimes.values).mean().reindex(range(n))
        self.regime_vol_ = means.interpolate(limit_direction="both").values
        self.train_counts_ = np.bincount(train_regimes.values, minlength=n)
        return self

    # -- assigning --------------------------------------------------------- #
    def _check(self):
        if not self.fitted_:
            raise RuntimeError("RegimeLabeler is not fitted yet")

    def assign(self, vol: pd.Series) -> pd.Series:
        """Regimes for a contiguous vol series. The HMM decodes the whole path
        (Viterbi), so only use this for TARGETS, never for inputs/predictions."""
        self._check()
        x = np.asarray(vol.values, dtype=float)
        if self.method == "gmm":
            reg = self._rank[self.model_.predict(x.reshape(-1, 1))]
        elif self.method == "percentile":
            reg = np.digitize(x, self.thresholds_, right=True)
        else:
            raw = self.model_.predict(np.log(np.maximum(x, 1e-6)).reshape(-1, 1))
            reg = self._rank[raw]
        return pd.Series(np.asarray(reg).astype(int), index=vol.index, name="regime")

    def assign_causal(self, vol: pd.Series) -> pd.Series:
        """Memoryless assignment (each day uses only its own value). Used wherever
        a regime is derived from information available at that day (persistence
        baseline, GARCH forecasts): no future observation can leak in."""
        self._check()
        if self.method != "hmm":
            return self.assign(vol)
        x = np.log(np.maximum(np.asarray(vol.values, dtype=float), 1e-6))
        mu = np.asarray(self.model_.means_).reshape(self.n_regimes)
        var = np.asarray(self.model_.covars_).reshape(self.n_regimes)
        loglik = -0.5 * (np.log(2 * np.pi * var)[None, :] + (x[:, None] - mu[None, :]) ** 2 / var[None, :])
        reg = self._rank[loglik.argmax(axis=1)]
        return pd.Series(reg.astype(int), index=vol.index, name="regime")

    # -- reporting --------------------------------------------------------- #
    def summary(self) -> dict:
        self._check()
        out = {
            "method": self.method,
            "n_regimes": self.n_regimes,
            "mean_vol_by_regime_on_train": [float(x) for x in self.regime_vol_],
            "train_counts": [int(c) for c in self.train_counts_],
        }
        if self.method == "gmm":
            out["component_means_vol"] = [float(m) for m in np.sort(self.model_.means_.ravel())]
        elif self.method == "percentile":
            out["thresholds_vol"] = [float(t) for t in self.thresholds_]
        else:
            inv = np.argsort(self._rank)  # ordered position -> raw state
            tm = self.model_.transmat_[np.ix_(inv, inv)]
            out["transition_matrix"] = [[float(p) for p in row] for row in tm]
            out["expected_duration_days"] = [float(1.0 / max(1.0 - tm[i, i], 1e-9)) for i in range(self.n_regimes)]
        return out


def labeler_from_cfg(cfg: dict) -> RegimeLabeler:
    return RegimeLabeler(
        method=cfg["labeling_method"],
        n_regimes=cfg["n_regimes"],
        percentile_bounds=cfg.get("percentile_bounds", [0.33, 0.66]),
        seed=cfg.get("random_seed", 42),
    )


# Convenience wrappers kept for quick experiments.
def label_gmm(vol: pd.Series, n_regimes: int, seed: int = 42) -> pd.Series:
    return RegimeLabeler("gmm", n_regimes, seed=seed).fit(vol).assign(vol)


def label_percentile(vol: pd.Series, bounds: list[float]) -> pd.Series:
    return RegimeLabeler("percentile", len(bounds) + 1, bounds).fit(vol).assign(vol)


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
def run(cfg: dict | None = None) -> pd.DataFrame:
    """Build the continuous target series, then fit the labeler on the PRIMARY
    split's training rows and write the resulting regimes for inspection."""
    import json

    from src.data_pipeline import get_price_data
    from src.dataset import load_frame, make_primary_fold

    cfg = cfg or load_config()
    prices = get_price_data(cfg)
    vol_frame = compute_vol_frame(prices, cfg)

    data_dir = resolve_path(cfg, "data_dir")
    out_dir = resolve_path(cfg, "outputs_dir")
    vol_frame.to_csv(data_dir / "labels.csv")

    frame, feature_cols = load_frame(cfg, prices=prices)
    fold = make_primary_fold(frame, feature_cols, cfg)

    # regimes over time for the timeline plot (train-fit labeler; purged rows stay NaN)
    regimes = pd.Series(np.nan, index=frame.index, name="regime")
    regimes.iloc[fold.train_pos] = fold.y_train
    regimes.iloc[fold.test_pos] = fold.y_test
    split = pd.Series("purged", index=frame.index, name="split")
    split.iloc[fold.train_pos] = "train"
    split.iloc[fold.test_pos] = "test"
    plot_df = pd.DataFrame({
        "target_vol": frame["target_vol"], "trailing_vol": frame["trailing_vol"],
        "regime": regimes, "split": split,
    })
    plot_df.to_csv(out_dir / "regime_labels_primary.csv")

    meta = fold.meta()
    meta["labeler"] = fold.labeler.summary()
    (out_dir / "primary_meta.json").write_text(json.dumps(meta, indent=2))

    k = cfg["n_regimes"]
    dist = lambda y: {REGIME_NAMES[i]: round(float(c) / max(len(y), 1), 3)  # noqa: E731
                      for i, c in enumerate(np.bincount(y, minlength=k))}
    log(f"Labels ({cfg['labeling_method']}, {cfg.get('label_mode', 'forward')}, labeler fit on TRAIN only)")
    log(f"  train distribution: {dist(fold.y_train)}")
    log(f"  test  distribution: {dist(fold.y_test)}")
    if (np.bincount(fold.y_test, minlength=k) == 0).any():
        log("  WARNING: at least one regime never occurs in the primary test window "
            "(walk-forward results give broader coverage).")
    return vol_frame


if __name__ == "__main__":
    df = run()
    print(df.tail())
