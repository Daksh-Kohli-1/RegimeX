"""
Builds the modelling table and the train/test "folds".

Everything that is FITTED (scaler, regime labeler) is fitted on the training
rows of a fold only, then applied forward (README 9.3).

A ``Fold`` is one complete experiment slice:

    [ ...... train ...... ][ purged gap ][ ...... test ...... ]

* purged gap (README 9.5): the last ``gap`` rows before the test period are
  dropped from training because their forward label window reaches into the test
  period. gap = vol_window for forward labels (see ``utils.effective_gap``).
* the single holdout split ("primary fold") and every walk-forward fold are the
  same object, so all models run through identical code paths.
* EVERY model is scored on exactly the same test dates. The sequence models
  (LSTM/Transformer) are allowed to look back into the rows before the test
  period for their input window - that is past information - so they do not
  lose the first `sequence_length` test days (this was the reason the old
  buy-and-hold figures differed between model rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.labeling import RegimeLabeler, compute_vol_frame, labeler_from_cfg
from src.utils import effective_gap, load_config


@dataclass
class Fold:
    name: str
    cfg: dict
    index: pd.DatetimeIndex            # dates of ALL rows in the frame
    train_pos: np.ndarray              # row positions used for training (after purge)
    test_pos: np.ndarray               # row positions scored
    gap: int
    feature_names: list
    X_all: np.ndarray                  # scaled features for every row (scaler fit on train)
    X_raw: pd.DataFrame                # unscaled features for every row
    y_all: np.ndarray                  # regime per row (-1 where undefined / purged)
    target_vol: np.ndarray             # continuous forecast target for every row
    trailing_vol: np.ndarray           # past realized vol for every row
    label_returns: pd.Series           # daily log returns of the label ticker (all rows)
    labeler: RegimeLabeler
    scaler: StandardScaler
    n_regimes: int = 3
    seq_len: int = 30
    extras: dict = field(default_factory=dict)

    # ---- flat (tabular) views -------------------------------------------- #
    @property
    def X_train(self): return self.X_all[self.train_pos]
    @property
    def X_test(self): return self.X_all[self.test_pos]
    @property
    def y_train(self): return self.y_all[self.train_pos]
    @property
    def y_test(self): return self.y_all[self.test_pos]
    @property
    def dates_train(self): return self.index[self.train_pos]
    @property
    def dates_test(self): return self.index[self.test_pos]
    @property
    def regime_vol(self): return self.labeler.regime_vol_

    # ---- sequence views --------------------------------------------------- #
    def windows(self, positions: np.ndarray) -> np.ndarray:
        """(N, seq_len, n_features) windows ENDING at each position (inclusive)."""
        return windows_at(self.X_all, positions, self.seq_len)

    def valid_train_positions(self) -> np.ndarray:
        return self.train_pos[self.train_pos >= self.seq_len - 1]

    def deep_train_val_positions(self, val_frac: float):
        """Chronological train / validation positions for early stopping, with a
        purge gap between them (the validation labels also look forward)."""
        pos = self.valid_train_positions()
        cut = int(len(pos) * (1 - val_frac))
        tr = pos[: max(cut - self.gap, 1)]
        val = pos[cut:]
        return tr, val

    # ---- helpers ----------------------------------------------------------- #
    def proba_to_expected_vol(self, proba: np.ndarray) -> np.ndarray:
        """Expected next-horizon vol = sum_k p_k * (mean training vol of regime k)."""
        return proba @ self.regime_vol

    def meta(self) -> dict:
        k = self.n_regimes
        tr = np.bincount(self.y_train, minlength=k)
        te = np.bincount(self.y_test, minlength=k)
        maj = int(tr.argmax())
        return {
            "name": self.name,
            "label_mode": self.cfg.get("label_mode", "forward"),
            "labeling_method": self.cfg["labeling_method"],
            "gap_days": int(self.gap),
            "n_train": int(len(self.train_pos)),
            "n_test": int(len(self.test_pos)),
            "train_start": str(self.dates_train[0].date()),
            "train_end": str(self.dates_train[-1].date()),
            "test_start": str(self.dates_test[0].date()),
            "test_end": str(self.dates_test[-1].date()),
            "train_class_share": [float(x) for x in tr / max(tr.sum(), 1)],
            "test_class_share": [float(x) for x in te / max(te.sum(), 1)],
            "test_classes_present": [int(i) for i in np.where(te > 0)[0]],
            "majority_train_class": maj,
            # accuracy of "always predict the most common TRAINING class" on the test set
            "majority_baseline_test_acc": float(te[maj] / max(te.sum(), 1)),
        }


def windows_at(X: np.ndarray, positions: np.ndarray, seq_len: int) -> np.ndarray:
    """Stack the `seq_len` rows ending at each position (inclusive)."""
    positions = np.asarray(positions)
    if len(positions) and positions.min() < seq_len - 1:
        raise ValueError("Some positions have fewer than seq_len rows of history")
    return np.stack([X[p - seq_len + 1 : p + 1] for p in positions]) if len(positions) else \
        np.empty((0, seq_len, X.shape[1]))


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling windows over a standalone array: X_seq[i] = X[i : i+seq_len],
    y_seq[i] = y[i+seq_len-1] (label of the LAST day in the window)."""
    n = len(X)
    if n <= seq_len:
        raise ValueError(f"Not enough rows ({n}) for sequence_length={seq_len}")
    X_seq = np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])
    return X_seq, y[seq_len - 1 :]


# --------------------------------------------------------------------------- #
# Building the modelling frame
# --------------------------------------------------------------------------- #
def load_frame(cfg: dict | None = None, prices: pd.DataFrame | None = None):
    """Features + continuous targets, always rebuilt from the cached prices (so a
    config change can never silently reuse stale feature/label files)."""
    from src.data_pipeline import get_price_data
    from src.features import build_features

    cfg = cfg or load_config()
    prices = prices if prices is not None else get_price_data(cfg)
    features = build_features(prices, cfg)
    vol_frame = compute_vol_frame(prices, cfg)
    frame = features.join(vol_frame, how="inner")
    feature_cols = list(features.columns)
    return frame, feature_cols


def make_fold(frame: pd.DataFrame, feature_cols: list, cfg: dict,
              test_start: int, test_end: int, name: str = "holdout") -> Fold:
    """Train on rows [0, test_start - gap), test on rows [test_start, test_end)."""
    gap = effective_gap(cfg)
    n = len(frame)
    if not (0 < test_start < test_end <= n):
        raise ValueError(f"bad fold bounds: test_start={test_start}, test_end={test_end}, n={n}")
    train_pos = np.arange(0, test_start - gap)
    test_pos = np.arange(test_start, test_end)
    if len(train_pos) < 100:
        raise ValueError(f"Fold {name}: only {len(train_pos)} training rows after purge")

    X_raw = frame[feature_cols]
    scaler = StandardScaler().fit(X_raw.values[train_pos])          # train only
    X_all = scaler.transform(X_raw.values)

    target_vol = frame["target_vol"].values
    trailing_vol = frame["trailing_vol"].values

    labeler = labeler_from_cfg(cfg).fit(frame["target_vol"].iloc[train_pos])  # train only
    y_all = np.full(n, -1, dtype=int)
    y_all[train_pos] = labeler.assign(frame["target_vol"].iloc[train_pos]).values
    y_all[test_pos] = labeler.assign(frame["target_vol"].iloc[test_pos]).values

    label_ret_col = f"{cfg['label_ticker']}_logret"
    return Fold(
        name=name, cfg=cfg, index=frame.index, train_pos=train_pos, test_pos=test_pos,
        gap=gap, feature_names=list(feature_cols), X_all=X_all, X_raw=X_raw, y_all=y_all,
        target_vol=target_vol, trailing_vol=trailing_vol,
        label_returns=frame[label_ret_col], labeler=labeler, scaler=scaler,
        n_regimes=cfg["n_regimes"], seq_len=cfg["sequence_length"],
    )


def make_primary_fold(frame: pd.DataFrame, feature_cols: list, cfg: dict) -> Fold:
    """The single chronological holdout: newest `test_size` fraction is the test set."""
    n = len(frame)
    test_start = int(n * (1 - cfg["test_size"]))
    return make_fold(frame, feature_cols, cfg, test_start, n, name="holdout")


def walk_forward_folds(frame: pd.DataFrame, feature_cols: list, cfg: dict):
    """Expanding-window walk-forward folds with non-overlapping test windows (README 9.4)."""
    wf = cfg["walk_forward"]
    n = len(frame)
    start, win, step = int(wf["min_train_days"]), int(wf["test_window_days"]), int(wf["step_days"])
    if step < win:
        raise ValueError("walk_forward.step_days must be >= test_window_days (overlapping tests would double count)")
    folds, test_start, i = [], start, 1
    while test_start < n:
        test_end = min(test_start + win, n)
        if test_end - test_start >= max(win // 2, 60):   # skip a tiny trailing sliver
            folds.append(make_fold(frame, feature_cols, cfg, test_start, test_end, name=f"fold{i}"))
            i += 1
        test_start += step
    return folds


def build_dataset(cfg: dict | None = None):
    """Convenience loader for the primary holdout: returns (fold, frame)."""
    cfg = cfg or load_config()
    frame, feature_cols = load_frame(cfg)
    return make_primary_fold(frame, feature_cols, cfg), frame
