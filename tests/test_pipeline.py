"""
Sanity checks for the pipeline. Run with: python -m pytest tests/ -v

Cheap by design (small synthetic data, no network, no deep-model training) so
they run in well under a minute. They guard the properties the README promises:
no look-ahead, labeler/scaler fit on training rows only, purge gap, identical test
dates for every model, honest backtest timing, and correct metrics.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import generate_synthetic_data
from src.dataset import (load_frame, make_fold, make_primary_fold, walk_forward_folds,
                         windows_at, make_sequences)
from src.features import build_features, log_returns, macd, realized_vol, rsi
from src.labeling import RegimeLabeler, compute_vol_frame
from src.metrics import classification_scores
from src.models_benchmarks import persistence_forecast
from src.strategy import (VOL_RULE, backtest, evaluate_strategies, predicted_regime_to_exposure,
                          sharpe_ratio, vol_target_exposure)
from src.utils import _merge_defaults, effective_gap, load_config

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c = copy.deepcopy(c)
    c.update({"tickers": TICKERS, "label_ticker": "SPY", "strategy_asset": "SPY", "vol_window": 20,
              "label_mode": "forward", "labeling_method": "gmm", "n_regimes": 3, "test_size": 0.2,
              "sequence_length": 30, "embargo_days": "auto", "random_seed": 42})
    c["walk_forward"].update({"min_train_days": 900, "test_window_days": 250, "step_days": 250})
    return c


@pytest.fixture(scope="module")
def prices():
    return generate_synthetic_data(TICKERS, "2012-01-01", "2021-01-01", seed=1)


@pytest.fixture(scope="module")
def frame_cols(prices, cfg):
    return load_frame(cfg, prices=prices)


@pytest.fixture(scope="module")
def fold(frame_cols, cfg):
    frame, cols = frame_cols
    return make_primary_fold(frame, cols, cfg)


# ------------------------------------------------------------------ data / features
def test_synthetic_data_shape(prices):
    assert prices.shape[0] > 1000
    assert set(prices.columns) == set(TICKERS)
    assert (prices > 0).all().all()


def test_log_returns_no_lookahead(prices):
    ret = log_returns(prices["SPY"])
    assert np.isnan(ret.iloc[0])
    assert abs(ret.iloc[5] - np.log(prices["SPY"].iloc[5] / prices["SPY"].iloc[4])) < 1e-9


def test_rsi_bounds(prices):
    assert rsi(prices["SPY"], 14).dropna().between(0, 100).all()


def test_macd_shapes(prices):
    line, sig, hist = macd(prices["SPY"])
    assert len(line) == len(prices)
    assert np.allclose((line - sig).dropna(), hist.dropna())


def test_build_features_no_nan(prices, cfg):
    feats = build_features(prices, cfg)
    assert not feats.isna().any().any() and feats.shape[0] > 0


def test_features_have_no_lookahead(prices, cfg):
    """Features at day t must not change if every price AFTER t is altered."""
    full = build_features(prices, cfg)
    t = full.index[len(full) // 2]
    cut = prices.loc[:t].copy()
    cut_feats = build_features(cut, cfg)
    pd.testing.assert_series_equal(full.loc[t], cut_feats.loc[t], check_names=False, rtol=1e-9)


def test_macd_is_price_level_invariant(prices, cfg):
    """With normalization, multiplying all prices by a constant leaves MACD features unchanged."""
    a = build_features(prices, cfg)
    b = build_features(prices * 5.0, cfg)
    for col in ["SPY_macd", "SPY_macd_signal", "SPY_macd_hist"]:
        assert np.allclose(a[col], b[col], rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------- targets / labels
def test_forward_target_uses_only_future_returns(prices, cfg):
    H = cfg["vol_window"]
    vf = compute_vol_frame(prices, cfg)
    ret = log_returns(prices["SPY"])
    t = vf.index[300]
    pos = prices.index.get_loc(t)
    manual = ret.iloc[pos + 1: pos + 1 + H].std() * np.sqrt(252)
    assert abs(vf.loc[t, "target_vol"] - manual) < 1e-12


def test_forward_target_shares_no_days_with_inputs(prices, cfg):
    """Rescaling the history up to t-1 changes only ret_t, so the target of day t cannot move."""
    vf = compute_vol_frame(prices, cfg)
    t = vf.index[300]
    pos = prices.index.get_loc(t)
    altered = prices.copy()
    altered.iloc[:pos, :] = altered.iloc[:pos, :] * 1.37
    vf2 = compute_vol_frame(altered, cfg)
    assert abs(vf.loc[t, "target_vol"] - vf2.loc[t, "target_vol"]) < 1e-12


def test_backward_mode_is_legacy_overlapping_label(prices, cfg):
    c = dict(cfg, label_mode="backward")
    vf = compute_vol_frame(prices, c)
    ret = log_returns(prices["SPY"])
    trailing = realized_vol(ret, c["vol_window"])
    common = vf.index
    assert np.allclose(vf["target_vol"], trailing.shift(-1).reindex(common))


def test_unknown_label_mode_raises(prices, cfg):
    with pytest.raises(ValueError):
        compute_vol_frame(prices, dict(cfg, label_mode="sideways"))


def test_gmm_labels_ordered_by_severity():
    rng = np.random.default_rng(0)
    vol = pd.Series(np.concatenate([rng.normal(0.05, 0.01, 200), rng.normal(0.20, 0.01, 200),
                                    rng.normal(0.45, 0.01, 200)]))
    lab = RegimeLabeler("gmm", 3, seed=1).fit(vol)
    reg = lab.assign(vol)
    means = vol.groupby(reg.values).mean()
    assert means.iloc[0] < means.iloc[1] < means.iloc[2]
    assert list(lab.regime_vol_) == sorted(lab.regime_vol_)


def test_percentile_labels_roughly_balanced():
    vol = pd.Series(np.random.default_rng(1).uniform(0, 1, 1000))
    reg = RegimeLabeler("percentile", 3, [0.33, 0.66]).fit(vol).assign(vol)
    for frac in reg.value_counts(normalize=True):
        assert 0.2 < frac < 0.5


def test_hmm_labeler_orders_states_and_returns_valid_transitions():
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(3)
    blocks = [rng.normal(0.08, 0.01, 150), rng.normal(0.18, 0.02, 150), rng.normal(0.45, 0.04, 150)] * 2
    vol = pd.Series(np.abs(np.concatenate(blocks)) + 0.01)
    lab = RegimeLabeler("hmm", 3, seed=2).fit(vol)
    reg = lab.assign(vol)
    assert vol.groupby(reg.values).mean().is_monotonic_increasing
    tm = np.array(lab.summary()["transition_matrix"])
    assert np.allclose(tm.sum(axis=1), 1.0)
    # persistence of regimes should be high for this block-structured series
    assert np.all(np.diag(tm) > 0.8)
    # causal assignment depends only on each day's own value
    c = lab.assign_causal(vol)
    assert len(c) == len(vol) and c.min() >= 0 and c.max() <= 2


# ----------------------------------------------------------------- folds / leakage
def test_purge_gap_equals_horizon_for_forward_labels(fold, cfg):
    assert fold.gap == effective_gap(cfg) == cfg["vol_window"]
    # last training label window ends strictly before the first test day
    assert fold.train_pos.max() + fold.gap < fold.test_pos.min()


def test_embargo_override_and_backward_default(cfg):
    assert effective_gap(dict(cfg, label_mode="backward", embargo_days="auto")) == 1
    assert effective_gap(dict(cfg, embargo_days=7)) == 7


def test_chronological_split_is_time_ordered(fold, frame_cols):
    frame, _ = frame_cols
    assert fold.dates_train.max() < fold.dates_test.min()
    assert fold.dates_test[-1] == frame.index[-1]
    assert (np.diff(fold.test_pos) == 1).all()


def test_scaler_is_fit_on_training_rows_only(fold):
    raw = fold.X_raw.values
    assert np.allclose(fold.scaler.mean_, raw[fold.train_pos].mean(axis=0))
    assert not np.allclose(fold.scaler.mean_, raw.mean(axis=0))


def test_labeler_ignores_test_period(frame_cols, cfg):
    """Changing target vol in the test rows must not change the fitted labeler."""
    frame, cols = frame_cols
    n = len(frame)
    test_start = int(n * 0.8)
    a = make_fold(frame, cols, cfg, test_start, n)
    altered = frame.copy()
    altered.iloc[test_start:, altered.columns.get_loc("target_vol")] *= 3.0
    b = make_fold(altered, cols, cfg, test_start, n)
    assert a.labeler.summary() == b.labeler.summary()
    assert np.array_equal(a.y_train, b.y_train)


def test_no_label_is_missing_in_train_or_test(fold):
    assert (fold.y_train >= 0).all() and (fold.y_test >= 0).all()


def test_labels_not_degenerate(fold):
    dist = np.bincount(fold.y_train, minlength=3) / len(fold.y_train)
    assert (dist > 0).all() and dist.max() < 0.9


def test_walk_forward_folds_are_expanding_and_non_overlapping(frame_cols, cfg):
    frame, cols = frame_cols
    folds = walk_forward_folds(frame, cols, cfg)
    assert len(folds) >= 3
    for prev, nxt in zip(folds[:-1], folds[1:]):
        assert prev.test_pos.max() < nxt.test_pos.min()               # test windows never overlap
        assert len(nxt.train_pos) > len(prev.train_pos)                # training window expands
    for f in folds:
        assert f.train_pos.max() + f.gap < f.test_pos.min()            # purge in every fold


def test_walk_forward_rejects_overlapping_windows(frame_cols, cfg):
    frame, cols = frame_cols
    bad = copy.deepcopy(cfg)
    bad["walk_forward"].update({"test_window_days": 250, "step_days": 100})
    with pytest.raises(ValueError):
        walk_forward_folds(frame, cols, bad)


# ------------------------------------------------------------------- sequences
def test_make_sequences_alignment():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10)
    X_seq, y_seq = make_sequences(X, y, seq_len=3)
    assert X_seq.shape == (8, 3, 2) and y_seq[0] == y[2]
    assert np.array_equal(X_seq[0], X[0:3])


def test_windows_end_at_each_position_and_cover_all_test_days(fold):
    w = fold.windows(fold.test_pos)
    assert w.shape == (len(fold.test_pos), fold.seq_len, len(fold.feature_names))
    p = fold.test_pos[0]
    assert np.array_equal(w[0], fold.X_all[p - fold.seq_len + 1: p + 1])
    # the last row of every window is the day itself (no peeking forward)
    assert np.array_equal(w[:, -1, :], fold.X_all[fold.test_pos])


def test_windows_reject_insufficient_history():
    with pytest.raises(ValueError):
        windows_at(np.zeros((10, 2)), np.array([1]), 5)


def test_deep_validation_split_is_chronological_and_purged(fold):
    tr, val = fold.deep_train_val_positions(0.1)
    assert tr.max() + fold.gap <= val.min()
    assert val.max() < fold.test_pos.min()


# -------------------------------------------------------------------- benchmarks
def test_persistence_uses_trailing_vol_only(fold):
    res = persistence_forecast(fold)
    trailing = pd.Series(fold.trailing_vol[fold.test_pos], index=fold.dates_test)
    assert np.array_equal(res["preds"], fold.labeler.assign_causal(trailing).values)
    assert np.allclose(res["pred_vol"], trailing.values)
    assert np.allclose(res["proba"].sum(axis=1), 1.0)


def test_garch_params_depend_only_on_training_returns(fold, cfg):
    pytest.importorskip("arch")
    from src.models_benchmarks import garch_forecast

    a = garch_forecast(fold, cfg)
    assert a["pred_vol"].shape == (len(fold.test_pos),) and (a["pred_vol"] > 0).all()
    assert 0 < a["params"]["persistence"] < 1.5
    # change returns after the training period -> estimated parameters must not move
    altered = copy.copy(fold)
    ret = fold.label_returns.copy()
    ret.iloc[fold.test_pos] = ret.iloc[fold.test_pos] * 4.0
    altered.label_returns = ret
    b = garch_forecast(altered, cfg)
    for k in ["omega", "alpha", "gamma", "beta"]:
        assert abs(a["params"][k] - b["params"][k]) < 1e-9


# -------------------------------------------------------------------- strategy
def test_exposure_rule_bounds():
    e = predicted_regime_to_exposure(np.array([0, 1, 2]), n_regimes=3)
    assert np.allclose(e, [1.0, 0.5, 0.0])


def test_vol_target_exposure_caps_and_scales():
    e = vol_target_exposure(np.array([0.06, 0.12, 0.24, 0.60]), 0.12, max_leverage=1.0, min_vol_floor=0.03)
    assert np.allclose(e, [1.0, 1.0, 0.5, 0.2])
    lev = vol_target_exposure(np.array([0.06]), 0.12, max_leverage=2.0)
    assert np.isclose(lev[0], 2.0)
    assert np.isfinite(vol_target_exposure(np.array([0.0]), 0.12)[0])    # floor prevents division by ~0


def test_sharpe_ratio_matches_manual_calc():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    rf_d = 0.02 / 252
    manual = np.sqrt(252) * (r - rf_d).mean() / (r - rf_d).std()
    assert abs(manual - sharpe_ratio(r, 0.02)) < 1e-9


def test_sharpe_ratio_zero_vol_returns_zero():
    assert sharpe_ratio(pd.Series([0.0] * 10)) == 0.0


def _toy_prices(n=30):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"SPY": 100 * np.cumprod(1 + np.linspace(-0.01, 0.012, n))}, index=idx)


def test_backtest_full_exposure_equals_buy_and_hold():
    px = _toy_prices()
    dates = px.index[:-1]
    bt = backtest(dates, np.ones(len(dates)), px, "SPY", cost_bps=0.0, cash_earns_rf=True)
    assert np.allclose(bt["strategy_return"], bt["buy_hold_return"])


def test_backtest_timing_decision_today_earns_tomorrow():
    px = _toy_prices()
    dates = px.index[:-1]
    expo = np.zeros(len(dates))
    expo[5] = 1.0                                   # invested only on the decision made at day 5
    bt = backtest(dates, expo, px, "SPY", cost_bps=0.0, risk_free_rate_annual=0.0)
    nonzero = bt.index[bt["strategy_return"] != 0]
    assert list(nonzero) == [dates[5]]
    assert np.isclose(bt.loc[dates[5], "strategy_return"], px["SPY"].iloc[6] / px["SPY"].iloc[5] - 1)


def test_transaction_costs_reduce_returns_by_turnover():
    px = _toy_prices()
    dates = px.index[:-1]
    expo = np.tile([1.0, 0.0], len(dates) // 2 + 1)[: len(dates)]
    free = backtest(dates, expo, px, "SPY", cost_bps=0.0)
    paid = backtest(dates, expo, px, "SPY", cost_bps=10.0)
    diff = free["strategy_return"] - paid["strategy_return"]
    assert np.allclose(diff, free["turnover"] * 10 / 1e4)
    assert paid["strategy_equity"].iloc[-1] < free["strategy_equity"].iloc[-1]


def test_cash_earns_risk_free_when_uninvested():
    px = _toy_prices()
    dates = px.index[:-1]
    bt = backtest(dates, np.zeros(len(dates)), px, "SPY", cost_bps=0.0, risk_free_rate_annual=0.05, cash_earns_rf=True)
    assert np.allclose(bt["strategy_return"], 0.05 / 252)


def test_buy_and_hold_is_identical_across_models(fold, cfg, prices):
    """The old report showed different buy&hold numbers per row; now it cannot."""
    res = {}
    for key, shift in [("a", 0), ("b", 3)]:
        res[key] = pd.DataFrame({
            "date": fold.dates_test, "y_pred": np.clip(fold.y_test + shift, 0, 2),
            "pred_vol": fold.target_vol[fold.test_pos] * (1 + 0.1 * shift),
        })
    rows, curves, bh = evaluate_strategies(res, {"a": "A", "b": "B"}, prices, cfg)
    bh_curves = [c["buy_hold_equity"] for c in curves.values()]
    for c in bh_curves[1:]:
        assert np.allclose(c.values, bh_curves[0].values)
    assert {r["rule"] for r in rows} >= {VOL_RULE}


# ----------------------------------------------------------------------- metrics
def test_macro_f1_ignores_regimes_absent_from_truth():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    s = classification_scores(y_true, y_pred, 3)
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0
    assert s["macro_f1_all_classes"] < 1.0           # strict version counts the absent class as 0


def test_config_defaults_fill_missing_keys():
    cfg = _merge_defaults({"walk_forward": {"enabled": False}}, {"walk_forward": {"enabled": True, "step_days": 5},
                                                                "x": 1})
    assert cfg["walk_forward"] == {"enabled": False, "step_days": 5} and cfg["x"] == 1
