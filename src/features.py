"""
Stage 1b — Feature engineering.

Pure formulas, no econometric theory: log returns, rolling realized
volatility, rolling momentum, RSI, MACD — computed per-ticker and then
assembled into one wide feature table indexed by date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import load_config, log, resolve_path


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling realized volatility from daily log returns."""
    return returns.rolling(window).std() * np.sqrt(252)


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """Simple rolling price momentum: pct change over `window` days."""
    return prices.pct_change(window)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def build_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Build a wide per-ticker feature table. Every feature at row `t` uses
    only information available up to and including day `t` (no look-ahead).
    The label built in labeling.py describes the FUTURE (days t+1 ... t+H when
    label_mode == "forward"), so inputs and target share no days.
    """
    vol_windows = cfg["vol_windows"]
    mom_windows = cfg["momentum_windows"]
    rsi_window = cfg["rsi_window"]
    macd_fast, macd_slow, macd_signal = cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]

    frames = []
    for ticker in prices.columns:
        px = prices[ticker]
        ret = log_returns(px)

        feat = pd.DataFrame(index=prices.index)
        feat[f"{ticker}_logret"] = ret
        for w in vol_windows:
            feat[f"{ticker}_vol_{w}d"] = realized_vol(ret, w)
        for w in mom_windows:
            feat[f"{ticker}_mom_{w}d"] = momentum(px, w)
        feat[f"{ticker}_rsi_{rsi_window}"] = rsi(px, rsi_window)
        macd_line, signal_line, hist = macd(px, macd_fast, macd_slow, macd_signal)
        if cfg.get("macd_normalize_by_price", True):
            # Raw MACD is in price units, so it drifts with the price level (SPY at $100 in
            # 2012 vs $500 in 2024). A model trained on one level would see test values
            # many standard deviations out of range. Dividing by price makes it stationary.
            macd_line, signal_line, hist = macd_line / px, signal_line / px, hist / px
        feat[f"{ticker}_macd"] = macd_line
        feat[f"{ticker}_macd_signal"] = signal_line
        feat[f"{ticker}_macd_hist"] = hist

        frames.append(feat)

    features = pd.concat(frames, axis=1)

    # Cross-asset feature: average pairwise rolling correlation of returns,
    # a cheap proxy for "market stress" (correlations rise in high-vol regimes).
    rets = prices.apply(log_returns)
    roll_corr = rets.rolling(20).corr()
    avg_corr = (
        roll_corr.groupby(level=0)
        .apply(lambda m: (m.values.sum() - np.trace(m.values)) / (m.shape[0] * (m.shape[0] - 1)))
    )
    features["avg_pairwise_corr_20d"] = avg_corr

    features = features.dropna(how="any")
    return features


def run(cfg: dict | None = None) -> pd.DataFrame:
    from src.data_pipeline import get_price_data

    cfg = cfg or load_config()
    prices = get_price_data(cfg)
    features = build_features(prices, cfg)

    data_dir = resolve_path(cfg, "data_dir")
    out_path = data_dir / "features.csv"
    features.to_csv(out_path)
    log(f"Built {features.shape[1]} features x {features.shape[0]} rows -> {out_path}")
    return features


if __name__ == "__main__":
    df = run()
    print(df.tail())
    print(df.shape)
