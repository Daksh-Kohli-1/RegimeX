"""
Stage 1a — Data pipeline.

Downloads daily OHLCV for a basket of liquid tickers via yfinance and caches
it to disk as parquet/csv. Falls back to a synthetic regime-switching price
generator when `use_synthetic: true` in config.yaml, or automatically if
yfinance / network access fails, so the rest of the pipeline is never
blocked by data access.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import active_start_date, active_tickers, load_config, log, resolve_path, set_seed

CACHE_FILENAME = "prices_raw.csv"
CACHE_META_FILENAME = "prices_meta.json"
SYNTHETIC_FLAG_FILENAME = "SYNTHETIC_DATA_WAS_USED.txt"


def generate_synthetic_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic-but-realistic multi-asset daily price panel with
    regime-switching volatility (a 3-state Markov-modulated GBM per asset,
    with a shared latent "market regime" driving correlated vol spikes
    across assets — this is what gives GMM something non-trivial to find).

    This is NOT meant to resemble real market statistics closely (no fat
    tails calibration, no real cross-asset correlation structure) — it
    exists purely so the full pipeline can run end-to-end without network
    access, with output clearly labeled as synthetic downstream.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, end_date)
    n = len(dates)

    # Shared latent regime path: 0 = calm, 1 = normal, 2 = turbulent.
    # Markov chain with sticky states (regimes persist for weeks/months).
    trans = np.array([
        [0.985, 0.013, 0.002],
        [0.02, 0.965, 0.015],
        [0.01, 0.04, 0.95],
    ])
    regime_vol_mult = {0: 0.6, 1: 1.0, 2: 2.4}
    regime_drift_mult = {0: 1.3, 1: 1.0, 2: -0.4}

    state = 1
    states = np.empty(n, dtype=int)
    for t in range(n):
        states[t] = state
        state = rng.choice(3, p=trans[state])

    panel = {}
    for ticker in tickers:
        base_drift = rng.uniform(0.0004, 0.0009)   # ~10-20% annualized
        base_vol = rng.uniform(0.008, 0.016)         # ~13-25% annualized
        beta_to_market = rng.uniform(0.6, 1.3)        # cross-asset correlation knob

        market_shock = rng.normal(0, 1, size=n)
        idio_shock = rng.normal(0, 1, size=n)

        returns = np.empty(n)
        for t in range(n):
            vmult = regime_vol_mult[states[t]]
            dmult = regime_drift_mult[states[t]]
            vol_t = base_vol * vmult
            shock = beta_to_market * market_shock[t] + np.sqrt(max(1 - beta_to_market**2, 0.05)) * idio_shock[t]
            returns[t] = base_drift * dmult + vol_t * shock

        price = 100 * np.exp(np.cumsum(returns))
        panel[ticker] = price

    prices = pd.DataFrame(panel, index=dates)
    prices.index.name = "Date"
    return prices


def _download_real_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    log(f"Downloading {len(tickers)} tickers from yfinance ({start_date} -> {end_date})...")
    raw = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if raw.empty:
        raise RuntimeError("yfinance returned an empty dataframe.")

    # yf.download with multiple tickers returns a MultiIndex column frame.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers
    close = close.dropna(how="all")
    close.index.name = "Date"
    return close


def get_price_data(cfg: dict | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point. Returns a DataFrame of daily close prices, columns =
    tickers, index = trading dates. Caches to data/prices_raw.csv.
    """
    cfg = cfg or load_config()
    set_seed(cfg.get("random_seed", 42))
    data_dir = resolve_path(cfg, "data_dir")
    cache_path = data_dir / CACHE_FILENAME
    meta_path = data_dir / CACHE_META_FILENAME
    flag_path = data_dir / SYNTHETIC_FLAG_FILENAME

    tickers = active_tickers(cfg)
    start_date, end_date = active_start_date(cfg), cfg["end_date"]
    wanted = {"tickers": tickers, "start": start_date, "end": end_date}

    if cache_path.exists() and not force_refresh:
        # Re-use the cache only if it was built for the same universe and dates.
        cache_ok = False
        if meta_path.exists():
            try:
                saved = json.loads(meta_path.read_text())
                cache_ok = all(saved.get(k) == v for k, v in wanted.items())
                # a synthetic cache must not silently stand in for real data (or vice versa)
                if cache_ok and flag_path.exists() != bool(cfg.get("use_synthetic", False)):
                    cache_ok = False
            except Exception:  # noqa: BLE001
                cache_ok = False
        if cache_ok:
            log(f"Loading cached price data from {cache_path}")
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if flag_path.exists():
                log("NOTE: cached data is SYNTHETIC (see config use_synthetic).")
            return prices
        log("Cached prices do not match the current config (tickers/dates); rebuilding.")

    used_synthetic = bool(cfg.get("use_synthetic", False))
    prices = None
    if not used_synthetic:
        try:
            prices = _download_real_data(tickers, start_date, end_date)
            if prices.isna().mean().mean() > 0.3:
                raise RuntimeError("Downloaded data mostly NaN — treating as failed fetch.")
        except Exception as e:  # noqa: BLE001
            log(f"Real data download failed ({e}); falling back to synthetic data.")
            used_synthetic = True

    if used_synthetic:
        prices = generate_synthetic_data(
            tickers, start_date, end_date, seed=cfg.get("random_seed", 42)
        )
        flag_path.write_text(
            "This dataset is SYNTHETIC (regime-switching GBM), not real market data.\n"
            "Generated because use_synthetic=true in config.yaml, or because the real\n"
            "data download failed/was unavailable (no network access, yfinance error,\n"
            "etc). See src/data_pipeline.py::generate_synthetic_data.\n"
        )
    elif flag_path.exists():
        flag_path.unlink()

    prices = prices.ffill().dropna(how="any")
    prices.to_csv(cache_path)
    meta_path.write_text(json.dumps(wanted))
    log(f"Saved price data: {prices.shape[0]} rows x {prices.shape[1]} tickers -> {cache_path}")
    return prices


if __name__ == "__main__":
    cfg = load_config()
    df = get_price_data(cfg, force_refresh=True)
    print(df.tail())
    print(df.shape)
