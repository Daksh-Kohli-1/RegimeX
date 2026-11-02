"""
Stage 4 — Strategy layer (framing only; explicitly NOT a real trading system).

Two exposure rules are evaluated for every model's predictions:

1. "Toy regime rule" (the original): exposure = 1 - predicted_regime / (K - 1).
   Only ever de-risks; ignores how big the predicted vol is.

2. "Volatility targeting" (README 9.6): exposure = target_vol / predicted_vol,
   capped at max_leverage (default 1.0 = no leverage). Hold less when the model
   expects turbulence, more when it expects calm, so portfolio risk stays near a
   constant level. Applied to the Persistence model this is plain
   "scale by recent realized vol" - the benchmark any regime forecaster has to
   beat to justify itself.

Realism switches (config.yaml):
  * transaction_cost_bps  : cost per unit of turnover, charged on every rule
  * cash_earns_rf         : the un-invested part earns the risk-free rate

Timing: the exposure decided at the close of day t (from information up to t)
earns the return of day t+1. Buy-and-hold is computed on exactly the same
days for every model, so it is identical across rows by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import load_config, log

TOY_RULE = "Toy regime rule"
VOL_RULE = "Vol targeting"


# ------------------------------------------------------------------ exposures
def predicted_regime_to_exposure(pred_regime: np.ndarray, n_regimes: int) -> np.ndarray:
    return 1 - np.asarray(pred_regime) / (n_regimes - 1)


def vol_target_exposure(pred_vol: np.ndarray, target_annual_vol: float,
                        max_leverage: float = 1.0, min_vol_floor: float = 0.03) -> np.ndarray:
    pv = np.maximum(np.asarray(pred_vol, dtype=float), min_vol_floor)
    return np.clip(target_annual_vol / pv, 0.0, max_leverage)


# ------------------------------------------------------------------- metrics
def sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float = 0.02, periods_per_year: int = 252) -> float:
    rf_daily = risk_free_rate_annual / periods_per_year
    excess = returns - rf_daily
    if len(excess) == 0 or excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] ** (periods_per_year / len(equity)) - 1.0)


# ------------------------------------------------------------------ backtest
def backtest(dates, exposure, prices: pd.DataFrame, asset: str, cost_bps: float = 0.0,
             risk_free_rate_annual: float = 0.02, cash_earns_rf: bool = True) -> pd.DataFrame:
    """Next-day backtest. Row `t` = decision made at close of t, return earned on t+1."""
    dates = pd.DatetimeIndex(dates)
    r_next = prices[asset].pct_change().shift(-1).reindex(dates)
    e = pd.Series(np.asarray(exposure, dtype=float), index=dates)
    ok = r_next.notna()
    r_next, e = r_next[ok], e[ok]

    turnover = e.diff().abs().fillna(0.0)           # initial entry is not charged (same for all rows)
    cost = turnover * cost_bps / 1e4
    rf_d = risk_free_rate_annual / 252.0
    gross = e * r_next + ((1.0 - e) * rf_d if cash_earns_rf else 0.0)

    out = pd.DataFrame({
        "exposure": e, "turnover": turnover, "asset_return": r_next,
        "strategy_gross_return": gross, "strategy_return": gross - cost,
        "buy_hold_return": r_next,
    })
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = (1 + out["buy_hold_return"]).cumprod()
    return out


def summarize(bt: pd.DataFrame, rf: float) -> dict:
    return {
        "sharpe_net": sharpe_ratio(bt["strategy_return"], rf),
        "sharpe_gross": sharpe_ratio(bt["strategy_gross_return"], rf),
        "total_return": float(bt["strategy_equity"].iloc[-1] - 1),
        "cagr": cagr(bt["strategy_equity"]),
        "ann_vol": float(bt["strategy_return"].std() * np.sqrt(252)),
        "max_drawdown": max_drawdown(bt["strategy_equity"]),
        "avg_exposure": float(bt["exposure"].mean()),
        "annual_turnover": float(bt["turnover"].mean() * 252),
    }


def evaluate_strategies(preds: dict[str, pd.DataFrame], names: dict[str, str],
                        prices: pd.DataFrame, cfg: dict):
    """Run both rules for every model on the SAME dates.

    preds: key -> DataFrame with columns date, y_pred, pred_vol
    returns (rows, curves, buy_hold_summary)
    """
    asset = cfg["strategy_asset"]
    rf = cfg["risk_free_rate_annual"]
    cost = cfg.get("transaction_cost_bps", 0.0)
    cash = cfg.get("cash_earns_rf", True)
    vt = cfg["vol_target"]
    k = cfg["n_regimes"]

    common = None
    for df in preds.values():
        d = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        common = d if common is None else common.intersection(d)
    if common is None or len(common) == 0:
        return [], {}, None

    rows, curves, bh = [], {}, None
    for key, df in preds.items():
        d = df.assign(date=pd.to_datetime(df["date"])).set_index("date").loc[common]
        exposures = {
            TOY_RULE: predicted_regime_to_exposure(d["y_pred"].values, k),
            VOL_RULE: vol_target_exposure(d["pred_vol"].values, vt["target_annual_vol"],
                                          vt["max_leverage"], vt["min_vol_floor"]),
        }
        for rule, expo in exposures.items():
            bt = backtest(common, expo, prices, asset, cost, rf, cash)
            s = summarize(bt, rf)
            rows.append({"key": key, "model": names[key], "rule": rule, **s})
            curves[(key, rule)] = bt
            if bh is None:
                bh = {
                    "sharpe": sharpe_ratio(bt["buy_hold_return"], rf),
                    "total_return": float(bt["buy_hold_equity"].iloc[-1] - 1),
                    "cagr": cagr(bt["buy_hold_equity"]),
                    "ann_vol": float(bt["buy_hold_return"].std() * np.sqrt(252)),
                    "max_drawdown": max_drawdown(bt["buy_hold_equity"]),
                    "n_days": int(len(bt)),
                    "start": str(bt.index[0].date()), "end": str(bt.index[-1].date()),
                }
    return rows, curves, bh


if __name__ == "__main__":
    from src.data_pipeline import get_price_data
    from src.utils import resolve_path

    cfg = load_config()
    prices = get_price_data(cfg)
    out = resolve_path(cfg, "outputs_dir")
    df = pd.read_csv(out / "preds_gbt.csv")
    rows, _, bh = evaluate_strategies({"gbt": df}, {"gbt": "GradientBoostedTrees"}, prices, cfg)
    for r in rows:
        log(f"{r['rule']}: net Sharpe={r['sharpe_net']:.3f}  (buy&hold {bh['sharpe']:.3f})")
