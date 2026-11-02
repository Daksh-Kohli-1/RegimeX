"""
Benchmarks that give the ML numbers context.

* Persistence (README 9.2): "the next period's regime equals today's regime",
  where today's regime is the regime of the TRAILING realized vol (the window that
  ends today). No learning. A model that cannot beat this adds nothing.
* GARCH (README 9.8): the standard finance volatility model. Parameters are
  estimated on the training rows only, then the variance recursion is run forward
  through the test rows WITHOUT re-estimating (so nothing from the test period
  influences the parameters). The h-step-ahead variance forecasts are averaged
  over the label horizon and mapped to a regime with the same train-fit labeler.

Both return the common result format (see ``metrics.py``), so they flow through
the same evaluation and strategy code as the ML models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset import build_dataset
from src.metrics import classification_scores, pred_frame
from src.utils import load_config, log, resolve_path, set_seed


# ------------------------------------------------------------------ persistence
def persistence_forecast(fold) -> dict:
    trailing = pd.Series(fold.trailing_vol[fold.test_pos], index=fold.dates_test)
    preds = fold.labeler.assign_causal(trailing).values
    proba = np.eye(fold.n_regimes)[preds]
    return {"name": "Persistence (naive)", "preds": preds, "proba": proba,
            "pred_vol": trailing.values.astype(float)}


# ------------------------------------------------------------------------ GARCH
def garch_forecast(fold, cfg: dict) -> dict | None:
    try:
        from arch import arch_model
    except ImportError:
        log("`arch` is not installed - skipping the GARCH benchmark (pip install arch).")
        return None

    variant = cfg.get("benchmarks", {}).get("garch_variant", "gjr")
    o = 1 if variant == "gjr" else 0
    horizon = int(cfg["vol_window"])

    r_pct = fold.label_returns.values * 100.0           # percent returns: friendlier scale for the optimiser
    last_train = int(fold.train_pos[-1])
    am = arch_model(r_pct[: last_train + 1], mean="Zero", vol="GARCH", p=1, o=o, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)
    prm = res.params
    omega, alpha, beta = float(prm["omega"]), float(prm["alpha[1]"]), float(prm["beta[1]"])
    gamma = float(prm["gamma[1]"]) if o else 0.0

    # variance recursion with FIXED parameters over the whole sample
    n = len(r_pct)
    s2 = np.empty(n)                                    # s2[t] = conditional variance of r_t
    s2[0] = float(np.var(r_pct[: last_train + 1]))
    f1 = np.empty(n)                                    # f1[t] = forecast of Var(r_{t+1}) made at t
    for t in range(n):
        shock = (alpha + (gamma if r_pct[t] < 0 else 0.0)) * r_pct[t] ** 2
        f1[t] = omega + shock + beta * s2[t]
        if t + 1 < n:
            s2[t + 1] = f1[t]

    phi = alpha + 0.5 * gamma + beta                    # persistence
    if phi < 0.9999:
        v_long = omega / (1.0 - phi)
        avg_factor = (1.0 - phi ** horizon) / (horizon * (1.0 - phi))   # mean of phi^(h-1), h=1..H
        avg_var = v_long + (f1 - v_long) * avg_factor
    else:                                               # (near-)integrated: flat forecast
        avg_var = f1
    ann_vol = np.sqrt(np.maximum(avg_var, 1e-12) * 252.0) / 100.0       # back to decimal annualized vol

    pred_vol = pd.Series(ann_vol[fold.test_pos], index=fold.dates_test)
    preds = fold.labeler.assign_causal(pred_vol).values
    name = "GJR-GARCH(1,1)" if o else "GARCH(1,1)"
    return {"name": name, "preds": preds, "proba": None, "pred_vol": pred_vol.values,
            "params": {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "persistence": phi}}


# ------------------------------------------------------------------ entry points
def fit_predict(fold, cfg: dict) -> dict:
    bcfg = cfg.get("benchmarks", {})
    out = {}
    if bcfg.get("persistence", True):
        out["persistence"] = persistence_forecast(fold)
    if bcfg.get("garch", True):
        g = garch_forecast(fold, cfg)
        if g is not None:
            out["garch"] = g
    return out


def run(cfg: dict | None = None):
    cfg = cfg or load_config()
    set_seed(cfg.get("random_seed", 42))
    fold, _ = build_dataset(cfg)
    results = fit_predict(fold, cfg)
    outputs_dir = resolve_path(cfg, "outputs_dir")
    for key, res in results.items():
        s = classification_scores(fold.y_test, res["preds"], fold.n_regimes)
        log(f"{res['name']}: accuracy={s['accuracy']:.4f}  macro-F1={s['macro_f1']:.4f}")
        pred_frame(fold, res).to_csv(outputs_dir / f"preds_{key}.csv", index=False)
    return results, fold


if __name__ == "__main__":
    run()
