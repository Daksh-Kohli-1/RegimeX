"""
Explainability (README 9.9): which inputs does the gradient-boosted model use?

* GBT impurity importance        (how much each feature reduces training loss)
* Permutation importance (test)  (how much test accuracy drops if one feature is shuffled)
* SHAP mean |value| (optional)   only if the `shap` package is installed

Features are also grouped by family (volatility / momentum / RSI / MACD / return /
cross-asset correlation) and by ticker, because "which kind of signal" is more
readable than a list of 100 column names. This is a sanity check: if a forward-label
model leaned on something implausible we would see it here.
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import models_baseline
from src.dataset import build_dataset
from src.utils import load_config, log, resolve_path, set_seed


def feature_family(name: str) -> str:
    if name.startswith("avg_pairwise_corr"):
        return "cross-asset correlation"
    m = re.match(r"^[A-Z0-9.\-]+_(logret|vol|mom|rsi|macd)", name)
    if not m:
        return "other"
    return {"logret": "daily return", "vol": "volatility", "mom": "momentum",
            "rsi": "RSI", "macd": "MACD"}[m.group(1)]


def feature_ticker(name: str) -> str:
    return "cross-asset" if name.startswith("avg_pairwise_corr") else name.split("_")[0]


def run(cfg: dict | None = None):
    cfg = cfg or load_config()
    ecfg = cfg.get("explain", {})
    if not ecfg.get("enabled", True):
        log("Explainability disabled in config - skipping.")
        return None
    seed = cfg.get("random_seed", 42)
    set_seed(seed)

    fold, _ = build_dataset(cfg)
    _, models = models_baseline.fit_predict(fold, cfg, return_models=True)
    gbt = models["gbt"]
    outputs_dir = resolve_path(cfg, "outputs_dir")
    fig_dir = outputs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("Computing permutation importance on the test window...")
    perm = permutation_importance(
        gbt, fold.X_test, fold.y_test, n_repeats=int(ecfg.get("permutation_repeats", 5)),
        random_state=seed, scoring="accuracy", n_jobs=1,
    )
    imp = pd.DataFrame({
        "feature": fold.feature_names,
        "family": [feature_family(f) for f in fold.feature_names],
        "ticker": [feature_ticker(f) for f in fold.feature_names],
        "gbt_impurity": gbt.feature_importances_,
        "perm_acc_drop_mean": perm.importances_mean,
        "perm_acc_drop_std": perm.importances_std,
    })

    if ecfg.get("use_shap", True):
        try:
            import shap

            sample = fold.X_test[:: max(1, len(fold.X_test) // 300)]
            sv = shap.TreeExplainer(gbt).shap_values(sample)
            arr = np.abs(np.array(sv))                      # list-of-classes or (n, f, k) array
            if arr.ndim == 3:
                axes = [a for a in range(3) if arr.shape[a] == len(fold.feature_names)]
                other = tuple(a for a in range(3) if a != axes[0])
                imp["shap_mean_abs"] = arr.mean(axis=other)
            log("SHAP values computed.")
        except ImportError:
            log("`shap` not installed - using impurity + permutation importance only "
                "(optional: pip install -r requirements-optional.txt).")
        except Exception as e:  # noqa: BLE001
            log(f"SHAP step skipped ({type(e).__name__}: {e}).")

    imp = imp.sort_values("perm_acc_drop_mean", ascending=False).reset_index(drop=True)
    imp.to_csv(outputs_dir / "feature_importance.csv", index=False)

    # by family / by ticker (share of total impurity importance, plus permutation drop)
    def group_table(col):
        g = imp.groupby(col).agg(impurity_share=("gbt_impurity", "sum"),
                                 perm_acc_drop=("perm_acc_drop_mean", "sum"),
                                 n_features=("feature", "count"))
        return g.sort_values("impurity_share", ascending=False)

    fam, tic = group_table("family"), group_table("ticker")
    fam.to_csv(outputs_dir / "feature_importance_by_family.csv")
    tic.to_csv(outputs_dir / "feature_importance_by_ticker.csv")

    # plots
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["perm_acc_drop_mean"], xerr=top["perm_acc_drop_std"], color="#2a6f97")
    ax.set_xlabel("Drop in test accuracy when feature is shuffled")
    ax.set_title("GBT permutation importance (top 15)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance_top.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    f = fam.iloc[::-1]
    ax.barh(f.index, f["impurity_share"], color="#e07a5f")
    ax.set_xlabel("Share of GBT impurity importance")
    ax.set_title("Importance by feature family")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance_groups.png", dpi=130)
    plt.close(fig)

    log("Top families: " + ", ".join(f"{i} {r.impurity_share:.0%}" for i, r in fam.head(3).iterrows()))
    return imp


if __name__ == "__main__":
    run()
