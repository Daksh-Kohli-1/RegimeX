"""
Stage 5 — Evaluation. Reads the saved predictions of every stage and produces:

  * outputs/model_comparison.csv, strategy_holdout.csv, strategy_walk_forward.csv
  * figures: regime timeline, confusion matrices, equity curves, walk-forward chart,
             feature importance (if explain.py was run)
  * outputs/report.md  - results AND limitations, stated every time

Nothing here trains a model, so `python -m src.evaluate` can be re-run cheaply.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data_pipeline import get_price_data
from src.labeling import REGIME_NAMES
from src.metrics import classification_scores
from src.strategy import TOY_RULE, VOL_RULE, evaluate_strategies
from src.utils import load_config, log, resolve_path

MODEL_ORDER = ["persistence", "garch", "logreg", "gbt", "lstm", "transformer"]
REGIME_COLORS = {0: "#8ecae6", 1: "#ffb703", 2: "#fb8500"}


def _model_names(cfg: dict) -> dict:
    garch = "GJR-GARCH(1,1)" if cfg.get("benchmarks", {}).get("garch_variant", "gjr") == "gjr" else "GARCH(1,1)"
    return {"persistence": "Persistence (naive)", "garch": garch, "logreg": "LogisticRegression",
            "gbt": "GradientBoostedTrees", "lstm": "LSTM", "transformer": "Transformer"}


def _load_preds(outputs_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    out = {}
    for key in MODEL_ORDER:
        p = outputs_dir / f"{prefix}_{key}.csv"
        if p.exists():
            out[key] = pd.read_csv(p, parse_dates=["date"])
    return out


# ------------------------------------------------------------------------ plots
def plot_regime_timeline(df: pd.DataFrame, label_mode: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["target_vol"], color="black", linewidth=0.8,
            label="Target: realized vol of the NEXT window (ann.)" if label_mode == "forward"
            else "Target: realized vol (ann.)")
    top = float(df["target_vol"].max()) * 1.05
    ax.set_ylim(0, top)
    for regime, name in REGIME_NAMES.items():
        mask = (df["regime"] == regime).values
        ax.fill_between(df.index, 0, top, where=mask, color=REGIME_COLORS[regime], alpha=0.25,
                        step="mid", label=name)
    test = df[df["split"] == "test"]
    if len(test):
        ax.axvline(test.index[0], color="red", linestyle="--", linewidth=1, label="test start")
    ax.set_ylabel("Realized volatility")
    ax.set_title("Regime labels over time (labeler fit on TRAIN rows only; purged rows unshaded)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, name: str, out_path: Path, n_regimes: int, title_suffix: str = ""):
    order = list(range(n_regimes))
    cm = confusion_matrix(y_true, y_pred, labels=order)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(order)
    ax.set_yticks(order)
    ax.set_xticklabels([REGIME_NAMES.get(i, i) for i in order], rotation=45, ha="right")
    ax.set_yticklabels([REGIME_NAMES.get(i, i) for i in order])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {name}{title_suffix}", fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_equity_curves(curves: dict, names: dict, title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, rule in zip(axes, [TOY_RULE, VOL_RULE]):
        bh_done = False
        for (key, r), bt in curves.items():
            if r != rule:
                continue
            ax.plot(bt.index, bt["strategy_equity"], label=names[key], linewidth=1.1)
            if not bh_done:
                ax.plot(bt.index, bt["buy_hold_equity"], color="black", linestyle="--", label="Buy & hold")
                bh_done = True
        ax.set_title(rule)
        ax.set_xlabel("Date")
    axes[0].set_ylabel("Growth of $1 (net of costs)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_walk_forward(scores: pd.DataFrame, out_path: Path):
    piv = scores.pivot(index="fold", columns="model", values="accuracy")
    order = sorted(piv.index, key=lambda f: int(f.replace("fold", "")))
    piv = piv.loc[order]
    starts = scores.drop_duplicates("fold").set_index("fold")["test_start"].loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for col in piv.columns:
        ax.plot(range(len(piv)), piv[col], marker="o", label=col, linewidth=1.3)
    ax.set_xticks(range(len(piv)))
    ax.set_xticklabels([s[:4] for s in starts], rotation=0)
    ax.set_xlabel("Test year (start of fold)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Walk-forward accuracy by fold")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- tables
def holdout_table(preds: dict, names: dict, n_regimes: int) -> pd.DataFrame:
    rows = []
    for key, df in preds.items():
        s = classification_scores(df["y_true"], df["y_pred"], n_regimes)
        rows.append({"key": key, "model": names[key], "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                     "macro_f1_all_classes": s["macro_f1_all_classes"]})
    t = pd.DataFrame(rows)
    if "persistence" in t["key"].values:
        pa = float(t.loc[t["key"] == "persistence", "accuracy"].iloc[0])
        t["acc_minus_persistence"] = t["accuracy"] - pa
    return t


def _strategy_md(rows: list, bh: dict | None) -> str:
    if not rows or bh is None:
        return "_No strategy results available._\n"
    df = pd.DataFrame(rows).drop(columns=["key"])
    df = df[["model", "rule", "sharpe_net", "sharpe_gross", "cagr", "ann_vol", "max_drawdown",
             "avg_exposure", "annual_turnover"]]
    df.columns = ["model", "rule", "Sharpe (net)", "Sharpe (gross)", "CAGR (net)", "ann. vol",
                  "max drawdown", "avg exposure", "turnover / yr"]
    bh_row = pd.DataFrame([{
        "model": "Buy & hold", "rule": "-", "Sharpe (net)": bh["sharpe"], "Sharpe (gross)": bh["sharpe"],
        "CAGR (net)": bh["cagr"], "ann. vol": bh["ann_vol"], "max drawdown": bh["max_drawdown"],
        "avg exposure": 1.0, "turnover / yr": 0.0}])
    df = pd.concat([bh_row, df], ignore_index=True)
    return df.to_markdown(index=False, floatfmt=".3f") + "\n"


def _confusion_counts(df: pd.DataFrame, k: int) -> str:
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=list(range(k)))
    return "\n".join("  " + " ".join(f"{v:5d}" for v in row) for row in cm)


# -------------------------------------------------------------------------- run
def run(cfg: dict | None = None, **_ignored):
    cfg = cfg or load_config()
    outputs_dir = resolve_path(cfg, "outputs_dir")
    fig_dir = outputs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = _model_names(cfg)
    k = cfg["n_regimes"]
    label_mode = cfg.get("label_mode", "forward")
    prices = get_price_data(cfg)

    meta_path = outputs_dir / "primary_meta.json"
    timeline_path = outputs_dir / "regime_labels_primary.csv"
    if not meta_path.exists() or not timeline_path.exists():
        raise FileNotFoundError("Run `python -m src.labeling` (or run_all.py) first: primary_meta.json is missing.")
    meta = json.loads(meta_path.read_text())
    timeline = pd.read_csv(timeline_path, index_col=0, parse_dates=True)
    plot_regime_timeline(timeline, label_mode, fig_dir / "regime_labels_timeline.png")

    # ---------------- holdout ----------------
    preds = _load_preds(outputs_dir, "preds")
    if not preds:
        raise FileNotFoundError("No preds_*.csv found - run the model stages first.")
    table = holdout_table(preds, names, k)
    table.sort_values("accuracy", ascending=False).to_csv(outputs_dir / "model_comparison.csv", index=False)
    for key, df in preds.items():
        plot_confusion_matrix(df["y_true"], df["y_pred"], names[key], fig_dir / f"confusion_matrix_{key}.png", k)

    h_rows, h_curves, h_bh = evaluate_strategies(preds, names, prices, cfg)
    if h_rows:
        pd.DataFrame(h_rows).to_csv(outputs_dir / "strategy_holdout.csv", index=False)
        plot_equity_curves(h_curves, names, "Holdout: strategy equity vs buy & hold",
                           fig_dir / "equity_curves.png")

    # ---------------- walk-forward ----------------
    wf_scores_path = outputs_dir / "walk_forward_fold_scores.csv"
    wf_scores = pd.read_csv(wf_scores_path, dtype={"classes_present": str}) if wf_scores_path.exists() else None
    wf_preds = _load_preds(outputs_dir, "walk_forward_preds")
    w_rows, w_curves, w_bh, wf_pooled = [], {}, None, None
    if wf_scores is not None and wf_preds:
        plot_walk_forward(wf_scores, fig_dir / "walk_forward_accuracy.png")
        pooled_rows = []
        for key, df in wf_preds.items():
            s = classification_scores(df["y_true"], df["y_pred"], k)
            pooled_rows.append({"key": key, "model": names[key], "accuracy": s["accuracy"],
                                "macro_f1_all_classes": s["macro_f1_all_classes"]})
            plot_confusion_matrix(df["y_true"], df["y_pred"], names[key],
                                  fig_dir / f"confusion_matrix_walkforward_{key}.png", k, " (walk-forward, pooled)")
        wf_pooled = pd.DataFrame(pooled_rows)
        if "persistence" in wf_pooled["key"].values:
            pa = float(wf_pooled.loc[wf_pooled["key"] == "persistence", "accuracy"].iloc[0])
            wf_pooled["acc_minus_persistence"] = wf_pooled["accuracy"] - pa
        wf_pooled.to_csv(outputs_dir / "walk_forward_pooled.csv", index=False)
        w_rows, w_curves, w_bh = evaluate_strategies(wf_preds, names, prices, cfg)
        if w_rows:
            pd.DataFrame(w_rows).to_csv(outputs_dir / "strategy_walk_forward.csv", index=False)
            plot_equity_curves(w_curves, names, "Walk-forward (pooled out-of-sample): strategy equity vs buy & hold",
                               fig_dir / "equity_curves_walk_forward.png")

    # ---------------- explainability files ----------------
    fam_path = outputs_dir / "feature_importance_by_family.csv"
    imp_path = outputs_dir / "feature_importance.csv"

    # ---------------- report ----------------
    is_synthetic = (resolve_path(cfg, "data_dir") / "SYNTHETIC_DATA_WAS_USED.txt").exists()
    report_path = outputs_dir / "report.md"
    out = []
    w = out.append
    w("# Volatility Regime Forecasting - Results Report\n")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    if is_synthetic:
        w("> **DATA NOTICE: this run used SYNTHETIC data** (regime-switching GBM, not real market prices). "
          "Results are for pipeline validation / demo purposes only and must not be read as market findings. "
          "Set `use_synthetic: false` in config.yaml and ensure network access to yfinance to reproduce with real data.\n")
    if label_mode == "backward":
        w("> **LABEL NOTICE: `label_mode: backward` (legacy).** The label window shares 19 of 20 days with the "
          "input features, so the accuracies below overstate real forecasting skill. Use `label_mode: forward` "
          "for an honest forecast.\n")

    w("## 1. Task and experimental design\n")
    horizon = cfg["vol_window"]
    w(f"Predict the volatility regime ({', '.join(REGIME_NAMES.values())}) of `{cfg['label_ticker']}` with "
      f"`{cfg['labeling_method']}` regimes. Label mode: **{label_mode}**"
      + (f" - the target for day *t* is the realized volatility of days *t+1 ... t+{horizon}*; the inputs "
         f"describe days up to *t*, so the two share no days. " if label_mode == "forward" else ". ")
      + f"The regime labeler and the feature scaler are fit on training rows only. "
        f"A purge gap of **{meta['gap_days']} trading days** separates train from test so no training label "
        f"window reaches into the test period.\n")
    w(f"Universe: {len(prices.columns)} tickers, {prices.index[0].date()} to {prices.index[-1].date()}, "
      f"daily adjusted closes.\n")

    # ---- holdout
    w("## 2. Holdout results (single chronological split)\n")
    w(f"Train: {meta['train_start']} to {meta['train_end']} ({meta['n_train']} rows). "
      f"Test: {meta['test_start']} to {meta['test_end']} ({meta['n_test']} rows). "
      f"Every model is scored on exactly these test dates.\n")
    share = ", ".join(f"{REGIME_NAMES[i]} {s:.1%}" for i, s in enumerate(meta["test_class_share"]))
    w(f"Test-window class shares: {share}.\n")
    missing = [REGIME_NAMES[i] for i in range(k) if i not in meta["test_classes_present"]]
    if missing:
        w(f"> **Regime(s) absent from the test window: {', '.join(missing)}.** Models were never tested on "
          f"them in this split, and macro-F1 averages only the regimes that occur. See the walk-forward "
          f"section for coverage of rarer regimes.\n")
    floors = f"random guessing = **{1.0 / k:.3f}**, always predicting the most common training regime = " \
             f"**{meta['majority_baseline_test_acc']:.3f}** (on this test window)"
    if "persistence" in preds:
        pa = float(table.loc[table["key"] == "persistence", "accuracy"].iloc[0])
        floors += f", persistence = **{pa:.3f}**"
    w(f"Reference floors: {floors}.\n")
    show = table.sort_values("accuracy", ascending=False).drop(columns=["key"]).rename(
        columns={"macro_f1": "macro-F1 (present classes)", "macro_f1_all_classes": "macro-F1 (all classes)",
                 "acc_minus_persistence": "acc - persistence"})
    w(show.to_markdown(index=False, floatfmt=".3f") + "\n")

    ml_keys = [kk for kk in ["logreg", "gbt", "lstm", "transformer"] if kk in preds]
    if "persistence" in preds and ml_keys:
        beat = [names[kk] for kk in ml_keys
                if float(table.loc[table["key"] == kk, "accuracy"].iloc[0]) > pa]
        if beat:
            w(f"Models with higher holdout accuracy than persistence: {', '.join(beat)}. "
              f"The margin over persistence, not the headline accuracy, is the measure of added skill.\n")
        else:
            w("**No ML model beats the persistence baseline on holdout accuracy.** On this split the "
              "models add no measurable skill over \"tomorrow's regime = today's regime\".\n")
    w("Confusion matrices: `outputs/figures/confusion_matrix_<model>.png` (rows = true, columns = predicted).\n")
    w("Counts below are rows = true regime (low, medium, high), columns = predicted:\n")
    for key, df in preds.items():
        w(f"```\n{names[key]}\n{_confusion_counts(df, k)}\n```\n")

    # ---- walk-forward
    w("## 3. Walk-forward validation (expanding window, purged)\n")
    if wf_scores is None or wf_pooled is None:
        w("_Walk-forward results not found. Run `python -m src.walkforward` (or `run_all.py`)._\n")
    else:
        folds_meta = json.loads((outputs_dir / "walk_forward_meta.json").read_text())
        w(f"{len(folds_meta)} folds, each testing on ~{cfg['walk_forward']['test_window_days']} trading days "
          f"that follow the training window. The regime labeler, scaler and every model are re-fit per fold "
          f"on that fold's training rows. **Regimes are relative to each fold's own training distribution** "
          f"(a calm training period labels smaller spikes as \"high\"), so the meaning of a regime shifts "
          f"slightly between folds.\n")
        piv = wf_scores.pivot(index="fold", columns="model", values="accuracy")
        info = wf_scores.drop_duplicates("fold").set_index("fold")[["test_start", "test_end", "classes_present", "high_vol_share"]]
        tab = info.join(piv)
        tab = tab.loc[sorted(tab.index, key=lambda f: int(f.replace("fold", "")))]
        tab = tab.rename(columns={"classes_present": "regimes in test", "high_vol_share": "high-vol share"}).reset_index()
        w("**Accuracy by fold**\n")
        w(tab.to_markdown(index=False, floatfmt=".3f") + "\n")

        agg = wf_scores.groupby("model").agg(
            acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std")).reset_index()
        if "persistence" in wf_scores["key"].values:
            pk = wf_scores[wf_scores["key"] == "persistence"].set_index("fold")["accuracy"]
            wins = {}
            for key, g in wf_scores.groupby("key"):
                wins[names[key]] = f"{int((g.set_index('fold')['accuracy'] > pk.reindex(g['fold']).values).sum())}/{len(g)}"
            agg["folds beating persistence"] = agg["model"].map(wins)
        agg = agg.rename(columns={"acc_mean": "accuracy mean", "acc_std": "accuracy std",
                                  "f1_mean": "macro-F1 mean", "f1_std": "macro-F1 std"})
        w("**Mean and spread across folds**\n")
        w(agg.sort_values("accuracy mean", ascending=False).to_markdown(index=False, floatfmt=".3f") + "\n")
        w("**Pooled out-of-sample predictions (all folds together)**\n")
        w(wf_pooled.drop(columns=["key"]).sort_values("accuracy", ascending=False)
          .rename(columns={"macro_f1_all_classes": "macro-F1 (all classes)",
                           "acc_minus_persistence": "acc - persistence"})
          .to_markdown(index=False, floatfmt=".3f") + "\n")
        n_hi = int((wf_scores.drop_duplicates("fold")["high_vol_share"] > 0).sum())
        w(f"The high-volatility regime appears in {n_hi} of {len(folds_meta)} test windows, so crisis behaviour "
          f"is tested here in a way the single holdout cannot. Figure: `outputs/figures/walk_forward_accuracy.png`.\n")

    # ---- labels
    w("## 4. Regime labels\n")
    lab = meta.get("labeler", {})
    w(f"Primary-split labeler (`{lab.get('method')}`, fit on training rows only): "
      f"mean volatility by regime on training data = "
      f"{', '.join(f'{REGIME_NAMES[i]} {v:.3f}' for i, v in enumerate(lab.get('mean_vol_by_regime_on_train', [])))}.\n")
    if "thresholds_vol" in lab:
        w(f"Percentile thresholds (vol): {', '.join(f'{t:.3f}' for t in lab['thresholds_vol'])}.\n")
    if "transition_matrix" in lab:
        tm = pd.DataFrame(lab["transition_matrix"], index=[REGIME_NAMES[i] for i in range(k)],
                          columns=[REGIME_NAMES[i] for i in range(k)])
        w("HMM transition matrix (rows = from, columns = to):\n")
        w(tm.to_markdown(floatfmt=".3f") + "\n")
        w("Expected regime duration (days): " + ", ".join(
            f"{REGIME_NAMES[i]} {d:.0f}" for i, d in enumerate(lab["expected_duration_days"])) + ".\n")
    w("![Regime timeline](figures/regime_labels_timeline.png)\n")

    # ---- strategy
    w("## 5. Strategy layer (illustrative only)\n")
    asset = cfg["strategy_asset"]
    w(f"Two rules on `{asset}`: the original **toy regime rule** (`exposure = 1 - regime/(K-1)`) and "
      f"**volatility targeting** (`exposure = {cfg['vol_target']['target_annual_vol']:.0%} / predicted vol`, "
      f"capped at {cfg['vol_target']['max_leverage']:.1f}x). Costs: {cfg.get('transaction_cost_bps', 0)} bps per unit "
      f"of turnover on every rule; un-invested cash "
      f"{'earns' if cfg.get('cash_earns_rf', True) else 'does not earn'} the risk-free rate "
      f"({cfg['risk_free_rate_annual']:.0%}). Exposure decided at the close of day *t* earns day *t+1*'s return. "
      f"All models and buy & hold are measured on identical days. "
      f"Vol targeting with the **Persistence** row is simply \"scale by recent realized vol\" - the no-ML "
      f"benchmark a forecaster has to beat to justify itself.\n")
    w(f"**Holdout ({h_bh['start']} to {h_bh['end']}, {h_bh['n_days']} days)**\n" if h_bh else "**Holdout**\n")
    w(_strategy_md(h_rows, h_bh))
    w("![Equity curves](figures/equity_curves.png)\n")
    if w_rows:
        w(f"**Walk-forward, pooled out-of-sample ({w_bh['start']} to {w_bh['end']}, {w_bh['n_days']} days)**\n")
        w(_strategy_md(w_rows, w_bh))
        w("![Walk-forward equity curves](figures/equity_curves_walk_forward.png)\n")
    w("Read these as a comparison of *rules and forecasts under the same assumptions*, not as evidence of a "
      "tradable edge: one asset, one history, simple cost model, no capacity or tax effects.\n")

    # ---- explain
    w("## 6. What drives the gradient-boosted model\n")
    if fam_path.exists() and imp_path.exists():
        fam = pd.read_csv(fam_path)
        imp = pd.read_csv(imp_path)
        fam = fam.rename(columns={"impurity_share": "share of GBT impurity importance",
                                  "perm_acc_drop": "sum of permutation accuracy drop"})
        w(fam.to_markdown(index=False, floatfmt=".3f") + "\n")
        w("Top individual features by permutation importance on the test window:\n")
        top = imp.head(10)[["feature", "family", "gbt_impurity", "perm_acc_drop_mean"]]
        w(top.to_markdown(index=False, floatfmt=".4f") + "\n")
        w("![Importance by family](figures/feature_importance_groups.png)\n")
        w("Importance is a description of what the fitted model uses, not proof of a causal driver.\n")
    else:
        w("_Run `python -m src.explain` to generate feature importance._\n")

    # ---- limitations
    w("## 7. Explicit limitations\n")
    lim = [
        "**Regime labels are a proxy**, produced by an unsupervised model on realized volatility, not ground-truth economic regimes. They are defined relative to each training window.",
        "**One history.** Walk-forward adds several test windows but they are still one path of one market; rarer regimes have few observations.",
        "**Single asset target and a small universe** (the configured ETFs), daily adjusted closes only. Intraday realized volatility is not implemented.",
        "**Simple cost model** (proportional bps on turnover): no slippage that grows with size, no market impact, no taxes, no borrowing spread beyond the risk-free rate.",
        "**Simple sizing rules.** Vol targeting is implemented; Kelly, risk parity and drawdown control are not.",
        "**Hyper-parameters are fixed** (not tuned inside each walk-forward fold), which is conservative but not optimal.",
        "**Persistence and GARCH are strong benchmarks.** If a model does not clearly beat them, the honest conclusion is that it adds little.",
    ]
    if label_mode == "backward":
        lim.insert(0, "**Label overlap** (`label_mode: backward`): 19 of 20 label days are already inside the inputs; accuracy is inflated.")
    if is_synthetic:
        lim.append("**Synthetic data**: see the data notice at the top.")
    w("\n".join(f"- {x}" for x in lim) + "\n")

    w("## 8. Extension points\n")
    w("- Intraday (e.g. 5-minute) data for a more precise realized-volatility target\n"
      "- Kelly / risk-parity sizing and drawdown control\n"
      "- Forecasting with HMM transition probabilities and multi-horizon targets\n"
      "- Tuning hyper-parameters inside each walk-forward fold\n")

    report_path.write_text("\n".join(out))
    log(f"Wrote report to {report_path}")
    return report_path


if __name__ == "__main__":
    run()
