"""
Stage 3 (baselines) — Logistic Regression and Gradient Boosted Trees, trained on
the flat (non-sequential) engineered feature table to predict the regime of the
forecast horizon. These set the bar the deep models need to clear.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset import build_dataset
from src.metrics import align_proba, classification_scores, pred_frame, result_dict
from src.utils import load_config, log, resolve_path, set_seed


def train_logreg(X_train, y_train, seed: int):
    # multinomial loss is the default for multi-class in modern scikit-learn
    return LogisticRegression(max_iter=2000, random_state=seed).fit(X_train, y_train)


def train_gbt(X_train, y_train, cfg: dict, seed: int):
    p = cfg.get("gbt", {})
    return GradientBoostingClassifier(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", 3),
        learning_rate=p.get("learning_rate", 0.05),
        random_state=seed,
    ).fit(X_train, y_train)


def fit_predict(fold, cfg: dict, return_models: bool = False):
    """Fit both baselines on the fold's training rows, predict its test rows."""
    seed = cfg.get("random_seed", 42)
    k = fold.n_regimes
    out, models = {}, {}

    lr = train_logreg(fold.X_train, fold.y_train, seed)
    out["logreg"] = result_dict(fold, "LogisticRegression", align_proba(lr.classes_, lr.predict_proba(fold.X_test), k))
    models["logreg"] = lr

    gbt = train_gbt(fold.X_train, fold.y_train, cfg, seed)
    out["gbt"] = result_dict(fold, "GradientBoostedTrees", align_proba(gbt.classes_, gbt.predict_proba(fold.X_test), k))
    models["gbt"] = gbt

    return (out, models) if return_models else out


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
