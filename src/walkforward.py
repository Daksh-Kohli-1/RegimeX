"""
Walk-forward validation (README 9.4) with purging/embargo (README 9.5).

Expanding training window, one-year test windows that move forward in time:

    fold1: train 2012-2015                  | test 2016
    fold2: train 2012-2016                  | test 2017
    ...

In EVERY fold: the regime labeler and the scaler are re-fit on that fold's
training rows only, the last `gap` training rows are purged, and every model
(LogReg, GBT, Persistence, GARCH, LSTM, Transformer) is re-trained from scratch.
Pooled out-of-sample predictions from all folds are saved so the strategy layer
can be evaluated across the whole walk-forward period (which includes 2020).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import models_baseline, models_benchmarks, models_deep
from src.dataset import load_frame, walk_forward_folds
from src.metrics import classification_scores, pred_frame
from src.utils import load_config, log, resolve_path, set_seed, torch_available


def run(cfg: dict | None = None) -> pd.DataFrame | None:
    cfg = cfg or load_config()
    wf = cfg["walk_forward"]
    if not wf.get("enabled", True):
        log("Walk-forward disabled in config - skipping.")
        return None
    set_seed(cfg.get("random_seed", 42))

    frame, feature_cols = load_frame(cfg)
    folds = walk_forward_folds(frame, feature_cols, cfg)
    if not folds:
        log("Not enough data for a single walk-forward fold - lower walk_forward.min_train_days.")
        return None

    use_deep = bool(wf.get("include_deep", True)) and torch_available()
    if wf.get("include_deep", True) and not torch_available():
        log("torch not installed - walk-forward will run without LSTM/Transformer.")

    outputs_dir = resolve_path(cfg, "outputs_dir")
    score_rows, pooled, metas = [], {}, []

    for fold in folds:
        log(f"Walk-forward {fold.name}: train {fold.dates_train[0].date()} -> {fold.dates_train[-1].date()} "
            f"(+{fold.gap}d purge) | test {fold.dates_test[0].date()} -> {fold.dates_test[-1].date()}")
        results = {}
        results.update(models_baseline.fit_predict(fold, cfg))
        results.update(models_benchmarks.fit_predict(fold, cfg))
        if use_deep:
            results.update(models_deep.fit_predict(fold, cfg, verbose=False))

        meta = fold.meta()
        meta["labeler"] = fold.labeler.summary()
        metas.append(meta)

        for key, res in results.items():
            s = classification_scores(fold.y_test, res["preds"], fold.n_regimes)
            score_rows.append({
                "fold": fold.name, "key": key, "model": res["name"],
                "test_start": meta["test_start"], "test_end": meta["test_end"],
                "n_test": meta["n_test"],
                "classes_present": "".join(str(c) for c in meta["test_classes_present"]),
                "high_vol_share": meta["test_class_share"][-1],
                **s,
            })
            pooled.setdefault(key, []).append(pred_frame(fold, res).assign(fold=fold.name))
        line = "  " + "  ".join(f"{r['model'][:6]}={r['accuracy']:.3f}" for r in score_rows[-len(results):])
        log(line)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(outputs_dir / "walk_forward_fold_scores.csv", index=False)
    for key, parts in pooled.items():
        pd.concat(parts, ignore_index=True).to_csv(outputs_dir / f"walk_forward_preds_{key}.csv", index=False)
    (outputs_dir / "walk_forward_meta.json").write_text(json.dumps(metas, indent=2))
    log(f"Walk-forward finished: {len(folds)} folds -> {outputs_dir}/walk_forward_*.csv")
    return scores


if __name__ == "__main__":
    run()
