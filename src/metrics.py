"""Small shared helpers: classification scores and the common prediction format.

Every model returns a dict with the same keys so that evaluation, walk-forward
and the strategy layer can treat all models identically:

    {"name": str,
     "preds": int array (n_test,)          predicted regime
     "proba": float array (n_test, K)|None predicted regime probabilities
     "pred_vol": float array (n_test,)     predicted annualized vol over the next horizon}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def classification_scores(y_true, y_pred, n_regimes: int) -> dict:
    """Accuracy and macro-F1.

    ``macro_f1`` averages ONLY over regimes that actually occur in y_true. A regime
    that never occurs in the test window cannot be scored (recall is undefined), and
    averaging in a forced zero would punish every model for something it cannot
    influence. ``macro_f1_all_classes`` is the strict version for reference.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "macro_f1_all_classes": float(
            f1_score(y_true, y_pred, labels=list(range(n_regimes)), average="macro", zero_division=0)
        ),
    }


def align_proba(classes: np.ndarray, proba: np.ndarray, n_regimes: int) -> np.ndarray:
    """Expand sklearn's predict_proba (only the classes seen in training) to K columns."""
    full = np.zeros((proba.shape[0], n_regimes))
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    return full


def result_dict(fold, name: str, proba: np.ndarray) -> dict:
    """Package a probability matrix into the common result format."""
    return {
        "name": name,
        "preds": proba.argmax(axis=1),
        "proba": proba,
        "pred_vol": fold.proba_to_expected_vol(proba),
    }


def pred_frame(fold, res: dict) -> pd.DataFrame:
    """Prediction table for one model on one fold's test dates."""
    pos = fold.test_pos
    df = pd.DataFrame({
        "date": fold.dates_test,
        "y_true": fold.y_test,
        "y_pred": res["preds"],
        "pred_vol": res["pred_vol"],
        "target_vol": fold.target_vol[pos],
        "trailing_vol": fold.trailing_vol[pos],
    })
    if res.get("proba") is not None:
        for i in range(res["proba"].shape[1]):
            df[f"proba_regime_{i}"] = res["proba"][:, i]
    return df
