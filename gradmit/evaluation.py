"""Cross-validation protocol, metrics and fold-level summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "brier"]
PRIMARY_METRIC = "balanced_accuracy"


def make_cv(n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )


def cv_splits(X, y, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> list[dict]:
    """Materialise the folds once so every model is scored on identical splits."""
    cv = make_cv(n_splits, n_repeats)
    return [
        {"repeat": i // n_splits, "fold": i % n_splits, "train": tr, "test": te}
        for i, (tr, te) in enumerate(cv.split(X, y))
    ]


def compute_metrics(y_true, y_pred, y_score=None, y_prob=None) -> dict:
    """Classification metrics for one validation fold.

    ``y_score`` is any ranking score (used for ROC-AUC, only when both classes are
    present). ``y_prob`` is a probability of the positive class (used for Brier).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "brier": np.nan,
    }
    both_classes = len(np.unique(y_true)) == 2
    if y_score is not None and both_classes:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    if y_prob is not None:
        out["brier"] = brier_score_loss(y_true, y_prob)
    return {k: float(v) for k, v in out.items()}


def corrected_ci(values, test_train_ratio: float = 1 / (N_SPLITS - 1), level: float = 0.95):
    """Mean, sample SD and a 95% CI for repeated-CV fold scores.

    Folds from repeated CV share training data, so a naive t-interval is too
    narrow. This uses the Nadeau-Bengio corrected resampled variance:
    ``se = sd * sqrt(1/J + n_test/n_train)``.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    j = len(v)
    if j == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(v.mean())
    if j == 1:
        return {"mean": mean, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    sd = float(v.std(ddof=1))
    se = sd * np.sqrt(1 / j + test_train_ratio)
    half = float(stats.t.ppf(0.5 + level / 2, df=j - 1) * se)
    return {"mean": mean, "std": sd, "ci_low": mean - half, "ci_high": mean + half}


def summarise_folds(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (target_threshold, model) with mean / SD / 95% CI per metric."""
    rows = []
    for (thr, model), grp in fold_scores.groupby(["target_threshold", "model"], sort=False):
        row = {"target_threshold": thr, "model": model, "n_folds": len(grp)}
        for m in METRICS:
            s = corrected_ci(grp[m])
            row[f"{m}_mean"] = s["mean"]
            row[f"{m}_std"] = s["std"]
            row[f"{m}_ci_low"] = s["ci_low"]
            row[f"{m}_ci_high"] = s["ci_high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["target_threshold", "balanced_accuracy_mean"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
