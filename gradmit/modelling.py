"""Model pipelines, nested cross-validation and final-model training."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .baselines import CGPA_CUTOFF, MAJORITY, CGPACutoffClassifier, make_majority
from .data import CLASSIFIER_THRESHOLD, FEATURES, make_target
from .evaluation import (
    N_REPEATS,
    N_SPLITS,
    PRIMARY_METRIC,
    RANDOM_STATE,
    compute_metrics,
    cv_splits,
)

INNER_SPLITS = 5


def model_specs() -> dict[str, tuple[Pipeline, dict]]:
    """Candidate pipelines and their hyperparameter grids.

    Preprocessing lives inside each pipeline, so scalers are only ever fitted on
    the training portion of whichever split is being evaluated.
    """
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=150, random_state=RANDOM_STATE, n_jobs=1
                        ),
                    )
                ]
            ),
            {"model__max_depth": [4, None], "model__min_samples_leaf": [1, 5]},
        ),
        "Hist Gradient Boosting": (
            Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=100, early_stopping=False, random_state=RANDOM_STATE
                        ),
                    )
                ]
            ),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
        ),
        "SVM (RBF)": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    # Platt-scaled SVC: sigmoid calibration fitted on internal
                    # cross-validated decision values gives probability support.
                    ("model", CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False, cv=5)),
                ]
            ),
            {"model__estimator__C": [0.1, 1.0, 10.0]},
        ),
        "K-Nearest Neighbours": (
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"]},
        ),
    }


MODEL_NAMES = tuple(model_specs())


def inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def make_search(name: str) -> GridSearchCV:
    """Hyperparameter search that runs entirely inside the data it is fitted on."""
    pipeline, grid = model_specs()[name]
    return GridSearchCV(
        clone(pipeline), grid, scoring=PRIMARY_METRIC, cv=inner_cv(), refit=True, n_jobs=1
    )


def positive_probability(estimator, X) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


def _evaluate_fold(name: str, X: pd.DataFrame, y: np.ndarray, split: dict) -> tuple[dict, dict]:
    train, test = split["train"], split["test"]
    X_tr, X_te, y_tr, y_te = X.iloc[train], X.iloc[test], y[train], y[test]
    extra = {"best_params": "", "cgpa_cutoff": np.nan}

    if name == MAJORITY:
        est = make_majority().fit(X_tr, y_tr)
        prob = positive_probability(est, X_te)
        pred, score = est.predict(X_te), prob
    elif name == CGPA_CUTOFF:
        est = CGPACutoffClassifier().fit(X_tr, y_tr)
        pred, score, prob = est.predict(X_te), est.decision_function(X_te), None
        extra["cgpa_cutoff"] = est.cutoff_
    else:
        est = make_search(name).fit(X_tr, y_tr)
        prob = positive_probability(est, X_te)
        pred, score = (prob >= CLASSIFIER_THRESHOLD).astype(int), prob
        extra["best_params"] = json.dumps(est.best_params_, sort_keys=True)

    metrics = compute_metrics(y_te, pred, y_score=score, y_prob=prob)
    row = {
        "model": name,
        "repeat": split["repeat"],
        "fold": split["fold"],
        "n_train": len(train),
        "n_test": len(test),
        **metrics,
        **extra,
    }
    preds = {
        "model": name,
        "repeat": split["repeat"],
        "row_id": test,
        "pred": np.asarray(pred).astype(int),
        "prob": np.asarray(prob if prob is not None else pred, dtype=float),
    }
    return row, preds


def evaluate_threshold(
    df: pd.DataFrame,
    target_threshold: float,
    models=None,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested repeated stratified CV for baselines and models at one target threshold.

    Returns fold-level scores and cross-fitted predictions: one row per record and
    model, holding the out-of-fold probability averaged over repeats and the class
    obtained by thresholding that average at 0.5. For the CGPA cutoff rule the
    "probability" is the share of repeats in which the rule predicted positive.
    """
    models = list(models) if models is not None else [MAJORITY, CGPA_CUTOFF, *MODEL_NAMES]
    X = df[FEATURES]
    y = make_target(df, target_threshold).to_numpy()
    splits = cv_splits(X, y, n_splits, n_repeats)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_fold)(name, X, y, split) for name in models for split in splits
    )
    folds = pd.DataFrame([r for r, _ in results])
    folds.insert(0, "target_threshold", float(target_threshold))

    long = pd.concat([pd.DataFrame(p) for _, p in results], ignore_index=True)
    oof = (
        long.groupby(["model", "row_id"], sort=False)["prob"]
        .mean()
        .rename("mean_oof_prob")
        .reset_index()
    )
    oof["pred"] = (oof["mean_oof_prob"] >= CLASSIFIER_THRESHOLD).astype(int)
    oof["y_true"] = y[oof["row_id"].to_numpy()]
    oof.insert(0, "target_threshold", float(target_threshold))
    oof["_order"] = oof["model"].map({m: i for i, m in enumerate(models)})
    oof = oof.sort_values(["_order", "row_id"]).reset_index(drop=True)
    return folds, oof[["target_threshold", "model", "row_id", "y_true", "mean_oof_prob", "pred"]]


def select_model(comparison: pd.DataFrame, target_threshold: float) -> str:
    """ML model with the highest mean repeated-CV balanced accuracy (baselines excluded)."""
    rows = comparison[
        np.isclose(comparison["target_threshold"], target_threshold)
        & comparison["model"].isin(MODEL_NAMES)
    ]
    return str(rows.sort_values(f"{PRIMARY_METRIC}_mean", ascending=False).iloc[0]["model"])


def train_final(df: pd.DataFrame, model_name: str, target_threshold: float):
    """Tune on all rows with the inner CV, then refit the best pipeline on all rows."""
    X = df[FEATURES]
    y = make_target(df, target_threshold).to_numpy()
    search = make_search(model_name).fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def rebuild_pipeline(model_name: str, params: dict) -> Pipeline:
    pipeline, _ = model_specs()[model_name]
    return clone(pipeline).set_params(**params)
