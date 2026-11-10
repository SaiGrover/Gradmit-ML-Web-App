import numpy as np
import pandas as pd
import pytest

from gradmit.baselines import (
    CGPA_CUTOFF,
    MAJORITY,
    CGPACutoffClassifier,
    make_majority,
    select_cgpa_cutoff,
)
from gradmit.data import FEATURES, make_target
from gradmit.evaluation import compute_metrics, corrected_ci, cv_splits
from gradmit.modelling import MODEL_NAMES, evaluate_threshold, model_specs


@pytest.fixture(scope="module")
def xy(df):
    return df[FEATURES], make_target(df, 0.75).to_numpy()


def test_repeated_cv_has_50_disjoint_folds(xy):
    X, y = xy
    splits = cv_splits(X, y)
    assert len(splits) == 50
    for s in splits:
        assert len(np.intersect1d(s["train"], s["test"])) == 0
        assert len(s["train"]) + len(s["test"]) == len(X)
    for repeat in range(10):
        tests = np.concatenate([s["test"] for s in splits if s["repeat"] == repeat])
        assert sorted(tests) == list(range(len(X)))  # each record validated once per repeat


def test_folds_are_stratified(xy):
    X, y = xy
    for s in cv_splits(X, y):
        assert abs(y[s["test"]].mean() - y.mean()) < 0.01


def test_majority_baseline_predicts_the_training_majority(xy):
    X, y = xy
    s = cv_splits(X, y)[0]
    model = make_majority().fit(X.iloc[s["train"]], y[s["train"]])
    pred = model.predict(X.iloc[s["test"]])
    assert set(pred) == {0}  # proxy-negative is the majority at 0.75
    metrics = compute_metrics(y[s["test"]], pred, y_score=np.zeros(len(pred)))
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(1 - y[s["test"]].mean())


def test_cgpa_cutoff_uses_training_rows_only(xy):
    X, y = xy
    s = cv_splits(X, y)[0]
    fitted = CGPACutoffClassifier().fit(X.iloc[s["train"]], y[s["train"]])
    assert fitted.n_train_ == len(s["train"])
    assert fitted.cutoff_ == select_cgpa_cutoff(X["CGPA"].to_numpy()[s["train"]], y[s["train"]])

    # Corrupting every validation label and feature must not move the cutoff.
    X_bad, y_bad = X.copy(), y.copy()
    X_bad.iloc[s["test"], X.columns.get_loc("CGPA")] = 0.0
    y_bad[s["test"]] = 1 - y_bad[s["test"]]
    refit = CGPACutoffClassifier().fit(X_bad.iloc[s["train"]], y_bad[s["train"]])
    assert refit.cutoff_ == fitted.cutoff_

    # The frozen cutoff is what gets applied to validation rows.
    expected = (X["CGPA"].to_numpy()[s["test"]] >= fitted.cutoff_).astype(int)
    assert (fitted.predict(X.iloc[s["test"]]) == expected).all()


def test_cgpa_cutoff_separates_a_clean_split():
    cgpa = np.array([7.0, 7.5, 8.0, 9.0, 9.5])
    y = np.array([0, 0, 0, 1, 1])
    assert select_cgpa_cutoff(cgpa, y) == pytest.approx(8.5)


def test_scalers_live_inside_pipelines():
    for name in ("Logistic Regression", "SVM (RBF)", "K-Nearest Neighbours"):
        pipeline, _ = model_specs()[name]
        assert pipeline.steps[0][0] == "scale"
    assert len(MODEL_NAMES) >= 5


def test_repeated_cv_outputs_are_deterministic(df):
    models = [MAJORITY, CGPA_CUTOFF, "Logistic Regression"]
    first, oof_first = evaluate_threshold(df, 0.75, models=models, n_repeats=2, n_jobs=1)
    second, oof_second = evaluate_threshold(df, 0.75, models=models, n_repeats=2, n_jobs=1)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(oof_first, oof_second)
    assert len(first) == 3 * 10
    assert len(oof_first) == 3 * len(df)


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_roc_auc_only_when_both_classes_present():
    one_class = compute_metrics([1, 1, 1], [1, 1, 0], y_score=[0.9, 0.8, 0.2], y_prob=[0.9, 0.8, 0.2])
    assert np.isnan(one_class["roc_auc"])
    both = compute_metrics([0, 1, 1], [0, 1, 1], y_score=[0.1, 0.8, 0.9], y_prob=[0.1, 0.8, 0.9])
    assert both["roc_auc"] == pytest.approx(1.0)


def test_corrected_interval_is_wider_than_naive():
    values = np.linspace(0.8, 0.9, 50)
    ci = corrected_ci(values)
    naive_half = 2.0096 * values.std(ddof=1) / np.sqrt(50)
    assert ci["mean"] == pytest.approx(0.85)
    assert (ci["ci_high"] - ci["mean"]) > naive_half
