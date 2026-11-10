"""Baselines evaluated on exactly the same folds as every ML model."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score

MAJORITY = "Majority baseline"
CGPA_CUTOFF = "CGPA cutoff baseline"
BASELINES = (MAJORITY, CGPA_CUTOFF)


def make_majority() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def select_cgpa_cutoff(cgpa, y) -> float:
    """Cutoff maximising balanced accuracy of ``cgpa >= cutoff`` on the given rows.

    Candidates are midpoints between consecutive distinct CGPA values. Ties go to
    the lowest cutoff, which keeps the choice deterministic.
    """
    cgpa = np.asarray(cgpa, dtype=float)
    y = np.asarray(y).astype(int)
    values = np.unique(cgpa)
    if len(values) == 1:
        return float(values[0])
    candidates = (values[:-1] + values[1:]) / 2
    scores = [balanced_accuracy_score(y, (cgpa >= c).astype(int)) for c in candidates]
    return float(candidates[int(np.argmax(scores))])


class CGPACutoffClassifier(ClassifierMixin, BaseEstimator):
    """Interpretable rule: predict positive when ``CGPA >= cutoff``.

    ``fit`` chooses the cutoff from the rows it is given and nothing else, so
    inside cross-validation the cutoff only ever sees the training fold.
    """

    def __init__(self, column: str = "CGPA"):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.cutoff_ = select_cgpa_cutoff(X[self.column].to_numpy(), y)
        self.n_train_ = len(X)
        return self

    def decision_function(self, X):
        return X[self.column].to_numpy(dtype=float) - self.cutoff_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
