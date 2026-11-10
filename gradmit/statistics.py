"""Paired resampling tests, multiplicity correction and proportion intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import make_target

N_RESAMPLES = 10_000


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def research_association(df: pd.DataFrame, target_threshold: float) -> pd.DataFrame:
    """P(proxy-positive | Research status) with Wilson 95% intervals.

    The conditioning is on Research: each row is a research group and the rate is
    the share of that group above the Chance-of-Admit threshold. This is an
    association in observational data, not a causal effect.
    """
    target = make_target(df, target_threshold)
    rows = []
    for value, label in ((1, "Has research"), (0, "No research")):
        group = target[df["Research"] == value]
        n, pos = int(len(group)), int(group.sum())
        low, high = wilson_interval(pos, n)
        rows.append(
            {
                "target_threshold": float(target_threshold),
                "research": value,
                "group": label,
                "n": n,
                "n_positive": pos,
                "positive_rate": pos / n,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _balanced_accuracy_rows(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Row-wise balanced accuracy for 2-D boolean arrays of labels and predictions."""
    pos = y
    neg = ~y
    tpr = (pred & pos).sum(axis=1) / pos.sum(axis=1)
    tnr = (~pred & neg).sum(axis=1) / neg.sum(axis=1)
    return (tpr + tnr) / 2


def paired_comparison(
    y_true, pred_a, pred_b, n_resamples: int = N_RESAMPLES, seed: int = 42
) -> dict:
    """Paired row-level comparison of balanced accuracy (model A minus model B).

    * 95% CI: percentile bootstrap that resamples rows, keeping each row's label
      and both models' predictions together.
    * p-value: two-sided paired permutation test that swaps the two models'
      predictions within randomly chosen rows.
    """
    y = np.asarray(y_true).astype(bool)
    a = np.asarray(pred_a).astype(bool)
    b = np.asarray(pred_b).astype(bool)
    n = len(y)
    rng = np.random.default_rng(seed)

    observed = float(
        _balanced_accuracy_rows(y[None], a[None])[0] - _balanced_accuracy_rows(y[None], b[None])[0]
    )

    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = _balanced_accuracy_rows(y[idx], a[idx]) - _balanced_accuracy_rows(y[idx], b[idx])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    swap = rng.random((n_resamples, n)) < 0.5
    y2 = np.broadcast_to(y, swap.shape)
    perm_a = np.where(swap, b, a)
    perm_b = np.where(swap, a, b)
    perm = _balanced_accuracy_rows(y2, perm_a) - _balanced_accuracy_rows(y2, perm_b)
    extreme = int((np.abs(perm) >= abs(observed) - 1e-12).sum())
    p_value = (extreme + 1) / (n_resamples + 1)

    return {
        "difference": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_resamples": int(n_resamples),
    }


def holm_correction(p_values) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
