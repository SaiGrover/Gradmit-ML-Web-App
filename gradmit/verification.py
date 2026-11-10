"""Checks that the committed outputs and artifact can be reproduced from the data.

The fast mode recomputes everything that is cheap and exactly deterministic
(class balance, the Research table, both baselines on all 50 folds, the fold
summaries and the paired tests) and re-runs the selected model on the first
repeat only. The full nested CV for every model is left to
``scripts/evaluate_models.py``.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .baselines import CGPA_CUTOFF, MAJORITY
from .data import (
    ARTIFACTS_DIR,
    FEATURES,
    OUTPUTS_DIR,
    TARGET_THRESHOLDS,
    class_balance,
    data_sha256,
    load_data,
    make_target,
)
from .evaluation import METRICS, summarise_folds
from .modelling import evaluate_threshold, positive_probability, rebuild_pipeline
from .reporting import build_paired_tests, build_research, load_outputs

EXACT = 2e-6  # committed CSVs are rounded to 6 decimals
MODEL_RERUN_TOLERANCE = 0.03


def _close(a, b, atol=EXACT) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, equal_nan=True))


def check_class_balance(df, outputs) -> None:
    for thr in TARGET_THRESHOLDS:
        committed = outputs["evaluation"]["thresholds"][f"{thr:.2f}"]["class_balance"]
        assert committed == class_balance(df, thr), f"class balance mismatch at {thr}"


def check_research(df, outputs) -> None:
    fresh = build_research(df)
    cols = ["n", "n_positive", "positive_rate", "ci_low", "ci_high"]
    assert _close(fresh[cols], outputs["research"][cols]), "research association mismatch"


def check_baselines(df, outputs) -> None:
    committed = outputs["folds"]
    for thr in TARGET_THRESHOLDS:
        fresh, _ = evaluate_threshold(df, thr, models=[MAJORITY, CGPA_CUTOFF], n_jobs=1)
        old = committed[
            np.isclose(committed["target_threshold"], thr)
            & committed["model"].isin([MAJORITY, CGPA_CUTOFF])
        ]
        key = ["model", "repeat", "fold"]
        merged = fresh.merge(old, on=key, suffixes=("_new", "_old"))
        assert len(merged) == len(fresh) == len(old), "baseline fold count mismatch"
        for col in [*METRICS, "cgpa_cutoff"]:
            assert _close(merged[f"{col}_new"], merged[f"{col}_old"]), f"baseline {col} at {thr}"


def check_summaries(outputs) -> None:
    fresh = summarise_folds(outputs["folds"])
    old = outputs["comparison"]
    merged = fresh.merge(old, on=["target_threshold", "model"], suffixes=("_new", "_old"))
    assert len(merged) == len(old), "model comparison rows mismatch"
    for metric in METRICS:
        for stat in ("mean", "std", "ci_low", "ci_high"):
            col = f"{metric}_{stat}"
            assert _close(merged[f"{col}_new"], merged[f"{col}_old"]), f"summary {col}"


def check_paired_tests(outputs) -> None:
    selected = outputs["evaluation"]["selected_model"]
    n = int(outputs["paired"]["n_resamples"].iloc[0])
    fresh = build_paired_tests(outputs["oof"], selected, n_resamples=n)
    for col in ("difference", "ci_low", "ci_high", "p_value", "p_value_holm"):
        assert _close(fresh[col], outputs["paired"][col]), f"paired test {col}"
    assert fresh["significant"].tolist() == outputs["paired"]["significant"].astype(bool).tolist()


def check_selected_model_rerun(df, outputs, threshold: float = 0.75) -> None:
    """Re-run the selected model on the first repeat and compare fold metrics."""
    selected = outputs["evaluation"]["selected_model"]
    fresh, _ = evaluate_threshold(df, threshold, models=[selected], n_repeats=1, n_jobs=1)
    committed = outputs["folds"]
    old = committed[
        np.isclose(committed["target_threshold"], threshold)
        & (committed["model"] == selected)
        & (committed["repeat"] == 0)
    ].sort_values("fold")
    fresh = fresh.sort_values("fold")
    for metric in ("accuracy", "balanced_accuracy", "f1", "roc_auc"):
        assert _close(fresh[metric], old[metric], atol=MODEL_RERUN_TOLERANCE), f"rerun {metric}"


def load_artifact():
    pipeline = joblib.load(ARTIFACTS_DIR / "gradmit_pipeline.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return pipeline, metadata


def check_artifact(df, outputs) -> None:
    pipeline, meta = load_artifact()
    assert meta["model_name"] == outputs["evaluation"]["selected_model"], "artifact model differs"
    assert meta["data_sha256"] == data_sha256(), "artifact trained on different data"
    saved = positive_probability(pipeline, df[FEATURES])
    parity = meta["parity"]
    assert _close(saved[:5], parity["first_rows_probability"], atol=1e-6), "artifact probabilities"
    assert int((saved >= meta["classifier_threshold"]).sum()) == parity["n_predicted_positive"]
    retrained = rebuild_pipeline(meta["model_name"], meta["best_params"]).fit(
        df[FEATURES], make_target(df, meta["target_threshold"]).to_numpy()
    )
    fresh = positive_probability(retrained, df[FEATURES])
    # A refit may differ in the last digits across BLAS builds; 1e-4 is far below any decision.
    assert _close(fresh, saved, atol=1e-4), "artifact differs from a fresh fit"


def run_all(fast: bool = True) -> list[str]:
    df = load_data()
    outputs = load_outputs(OUTPUTS_DIR)
    checks = [
        ("class balance", lambda: check_class_balance(df, outputs)),
        ("research association", lambda: check_research(df, outputs)),
        ("baseline fold scores (50 folds x 3 thresholds)", lambda: check_baselines(df, outputs)),
        ("fold summaries", lambda: check_summaries(outputs)),
        ("paired tests", lambda: check_paired_tests(outputs)),
        ("selected model, first repeat", lambda: check_selected_model_rerun(df, outputs)),
        ("artifact parity", lambda: check_artifact(df, outputs)),
    ]
    passed = []
    for name, fn in checks:
        fn()
        passed.append(name)
    if not fast:
        frames = [evaluate_threshold(df, thr)[0] for thr in TARGET_THRESHOLDS]
        fresh = summarise_folds(pd.concat(frames, ignore_index=True))
        merged = fresh.merge(
            outputs["comparison"], on=["target_threshold", "model"], suffixes=("_new", "_old")
        )
        assert _close(
            merged["balanced_accuracy_mean_new"],
            merged["balanced_accuracy_mean_old"],
            atol=MODEL_RERUN_TOLERANCE,
        ), "full re-evaluation differs"
        passed.append("full nested CV for every model")
    return passed
