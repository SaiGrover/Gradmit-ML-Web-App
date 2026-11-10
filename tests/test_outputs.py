import json
import subprocess
import sys

import numpy as np
import pytest

from gradmit import verification
from gradmit.baselines import CGPA_CUTOFF, MAJORITY
from gradmit.data import FEATURES, TARGET_THRESHOLDS
from gradmit.evaluation import METRICS
from gradmit.modelling import MODEL_NAMES

ALL_MODELS = {MAJORITY, CGPA_CUTOFF, *MODEL_NAMES}


def test_fold_scores_cover_every_model_and_fold(outputs):
    folds = outputs["folds"]
    assert set(folds["model"]) == ALL_MODELS
    counts = folds.groupby(["target_threshold", "model"]).size()
    assert (counts == 50).all()
    assert len(counts) == 3 * len(ALL_MODELS)
    # Paired design: identical validation fold sizes for every model.
    sizes = folds.pivot_table(index=["target_threshold", "repeat", "fold"], columns="model", values="n_test")
    assert (sizes.nunique(axis=1) == 1).all()


def test_majority_baseline_results(outputs):
    comp = outputs["comparison"]
    row = comp[np.isclose(comp["target_threshold"], 0.75) & (comp["model"] == MAJORITY)].iloc[0]
    assert row["accuracy_mean"] == pytest.approx(0.564, abs=1e-6)
    assert row["balanced_accuracy_mean"] == pytest.approx(0.5)


def test_predictions_oof_has_one_row_per_record_and_model(outputs):
    oof = outputs["oof"]
    counts = oof.groupby(["target_threshold", "model"])["row_id"].nunique()
    assert (counts == 500).all()
    assert oof["mean_oof_prob"].between(0, 1).all()
    assert set(oof["pred"]) <= {0, 1}


def test_metrics_json_schema(outputs):
    ev = outputs["evaluation"]
    for key in ("schema_version", "target", "data", "protocol", "selected_model", "thresholds", "legacy", "environment"):
        assert key in ev
    assert ev["target"]["name"] == "High_Admission_Chance"
    assert ev["target"]["classifier_threshold"] == 0.5
    assert ev["target"]["default_target_threshold"] == 0.75
    assert ev["data"]["features"] == FEATURES and ev["data"]["n_rows"] == 500
    assert ev["protocol"]["n_folds"] == 50 and ev["protocol"]["n_resamples"] >= 5000
    assert ev["selected_model"] in MODEL_NAMES
    assert set(ev["thresholds"]) == {f"{t:.2f}" for t in TARGET_THRESHOLDS}
    for block in ev["thresholds"].values():
        for section in ("selected_model_cv", "majority_baseline_cv", "cgpa_cutoff_baseline_cv"):
            assert set(block[section]) == set(METRICS)
            for stats in block[section].values():
                assert set(stats) == {"mean", "std", "ci_low", "ci_high"}
        assert {t["baseline"] for t in block["paired_tests"]} == {MAJORITY, CGPA_CUTOFF}
        cm = block["cross_fitted"]["confusion_matrix"]
        assert sum(cm.values()) == 500
    json.dumps(ev)  # serialisable, no NaN objects


def test_significance_flags_follow_the_stated_rule(outputs):
    for row in outputs["paired"].itertuples():
        excludes_zero = row.ci_low > 0 or row.ci_high < 0
        assert bool(row.significant) == (excludes_zero and row.p_value_holm < 0.05)
        assert row.p_value_holm >= row.p_value - 1e-9


def test_threshold_sensitivity_matches_class_balance(outputs):
    sens = outputs["sensitivity"]
    assert sens["n_positive"].tolist() == [300, 218, 155]
    assert sens["majority_accuracy"].tolist() == pytest.approx([0.60, 0.564, 0.69])
    assert (sens["majority_balanced_accuracy"] == 0.5).all()


def test_committed_outputs_reproduce(df, outputs):
    verification.check_class_balance(df, outputs)
    verification.check_research(df, outputs)
    verification.check_baselines(df, outputs)
    verification.check_summaries(outputs)
    verification.check_paired_tests(outputs)


def test_selected_model_first_repeat_reproduces(df, outputs):
    verification.check_selected_model_rerun(df, outputs)


def test_notebooks_are_executed(root):
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_notebooks.py")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


def test_readme_numbers_trace_to_outputs(root, outputs):
    readme = (root / "README.md").read_text(encoding="utf-8")
    ev = outputs["evaluation"]
    selected = ev["selected_model"]
    assert selected in readme
    for thr, block in ev["thresholds"].items():
        for section in ("selected_model_cv", "cgpa_cutoff_baseline_cv"):
            assert f"{block[section]['balanced_accuracy']['mean']:.3f}" in readme, (thr, section)
        for test in block["paired_tests"]:
            assert f"{test['difference']:+.3f}" in readme, (thr, test["baseline"])
    for row in outputs["research"].itertuples():
        assert f"{row.positive_rate * 100:.1f}%" in readme
    assert "Flask" not in readme
