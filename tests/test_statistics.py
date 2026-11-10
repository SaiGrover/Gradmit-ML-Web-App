import numpy as np
import pytest

from gradmit.data import make_target
from gradmit.statistics import (
    holm_correction,
    paired_comparison,
    research_association,
    wilson_interval,
)


def test_research_rates_condition_on_research(df):
    """P(proxy-positive | Research), not P(Research | proxy-positive)."""
    table = research_association(df, 0.75).set_index("research")
    target = make_target(df, 0.75)

    with_research = target[df["Research"] == 1]
    without_research = target[df["Research"] == 0]
    assert table.loc[1, "n"] == len(with_research) == 280
    assert table.loc[0, "n"] == len(without_research) == 220
    assert table.loc[1, "positive_rate"] == pytest.approx(with_research.mean())
    assert table.loc[0, "positive_rate"] == pytest.approx(without_research.mean())
    assert table.loc[1, "positive_rate"] == pytest.approx(0.657, abs=5e-4)
    assert table.loc[0, "positive_rate"] == pytest.approx(0.155, abs=5e-4)

    # The reversed conditional gives the retired, wrong numbers.
    reversed_positive = df.loc[target == 1, "Research"].mean()
    reversed_negative = df.loc[target == 0, "Research"].mean()
    assert reversed_positive == pytest.approx(0.844, abs=5e-4)
    assert reversed_negative == pytest.approx(0.340, abs=5e-4)
    assert table.loc[1, "positive_rate"] != pytest.approx(reversed_positive, abs=0.05)
    assert table.loc[0, "positive_rate"] != pytest.approx(reversed_negative, abs=0.05)


def test_research_intervals_contain_the_rate(df):
    for threshold in (0.70, 0.75, 0.80):
        table = research_association(df, threshold)
        assert (table["ci_low"] < table["positive_rate"]).all()
        assert (table["positive_rate"] < table["ci_high"]).all()
        assert table["n"].sum() == len(df)


def test_wilson_interval_known_value():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)


def test_holm_correction():
    assert holm_correction([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm_correction([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert holm_correction([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_paired_comparison_detects_a_real_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(rng.random(400) < 0.9, y, 1 - y)
    poor = np.where(rng.random(400) < 0.6, y, 1 - y)
    result = paired_comparison(y, good, poor, n_resamples=2000)
    assert result["difference"] > 0.2
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01


def test_paired_comparison_of_identical_models_is_null():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.8, y, 1 - y)
    result = paired_comparison(y, pred, pred, n_resamples=1000)
    assert result["difference"] == 0
    assert result["ci_low"] == result["ci_high"] == 0
    assert result["p_value"] == 1.0


def test_paired_comparison_is_seeded():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    a = np.where(rng.random(200) < 0.8, y, 1 - y)
    b = np.where(rng.random(200) < 0.7, y, 1 - y)
    assert paired_comparison(y, a, b, n_resamples=500) == paired_comparison(y, a, b, n_resamples=500)
