import pytest

from gradmit.data import (
    EXPECTED_ROWS,
    FEATURES,
    SCORE_COLUMN,
    TARGET_NAME,
    class_balance,
    feature_ranges,
    make_target,
    out_of_range,
    validate_profile,
)

PROFILE = {
    "GRE_Score": 320,
    "TOEFL_Score": 110,
    "University_Rating": 3,
    "SOP": 3.5,
    "LOR": 4.0,
    "CGPA": 8.7,
    "Research": 1,
}


def test_schema_and_row_count(df):
    assert list(df.columns) == FEATURES + [SCORE_COLUMN]
    assert len(df) == EXPECTED_ROWS
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert df[SCORE_COLUMN].between(0, 1).all()
    assert set(df["Research"].unique()) == {0, 1}


@pytest.mark.parametrize(
    ("threshold", "positives", "majority_accuracy"),
    [(0.70, 300, 0.600), (0.75, 218, 0.564), (0.80, 155, 0.690)],
)
def test_target_counts(df, threshold, positives, majority_accuracy):
    target = make_target(df, threshold)
    assert target.name == TARGET_NAME
    assert int(target.sum()) == positives
    balance = class_balance(df, threshold)
    assert balance["n_positive"] + balance["n_negative"] == EXPECTED_ROWS
    assert balance["majority_accuracy"] == pytest.approx(majority_accuracy)


def test_target_is_inclusive_at_the_threshold(df):
    at_cut = df[SCORE_COLUMN] == 0.75
    assert at_cut.any()
    assert make_target(df, 0.75)[at_cut].all()


def test_validate_profile_accepts_a_complete_profile():
    row = validate_profile(PROFILE)
    assert list(row.columns) == FEATURES
    assert row.shape == (1, 7)


@pytest.mark.parametrize(
    "change",
    [{"CGPA": "high"}, {"Research": 2}, {"GRE_Score": float("nan")}, {"TOEFL_Score": None}],
)
def test_validate_profile_rejects_bad_values(change):
    with pytest.raises(ValueError):
        validate_profile({**PROFILE, **change})


def test_validate_profile_rejects_missing_features():
    incomplete = {k: v for k, v in PROFILE.items() if k != "LOR"}
    with pytest.raises(ValueError, match="LOR"):
        validate_profile(incomplete)


def test_out_of_range_flags_extrapolation(df):
    ranges = feature_ranges(df)
    assert out_of_range(PROFILE, ranges) == []
    assert out_of_range({**PROFILE, "GRE_Score": 270, "CGPA": 9.99}, ranges) == ["GRE_Score", "CGPA"]
