import numpy as np
import pytest

from gradmit import verification
from gradmit.data import FEATURES, data_sha256, feature_ranges, validate_profile
from gradmit.modelling import positive_probability


@pytest.fixture(scope="module")
def artifact():
    return verification.load_artifact()


def test_artifact_loads_and_predicts(artifact, df):
    pipeline, meta = artifact
    prob = positive_probability(pipeline, df[FEATURES])
    assert prob.shape == (len(df),)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert meta["features"] == FEATURES


def test_metadata_describes_the_proxy_target(artifact, df, outputs):
    _, meta = artifact
    assert meta["target_name"] == "High_Admission_Chance"
    assert meta["target_threshold"] == 0.75
    assert meta["classifier_threshold"] == 0.5
    assert meta["model_name"] == outputs["evaluation"]["selected_model"]
    assert meta["n_training_rows"] == 500 and meta["n_training_positive"] == 218
    assert meta["data_sha256"] == data_sha256()
    assert meta["feature_ranges"] == feature_ranges(df)


def test_artifact_matches_a_fresh_fit(df, outputs):
    verification.check_artifact(df, outputs)


def test_prediction_moves_with_cgpa_in_the_expected_direction(artifact):
    pipeline, _ = artifact
    base = {"GRE_Score": 317, "TOEFL_Score": 107, "University_Rating": 3, "SOP": 3.5, "LOR": 3.5, "Research": 1}
    low = positive_probability(pipeline, validate_profile({**base, "CGPA": 7.5}))[0]
    high = positive_probability(pipeline, validate_profile({**base, "CGPA": 9.5}))[0]
    assert 0 <= low < high <= 1
    assert np.isfinite([low, high]).all()
