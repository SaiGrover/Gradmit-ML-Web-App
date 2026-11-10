"""Dataset loading, schema and proxy-target construction.

The dataset has no observed admission decisions. ``Chance_of_Admit`` is itself an
estimated score, so the binary target built here is a *proxy*:
``High_Admission_Chance = Chance_of_Admit >= target_threshold``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Admission_Predict.csv"
OUTPUTS_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"

FEATURES = [
    "GRE_Score",
    "TOEFL_Score",
    "University_Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]
SCORE_COLUMN = "Chance_of_Admit"
TARGET_NAME = "High_Admission_Chance"

TARGET_THRESHOLDS = (0.70, 0.75, 0.80)
DEFAULT_TARGET_THRESHOLD = 0.75
CLASSIFIER_THRESHOLD = 0.50
EXPECTED_ROWS = 500
RANDOM_STATE = 42

FEATURE_LABELS = {
    "GRE_Score": "GRE score",
    "TOEFL_Score": "TOEFL score",
    "University_Rating": "University rating",
    "SOP": "Statement of purpose strength",
    "LOR": "Letter of recommendation strength",
    "CGPA": "CGPA (10-point scale)",
    "Research": "Research experience",
}


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV with normalised column names and no serial column."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Serial_No")])
    return df[FEATURES + [SCORE_COLUMN]].reset_index(drop=True)


def make_target(df: pd.DataFrame, target_threshold: float) -> pd.Series:
    """Proxy target: 1 when the dataset's estimated Chance of Admit >= threshold."""
    return (df[SCORE_COLUMN] >= target_threshold).astype(int).rename(TARGET_NAME)


def class_balance(df: pd.DataFrame, target_threshold: float) -> dict:
    y = make_target(df, target_threshold)
    n, pos = int(len(y)), int(y.sum())
    return {
        "target_threshold": float(target_threshold),
        "n": n,
        "n_positive": pos,
        "n_negative": n - pos,
        "positive_share": pos / n,
        "majority_accuracy": max(pos, n - pos) / n,
    }


def data_sha256(path: Path | str = DATA_PATH) -> str:
    """Hash of the CSV with line endings normalised (stable across platforms)."""
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def feature_ranges(df: pd.DataFrame) -> dict:
    return {f: {"min": float(df[f].min()), "max": float(df[f].max())} for f in FEATURES}


def validate_profile(profile: dict) -> pd.DataFrame:
    """Turn a dict of the seven features into a one-row frame, or raise ValueError."""
    missing = [f for f in FEATURES if f not in profile]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    row = {}
    for f in FEATURES:
        try:
            value = float(profile[f])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{f} must be numeric, got {profile[f]!r}") from exc
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{f} must be a finite number")
        row[f] = value
    if row["Research"] not in (0.0, 1.0):
        raise ValueError("Research must be 0 or 1")
    row["Research"] = int(row["Research"])
    return pd.DataFrame([row], columns=FEATURES)


def out_of_range(profile: dict, ranges: dict) -> list[str]:
    """Features whose value lies outside the range seen in training."""
    flagged = []
    for f in FEATURES:
        lo, hi = ranges[f]["min"], ranges[f]["max"]
        if not lo <= float(profile[f]) <= hi:
            flagged.append(f)
    return flagged
