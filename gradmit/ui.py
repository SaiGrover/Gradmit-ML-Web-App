"""Streamlit helpers: cached loaders for committed files and small shared widgets.

Nothing here trains or evaluates a model. Pages read ``outputs/`` and
``artifacts/`` exactly as committed.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .data import (
    ARTIFACTS_DIR,
    DEFAULT_TARGET_THRESHOLD,
    OUTPUTS_DIR,
    TARGET_THRESHOLDS,
    load_data,
)
from .reporting import OUTPUT_FILES

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(max_entries=1)
def dataset() -> pd.DataFrame:
    return load_data()


@st.cache_data(max_entries=8)
def output(key: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / OUTPUT_FILES[key])


@st.cache_data(max_entries=1)
def evaluation() -> dict:
    return json.loads((OUTPUTS_DIR / OUTPUT_FILES["evaluation"]).read_text(encoding="utf-8"))


@st.cache_data(max_entries=1)
def metadata() -> dict:
    return json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text(encoding="utf-8"))


@st.cache_resource(max_entries=1)
def pipeline():
    return joblib.load(ARTIFACTS_DIR / "gradmit_pipeline.joblib")


def at_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return frame[np.isclose(frame["target_threshold"], threshold)]


def threshold_control(page: str) -> float:
    """Target-threshold switch whose choice carries across pages."""
    current = st.session_state.get("target_threshold", DEFAULT_TARGET_THRESHOLD)
    choice = st.segmented_control(
        "Target threshold on Chance of Admit",
        options=list(TARGET_THRESHOLDS),
        default=current,
        format_func=lambda t: f"{t:.2f}",
        required=True,
        key=f"target_threshold_{page}",
        help=(
            "A record is proxy-positive when the dataset's estimated Chance of Admit is at or "
            "above this value. This is not the classifier's 0.50 decision threshold."
        ),
    )
    st.session_state["target_threshold"] = choice
    return float(choice)


def chart(fig, alt: str) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG, alt=alt)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def interval(low: float, high: float, digits: int = 3, signed: bool = False) -> str:
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f"{fmt.format(low)} to {fmt.format(high)}"


def holm(p: float) -> str:
    """Phrase a Holm-corrected p-value, for example 'Holm p = 0.038' or 'Holm p < 0.001'."""
    return "Holm p < 0.001" if p < 0.001 else f"Holm p = {p:.3f}"
