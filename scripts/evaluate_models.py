"""Run the full evaluation and write every file in ``outputs/``.

    python scripts/evaluate_models.py

Nested repeated stratified cross-validation (5 folds x 10 repeats) for both
baselines and all candidate models at target thresholds 0.70, 0.75 and 0.80.
Takes a few minutes; results are deterministic for the pinned dependencies.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradmit.data import DEFAULT_TARGET_THRESHOLD, OUTPUTS_DIR, TARGET_THRESHOLDS, load_data  # noqa: E402
from gradmit.evaluation import summarise_folds  # noqa: E402
from gradmit.modelling import evaluate_threshold, select_model  # noqa: E402
from gradmit.reporting import (  # noqa: E402
    OUTPUT_FILES,
    build_evaluation_json,
    build_paired_tests,
    build_research,
    build_threshold_sensitivity,
    write_csv,
    write_json,
)


def main() -> None:
    df = load_data()
    OUTPUTS_DIR.mkdir(exist_ok=True)

    fold_frames, oof_frames = [], []
    for thr in TARGET_THRESHOLDS:
        start = time.time()
        folds, oof = evaluate_threshold(df, thr)
        fold_frames.append(folds)
        oof_frames.append(oof)
        print(f"target threshold {thr:.2f}: {len(folds)} fold scores in {time.time() - start:.0f}s")

    folds = pd.concat(fold_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    comparison = summarise_folds(folds)
    selected = select_model(comparison, DEFAULT_TARGET_THRESHOLD)
    paired = build_paired_tests(oof, selected)
    sensitivity = build_threshold_sensitivity(df, comparison, paired, selected)
    research = build_research(df)

    write_csv(folds, OUTPUTS_DIR / OUTPUT_FILES["folds"])
    write_csv(oof, OUTPUTS_DIR / OUTPUT_FILES["oof"])
    write_csv(comparison, OUTPUTS_DIR / OUTPUT_FILES["comparison"])
    write_csv(paired, OUTPUTS_DIR / OUTPUT_FILES["paired"])
    write_csv(sensitivity, OUTPUTS_DIR / OUTPUT_FILES["sensitivity"])
    write_csv(research, OUTPUTS_DIR / OUTPUT_FILES["research"])
    write_json(
        build_evaluation_json(df, comparison, paired, oof, selected),
        OUTPUTS_DIR / OUTPUT_FILES["evaluation"],
    )

    print(f"\nSelected model: {selected}")
    cols = ["model", "balanced_accuracy_mean", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high"]
    view = comparison[comparison["target_threshold"] == DEFAULT_TARGET_THRESHOLD][cols]
    print(view.round(4).to_string(index=False))
    print("\nPaired tests (selected model minus baseline, balanced accuracy):")
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
