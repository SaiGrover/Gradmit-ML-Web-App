"""Turn fold scores and cross-fitted predictions into the committed report files."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import CGPA_CUTOFF, MAJORITY
from .data import (
    CLASSIFIER_THRESHOLD,
    DEFAULT_TARGET_THRESHOLD,
    FEATURES,
    OUTPUTS_DIR,
    SCORE_COLUMN,
    TARGET_NAME,
    TARGET_THRESHOLDS,
    class_balance,
    data_sha256,
    make_target,
)
from .evaluation import METRICS, N_REPEATS, N_SPLITS, PRIMARY_METRIC, RANDOM_STATE, compute_metrics
from .modelling import INNER_SPLITS, MODEL_NAMES, model_specs
from .statistics import N_RESAMPLES, holm_correction, paired_comparison, research_association

DECIMALS = 6
OUTPUT_FILES = {
    "folds": "repeated_cv_fold_scores.csv",
    "comparison": "model_comparison.csv",
    "sensitivity": "threshold_sensitivity.csv",
    "paired": "paired_tests.csv",
    "research": "research_association.csv",
    "oof": "predictions_oof.csv",
    "evaluation": "model_evaluation.json",
}


def _at(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return frame[np.isclose(frame["target_threshold"], threshold)]


def oof_for(oof: pd.DataFrame, threshold: float, model: str) -> pd.DataFrame:
    rows = _at(oof, threshold)
    return rows[rows["model"] == model].sort_values("row_id").reset_index(drop=True)


def build_paired_tests(
    oof: pd.DataFrame, selected_model: str, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Selected model versus each baseline at every threshold, Holm-corrected per threshold."""
    rows = []
    for thr in TARGET_THRESHOLDS:
        full = oof_for(oof, thr, selected_model)
        block = []
        for baseline in (MAJORITY, CGPA_CUTOFF):
            base = oof_for(oof, thr, baseline)
            res = paired_comparison(full["y_true"], full["pred"], base["pred"], n_resamples)
            block.append(
                {
                    "target_threshold": float(thr),
                    "model": selected_model,
                    "baseline": baseline,
                    "metric": PRIMARY_METRIC,
                    **res,
                }
            )
        for row, p_adj in zip(block, holm_correction([r["p_value"] for r in block])):
            row["p_value_holm"] = p_adj
            row["significant"] = bool(
                (row["ci_low"] > 0 or row["ci_high"] < 0) and row["p_value_holm"] < 0.05
            )
        rows.extend(block)
    cols = [
        "target_threshold",
        "model",
        "baseline",
        "metric",
        "difference",
        "ci_low",
        "ci_high",
        "p_value",
        "p_value_holm",
        "significant",
        "n_resamples",
    ]
    return pd.DataFrame(rows)[cols]


def build_research(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [research_association(df, thr) for thr in TARGET_THRESHOLDS], ignore_index=True
    )


def build_threshold_sensitivity(
    df: pd.DataFrame, comparison: pd.DataFrame, paired: pd.DataFrame, selected_model: str
) -> pd.DataFrame:
    rows = []
    for thr in TARGET_THRESHOLDS:
        comp = _at(comparison, thr).set_index("model")
        tests = _at(paired, thr).set_index("baseline")
        ml = comp.loc[[m for m in comp.index if m in MODEL_NAMES]]
        bal = class_balance(df, thr)
        row = {
            "target_threshold": float(thr),
            "n_positive": bal["n_positive"],
            "positive_share": bal["positive_share"],
            "majority_accuracy": bal["majority_accuracy"],
            "selected_model": selected_model,
            "best_model_at_threshold": str(ml[f"{PRIMARY_METRIC}_mean"].idxmax()),
        }
        for label, name in (("majority", MAJORITY), ("cgpa", CGPA_CUTOFF), ("model", selected_model)):
            for metric in ("balanced_accuracy", "f1", "roc_auc"):
                row[f"{label}_{metric}"] = comp.loc[name, f"{metric}_mean"]
        row["model_balanced_accuracy_ci_low"] = comp.loc[selected_model, "balanced_accuracy_ci_low"]
        row["model_balanced_accuracy_ci_high"] = comp.loc[selected_model, "balanced_accuracy_ci_high"]
        for label, name in (("cgpa", CGPA_CUTOFF), ("majority", MAJORITY)):
            t = tests.loc[name]
            row[f"improvement_over_{label}"] = t["difference"]
            row[f"improvement_over_{label}_ci_low"] = t["ci_low"]
            row[f"improvement_over_{label}_ci_high"] = t["ci_high"]
            row[f"improvement_over_{label}_p_holm"] = t["p_value_holm"]
            row[f"improvement_over_{label}_significant"] = bool(t["significant"])
        rows.append(row)
    return pd.DataFrame(rows)


def cross_fitted_summary(oof: pd.DataFrame, threshold: float, model: str) -> dict:
    """Confusion matrix and metrics from one cross-fitted prediction per record."""
    rows = oof_for(oof, threshold, model)
    y, pred = rows["y_true"].to_numpy(), rows["pred"].to_numpy()
    prob = rows["mean_oof_prob"].to_numpy()
    use_prob = model not in (CGPA_CUTOFF,)
    metrics = compute_metrics(y, pred, y_score=prob, y_prob=prob if use_prob else None)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "metrics": {k: (None if np.isnan(v) else round(v, DECIMALS)) for k, v in metrics.items()},
    }


def error_frame(df: pd.DataFrame, oof: pd.DataFrame, threshold: float, model: str) -> pd.DataFrame:
    """Records joined with their cross-fitted prediction and error flags."""
    rows = oof_for(oof, threshold, model)
    out = df.copy()
    out["row_id"] = np.arange(len(df))
    out[TARGET_NAME] = make_target(df, threshold).to_numpy()
    out = out.merge(rows[["row_id", "mean_oof_prob", "pred"]], on="row_id")
    out["error"] = out["pred"] != out[TARGET_NAME]
    out["error_type"] = np.select(
        [(out["pred"] == 1) & out["error"], (out["pred"] == 0) & out["error"]],
        ["False positive", "False negative"],
        default="Correct",
    )
    out["distance_to_threshold"] = (out[SCORE_COLUMN] - threshold).round(4)
    out["confidence"] = (out["mean_oof_prob"] - CLASSIFIER_THRESHOLD).abs() + CLASSIFIER_THRESHOLD
    return out


def error_slices(errors: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Cross-fitted error rate within interpretable slices of the data."""
    e = errors.copy()
    e["Research status"] = e["Research"].map({1: "Has research", 0: "No research"})
    e["University rating"] = e["University_Rating"].astype(int).astype(str)
    e["CGPA band"] = pd.cut(
        e["CGPA"], [0, 8.0, 8.5, 9.0, 10.0], labels=["< 8.0", "8.0-8.5", "8.5-9.0", "> 9.0"]
    ).astype(str)
    gap = e["distance_to_threshold"].abs()
    e["Gap to target threshold"] = np.where(gap <= 0.05, "Within 0.05", "More than 0.05")
    rows = []
    for dim in ("Gap to target threshold", "Research status", "CGPA band", "University rating"):
        for value, grp in e.groupby(dim, sort=True):
            rows.append(
                {
                    "slice": dim,
                    "value": value,
                    "n": int(len(grp)),
                    "errors": int(grp["error"].sum()),
                    "error_rate": float(grp["error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_evaluation_json(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    oof: pd.DataFrame,
    selected_model: str,
) -> dict:
    import scipy
    import sklearn

    def stats_for(thr: float, model: str) -> dict:
        row = _at(comparison, thr).set_index("model").loc[model]
        return {
            m: {
                k: (None if pd.isna(row[f"{m}_{k}"]) else round(float(row[f"{m}_{k}"]), DECIMALS))
                for k in ("mean", "std", "ci_low", "ci_high")
            }
            for m in METRICS
        }

    thresholds = {}
    for thr in TARGET_THRESHOLDS:
        comp = _at(comparison, thr)
        ranking = comp[comp["model"].isin(MODEL_NAMES)]["model"].tolist()
        tests = _at(paired, thr).drop(columns=["target_threshold"]).to_dict(orient="records")
        thresholds[f"{thr:.2f}"] = {
            "class_balance": class_balance(df, thr),
            "selected_model_cv": stats_for(thr, selected_model),
            "majority_baseline_cv": stats_for(thr, MAJORITY),
            "cgpa_cutoff_baseline_cv": stats_for(thr, CGPA_CUTOFF),
            "model_ranking": ranking,
            "paired_tests": [
                {k: (round(v, DECIMALS) if isinstance(v, float) else v) for k, v in t.items()}
                for t in tests
            ],
            "cross_fitted": cross_fitted_summary(oof, thr, selected_model),
        }

    return {
        "schema_version": 1,
        "target": {
            "name": TARGET_NAME,
            "definition": (
                "1 when the dataset's estimated Chance of Admit is at or above the target "
                "threshold; a proxy label, not an observed admission decision"
            ),
            "target_thresholds": list(TARGET_THRESHOLDS),
            "default_target_threshold": DEFAULT_TARGET_THRESHOLD,
            "classifier_threshold": CLASSIFIER_THRESHOLD,
        },
        "data": {
            "path": "data/Admission_Predict.csv",
            "sha256": data_sha256(),
            "n_rows": int(len(df)),
            "features": FEATURES,
        },
        "protocol": {
            "outer_cv": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "n_folds": N_SPLITS * N_REPEATS,
            "random_state": RANDOM_STATE,
            "inner_cv": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True)",
            "selection_metric": PRIMARY_METRIC,
            "confidence_interval": "Nadeau-Bengio corrected resampled t-interval, 95%",
            "paired_test": (
                "row-level paired bootstrap CI and paired permutation p-value on cross-fitted "
                "predictions, Holm-corrected across the two baselines"
            ),
            "n_resamples": int(paired["n_resamples"].iloc[0]),
            "hyperparameter_grids": {name: grid for name, (_, grid) in model_specs().items()},
        },
        "selected_model": selected_model,
        "selection_rule": (
            "highest mean balanced accuracy over the 50 outer validation folds at the default "
            "target threshold; hyperparameters are tuned inside each training fold"
        ),
        "thresholds": thresholds,
        "legacy": {
            "retired_result": "91.2% accuracy from a single 75/25 split",
            "note": (
                "Retired. It came from one split with no baseline and described the proxy "
                "label as admission. It is not a result of this evaluation."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.round(DECIMALS).to_csv(path, index=False, lineterminator="\n")


def write_json(payload: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_outputs(outputs_dir: Path = OUTPUTS_DIR) -> dict:
    """Read every committed report file."""
    out = {}
    for key, name in OUTPUT_FILES.items():
        path = Path(outputs_dir) / name
        if name.endswith(".json"):
            out[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            out[key] = pd.read_csv(path, keep_default_na=True)
    return out
