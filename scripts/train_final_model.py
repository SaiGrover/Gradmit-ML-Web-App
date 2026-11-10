"""Train the deployed pipeline and write ``artifacts/``.

    python scripts/train_final_model.py

Reads the model family chosen by ``scripts/evaluate_models.py`` from
``outputs/model_evaluation.json``, tunes it with the inner CV on all 500 rows at
the default target threshold, refits on all rows and saves the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradmit.data import (  # noqa: E402
    ARTIFACTS_DIR,
    CLASSIFIER_THRESHOLD,
    DEFAULT_TARGET_THRESHOLD,
    FEATURES,
    OUTPUTS_DIR,
    TARGET_NAME,
    data_sha256,
    feature_ranges,
    load_data,
    make_target,
)
from gradmit.modelling import positive_probability, train_final  # noqa: E402
from gradmit.reporting import write_json  # noqa: E402

PIPELINE_PATH = ARTIFACTS_DIR / "gradmit_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


def main() -> None:
    evaluation = json.loads((OUTPUTS_DIR / "model_evaluation.json").read_text(encoding="utf-8"))
    model_name = evaluation["selected_model"]
    df = load_data()
    pipeline, params, inner_score = train_final(df, model_name, DEFAULT_TARGET_THRESHOLD)

    prob = positive_probability(pipeline, df[FEATURES])
    y = make_target(df, DEFAULT_TARGET_THRESHOLD)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    metadata = {
        "schema_version": 1,
        "model_name": model_name,
        "target_name": TARGET_NAME,
        "target_threshold": DEFAULT_TARGET_THRESHOLD,
        "classifier_threshold": CLASSIFIER_THRESHOLD,
        "features": FEATURES,
        "best_params": params,
        "inner_cv_balanced_accuracy": round(inner_score, 6),
        "n_training_rows": int(len(df)),
        "n_training_positive": int(y.sum()),
        "feature_ranges": feature_ranges(df),
        "data_sha256": data_sha256(),
        "scikit_learn_version": sklearn.__version__,
        "parity": {
            "description": "in-sample probabilities used to check the saved pipeline",
            "first_rows_probability": [round(float(p), 8) for p in prob[:5]],
            "mean_probability": round(float(prob.mean()), 8),
            "n_predicted_positive": int((prob >= CLASSIFIER_THRESHOLD).sum()),
        },
        "note": (
            "In-sample numbers above are only a reproducibility check. Performance "
            "estimates come from outputs/model_evaluation.json."
        ),
    }
    write_json(metadata, METADATA_PATH)
    print(f"Saved {model_name} with {params} to {PIPELINE_PATH.relative_to(ARTIFACTS_DIR.parent)}")


if __name__ == "__main__":
    main()
