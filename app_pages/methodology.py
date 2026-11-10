import streamlit as st

from gradmit import ui

evaluation = ui.evaluation()
meta = ui.metadata()
protocol = evaluation["protocol"]
env = evaluation["environment"]

st.title("Methodology and limitations")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("What is being predicted")
    st.markdown(
        f"The target is `{evaluation['target']['name']}`: 1 when the dataset's estimated "
        "`Chance of Admit` is at or above a target threshold, otherwise 0. The dataset records "
        "no admission decisions, so this is a proxy label derived from an estimate.\n\n"
        f"- Target thresholds evaluated: {', '.join(f'{t:.2f}' for t in evaluation['target']['target_thresholds'])}\n"
        f"- Target threshold of the deployed model: {evaluation['target']['default_target_threshold']:.2f}\n"
        f"- Classifier decision threshold: {evaluation['target']['classifier_threshold']:.2f}"
    )

    st.subheader("Data")
    st.markdown(
        "Graduate Admission dataset (Mohan S. Acharya, Asfia Armaan and Aneeta S. Antony, 2019), "
        f"distributed on Kaggle. {evaluation['data']['n_rows']} records, seven features: GRE, "
        "TOEFL, university rating, statement-of-purpose and recommendation-letter strength, "
        "CGPA and a research-experience flag. No missing values and no duplicate rows."
    )

    st.subheader("Evaluation protocol")
    st.markdown(
        f"1. **Outer loop.** `{protocol['outer_cv']}` with {protocol['n_splits']} folds and "
        f"{protocol['n_repeats']} repeats, seed {protocol['random_state']}: "
        f"{protocol['n_folds']} validation folds shared by every model and baseline.\n"
        f"2. **Inner loop.** Hyperparameters are chosen by `{protocol['inner_cv']}` grid search "
        "on the training fold only. Scalers sit inside each pipeline and are fitted on training "
        "rows only.\n"
        "3. **Selection.** The model family with the highest mean balanced accuracy over the "
        "outer validation folds at the default threshold is selected.\n"
        "4. **Baselines.** A majority-class rule and a CGPA cutoff whose cutoff is chosen on each "
        "training fold.\n"
        "5. **Intervals.** 95% intervals on fold means use the Nadeau-Bengio correction, because "
        "repeated folds share training data.\n"
        f"6. **Paired tests.** One cross-fitted prediction per record, then a paired bootstrap "
        f"interval and a paired permutation p-value ({protocol['n_resamples']:,} resamples each), "
        "Holm-corrected across the two baseline comparisons."
    )

with right:
    st.subheader("Limitations")
    st.markdown(
        "- **No real outcomes.** Nothing here was validated against an admission decision.\n"
        "- **The label is a cut on a continuous estimate.** Records near the cut are labelled "
        "almost arbitrarily, which caps achievable accuracy.\n"
        "- **Small, narrow sample.** 500 records from one source, with scores concentrated in a "
        "high range. Estimates outside that range are extrapolation.\n"
        "- **Associations only.** Features are strongly correlated; no coefficient, chart or "
        "estimate supports a causal reading.\n"
        "- **Selection optimism.** Choosing the best of five families on the same outer folds "
        "makes its reported score slightly optimistic. The paired tests compare it with the "
        "baselines on equal terms.\n"
        "- **Calibration is uncorrected.** Probabilities are the fitted model's own; the "
        "cross-fitted calibration curve and Brier score are reported so they can be judged.\n"
        "- **Not advice.** The predictor describes a dataset. It is not guidance on applying."
    )

    st.subheader("Deployed model")
    st.markdown(
        f"- Model: {meta['model_name']} with `{meta['best_params']}`\n"
        f"- Trained on all {meta['n_training_rows']} records at target threshold "
        f"{meta['target_threshold']:.2f} ({meta['n_training_positive']} proxy-positive)\n"
        f"- Loaded from `artifacts/gradmit_pipeline.joblib`; the app never trains a model"
    )

    st.subheader("Reproducing the results")
    st.code(
        "python scripts/evaluate_models.py\n"
        "python scripts/train_final_model.py\n"
        "python scripts/verify_outputs.py\n"
        "pytest -q",
        language="bash",
    )
    st.caption(
        f"Committed results were produced with Python {env['python']}, scikit-learn "
        f"{env['scikit_learn']}, NumPy {env['numpy']}, pandas {env['pandas']} and SciPy "
        f"{env['scipy']}. Data SHA-256 `{evaluation['data']['sha256'][:16]}`."
    )
