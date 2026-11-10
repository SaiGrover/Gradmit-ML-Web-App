import streamlit as st

from gradmit import charts, ui
from gradmit.baselines import BASELINES
from gradmit.evaluation import METRICS
from gradmit.reporting import error_frame, error_slices, oof_for

evaluation = ui.evaluation()
selected = evaluation["selected_model"]
protocol = evaluation["protocol"]

st.title("Model evaluation", icon=":material/monitoring:")
st.markdown(
    f"Five model families and two baselines, scored on the same {protocol['n_folds']} validation "
    f"folds ({protocol['n_splits']}-fold stratified cross-validation repeated "
    f"{protocol['n_repeats']} times). Hyperparameters are tuned inside each training fold, so no "
    "validation fold informs its own model."
)
threshold = ui.threshold_control("evaluation")

comparison = ui.at_threshold(ui.output("comparison"), threshold)
folds = ui.at_threshold(ui.output("folds"), threshold)
paired = ui.at_threshold(ui.output("paired"), threshold)
block = evaluation["thresholds"][f"{threshold:.2f}"]
order = comparison["model"].tolist()

st.subheader("Model comparison", icon=":material/leaderboard:")
table = comparison[["model"] + [f"{m}_mean" for m in METRICS] + ["balanced_accuracy_std"]].copy()
table["Kind"] = ["Baseline" if m in BASELINES else ("Selected" if m == selected else "Model") for m in table["model"]]
table["Balanced accuracy 95% CI"] = [
    ui.interval(lo, hi) for lo, hi in zip(comparison["balanced_accuracy_ci_low"], comparison["balanced_accuracy_ci_high"])
]
table = table.rename(columns={"model": "Model", "balanced_accuracy_std": "Balanced accuracy SD"})
table = table.rename(columns={f"{m}_mean": charts.METRIC_LABELS[m] for m in METRICS})
st.dataframe(
    table[["Model", "Kind", "Balanced accuracy", "Balanced accuracy SD", "Balanced accuracy 95% CI",
           "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier score"]],
    hide_index=True,
    column_config={
        c: st.column_config.NumberColumn(format="%.3f")
        for c in ["Balanced accuracy", "Balanced accuracy SD", "Accuracy", "Precision", "Recall",
                  "F1", "ROC-AUC", "Brier score"]
    },
    alt="Repeated cross-validation metrics for every model and baseline",
)
st.caption(
    "Means over 50 validation folds, sorted by balanced accuracy, the selection metric. The CGPA "
    "cutoff outputs a class, not a probability, so it has no Brier score; its ROC-AUC ranks "
    "records by CGPA."
)

st.subheader("Repeated cross-validation distributions", icon=":material/scatter_plot:")
metric = st.segmented_control(
    "Metric",
    options=METRICS,
    default="balanced_accuracy",
    format_func=lambda m: charts.METRIC_LABELS[m],
    required=True,
    key="evaluation_metric",
)
left, right = st.columns(2, gap="large")
with left:
    ui.chart(
        charts.fold_distribution(folds, metric, selected, order),
        alt=f"{charts.METRIC_LABELS[metric]} on each of the 50 validation folds, per model",
    )
    st.caption("Each dot is one validation fold. Every model sees the same 50 folds.")
with right:
    ui.chart(
        charts.interval_chart(comparison, metric, selected),
        alt=f"Mean {charts.METRIC_LABELS[metric]} with 95% confidence interval, per model",
    )
    st.caption(f"{protocol['confidence_interval']}.")

st.subheader(f"Paired tests: {selected} against each baseline", icon=":material/balance:")
tests_col, forest_col = st.columns([7, 5], gap="large")
with tests_col:
    view = paired.assign(
        ci=[ui.interval(lo, hi, signed=True) for lo, hi in zip(paired["ci_low"], paired["ci_high"])],
        verdict=["Supported" if s else "Not supported" for s in paired["significant"]],
    )[["baseline", "difference", "ci", "p_value", "p_value_holm", "verdict"]]
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "baseline": "Compared with",
            "difference": st.column_config.NumberColumn("Difference", format="%+.3f"),
            "ci": "95% bootstrap CI",
            "p_value": st.column_config.NumberColumn("p", format="%.4f"),
            "p_value_holm": st.column_config.NumberColumn("Holm p", format="%.4f"),
            "verdict": "Improvement",
        },
        alt="Paired comparison of the selected model with each baseline",
    )
    st.caption(
        f"Difference in balanced accuracy on one cross-fitted prediction per record. "
        f"{int(paired['n_resamples'].iloc[0]):,} paired bootstrap resamples for the interval and "
        "as many paired permutations for the p-value. An improvement is called supported only "
        "when the interval excludes zero and the Holm-corrected p-value is below 0.05."
    )
with forest_col:
    ui.chart(charts.paired_forest(paired), alt="Paired differences in balanced accuracy with 95% confidence intervals")

st.subheader(f"Cross-fitted diagnostics for {selected}", icon=":material/stethoscope:")
st.markdown(
    "Each record's out-of-fold probability is averaged over the ten repeats, giving one "
    "prediction per record that never came from a model trained on that record."
)
rows = oof_for(ui.output("oof"), threshold, selected)
cm = block["cross_fitted"]["confusion_matrix"]
a, b = st.columns(2, gap="large")
with a:
    st.markdown("**Confusion matrix**")
    ui.chart(charts.confusion_chart(cm), alt="Cross-fitted confusion matrix")
with b:
    st.markdown("**Calibration**")
    ui.chart(charts.calibration_chart(rows["y_true"], rows["mean_oof_prob"]), alt="Calibration curve of cross-fitted probabilities")
c, d = st.columns(2, gap="large")
with c:
    st.markdown("**ROC curve**")
    ui.chart(charts.roc_chart(rows["y_true"], rows["mean_oof_prob"]), alt="Cross-fitted ROC curve")
with d:
    st.markdown("**Precision-recall curve**")
    ui.chart(charts.pr_chart(rows["y_true"], rows["mean_oof_prob"]), alt="Cross-fitted precision-recall curve")

st.subheader("Error analysis", icon=":material/error_med:")
errors = error_frame(ui.dataset(), ui.output("oof"), threshold, selected)
n_err = int(errors["error"].sum())
st.markdown(
    f"{n_err} of {len(errors)} cross-fitted predictions are wrong: {cm['fp']} false positives and "
    f"{cm['fn']} false negatives. Because the label is a cut on a continuous score, records close "
    "to the cut are the hard ones."
)
slices_col, worst_col = st.columns(2, gap="large")
with slices_col:
    st.markdown("**Error rate by slice**")
    st.dataframe(
        error_slices(errors, threshold),
        hide_index=True,
        column_config={
            "slice": "Slice",
            "value": "Group",
            "n": "Records",
            "errors": "Errors",
            "error_rate": st.column_config.ProgressColumn("Error rate", format="percent", min_value=0, max_value=1),
        },
        alt="Cross-fitted error rate within slices of the data",
    )
with worst_col:
    st.markdown("**Most confident errors**")
    worst = errors[errors["error"]].sort_values("confidence", ascending=False).head(10)
    st.dataframe(
        worst[["error_type", "mean_oof_prob", "Chance_of_Admit", "CGPA", "GRE_Score", "TOEFL_Score", "Research"]],
        hide_index=True,
        column_config={
            "error_type": "Error",
            "mean_oof_prob": st.column_config.NumberColumn("Predicted probability", format="%.2f"),
            "Chance_of_Admit": st.column_config.NumberColumn("Chance of Admit", format="%.2f"),
            "GRE_Score": "GRE",
            "TOEFL_Score": "TOEFL",
        },
        alt="The ten errors where the model was most confident",
    )
    st.caption("Chance of Admit shows how far each record's score is from the target threshold.")
