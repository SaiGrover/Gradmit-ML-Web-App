import pandas as pd
import streamlit as st

from gradmit import charts, ui

evaluation = ui.evaluation()
selected = evaluation["selected_model"]
sensitivity = ui.output("sensitivity")

st.title("Threshold sensitivity", icon=":material/tune:")
st.markdown(
    "The label depends on where `Chance of Admit` is cut. The whole evaluation was repeated at "
    "three target thresholds. Switching below reads committed results; nothing is retrained in "
    "the browser."
)
threshold = ui.threshold_control("sensitivity")
row = ui.at_threshold(sensitivity, threshold).iloc[0]

with st.container(horizontal=True):
    st.metric("Proxy-positive", f"{int(row['n_positive'])} ({ui.pct(row['positive_share'])})", border=True)
    st.metric("Majority accuracy", ui.pct(row["majority_accuracy"]), border=True)
    st.metric("CGPA cutoff balanced accuracy", f"{row['cgpa_balanced_accuracy']:.3f}", border=True)
    st.metric(
        f"{selected} balanced accuracy",
        f"{row['model_balanced_accuracy']:.3f}",
        delta=f"{row['improvement_over_cgpa']:+.3f} vs CGPA cutoff",
        delta_color="off",
        border=True,
    )

supported = bool(row["improvement_over_cgpa_significant"])
message = (
    f"At {threshold:.2f} the paired difference between {selected} and the CGPA cutoff is "
    f"{row['improvement_over_cgpa']:+.3f} balanced accuracy "
    f"(95% CI {ui.interval(row['improvement_over_cgpa_ci_low'], row['improvement_over_cgpa_ci_high'], signed=True)}, "
    f"{ui.holm(row['improvement_over_cgpa_p_holm'])}). "
)
if supported:
    st.success(message + "The interval excludes zero and the corrected p-value is below 0.05.",
               icon=":material/check_circle:", title="Improvement over the CGPA cutoff is supported")
else:
    st.info(message + "The interval includes zero, so no improvement is claimed.",
            icon=":material/balance:", title="No supported improvement over the CGPA cutoff")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Balanced accuracy across thresholds")
    ui.chart(charts.sensitivity_chart(sensitivity, selected),
             alt="Balanced accuracy of the majority baseline, CGPA cutoff and selected model at each target threshold")
with right:
    st.subheader("Class balance across thresholds")
    ui.chart(charts.class_balance_chart(sensitivity),
             alt="Share of proxy-positive and proxy-negative records at each target threshold")

st.subheader("All three thresholds")
table = pd.DataFrame(
    {
        "Threshold": sensitivity["target_threshold"].map("{:.2f}".format),
        "Positive": sensitivity["n_positive"],
        "Positive share": sensitivity["positive_share"] * 100,
        "Majority accuracy": sensitivity["majority_accuracy"] * 100,
        "CGPA cutoff BA": sensitivity["cgpa_balanced_accuracy"],
        "Model BA": sensitivity["model_balanced_accuracy"],
        "Model BA 95% CI": [ui.interval(a, b) for a, b in zip(sensitivity["model_balanced_accuracy_ci_low"], sensitivity["model_balanced_accuracy_ci_high"])],
        "Model F1": sensitivity["model_f1"],
        "Model ROC-AUC": sensitivity["model_roc_auc"],
        "Model minus CGPA": sensitivity["improvement_over_cgpa"],
        "Difference 95% CI": [ui.interval(a, b, signed=True) for a, b in zip(sensitivity["improvement_over_cgpa_ci_low"], sensitivity["improvement_over_cgpa_ci_high"])],
        "Holm p": sensitivity["improvement_over_cgpa_p_holm"],
    }
)
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Positive share": st.column_config.NumberColumn(format="%.1f%%"),
        "Majority accuracy": st.column_config.NumberColumn(format="%.1f%%"),
        "CGPA cutoff BA": st.column_config.NumberColumn(format="%.3f"),
        "Model BA": st.column_config.NumberColumn(format="%.3f"),
        "Model F1": st.column_config.NumberColumn(format="%.3f"),
        "Model ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Model minus CGPA": st.column_config.NumberColumn(format="%+.3f"),
        "Holm p": st.column_config.NumberColumn(format="%.4f"),
    },
    alt="Threshold sensitivity results for target thresholds 0.70, 0.75 and 0.80",
)
st.caption(f"BA is balanced accuracy. Model is {selected}, the family selected at the default threshold and reused at the others.")

st.subheader("Is the model ranking stable?")
ranking = pd.DataFrame(
    {f"{float(t):.2f}": block["model_ranking"] for t, block in evaluation["thresholds"].items()}
)
ranking.insert(0, "Rank", range(1, len(ranking) + 1))
st.dataframe(ranking, hide_index=True, alt="Model families ranked by mean balanced accuracy at each target threshold")
st.caption(
    "Model families ordered by mean balanced accuracy. The gaps between neighbours are small "
    "compared with their fold-to-fold spread, so the order below first place is not meaningful."
)
