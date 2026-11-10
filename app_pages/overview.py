import streamlit as st

from gradmit import charts, ui
from gradmit.baselines import CGPA_CUTOFF, MAJORITY
from gradmit.data import DEFAULT_TARGET_THRESHOLD

evaluation = ui.evaluation()
comparison = ui.at_threshold(ui.output("comparison"), DEFAULT_TARGET_THRESHOLD)
paired = ui.at_threshold(ui.output("paired"), DEFAULT_TARGET_THRESHOLD).set_index("baseline")
selected = evaluation["selected_model"]
block = evaluation["thresholds"][f"{DEFAULT_TARGET_THRESHOLD:.2f}"]
balance = block["class_balance"]
model_ba = block["selected_model_cv"]["balanced_accuracy"]
cgpa_ba = block["cgpa_cutoff_baseline_cv"]["balanced_accuracy"]
vs_cgpa = paired.loc[CGPA_CUTOFF]
vs_majority = paired.loc[MAJORITY]

st.title("How much does a full model add over one CGPA cutoff?", icon=":material/home:")
st.markdown(
    "GradMit studies 500 applicant profiles from the Graduate Admission dataset. The dataset "
    "holds no admission decisions. Its `Chance of Admit` column is an estimated score, so this "
    "project predicts a proxy label: **whether that estimated score is at or above a chosen "
    "threshold**."
)

lead, side = st.columns([3, 2], gap="large")

with lead:
    st.subheader(f"Balanced accuracy at target threshold {DEFAULT_TARGET_THRESHOLD:.2f}", icon=":material/bar_chart:")
    ui.chart(
        charts.interval_chart(comparison, "balanced_accuracy", selected),
        alt="Mean balanced accuracy and 95% confidence interval for every model and baseline",
    )
    st.caption(
        "Mean of 50 validation folds from 5-fold stratified cross-validation repeated 10 times. "
        "Bars are 95% intervals with the Nadeau-Bengio correction. Teal is the selected model, "
        "ochre the two baselines."
    )

with side, st.container(border=True):
    st.subheader("What the evaluation found", icon=":material/insights:")
    if vs_cgpa["significant"]:
        verdict = (
            f"{selected} improves on the CGPA cutoff by {vs_cgpa['difference']:+.3f} balanced "
            f"accuracy (95% CI {ui.interval(vs_cgpa['ci_low'], vs_cgpa['ci_high'], signed=True)}, "
            f"{ui.holm(vs_cgpa['p_value_holm'])})."
        )
    else:
        verdict = (
            f"{selected} is **not distinguishable from a single CGPA cutoff**: the paired "
            f"difference is {vs_cgpa['difference']:+.3f} balanced accuracy "
            f"(95% CI {ui.interval(vs_cgpa['ci_low'], vs_cgpa['ci_high'], signed=True)}, "
            f"{ui.holm(vs_cgpa['p_value_holm'])})."
        )
    st.markdown(
        f"- {selected} had the highest mean balanced accuracy of five model families: "
        f"**{model_ba['mean']:.3f}** (95% CI {ui.interval(model_ba['ci_low'], model_ba['ci_high'])}).\n"
        f"- The CGPA cutoff rule alone reaches **{cgpa_ba['mean']:.3f}**.\n"
        f"- {verdict}\n"
        f"- Against always guessing the majority class the gain is "
        f"{vs_majority['difference']:+.3f} "
        f"(95% CI {ui.interval(vs_majority['ci_low'], vs_majority['ci_high'], signed=True)}, "
        f"{ui.holm(vs_majority['p_value_holm'])})."
    )
    st.page_link("app_pages/evaluation.py", label="See the full model evaluation", icon=":material/monitoring:")
    st.page_link("app_pages/sensitivity.py", label="Compare thresholds 0.70, 0.75 and 0.80", icon=":material/tune:")

st.subheader("The data and the label", icon=":material/dataset:")
with st.container(horizontal=True):
    st.metric("Records", f"{balance['n']}", border=True)
    st.metric("Features", f"{len(evaluation['data']['features'])}", border=True)
    st.metric(
        f"Proxy-positive at {DEFAULT_TARGET_THRESHOLD:.2f}",
        f"{balance['n_positive']} ({ui.pct(balance['positive_share'])})",
        border=True,
    )
    st.metric("Majority-class accuracy", ui.pct(balance["majority_accuracy"]), border=True)

left, right = st.columns(2, gap="large")
with left, st.container(border=True, height="stretch"):
    st.markdown(
        "**Two thresholds, kept separate**\n\n"
        "- The **target threshold** (0.70, 0.75 or 0.80) defines the label from "
        "`Chance of Admit`. The deployed model uses 0.75.\n"
        "- The **classifier decision threshold** (0.50) turns the model's probability into a "
        "predicted class."
    )
with right, st.container(border=True, height="stretch"):
    st.markdown(
        "**What this app does not claim**\n\n"
        "- It does not say whether any university would accept an applicant.\n"
        "- It does not show that changing a score changes an outcome. Every relationship here "
        "is an association in observational data."
    )

st.info(
    f"An earlier version of this project reported {evaluation['legacy']['retired_result']} "
    "and described it as admission prediction. That figure is retired: it came from one split, "
    "had no baseline, and treated the proxy label as a real outcome.",
    icon=":material/history:",
    title="Retired legacy result",
)
