import pandas as pd
import streamlit as st

from gradmit import charts, ui
from gradmit.baselines import CGPA_CUTOFF, MAJORITY

evaluation = ui.evaluation()
selected = evaluation["selected_model"]

st.title("Baselines", icon=":material/straighten:")
st.markdown(
    "A model result means little without something simple to beat. Both baselines are scored on "
    "exactly the same 50 validation folds as every model."
)
threshold = ui.threshold_control("baselines")

block = evaluation["thresholds"][f"{threshold:.2f}"]
balance = block["class_balance"]
folds = ui.at_threshold(ui.output("folds"), threshold)
paired = ui.at_threshold(ui.output("paired"), threshold)
majority_cv = block["majority_baseline_cv"]
cgpa_cv = block["cgpa_cutoff_baseline_cv"]
model_cv = block["selected_model_cv"]
cutoffs = folds.loc[folds["model"] == CGPA_CUTOFF, "cgpa_cutoff"]

left, right = st.columns(2, gap="large")
with left, st.container(border=True, height="stretch"):
    st.subheader("Majority baseline", icon=":material/groups:")
    st.markdown(
        "`DummyClassifier(strategy=\"most_frequent\")` predicts the larger class of the training "
        "fold for every record."
    )
    majority_class = "proxy-negative" if balance["n_negative"] >= balance["n_positive"] else "proxy-positive"
    st.markdown(
        f"- Proxy-positive: {balance['n_positive']}/{balance['n']} = {ui.pct(balance['positive_share'])}\n"
        f"- Proxy-negative: {balance['n_negative']}/{balance['n']} = {ui.pct(1 - balance['positive_share'])}\n"
        f"- Always predicts **{majority_class}**"
    )
    with st.container(horizontal=True):
        st.metric("Accuracy", ui.pct(majority_cv["accuracy"]["mean"]))
        st.metric("Balanced accuracy", ui.pct(majority_cv["balanced_accuracy"]["mean"]))
with right, st.container(border=True, height="stretch"):
    st.subheader("CGPA cutoff baseline", icon=":material/school:")
    st.markdown(
        "One rule: predict proxy-positive when `CGPA >= cutoff`. In each fold the cutoff is "
        "chosen on the training rows to maximise balanced accuracy, frozen, then applied to the "
        "validation rows. It is never chosen with all 500 labels."
    )
    st.markdown(
        f"- Cutoffs chosen across folds: {cutoffs.min():.2f} to {cutoffs.max():.2f} "
        f"(median {cutoffs.median():.2f})"
    )
    with st.container(horizontal=True):
        st.metric("Accuracy", ui.pct(cgpa_cv["accuracy"]["mean"]))
        st.metric("Balanced accuracy", ui.pct(cgpa_cv["balanced_accuracy"]["mean"]))

st.subheader("Side by side", icon=":material/compare_arrows:")
summary = pd.DataFrame(
    [
        {
            "Approach": name,
            "Accuracy": cv["accuracy"]["mean"],
            "Balanced accuracy": cv["balanced_accuracy"]["mean"],
            "95% CI": ui.interval(cv["balanced_accuracy"]["ci_low"], cv["balanced_accuracy"]["ci_high"]),
            "F1": cv["f1"]["mean"],
            "ROC-AUC": cv["roc_auc"]["mean"],
        }
        for name, cv in ((MAJORITY, majority_cv), (CGPA_CUTOFF, cgpa_cv), (f"{selected} (selected)", model_cv))
    ]
)
st.dataframe(
    summary,
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Accuracy", "Balanced accuracy", "F1", "ROC-AUC"]},
    alt="Baselines and the selected model compared on repeated cross-validation metrics",
)

a, b = st.columns(2, gap="large")
with a:
    ui.chart(
        charts.paired_forest(paired), alt="Paired differences in balanced accuracy against each baseline",
        title="Does the full model beat each baseline?", sub="model minus baseline, paired 95% interval",
        what="The selected model's balanced accuracy minus each baseline's, on the same records.",
        why="This is the question the project exists to answer. Zero inside the interval means no supported gain.",
    )
    for _, row in paired.iterrows():
        claim = "a supported improvement" if row["significant"] else "no supported difference"
        st.markdown(
            f"- Against the {row['baseline'].lower()}: {row['difference']:+.3f} "
            f"(95% CI {ui.interval(row['ci_low'], row['ci_high'], signed=True)}, "
            f"{ui.holm(row['p_value_holm'])}), {claim}."
        )
with b:
    ui.chart(
        charts.cutoff_histogram(cutoffs),
        alt="Histogram of the CGPA cutoff selected in each of the 50 training folds",
        title="How stable is the CGPA cutoff?", sub="the cutoff chosen in each of the 50 training folds",
        what="Each fold picks its own CGPA cutoff from its own training rows; this is where they landed.",
        why="A tight spread means the rule barely depends on which records it was fitted on.",
    )
