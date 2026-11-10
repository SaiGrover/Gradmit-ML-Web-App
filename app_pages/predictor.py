import pandas as pd
import streamlit as st

from gradmit import charts, ui
from gradmit.data import FEATURE_LABELS, FEATURES, out_of_range, validate_profile
from gradmit.modelling import positive_probability

meta = ui.metadata()
evaluation = ui.evaluation()
df = ui.dataset()
ranges = meta["feature_ranges"]
target_threshold = meta["target_threshold"]
classifier_threshold = meta["classifier_threshold"]
cv = evaluation["thresholds"][f"{target_threshold:.2f}"]["selected_model_cv"]

st.title("Profile predictor")
st.markdown(
    f"Enter a profile to estimate the probability that the dataset's `Chance of Admit` score "
    f"for it would be at or above **{target_threshold:.2f}**. This is a dataset-proxy estimate, "
    "not a real admission decision."
)

with st.container(horizontal=True):
    st.badge(f"Target threshold {target_threshold:.2f}", icon=":material/flag:", color="primary")
    st.badge(f"Classifier threshold {classifier_threshold:.2f}", icon=":material/rule:", color="gray")
    st.badge(meta["model_name"], icon=":material/model_training:", color="gray")

form_col, result_col = st.columns([5, 6], gap="large")

with form_col, st.form("profile"):
    st.subheader("Profile")
    a, b = st.columns(2)
    gre = a.number_input("GRE score", min_value=260, max_value=340, value=317, step=1,
                         help=f"Training range {ranges['GRE_Score']['min']:.0f} to {ranges['GRE_Score']['max']:.0f}")
    toefl = b.number_input("TOEFL score", min_value=0, max_value=120, value=107, step=1,
                           help=f"Training range {ranges['TOEFL_Score']['min']:.0f} to {ranges['TOEFL_Score']['max']:.0f}")
    c, d = st.columns(2)
    cgpa = c.number_input("CGPA (10-point scale)", min_value=0.0, max_value=10.0, value=8.56,
                          step=0.01, format="%.2f",
                          help=f"Training range {ranges['CGPA']['min']:.2f} to {ranges['CGPA']['max']:.2f}")
    rating = d.number_input("University rating (1 to 5)", min_value=1, max_value=5, value=3, step=1)
    e, f = st.columns(2)
    sop = e.slider("Statement of purpose strength", 1.0, 5.0, 3.5, 0.5)
    lor = f.slider("Letter of recommendation strength", 1.0, 5.0, 3.5, 0.5)
    research = st.toggle("Has research experience", value=True)
    submitted = st.form_submit_button("Estimate probability", type="primary", icon=":material/calculate:")

profile = {
    "GRE_Score": gre,
    "TOEFL_Score": toefl,
    "University_Rating": rating,
    "SOP": sop,
    "LOR": lor,
    "CGPA": cgpa,
    "Research": int(research),
}

with result_col:
    st.subheader("Estimate")
    try:
        row = validate_profile(profile)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:", title="Check the profile")
        st.stop()

    model = ui.pipeline()
    probability = float(positive_probability(model, row)[0])
    above = probability >= classifier_threshold

    st.metric(
        f"Estimated probability that Chance of Admit is at or above {target_threshold:.2f}",
        ui.pct(probability),
    )
    ui.chart(
        charts.probability_bar(probability, classifier_threshold),
        alt=f"Estimated probability {probability:.0%} against the {classifier_threshold:.2f} classifier threshold",
    )
    st.markdown(
        f"Predicted class: **{'above' if above else 'below'} the target threshold**, because the "
        f"estimate is {'at or above' if above else 'below'} the classifier threshold of "
        f"{classifier_threshold:.2f}."
    )

    flagged = out_of_range(profile, ranges)
    if flagged:
        details = "; ".join(
            f"{FEATURE_LABELS[f]} {profile[f]:g} (training range {ranges[f]['min']:g} to {ranges[f]['max']:g})"
            for f in flagged
        )
        st.warning(
            f"{details}. The model has seen no records like this, so the estimate is an "
            "extrapolation and should not be relied on.",
            icon=":material/warning:",
            title="Outside the training range",
        )

    st.caption(
        f"In repeated cross-validation this model family reached balanced accuracy "
        f"{cv['balanced_accuracy']['mean']:.3f} "
        f"(95% CI {ui.interval(cv['balanced_accuracy']['ci_low'], cv['balanced_accuracy']['ci_high'])}) "
        f"and Brier score {cv['brier']['mean']:.3f} on the proxy label. Probabilities come "
        "straight from the fitted model; the cross-fitted calibration curve is on the Model "
        "evaluation page."
    )

st.subheader("How this profile sits in the data")
percentiles = pd.DataFrame(
    {
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Your value": [float(profile[f]) for f in FEATURES],
        "Training minimum": [ranges[f]["min"] for f in FEATURES],
        "Training maximum": [ranges[f]["max"] for f in FEATURES],
        "Share of records at or below": [float((df[f] <= profile[f]).mean()) * 100 for f in FEATURES],
    }
)
st.dataframe(
    percentiles,
    hide_index=True,
    column_config={
        "Your value": st.column_config.NumberColumn(format="%.2f"),
        "Training minimum": st.column_config.NumberColumn(format="%.2f"),
        "Training maximum": st.column_config.NumberColumn(format="%.2f"),
        "Share of records at or below": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=100
        ),
    },
    alt="The entered profile compared with the range of the 500 training records",
)

steps = getattr(model, "named_steps", {})
if "scale" in steps and hasattr(steps.get("model"), "coef_"):
    z = steps["scale"].transform(row)[0]
    contributions = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Log-odds contribution": steps["model"].coef_[0] * z,
        }
    ).sort_values("Log-odds contribution")
    with st.expander("How the model weighed this profile"):
        st.bar_chart(contributions, x="Feature", y="Log-odds contribution", horizontal=True,
                     color=charts.TEAL)
        st.caption(
            "Each bar is the model coefficient times how far the value sits from the dataset "
            "mean, in standard deviations. It describes the fitted model's arithmetic. It is not "
            "evidence that changing a value would change a real decision."
        )
