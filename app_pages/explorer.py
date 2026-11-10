import streamlit as st

from gradmit import charts, ui
from gradmit.data import FEATURE_LABELS, FEATURES, SCORE_COLUMN, TARGET_NAME, make_target

df = ui.dataset()
SHORT_LABELS = {
    "GRE_Score": "GRE", "TOEFL_Score": "TOEFL", "University_Rating": "Rating", "SOP": "SOP",
    "LOR": "LOR", "CGPA": "CGPA", "Research": "Research", SCORE_COLUMN: "Chance of Admit",
}

st.title("Data explorer")
st.markdown(
    "All 500 records of the Graduate Admission dataset. `Chance of Admit` is the dataset's own "
    "estimated score; the proxy label is computed from it at the threshold you choose."
)
threshold = ui.threshold_control("explorer")
target = make_target(df, threshold)
data = df.assign(**{TARGET_NAME: target})

st.subheader("Research experience and high estimated admission chance")
research = ui.at_threshold(ui.output("research"), threshold)
has = research[research["research"] == 1].iloc[0]
none = research[research["research"] == 0].iloc[0]
chart_col, text_col = st.columns([3, 2], gap="large")
with chart_col:
    ui.chart(
        charts.research_chart(research),
        alt="Share of records above the Chance-of-Admit threshold, for records with and without research experience",
    )
with text_col:
    st.markdown(
        f"Of the {int(has['n'])} records **with** research experience, "
        f"{ui.pct(has['positive_rate'])} are above the {threshold:.2f} threshold "
        f"(Wilson 95% CI {ui.pct(has['ci_low'])} to {ui.pct(has['ci_high'])}).\n\n"
        f"Of the {int(none['n'])} records **without**, {ui.pct(none['positive_rate'])} are "
        f"(Wilson 95% CI {ui.pct(none['ci_low'])} to {ui.pct(none['ci_high'])})."
    )
    st.info(
        "Records with research experience also have higher CGPA and test scores on average, and "
        "the score being cut was itself an estimate. The gap does not show what research "
        "experience does to anyone's chances.",
        icon=":material/info:",
        title="An association, not a causal effect",
    )

st.subheader("Feature distributions")
feature = st.selectbox("Feature", FEATURES, index=FEATURES.index("CGPA"), format_func=FEATURE_LABELS.get)
left, right = st.columns(2, gap="large")
with left:
    ui.chart(
        charts.feature_histogram(df, feature, target, FEATURE_LABELS[feature]),
        alt=f"Distribution of {FEATURE_LABELS[feature]} for proxy-positive and proxy-negative records",
    )
with right:
    ui.chart(
        charts.correlation_chart(df[FEATURES + [SCORE_COLUMN]].corr(), SHORT_LABELS),
        alt="Pearson correlation between every pair of features and Chance of Admit",
    )
    st.caption(
        "The features are strongly correlated with one another, so no single one can be credited "
        "with the score. Correlation here describes this sample only."
    )

st.subheader("Records")
with st.container(horizontal=True, vertical_alignment="bottom"):
    label_filter = st.segmented_control(
        "Proxy label", ["All", "Proxy-positive", "Proxy-negative"], default="All", required=True
    )
    research_filter = st.segmented_control(
        "Research experience", ["All", "Has research", "No research"], default="All", required=True
    )
    ratings = st.pills("University rating", [1, 2, 3, 4, 5], selection_mode="multi", default=[1, 2, 3, 4, 5])
cgpa_low, cgpa_high = st.slider(
    "CGPA range", float(df["CGPA"].min()), float(df["CGPA"].max()),
    (float(df["CGPA"].min()), float(df["CGPA"].max())), step=0.01,
)

view = data[data["CGPA"].between(cgpa_low, cgpa_high) & data["University_Rating"].isin(ratings)]
if label_filter != "All":
    view = view[view[TARGET_NAME] == int(label_filter == "Proxy-positive")]
if research_filter != "All":
    view = view[view["Research"] == int(research_filter == "Has research")]

if view.empty:
    st.info("No records match these filters. Widen the CGPA range or select more ratings.", icon=":material/filter_alt_off:")
else:
    st.caption(f"{len(view)} of {len(data)} records")
    st.dataframe(
        view,
        hide_index=True,
        height=420,
        column_config={
            "GRE_Score": "GRE",
            "TOEFL_Score": "TOEFL",
            "University_Rating": "University rating",
            "SOP": st.column_config.NumberColumn("SOP", format="%.1f"),
            "LOR": st.column_config.NumberColumn("LOR", format="%.1f"),
            "CGPA": st.column_config.NumberColumn(format="%.2f"),
            "Research": st.column_config.CheckboxColumn("Research"),
            SCORE_COLUMN: st.column_config.ProgressColumn("Chance of Admit", format="%.2f", min_value=0, max_value=1),
            TARGET_NAME: st.column_config.CheckboxColumn(f"At or above {threshold:.2f}"),
        },
        alt="Filtered records of the Graduate Admission dataset",
    )
    st.download_button(
        "Download filtered records",
        view.to_csv(index=False),
        file_name="gradmit_filtered_records.csv",
        mime="text/csv",
        icon=":material/download:",
    )
