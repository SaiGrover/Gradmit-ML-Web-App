import streamlit as st

from gradmit import ui
from gradmit.baselines import CGPA_CUTOFF, MAJORITY
from gradmit.data import DEFAULT_TARGET_THRESHOLD

evaluation = ui.evaluation()
paired = ui.output("paired")
research = ui.at_threshold(ui.output("research"), DEFAULT_TARGET_THRESHOLD).set_index("group")
target = evaluation["target"]
protocol = evaluation["protocol"]
selected = evaluation["selected_model"]
block = evaluation["thresholds"][f"{DEFAULT_TARGET_THRESHOLD:.2f}"]
balance = block["class_balance"]
thresholds = [f"{t:.2f}" for t in target["target_thresholds"]]


def beats(baseline: str) -> list[str]:
    rows = paired[(paired["baseline"] == baseline) & paired["significant"].astype(bool)]
    return [f"{t:.2f}" for t in rows["target_threshold"]]


def joined(items: list[str]) -> str:
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


ui.page_header(
    "Problem statement",
    icon=":material/assignment:",
    lede=(
        "What this project sets out to answer, why the question had to be reframed, and how an "
        "answer is judged. Every figure on this page is read from the committed evaluation outputs."
    ),
)

with st.container(border=True):
    st.markdown("**The problem, in one sentence**")
    st.markdown(
        f"Given an applicant's seven profile features, predict whether the dataset's estimated "
        f"`Chance of Admit` is at or above a chosen threshold (`{target['name']}`), and establish "
        "whether a multi-feature model does measurably better than two simple rules: always "
        "predicting the majority class, and a single CGPA cutoff."
    )

background, reframe = st.columns(2, gap="large")

with background:
    st.subheader("Background", icon=":material/school:")
    st.markdown(
        "Applicants to graduate programmes, and the people advising them, want to know which "
        "parts of a profile line up with a strong application. The Graduate Admission dataset "
        f"offers {evaluation['data']['n_rows']} profiles with GRE and TOEFL scores, university "
        "rating, statement-of-purpose and recommendation strength, CGPA and research experience, "
        "each paired with an estimated `Chance of Admit` between 0 and 1."
    )
    st.markdown(
        "That score is an estimate, not a decision. No record says whether anyone was admitted, so "
        "any model trained here learns to reproduce the estimate, not the outcome."
    )

with reframe:
    st.subheader("Why the question was reframed", icon=":material/rule:")
    st.markdown(
        f"An earlier version of this project reported {evaluation['legacy']['retired_result']} "
        "and called it admission prediction. Three things were wrong with that:\n\n"
        "- The label was read as a record of who was admitted, which the data does not contain.\n"
        "- One random split gives one noisy number with no measure of its spread.\n"
        "- Without a baseline, there was no way to tell whether 91.2% was hard to reach. A single "
        "CGPA cutoff turns out to score almost as well."
    )

st.subheader("Research questions", icon=":material/help:")
st.caption("Each question links to the page that answers it. The short answers are computed from the committed outputs.")

model_ba = block["selected_model_cv"]["balanced_accuracy"]
has, no = research.loc["Has research"], research.loc["No research"]
questions = [
    (
        "Which model family scores best?",
        f"Under repeated nested cross-validation ({protocol['n_folds']} validation folds), which "
        "of five model families reaches the highest balanced accuracy?",
        f"{selected}, at {model_ba['mean']:.3f} balanced accuracy "
        f"(95% CI {ui.interval(model_ba['ci_low'], model_ba['ci_high'])}) at target threshold "
        f"{DEFAULT_TARGET_THRESHOLD:.2f}.",
        "app_pages/evaluation.py",
        ":material/monitoring:",
    ),
    (
        "Does it beat simple rules?",
        "In paired tests on the same records, does the selected model improve on the majority "
        "baseline and on a CGPA cutoff chosen from training data only?",
        f"It beats the majority baseline at {joined(beats(MAJORITY))}. It beats the CGPA cutoff "
        f"only at {joined(beats(CGPA_CUTOFF))}.",
        "app_pages/baselines.py",
        ":material/straighten:",
    ),
    (
        "Does the threshold change the answer?",
        f"How do class balance, baselines and the model compare when the label is cut at "
        f"{joined(thresholds)}?",
        "Yes. The share of proxy-positive records and the gap to the CGPA cutoff both move with "
        "the threshold.",
        "app_pages/sensitivity.py",
        ":material/tune:",
    ),
    (
        "How does research experience relate to the label?",
        "What share of applicants with and without research experience are proxy-positive?",
        f"{ui.pct(has['positive_rate'])} with research (n={int(has['n'])}) against "
        f"{ui.pct(no['positive_rate'])} without (n={int(no['n'])}) at "
        f"{DEFAULT_TARGET_THRESHOLD:.2f}. This is an association, not a cause.",
        "app_pages/explorer.py",
        ":material/table_chart:",
    ),
]

for row in (questions[:2], questions[2:]):
    cols = st.columns(2, gap="medium")
    for col, (title, question, answer, page, icon) in zip(cols, row):
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.markdown(question)
            st.markdown(f":primary-badge[Answer] {answer}")
            st.page_link(page, label="Open the evidence", icon=icon)

judged, scope = st.columns(2, gap="large")

with judged:
    st.subheader("How an answer is judged", icon=":material/fact_check:")
    st.markdown(
        f"- **Primary metric: balanced accuracy.** At {DEFAULT_TARGET_THRESHOLD:.2f}, "
        f"{ui.pct(balance['positive_share'])} of records are proxy-positive, so plain accuracy "
        f"rewards guessing the majority class ({ui.pct(balance['majority_accuracy'])}).\n"
        f"- **Repeated, nested evaluation.** {protocol['n_splits']}-fold stratified "
        f"cross-validation repeated {protocol['n_repeats']} times, with tuning inside each "
        "training fold.\n"
        "- **An improvement counts only if** its paired 95% bootstrap interval excludes zero "
        f"and its Holm-corrected permutation p-value is below 0.05 "
        f"({protocol['n_resamples']:,} resamples each)."
    )

with scope:
    st.subheader("Scope", icon=":material/crop_free:")
    st.markdown(
        "**In scope:** modelling the dataset's own proxy label, comparing models with baselines "
        "on equal terms, and describing associations in the data.\n\n"
        "**Out of scope:** predicting real admission decisions, advising applicants, and any "
        "causal claim about what raises a chance of admission."
    )

st.subheader("The data at a glance", icon=":material/dataset:")
with st.container(horizontal=True):
    st.metric("Profiles", f"{balance['n']}", border=True)
    st.metric("Features", f"{len(evaluation['data']['features'])}", border=True)
    st.metric(
        f"Proxy-positive at {DEFAULT_TARGET_THRESHOLD:.2f}",
        f"{balance['n_positive']} ({ui.pct(balance['positive_share'])})",
        border=True,
    )
    st.metric("Target thresholds studied", joined(thresholds), border=True)

st.page_link("app_pages/home.py", label="Back to the project and its headline result", icon=":material/arrow_forward:")
