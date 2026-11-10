import streamlit as st

from gradmit import charts, style, ui
from gradmit.baselines import CGPA_CUTOFF, MAJORITY
from gradmit.data import ARTIFACTS_DIR, DATA_PATH, DEFAULT_TARGET_THRESHOLD, OUTPUTS_DIR, ROOT
from gradmit.reporting import OUTPUT_FILES

evaluation = ui.evaluation()
meta = ui.metadata()
comparison = ui.at_threshold(ui.output("comparison"), DEFAULT_TARGET_THRESHOLD)
paired = ui.at_threshold(ui.output("paired"), DEFAULT_TARGET_THRESHOLD).set_index("baseline")
research = ui.at_threshold(ui.output("research"), DEFAULT_TARGET_THRESHOLD).set_index("group")
protocol = evaluation["protocol"]
selected = evaluation["selected_model"]
block = evaluation["thresholds"][f"{DEFAULT_TARGET_THRESHOLD:.2f}"]
balance = block["class_balance"]
model_ba = block["selected_model_cv"]["balanced_accuracy"]
cgpa_ba = block["cgpa_cutoff_baseline_cv"]["balanced_accuracy"]
vs_cgpa, vs_majority = paired.loc[CGPA_CUTOFF], paired.loc[MAJORITY]
has, no = research.loc["Has research"], research.loc["No research"]
n_rows = evaluation["data"]["n_rows"]
t = f"{DEFAULT_TARGET_THRESHOLD:.2f}"

# ------------------------------------------------------------------ hero
style.hero("A machine-learning project on 500 graduate applicant profiles", "")
st.title("Where does a graduate application profile stand?")
st.html(
    '<p class="lede">GradMit takes seven parts of an applicant&rsquo;s profile (GRE, TOEFL, university '
    "rating, statement of purpose, recommendations, CGPA and research experience) and estimates whether "
    f"the dataset would rate that profile&rsquo;s chance of admission at {t} or higher. Behind the "
    "predictor sits a reproducible study of how much a model really adds over simple rules.</p>"
)
with st.container(horizontal=True, gap="small"):
    if st.button("Try the profile predictor", type="primary", icon=":material/person_search:"):
        st.switch_page("app_pages/predictor.py")
    if st.button("Read the problem statement", icon=":material/assignment:"):
        st.switch_page("app_pages/problem.py")

style.tiles([
    ("Applicant profiles", f"{n_rows}", f"{len(evaluation['data']['features'])} features, no missing values", style.SKY),
    (f"{selected}", f"{model_ba['mean']:.3f}", "balanced accuracy, mean of 50 folds", style.MINT),
    ("One CGPA cutoff", f"{cgpa_ba['mean']:.3f}", "balanced accuracy, same 50 folds", style.GOLD),
    ("Validation folds", f"{protocol['n_folds']}", f"{protocol['n_splits']}-fold CV, repeated {protocol['n_repeats']} times", style.LILAC),
    ("Research experience", f"{ui.pct(has['positive_rate'])} vs {ui.pct(no['positive_rate'])}",
     f"rated {t}+ with vs without research", style.CORAL),
])

# ------------------------------------------------------------------ what is inside
style.eyebrow("What's inside", "four parts you can use, not just read")
cols = st.columns(4, gap="medium")
inside = [
    ("Profile predictor", "Enter a profile and get an estimate straight away, with a warning when an input "
     "falls outside the range the model was trained on.", style.MINT, "app_pages/predictor.py", ":material/person_search:"),
    ("Problem statement", "The question this project answers, why the first version had to be reframed, and how "
     "an answer is judged.", style.SKY, "app_pages/problem.py", ":material/assignment:"),
    ("Data explorer", "Filter the 500 profiles, compare feature distributions and see how research experience "
     "relates to a high estimated chance.", style.LILAC, "app_pages/explorer.py", ":material/table_chart:"),
    ("Evaluation", "Five model families, two baselines and three thresholds, each scored on the same 50 "
     "validation folds.", style.GOLD, "app_pages/evaluation.py", ":material/monitoring:"),
]
for col, (title, body, colour, target, icon) in zip(cols, inside):
    with col:
        style.panel(title, body, colour)
        st.page_link(target, label=f"Open {title.lower()}", icon=icon)

# ------------------------------------------------------------------ headline
style.eyebrow("The headline finding", "the honest answer, read from the committed outputs")
st.markdown(
    "The dataset holds no admission decisions, only an estimated `Chance of Admit`, so GradMit "
    "predicts a **proxy label**: whether that estimate is at or above a threshold. It makes "
    "no admission decisions and no causal claims."
)
lead, side = st.columns([3, 2], gap="large")
with lead:
    ui.chart(
        charts.interval_chart(comparison, "balanced_accuracy", selected),
        alt="Mean balanced accuracy and 95% confidence interval for every model and baseline",
        title=f"Balanced accuracy at target threshold {t}",
        sub="mean and 95% interval over 50 validation folds",
        what=(f"Each dot is a model's mean balanced accuracy over the same {protocol['n_folds']} validation "
              "folds; the bar is its 95% interval with the Nadeau-Bengio correction. Mint is the selected "
              "model, gold the two baselines."),
        why=("The interesting comparison is not model against model but model against the CGPA cutoff. "
             "Their intervals overlap almost completely."),
    )
with side:
    if vs_cgpa["significant"]:
        line = (f"<b>{selected} beats a single CGPA cutoff</b> by {vs_cgpa['difference']:+.3f} balanced accuracy "
                f"(95% CI {ui.interval(vs_cgpa['ci_low'], vs_cgpa['ci_high'], signed=True)}, {ui.holm(vs_cgpa['p_value_holm'])}).")
    else:
        line = (f"<b>{selected} is not distinguishable from a single CGPA cutoff.</b> The paired difference is "
                f"{vs_cgpa['difference']:+.3f} balanced accuracy (95% CI "
                f"{ui.interval(vs_cgpa['ci_low'], vs_cgpa['ci_high'], signed=True)}, {ui.holm(vs_cgpa['p_value_holm'])}).")
    all_paired = ui.output("paired")
    wins = all_paired[(all_paired["baseline"] == CGPA_CUTOFF) & all_paired["significant"].astype(bool)]
    win_at = ", ".join(f"{x:.2f}" for x in wins["target_threshold"])
    beyond_cgpa = (
        f"Across the three thresholds, the model beats the CGPA cutoff only at {win_at}." if win_at
        else "At none of the three thresholds does the model beat the CGPA cutoff."
    )
    style.verdict(
        f"{line}<br><br>Against always guessing the majority class, the gain is clear: "
        f"{vs_majority['difference']:+.3f} (95% CI "
        f"{ui.interval(vs_majority['ci_low'], vs_majority['ci_high'], signed=True)}, {ui.holm(vs_majority['p_value_holm'])})."
        f"<br><br>{beyond_cgpa}"
    )
    st.space("small")
    st.page_link("app_pages/baselines.py", label="See the baselines", icon=":material/straighten:")
    st.page_link("app_pages/sensitivity.py", label="Compare thresholds 0.70, 0.75 and 0.80", icon=":material/tune:")

# ------------------------------------------------------------------ how it was built
style.eyebrow("How it was built", "from one csv to a working predictor")
style.method(
    steps=[
        ("Load and audit the data",
         f"{n_rows} records and {len(evaluation['data']['features'])} features, checked for missing values, "
         "duplicates and out-of-range scores. The file's SHA-256 is recorded so every result can be traced to it.",
         "data/Admission_Predict.csv"),
        ("Define the proxy label",
         f"<code>High_Admission_Chance</code> is 1 when <code>Chance of Admit</code> is at or above {t}. The "
         "study is repeated at 0.70 and 0.80 to see whether the conclusions hold.",
         f"{balance['n_positive']} of {balance['n']} positive at {t}"),
        ("Compare five model families",
         "Logistic regression, random forest, gradient boosting, SVM and k-nearest neighbours, tuned inside each "
         "training fold and scored on the held-out fold.",
         f"{protocol['n_folds']} validation folds"),
        ("Test against simple rules",
         "A majority-class rule and a CGPA cutoff are scored on the same folds. Paired bootstrap and permutation "
         "tests, with a Holm correction, decide whether a gap is real.",
         f"{protocol['n_resamples']:,} resamples"),
        ("Train once and ship",
         f"The selected model is refitted on all {meta['n_training_rows']} records and saved. This app only loads "
         "that file; it never trains at start-up.",
         "artifacts/gradmit_pipeline.joblib"),
    ],
    engine=f"scikit-learn pipeline · {selected}",
    ins=[("Admission_Predict.csv", DATA_PATH)],
    outs=[(p.relative_to(ROOT).as_posix(), p) for p in (
        ARTIFACTS_DIR / "gradmit_pipeline.joblib",
        ARTIFACTS_DIR / "model_metadata.json",
        OUTPUTS_DIR / OUTPUT_FILES["comparison"],
        OUTPUTS_DIR / OUTPUT_FILES["paired"],
        OUTPUTS_DIR / OUTPUT_FILES["sensitivity"],
        OUTPUTS_DIR / OUTPUT_FILES["evaluation"],
    )],
)

# ------------------------------------------------------------------ history
style.eyebrow("What changed from the first version", "a retired number, and why")
st.markdown(
    f"The first GradMit was a Flask page that reported **{evaluation['legacy']['retired_result']}** and "
    "called it admission prediction. That figure is retired: it came from one split, had no baseline to "
    "compare against, and read the proxy label as a real outcome. This version replaces it with repeated, "
    "nested evaluation, two baselines and paired tests, all reproducible from the scripts in the repository."
)
