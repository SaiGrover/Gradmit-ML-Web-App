"""GradMit Streamlit application.

    streamlit run app.py

The app only reads committed files in ``outputs/`` and ``artifacts/``. Models are
evaluated and trained by ``scripts/evaluate_models.py`` and
``scripts/train_final_model.py``, never at app start-up.
"""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="GradMit",
    page_icon=str(ROOT / "assets" / "gradmit-icon.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.logo(str(ROOT / "assets" / "gradmit-logo.svg"), size="large")

PAGES = {
    "": [
        st.Page("app_pages/problem.py", title="Problem statement", icon=":material/assignment:", default=True),
        st.Page("app_pages/overview.py", title="Overview", icon=":material/home:"),
    ],
    "Results": [
        st.Page("app_pages/evaluation.py", title="Model evaluation", icon=":material/monitoring:"),
        st.Page("app_pages/baselines.py", title="Baselines", icon=":material/straighten:"),
        st.Page("app_pages/sensitivity.py", title="Threshold sensitivity", icon=":material/tune:"),
    ],
    "Explore": [
        st.Page("app_pages/predictor.py", title="Profile predictor", icon=":material/person_search:"),
        st.Page("app_pages/explorer.py", title="Data explorer", icon=":material/table_chart:"),
    ],
    "About": [
        st.Page("app_pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
    ],
}

page = st.navigation(PAGES, position="top")
page.run()

st.space("large")
with st.container(border=True):
    st.caption(
        "GradMit estimates a proxy label derived from the dataset's own Chance of Admit score. "
        "It does not predict real admission decisions. Built by Saanvi Grover, JIIT Noida."
    )
    with st.container(horizontal=True, gap="medium"):
        st.page_link("app_pages/methodology.py", label="Methodology", icon=":material/menu_book:")
        st.page_link(
            "https://github.com/SaiGrover/Gradmit-ML-Web-App",
            label="Source on GitHub",
            icon=":material/code:",
        )
