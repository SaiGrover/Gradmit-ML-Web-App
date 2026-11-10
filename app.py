"""GradMit Streamlit application.

    streamlit run app.py

The app only reads committed files in ``outputs/`` and ``artifacts/``. Models are
evaluated and trained by ``scripts/evaluate_models.py`` and
``scripts/train_final_model.py``, never at app start-up.
"""

import streamlit as st

st.set_page_config(
    page_title="GradMit",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGES = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("app_pages/predictor.py", title="Profile predictor", icon=":material/person_search:"),
    st.Page("app_pages/evaluation.py", title="Model evaluation", icon=":material/monitoring:"),
    st.Page("app_pages/baselines.py", title="Baselines", icon=":material/straighten:"),
    st.Page("app_pages/sensitivity.py", title="Threshold sensitivity", icon=":material/tune:"),
    st.Page("app_pages/explorer.py", title="Data explorer", icon=":material/table_chart:"),
    st.Page("app_pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
]

page = st.navigation(PAGES, position="top")
page.run()

st.divider()
st.caption(
    "GradMit estimates a proxy label derived from the dataset's own Chance of Admit score. "
    "It does not predict real admission decisions. Built by Saanvi Grover, JIIT Noida."
)
