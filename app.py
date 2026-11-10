"""GradMit Streamlit application.

    streamlit run app.py

The app only reads committed files in ``outputs/`` and ``artifacts/``. Models are
evaluated and trained by ``scripts/evaluate_models.py`` and
``scripts/train_final_model.py``, never at app start-up.
"""

from pathlib import Path

import streamlit as st

from gradmit import style, ui
from gradmit.data import DEFAULT_TARGET_THRESHOLD

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="GradMit",
    page_icon=str(ROOT / "assets" / "gradmit-icon.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
style.inject_css()

evaluation = ui.evaluation()
block = evaluation["thresholds"][f"{DEFAULT_TARGET_THRESHOLD:.2f}"]
model_ba = block["selected_model_cv"]["balanced_accuracy"]["mean"]
cgpa_ba = block["cgpa_cutoff_baseline_cv"]["balanced_accuracy"]["mean"]
thresholds = " / ".join(f"{t:.2f}" for t in evaluation["target"]["target_thresholds"])

# The project pipeline doubles as the navigation: every node is a real page.
# (file, title, tag, short metric read from the committed outputs, colour)
NODES = [
    ("home", "Project", "Start here", f"{evaluation['data']['n_rows']} profiles", style.MINT),
    ("problem", "Problem statement", "01 · Question", "a proxy label", style.SKY),
    ("explorer", "Data explorer", "02 · Data", f"{len(evaluation['data']['features'])} features", style.SKY),
    ("evaluation", "Model evaluation", "03 · Models", f"{model_ba:.3f} balanced acc.", style.MINT),
    ("baselines", "Baselines", "04 · Baselines", f"CGPA rule {cgpa_ba:.3f}", style.GOLD),
    ("sensitivity", "Threshold sensitivity", "05 · Thresholds", thresholds, style.LILAC),
    ("predictor", "Profile predictor", "06 · Try it", "live estimate", style.MINT),
    ("methodology", "Methodology", "Notes", "limits, stated plainly", style.SLATE),
]
PAGES = [
    st.Page(f"app_pages/{name}.py", title=title, default=name == "home")
    for name, title, *_ in NODES
]
page = st.navigation(PAGES, position="hidden")
active = next(i for i, p in enumerate(PAGES) if p.title == page.title)

style.topbar()
st.html(style.rail_css([n[4] for n in NODES], active))
with st.container(key="navrail"):
    widths = []
    for i in range(len(NODES)):
        widths += [5, 0.55] if i < len(NODES) - 1 else [5]
    slots = st.columns(widths, gap="small", vertical_alignment="center")
    for i, (_name, title, tag, metric, _) in enumerate(NODES):
        with slots[i * 2]:
            if st.button(f"{tag}  \n**{title}**  \n*{metric}*", key=f"nav{i}", width="stretch"):
                st.switch_page(PAGES[i])
        if i < len(NODES) - 1:
            slots[i * 2 + 1].html('<div class="flowarrow">&#10230;</div>')

page.run()
style.footer()
