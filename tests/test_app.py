import re

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["home", "problem", "predictor", "evaluation", "baselines", "sensitivity", "explorer", "methodology"]
FORBIDDEN = [
    r"will be admitted",
    r"admission outcome",
    r"actual admission",
    r"research causes",
    r"accuracy on unseen data",
    r"use_container_width",
]


def submit(at: AppTest) -> None:
    next(b for b in at.button if b.label == "Estimate probability").click().run()


def run_page(root, page: str) -> AppTest:
    at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    if page != "home":
        at.switch_page(f"app_pages/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(root, page):
    at = run_page(root, page)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_problem_statement_reads_its_answers_from_outputs(root):
    at = run_page(root, "problem")
    assert at.title[0].value == "Problem statement"
    text = " ".join(m.value for m in at.markdown)
    assert "Chance of Admit" in text and "CGPA cutoff" in text
    # Short answers are read from committed outputs, not typed in.
    assert "beats the CGPA cutoff only at 0.70" in text.replace("It beats", "beats")
    assert "65.7% with research (n=280)" in text and "15.5% without (n=220)" in text


def test_home_states_the_proxy_framing(root):
    at = run_page(root, "home")
    text = " ".join(m.value for m in at.markdown)
    assert "proxy label" in text
    assert "no admission decisions" in text


def test_predictor_returns_a_probability_and_both_thresholds(root):
    at = run_page(root, "predictor")
    assert re.fullmatch(r"\d{1,3}\.\d%", at.metric[0].value)
    assert "at or above 0.75" in at.metric[0].label
    text = " ".join(m.value for m in at.markdown)
    assert "not a real admission decision" in text
    assert "classifier threshold of 0.50" in text
    assert not at.warning  # default profile is inside the training range


def test_predictor_warns_outside_training_range(root):
    at = run_page(root, "predictor")
    at.number_input[0].set_value(270)  # GRE below the training minimum of 290
    submit(at)
    assert not at.exception
    assert len(at.warning) == 1
    assert "GRE score 270" in at.warning[0].value


def test_predictor_probability_responds_to_inputs(root):
    at = run_page(root, "predictor")
    before = float(at.metric[0].value.rstrip("%"))
    at.number_input[2].set_value(9.6)  # CGPA
    submit(at)
    after = float(at.metric[0].value.rstrip("%"))
    assert after > before


def test_threshold_switch_reads_committed_results(root, outputs):
    at = run_page(root, "sensitivity")
    at.button_group[0].select(0.80).run()
    assert not at.exception
    assert at.metric[0].value.startswith("155")


def test_no_misleading_language_in_app_sources(root):
    sources = [root / "app.py", *sorted((root / "app_pages").glob("*.py"))]
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        for pattern in FORBIDDEN:
            assert not re.search(pattern, text), f"{pattern!r} found in {path.name}"


def test_app_does_not_train_models(root):
    for path in [root / "app.py", root / "gradmit" / "ui.py", *sorted((root / "app_pages").glob("*.py"))]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\.fit\(|evaluate_threshold\(|train_final\(|GridSearchCV", text), path.name


def test_pipeline_rail_navigates_between_pages(root):
    at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    rail = [b for b in at.button if b.key and b.key.startswith("nav")]
    assert len(rail) == 8
    at.button(key="nav6").click().run()
    assert not at.exception
    assert at.title[0].value == "Profile predictor"
