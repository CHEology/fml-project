from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_app_entrypoint_stays_thin() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")

    assert len(app_source.splitlines()) < 150
    assert "st.navigation" in app_source
    assert "render_demo_page" in app_source
    assert "def inject_styles" not in app_source
    assert "def assess_quality" not in app_source


def test_extracted_app_modules_are_importable() -> None:
    from app.pages import demo, home, market
    from app.styles import inject_styles

    from app import app
    from ml import resume_assessment

    assert callable(app.main)
    assert callable(demo.render_demo_page)
    assert callable(home.render_home_page)
    assert callable(market.render_market_overview_page)
    assert callable(inject_styles)
    assert callable(resume_assessment.assess_resume_text)


def test_stylesheet_keeps_demo_navigation_and_input_method_rules() -> None:
    css = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "app" / "styles").glob("*.css"))
    )

    assert '[data-testid="stRadio"] [role="radiogroup"]' in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert ".st-key-demo-floating-nav" in css
    assert "position: fixed;" in css


def test_stylesheets_are_split_into_readable_files() -> None:
    style_files = sorted((PROJECT_ROOT / "app" / "styles").glob("*.css"))

    assert len(style_files) >= 4
    assert all(path.name != "base.css" for path in style_files)
    assert [
        path.name
        for path in style_files
        if len(path.read_text(encoding="utf-8").splitlines()) > 450
    ] == []


def test_stylesheets_do_not_include_python_or_style_delimiters() -> None:
    forbidden = ('"""', "<style>", "</style>")

    for path in sorted((PROJECT_ROOT / "app" / "styles").glob("*.css")):
        css = path.read_text(encoding="utf-8")
        assert [marker for marker in forbidden if marker in css] == []


def test_stylesheets_have_complete_rule_boundaries() -> None:
    for path in sorted((PROJECT_ROOT / "app" / "styles").glob("*.css")):
        depth = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("/*") or stripped.endswith("*/"):
                continue

            if (
                depth == 0
                and ":" in stripped
                and stripped.endswith(";")
                and not stripped.startswith("@")
            ):
                raise AssertionError(
                    f"{path.name}:{line_number} starts with a declaration outside a rule"
                )

            depth += stripped.count("{")
            depth -= stripped.count("}")
            if depth < 0:
                raise AssertionError(
                    f"{path.name}:{line_number} closes an unopened rule"
                )

        assert depth == 0, f"{path.name} has an unclosed CSS rule"


def test_demo_sample_resume_analysis_reaches_results() -> None:
    script = """
import numpy as np
import pandas as pd

from app.demo.state import initialize_session_state
from app.styles import inject_styles
import app.pages.demo as demo

initialize_session_state()
inject_styles("Lavender")

jobs = pd.DataFrame(
    [
        {
            "title": "Software Engineer",
            "company_name": "Example Co",
            "location": "New York, NY",
            "state": "NY",
            "text": "Python Kubernetes distributed systems observability",
            "salary_annual": 120000,
        }
    ]
)
status = [{"label": "retrieval", "path": "", "ready": True, "required_for": "retrieval"}]

demo.load_public_assessment_resource = lambda: None
demo.validate_resume = lambda public_models, text: {"is_resume": True, "reasons": []}
demo.public_resume_signals = lambda public_models, text: None
demo.load_retriever_resource = lambda: ("retriever", "encoder")
demo.encode_resume = lambda encoder, text: np.zeros(4, dtype=np.float32)
demo.retrieve_matches = lambda retriever, jobs, embedding, target_seniority, top_k: pd.DataFrame()
demo.apply_public_ats_fit = lambda public_models, resume_text, matches: matches
demo.salary_artifacts_ready = lambda project_root: False
demo.load_occupation_resource = lambda encoder: None
demo.load_wage_resource = lambda: None
demo.seniority_filtered_salary_matches = lambda matches: (pd.DataFrame(), None)
demo.hybrid_salary_band = lambda salary_matches, neural_band=None, bls_band=None, occupation_match=None: None
demo.add_salary_evidence_note = lambda band, note: band
demo.apply_quality_discount = lambda band, quality: band
demo.apply_capability_adjustment = lambda band, capability: band
demo.feedback_terms = lambda resume_text, matches, cluster: []

demo.render_demo_page(jobs, True, status)
"""
    at = AppTest.from_string(script, default_timeout=10).run()

    at.radio[0].set_value("Use a random sample resume").run()
    at.button[0].click().run()
    assert at.session_state["sample_resume_text"].endswith("\n")

    at.button(key="analyze_sample_resume").click().run()

    assert at.session_state["demo_stage"] == "results"
    assert not at.error
    assert not at.exception
