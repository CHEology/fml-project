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
    from app.components import quality
    from app.pages import demo, home, market
    from app.styles import inject_styles

    from app import app
    from ml import resume_assessment

    assert callable(app.main)
    assert callable(demo.render_demo_page)
    assert callable(demo.render_scroll_to_top)
    assert callable(demo.seniority_ladder_html)
    assert callable(demo.focus_evidence_html)
    assert callable(demo.seniority_evidence_html)
    assert callable(quality.render_profile_quality_section)
    assert callable(home.render_home_page)
    assert callable(market.render_market_overview_page)
    assert callable(inject_styles)
    assert callable(resume_assessment.assess_resume_text)


def test_demo_seniority_ladder_uses_shared_runtime_levels() -> None:
    import app.pages.demo as demo
    from app.runtime.ml import SENIORITY_RANKS

    ladder = demo.seniority_ladder_html("Associate")

    for level in SENIORITY_RANKS:
        assert level in ladder
    assert ladder.index("Lead / Executive") < ladder.index("Intern / Entry")
    assert 'class="seniority-step current"' in ladder
    assert "Current" not in ladder


def test_demo_snapshot_evidence_helpers_render_specific_reasons() -> None:
    import app.pages.demo as demo

    focus_html = demo.focus_evidence_html(
        {
            "skills_present": ["Python", "Kubernetes", "APIs"],
            "skills_missing": ["Leadership"],
            "confidence": 82,
        }
    )
    seniority_html = demo.seniority_evidence_html(
        "Associate",
        {
            "weighted_ft_months": 24,
            "role_count": 3,
            "max_seniority_keyword": "Associate",
        },
    )
    capability_html = demo.capability_evidence_html(
        {
            "summary": "thin within-level evidence",
            "skill_hits": ["Python", "Kubernetes"],
            "notes": ["Impact evidence is light relative to similar-level candidates."],
        },
        {
            "impact_score": 31,
            "specificity_score": 72,
        },
    )

    assert "Matched skills" in focus_html
    assert "Python, Kubernetes, APIs" in focus_html
    assert "Less visible" in focus_html
    assert "Highest title signal" in seniority_html
    assert "24 weighted months" in seniority_html
    assert "Tier readout" in capability_html
    assert "thin within-level evidence" in capability_html
    assert "Impact signal" in capability_html
    assert "Why this tier" in capability_html


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
    assert at.session_state["demo_scroll_to_top"] is False
    assert not at.error
    assert not at.exception


def test_profile_quality_section_renders_consolidated_content(monkeypatch) -> None:
    import streamlit as st
    from app.components.quality import render_profile_quality_section

    rendered: list[str] = []

    class DummyExpander:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

    def capture_markdown(body: str, *args, **kwargs) -> None:
        rendered.append(body)

    def capture_expander(label: str, *args, **kwargs) -> DummyExpander:
        rendered.append(label)
        return DummyExpander()

    monkeypatch.setattr(st, "markdown", capture_markdown)
    monkeypatch.setattr(st, "expander", capture_expander)

    render_profile_quality_section(
        quality={
            "overall": 39,
            "band_label": "Weak",
            "experience_score": 22,
            "impact_score": 0,
            "specificity_score": 100,
            "structure_score": 63,
            "strengths": ["History includes recognized, selective employers or labs."],
            "red_flags": ["Resume organization could be clearer; missing Projects."],
        },
        learned_quality={"score": 37, "label": "weak"},
        public_signals={
            "ready": True,
            "domain": {"label": "Information-Technology", "confidence": 0.09},
            "sections": {"counts": {"Exp": 44, "Edu": 2, "Sum": 2}},
            "entities": {
                "counts": {"College Name": 1, "Degree": 1, "Companies worked at": 7}
            },
        },
        resume_stats={
            "word_count": 582,
            "bullet_count": 0,
            "link_count": 0,
            "found_sections_count": 4,
            "total_sections_count": 5,
        },
        strengths=["Python", "SQL"],
        sections=["Experience", "Education", "Skills", "Summary"],
        missing_sections=["Projects"],
        missing_terms=["client", "conversion"],
    )

    html = "\n".join(rendered)

    assert "Profile Quality" in html
    assert "Read more about Profile Quality" in html
    assert "582 words" in html
    assert "Learned MLP cross-check" in html
    assert "Public-data model checks" in html
    assert "Evidence tags" in html
    assert "Resume organization" in html
    assert "Market-language gaps" in html
    assert "Add stronger evidence for client" in html
    assert '<div class="snapshot-label">Detected strengths</div>' not in html
    assert '<div class="snapshot-label">Gaps to close</div>' not in html
    assert "Evidence used in this snapshot" not in html
    assert "Uploaded file:" not in html


def test_demo_results_composition_removes_obsolete_profile_quality_sections() -> None:
    demo_source = (PROJECT_ROOT / "app" / "pages" / "demo.py").read_text(
        encoding="utf-8"
    )
    results_source = demo_source.split('if st.session_state.demo_stage == "results":')[
        1
    ]

    assert "render_profile_quality_section" in demo_source
    assert "Evidence used in this snapshot" not in results_source
    assert "Uploaded file:" not in results_source
    assert '"Gaps to close",' not in results_source
