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


def test_sidebar_artifact_summary_surfaces_pipeline_state() -> None:
    from app.components.sidebar import artifact_readiness_summary

    ready_status = [
        {
            "label": "Quality model",
            "path": "models/quality_model.pt",
            "ready": True,
            "required_for": "quality",
            "modified_label": "Created May 01, 2026 12:00 PM",
            "setup_command": "uv run python scripts/train_quality_model.py",
            "important": True,
        }
    ]

    summary, details = artifact_readiness_summary(ready_status)

    assert summary == "Full pipeline established"
    assert any("Quality model" in detail for detail in details)
    assert any("Created May 01, 2026 12:00 PM" in detail for detail in details)

    missing_status = [
        {
            "label": "Public assessment metrics",
            "path": "models/public_assessment_metrics.json",
            "ready": False,
            "required_for": "public_assessment",
            "modified_label": "Not created",
            "setup_command": "uv run python scripts/train_public_assessment_models.py",
            "important": True,
        }
    ]

    summary, details = artifact_readiness_summary(missing_status)

    assert summary == "Pipeline needs setup"
    assert any("Public Assessment: 0/1 ready" in detail for detail in details)
    assert any(
        "uv run python scripts/train_public_assessment_models.py" in detail
        for detail in details
    )


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


def test_job_rows_show_profile_signals(monkeypatch) -> None:
    import pandas as pd
    import streamlit as st
    from app.components.job_results import render_job_row

    rendered: list[str] = []

    def capture_markdown(body: str, *args, **kwargs) -> None:
        rendered.append(body)

    monkeypatch.setattr(st, "markdown", capture_markdown)

    render_job_row(
        pd.Series(
            {
                "title": "Data Scientist",
                "company_name": "Example Co",
                "location": "New York, NY",
                "work_type": "FULL_TIME",
                "experience_level": "Mid",
                "salary_annual": 130000,
                "text": "Build Python dashboards and SQL models for analytics teams.",
                "similarity": 0.42,
            }
        ),
        profile_terms=["Python", "SQL", "Salesforce"],
    )

    html = "\n".join(rendered)
    assert "Profile signals" in html
    assert "Python" in html
    assert "SQL" in html
    assert "Salesforce" not in html


def test_live_job_rows_show_provider_link(monkeypatch) -> None:
    import pandas as pd
    import streamlit as st
    from app.components.job_results import render_live_job_results

    rendered: list[str] = []

    def capture_markdown(body: str, *args, **kwargs) -> None:
        rendered.append(body)

    monkeypatch.setattr(st, "markdown", capture_markdown)

    render_live_job_results(
        pd.DataFrame(
            [
                {
                    "title": "Machine Learning Engineer",
                    "company_name": "Example AI",
                    "location": "New York, NY",
                    "posting_date": "2026-05-01",
                    "live_match_score": 91.2,
                    "job_link": "https://remotive.com/remote-jobs/ml-engineer-1",
                    "source": "Remotive",
                }
            ]
        )
    )

    html = "\n".join(rendered)
    assert "Live results are fetched from public job feeds" in html
    assert "Machine Learning Engineer" in html
    assert "91% live fit" in html
    assert "Current listing from Remotive" in html
    assert "View posting" in html
    assert "https://remotive.com/remote-jobs/ml-engineer-1" in html


def test_live_job_rows_hide_invalid_links(monkeypatch) -> None:
    import pandas as pd
    import streamlit as st
    from app.components.job_results import render_live_job_results

    rendered: list[str] = []

    def capture_markdown(body: str, *args, **kwargs) -> None:
        rendered.append(body)

    monkeypatch.setattr(st, "markdown", capture_markdown)

    render_live_job_results(
        pd.DataFrame(
            [
                {
                    "title": "Machine Learning Engineer",
                    "company_name": "Example AI",
                    "location": "New York, NY",
                    "posting_date": "2026-05-01",
                    "live_match_score": 91.2,
                    "job_link": "javascript:alert(1)",
                }
            ]
        )
    )

    assert "View posting" not in "\n".join(rendered)


def test_live_job_status_explains_provider_lookup_issue(monkeypatch) -> None:
    import streamlit as st
    from app.components.job_results import render_live_job_status

    rendered: list[str] = []

    def capture_markdown(body: str, *args, **kwargs) -> None:
        rendered.append(body)

    monkeypatch.setattr(st, "markdown", capture_markdown)

    render_live_job_status(
        {
            "query": "Machine Learning Engineer Python",
            "reason": "No keyless live job providers returned usable rows.",
        }
    )

    html = "\n".join(rendered)
    assert "Live job status" in html
    assert "keyless live job providers" in html
    assert "Machine Learning Engineer Python" in html


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
demo.validate_resume = lambda public_models, text: {
    "is_resume": True,
    "confidence": "high",
    "score": 1.0,
    "reasons": [],
    "signals": [],
}
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
demo.serpdog_api_key = lambda: ""
demo.fetch_live_jobs_resource = lambda query, exp_level, api_key: pd.DataFrame()

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


def test_demo_pasted_resume_analysis_runs_on_first_click() -> None:
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
demo.validate_resume = lambda public_models, text: {
    "is_resume": True,
    "confidence": "high",
    "score": 1.0,
    "reasons": [],
    "signals": [],
}
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
demo.serpdog_api_key = lambda: ""
demo.fetch_live_jobs_resource = lambda query, exp_level, api_key: pd.DataFrame()

demo.render_demo_page(jobs, True, status)
"""
    at = AppTest.from_string(script, default_timeout=10).run()

    at.radio[0].set_value("Paste resume / CV text").run()
    at.text_area[0].set_value(
        "Jane Doe\\nExperience\\n2022 - Present Software Engineer at Example Co\\n"
        "Education\\nBS Computer Science\\nSkills: Python, SQL"
    ).run()
    at.button(key="analyze_pasted_resume").click().run()

    assert at.session_state["demo_stage"] == "results"
    assert at.session_state["resume_text"].startswith("Jane Doe")
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
        resume_text=(
            "Summary\nPython analyst with SQL experience.\n"
            "Experience\nResponsible for client reporting and various trading projects.\n"
            "</ div>"
        ),
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
    assert html.index("Public-data model checks") < html.index("Resume organization")
    assert html.index("Resume organization") < html.index("What stood out positively")
    assert "Submitted resume/profile evidence" in html
    assert "quality-highlight-good" in html
    assert "quality-highlight-risk" in html
    assert "Python" in html
    assert "Responsible for" in html
    assert "</ div>" not in html
    assert "&lt;/ div&gt;" not in html
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


def test_demo_results_can_advance_to_actions_stage() -> None:
    demo_source = (PROJECT_ROOT / "app" / "pages" / "demo.py").read_text(
        encoding="utf-8"
    )
    nav_source = (PROJECT_ROOT / "app" / "demo" / "components.py").read_text(
        encoding="utf-8"
    )

    assert 'valid_stages = {"input", "results", "actions"}' in demo_source
    assert 'next_label="Choose actions"' in demo_source
    assert 'next_stage="actions"' in demo_source
    assert "st.session_state.demo_scroll_to_top = True" in nav_source
    assert 'if st.session_state.demo_stage == "actions":' in demo_source
    assert (
        "render_scroll_to_top()"
        in demo_source.split('if st.session_state.demo_stage == "actions":')[1]
    )
    assert "open_profile_actions_bottom" not in demo_source


def test_demo_actions_stage_renders_action_options_and_cluster_selector() -> None:
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
            "salary_annual": 120000,
            "text": "python apis",
        },
        {
            "title": "Principal Software Engineer",
            "salary_annual": 220000,
            "text": "python kubernetes platform ownership ms degree 8+ years of software engineering experience",
        },
        {
            "title": "Business Analyst",
            "salary_annual": 130000,
            "text": "sql dashboards",
        },
    ]
)
st = __import__("streamlit")
st.session_state.demo_stage = "actions"
st.session_state.resume_text = "Python engineer"
st.session_state.assessment = {
    "resume_text": "Python engineer",
    "cluster": {
        "cluster_id": 0,
        "label": "Software / Engineering",
        "top_terms": ["python"],
        "next_best_cluster_id": 1,
    },
    "cluster_assignments": np.array([0, 0, 1]),
    "cluster_labels": {
        "0": {"label": "Software / Engineering", "top_terms": ["python"]},
        "1": {"label": "Business / Data Analysis", "top_terms": ["sql"]},
    },
}

demo.render_demo_page(jobs, True, [])
"""
    at = AppTest.from_string(script, default_timeout=10).run()

    assert not at.error
    assert not at.exception
    action_radios = [
        radio for radio in at.radio if radio.label == "Action to improve your profile"
    ]
    assert len(action_radios) == 1
    assert action_radios[0].options == [
        "Improve my salary",
        "Move to a different cluster",
    ]
    assert action_radios[0].value == "Improve my salary"
    assert len(at.selectbox) == 0
    assert any(
        "Improve salary within Software / Engineering" in markdown.value
        for markdown in at.markdown
    )
    assert not any(
        "Future-state resume/profile draft" in markdown.value
        for markdown in at.markdown
    )
    assert any("Recommended actions" in markdown.value for markdown in at.markdown)

    action_radios[0].set_value("Move to a different cluster").run()

    assert len(at.selectbox) == 1
    assert at.selectbox[0].options == ["Business / Data Analysis (Cluster 1)"]
    assert "Software / Engineering (Cluster 0)" not in at.selectbox[0].options
