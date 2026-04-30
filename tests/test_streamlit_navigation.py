from __future__ import annotations

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _function_source(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    next_def = source.index("\ndef ", start + 1)
    return source[start:next_def]


def _assigned_string_list(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List):
            continue
        values: list[str] = []
        for item in node.value.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                raise AssertionError(f"{name} must be a literal string list")
            values.append(item.value)
        return values
    raise AssertionError(f"{name} assignment not found")


def test_methodology_is_extracted_to_shared_renderer() -> None:
    methodology_component = PROJECT_ROOT / "app" / "components" / "methodology.py"

    assert methodology_component.exists()
    assert "render_methodology_page" in methodology_component.read_text(
        encoding="utf-8"
    )


def test_sidebar_navigation_exposes_named_pages() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")

    assert "st.navigation" in app_source
    assert 'position="hidden"' in app_source
    assert "st.page_link" in app_source
    assert 'title="Home"' in app_source
    assert 'title="Demo"' in app_source
    assert 'title="Market Overview"' in app_source
    assert 'title="Methodology"' in app_source
    assert "st.tabs(" not in app_source


def test_home_page_has_team_and_mascot_asset() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    team_source = (PROJECT_ROOT / "app" / "components" / "team.py").read_text(
        encoding="utf-8"
    )
    mascot = PROJECT_ROOT / "app" / "assets" / "resumatch-mascot.png"

    assert "render_home_page" in app_source
    assert "Lucky Hamsters" in team_source
    for contributor in (
        "Omer Hortig",
        "Tanvi Patel",
        "Ryan Lu",
        "Alan He",
        "Eliguli Han",
    ):
        assert contributor in team_source
    assert mascot.exists()


def test_demo_page_uses_single_stage_wizard() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")
    nav_source = _function_source(app_source, "render_demo_floating_nav")

    assert 'st.session_state.demo_stage == "input"' in demo_source
    assert 'st.session_state.demo_stage == "results"' in demo_source
    assert 'st.session_state.demo_stage = "results"' in demo_source
    assert 'st.session_state.demo_stage == "snapshot"' not in demo_source
    assert 'st.session_state.demo_stage == "market"' not in demo_source
    assert 'st.session_state.demo_stage == "gaps"' not in demo_source
    assert 'next_stage="market"' not in demo_source
    assert 'next_stage="gaps"' not in demo_source
    assert "st.session_state.demo_stage = next_stage" in nav_source
    assert "Start over with new resume/profile" in nav_source
    assert "← Previous" in nav_source
    assert "{next_label} →" in nav_source


def test_demo_stage_navigation_is_floating() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    nav_source = _function_source(app_source, "render_demo_floating_nav")

    assert 'key="demo-floating-nav"' in nav_source
    assert ".st-key-demo-floating-nav" in app_source
    assert "position: fixed;" in app_source
    assert "right: 1rem;" in app_source
    assert "transform: none;" in app_source
    assert "width: min(32rem, calc(100vw - 2rem)) !important;" in app_source
    assert "background: transparent;" in app_source
    assert "border: 0;" in app_source
    assert "box-shadow: none;" in app_source
    assert "background: #175CD3;" in app_source
    assert "display: grid !important;" in app_source
    assert (
        "grid-template-columns: minmax(0, 1fr) minmax(0, 1.35fr) minmax(0, 1fr);"
        in app_source
    )
    assert "max-width: 28rem;" in app_source
    assert "margin: 0 auto;" in app_source
    assert "text-overflow: ellipsis;" in app_source
    assert "render_demo_floating_nav(" in app_source


def test_demo_results_sections_have_explainable_headers() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    assert "render_demo_section_header" in app_source
    assert "info_dot" in app_source
    assert 'with st.expander(f"Read more about {title}")' in app_source
    assert 'with st.expander("Read more about Candidate Snapshot")' in demo_source
    assert "{info_dot(snapshot_info)}" not in demo_source
    section_header_source = _function_source(app_source, "render_demo_section_header")
    assert "info_dot(" not in section_header_source
    for title in (
        "Candidate Snapshot",
        "Market readout",
        "Market segment and match evidence",
        "Gaps to close",
        "Top matching roles",
    ):
        assert title in demo_source
    for provenance in (
        "ml/retrieval.py",
        "ml/salary_model.py",
        "ml/clustering.py",
        "app/ml_runtime.py::feedback_terms()",
        "scripts/build_index.py",
    ):
        assert provenance in demo_source


def test_demo_market_readout_renders_cluster_salary_distribution() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    assert "render_cluster_salary_distribution" in app_source
    assert '"cluster_assignments": cluster_assignments' in demo_source
    assert '"cluster_labels": cluster_labels' in demo_source
    assert '"job_embeddings": job_embeddings' in demo_source
    assert '"resume_embedding": resume_embedding' in demo_source
    assert 'assessment.get("cluster_assignments")' in demo_source
    assert 'assessment.get("cluster_labels")' in demo_source
    assert 'assessment.get("job_embeddings")' in demo_source
    assert 'assessment.get("resume_embedding")' in demo_source

    salary_band_index = demo_source.index("render_salary_band(band)")
    distribution_index = demo_source.index("render_cluster_salary_distribution(")
    segment_index = demo_source.index(
        'render_demo_section_header(\n            "Market segment and match evidence"'
    )
    assert salary_band_index < distribution_index < segment_index


def test_demo_market_readout_explains_cluster_salary_distribution() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    for phrase in (
        "sentence-transformer vectors",
        "K-Means groups job embeddings into 8 role-family clusters",
        "PCA-reduced embedding components",
        "TF-IDF top terms and common titles",
        "nearest centroid",
        "2D cluster map",
        "random visualization sample",
        "hybrid band q50",
    ):
        assert phrase in demo_source


def test_demo_removes_intro_hero_and_static_metrics() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")
    market_source = _function_source(app_source, "render_market_overview_page")

    assert "Understand role fit, salary range, and market position." not in demo_source
    assert "Resume market intelligence" not in demo_source
    assert "Jobs loaded" not in demo_source
    assert "Median salary" not in demo_source
    assert "Jobs loaded" in market_source
    assert "Median salary" in market_source


def test_demo_rechecks_resume_text_after_upload_before_enabling_analysis() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    upload_index = demo_source.index("extract_uploaded_text(uploader)")
    recompute_index = demo_source.index(
        "current_text = st.session_state.uploaded_resume_text.strip()", upload_index
    )
    upload_button_index = demo_source.index(
        'key="analyze_upload_resume"', recompute_index
    )

    assert upload_index < recompute_index < upload_button_index


def test_demo_input_stage_uses_accordion_methods() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")
    input_methods = _assigned_string_list(demo_source, "input_methods")

    input_stage_source = demo_source[: demo_source.index("if analyze_clicked:")]
    assert input_stage_source.count("st.expander(") == 2
    assert "st.radio(" in demo_source
    assert 'key="demo_input_method"' in demo_source
    assert input_methods == [
        "Upload a PDF or TXT resume",
        "Paste resume/profile text",
        "Import a public portfolio or resume page",
        "Use a random sample resume",
    ]
    assert "Choose an input method" in demo_source
    assert "Clear current input" not in demo_source
    assert "demo-readiness" not in demo_source


def test_demo_analysis_buttons_are_inside_each_input_method() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    for key in (
        "analyze_upload_resume",
        "analyze_pasted_resume",
        "analyze_imported_profile",
        "analyze_sample_resume",
    ):
        assert key in demo_source

    assert "uploaded_resume_text" in demo_source
    assert "pasted_resume_text" in demo_source
    assert "imported_profile_text" in demo_source
    assert "sample_resume_text" in demo_source


def test_demo_sample_resume_preview_is_shown_after_loading() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    source_note_index = demo_source.index("st.info(SAMPLE_RESUME_SOURCE_SUMMARY)")
    details_index = demo_source.index("Read more about sample resume generation")
    load_index = demo_source.index("Load random sample resume")
    preview_index = demo_source.index("Random sample resume text", load_index)
    value_index = demo_source.index("value=st.session_state.sample_resume_text")

    assert source_note_index < load_index
    assert source_note_index < details_index < load_index
    assert load_index < preview_index
    assert value_index > load_index
    assert "Load a random sample resume to preview it here." in demo_source
    assert "help=SAMPLE_RESUME_SOURCE_HELP" not in demo_source
    assert "st.markdown(SAMPLE_RESUME_SOURCE_HELP)" in demo_source
    assert "SAMPLE_RESUME_SOURCE_SUMMARY" in app_source
    assert "synthetic demo examples for trying the workflow" in app_source


def test_demo_sample_resume_load_syncs_analysis_input() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    load_index = demo_source.index("Load random sample resume")
    sample_text_index = demo_source.index(
        "st.session_state.sample_resume_text = sample_text", load_index
    )
    sample_source_index = demo_source.index(
        "st.session_state.sample_resume_source = sample_source", sample_text_index
    )
    resume_text_index = demo_source.index(
        "st.session_state.resume_text = sample_text", sample_source_index
    )
    resume_source_index = demo_source.index(
        "st.session_state.resume_source = sample_source", resume_text_index
    )
    rerun_index = demo_source.index("st.rerun()", resume_source_index)

    assert (
        load_index
        < sample_text_index
        < sample_source_index
        < resume_text_index
        < resume_source_index
        < rerun_index
    )


def test_demo_sample_resume_analysis_reaches_snapshot_after_trailing_newline() -> None:
    script = """
import numpy as np
import pandas as pd
from app import app

app.initialize_session_state()
app.inject_styles("Lavender")

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

app.load_public_assessment_resource = lambda: None
app.validate_resume = lambda public_models, text: {"is_resume": True, "reasons": []}
app.public_resume_signals = lambda public_models, text: None
app.load_retriever_resource = lambda: ("retriever", "encoder")
app.encode_resume = lambda encoder, text: np.zeros(4, dtype=np.float32)
app.retrieve_matches = lambda retriever, jobs, embedding, target_seniority, top_k: []
app.apply_public_ats_fit = lambda public_models, resume_text, matches: matches
app.salary_artifacts_ready = lambda project_root: False
app.load_occupation_resource = lambda encoder: None
app.load_wage_resource = lambda: None
app.seniority_filtered_salary_matches = lambda matches: ([], None)
app.hybrid_salary_band = lambda salary_matches, neural_band=None, bls_band=None, occupation_match=None: None
app.add_salary_evidence_note = lambda band, note: band
app.apply_quality_discount = lambda band, quality: band
app.apply_capability_adjustment = lambda band, capability: band
app.feedback_terms = lambda resume_text, matches, cluster: []

app.render_demo_page(jobs, True, status)
"""
    at = AppTest.from_string(script, default_timeout=10).run()

    at.radio[0].set_value("Use a random sample resume").run()
    at.button[0].click().run()
    assert at.session_state["sample_resume_text"].endswith("\n")

    at.button(key="analyze_sample_resume").click().run()

    assert at.session_state["demo_stage"] == "results"
    assert not at.error
    assert not at.exception


def test_demo_uploaded_resume_preview_is_shown_after_uploading() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    upload_index = demo_source.index("Upload resume file")
    preview_index = demo_source.index("Uploaded resume text", upload_index)
    value_index = demo_source.index("value=st.session_state.uploaded_resume_text")

    assert upload_index < preview_index
    assert value_index > upload_index
    assert (
        "Upload a PDF or TXT resume to preview the extracted text here." in demo_source
    )


def test_demo_imported_page_preview_is_shown_after_importing() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    import_index = demo_source.index("Import page")
    preview_index = demo_source.index("Imported page text", import_index)
    value_index = demo_source.index("value=st.session_state.imported_profile_text")

    assert import_index < preview_index
    assert value_index > import_index
    assert (
        "Import a public portfolio or resume page to preview the extracted text here."
        in demo_source
    )


def test_demo_input_method_selector_is_styled_as_cards() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")

    assert '[data-testid="stRadio"] [role="radiogroup"]' in app_source
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in app_source
    assert '[data-testid="stRadio"] label:has(input:checked)' in app_source


def test_demo_selected_method_labels_match_render_branches() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")
    input_methods = _assigned_string_list(demo_source, "input_methods")

    for label in input_methods[:-1]:
        assert f'selected_method == "{label}"' in demo_source
    assert f'selected_method == "{input_methods[-1]}"' not in demo_source
