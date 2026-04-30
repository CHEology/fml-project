from __future__ import annotations

from typing import Any

import streamlit as st
from ml.resume_assessment import (
    add_salary_evidence_note,
    apply_capability_adjustment,
    apply_quality_discount,
    assess_capability_tier,
    assess_quality,
    detect_profile,
    enhance_structure_with_public_sections,
    extract_work_history,
    resume_structure,
    score_projects,
    seniority_filtered_salary_matches,
)

from app.components.methodology import render_methodology_page
from app.components.quality import render_public_model_card, render_quality_scorecard
from app.components.sidebar import render_app_sidebar
from app.config import PROJECT_ROOT
from app.demo.state import initialize_session_state
from app.pages.demo import render_demo_page
from app.pages.home import render_home_page
from app.pages.market import render_market_overview_page
from app.runtime import ml as runtime
from app.runtime.cache import (
    apply_public_ats_fit,
    artifact_status,
    artifacts_ready,
    cluster_position,
    encode_resume,
    feedback_terms,
    hybrid_salary_band,
    learned_quality_signal,
    load_cluster_resource,
    load_job_embedding_resource,
    load_jobs,
    load_occupation_resource,
    load_public_assessment_resource,
    load_quality_resource,
    load_retriever_resource,
    load_salary_resource,
    load_wage_resource,
    public_resume_signals,
    retrieve_matches,
    salary_artifacts_ready,
    salary_band_from_model,
    validate_resume,
)
from app.styles import inject_styles

__all__ = [
    "PROJECT_ROOT",
    "add_salary_evidence_note",
    "apply_capability_adjustment",
    "apply_public_ats_fit",
    "apply_quality_discount",
    "artifact_status",
    "artifacts_ready",
    "assess_capability_tier",
    "assess_quality",
    "cluster_position",
    "detect_profile",
    "encode_resume",
    "enhance_structure_with_public_sections",
    "extract_work_history",
    "feedback_terms",
    "hybrid_salary_band",
    "inject_styles",
    "initialize_session_state",
    "learned_quality_signal",
    "load_cluster_resource",
    "load_job_embedding_resource",
    "load_jobs",
    "load_occupation_resource",
    "load_public_assessment_resource",
    "load_quality_resource",
    "load_retriever_resource",
    "load_salary_resource",
    "load_wage_resource",
    "main",
    "public_resume_signals",
    "render_demo_page",
    "render_public_model_card",
    "render_quality_scorecard",
    "resume_structure",
    "retrieve_matches",
    "runtime",
    "salary_artifacts_ready",
    "salary_band_from_model",
    "score_projects",
    "seniority_filtered_salary_matches",
    "validate_resume",
]

st.set_page_config(
    page_title="ResuMatch",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    initialize_session_state()
    inject_styles("Lavender")
    jobs, data_source, has_real_data = load_jobs()
    status = artifact_status()

    pages: dict[str, Any] = {}
    pages["home"] = st.Page(
        lambda: render_home_page(pages, jobs, data_source, has_real_data, status),
        title="Home",
        url_path="home",
        default=True,
    )
    pages["demo"] = st.Page(
        lambda: render_demo_page(jobs, has_real_data, status),
        title="Demo",
        url_path="demo",
    )
    pages["market"] = st.Page(
        lambda: render_market_overview_page(jobs, data_source, has_real_data, status),
        title="Market Overview",
        url_path="market-overview",
    )
    pages["methodology"] = st.Page(
        render_methodology_page,
        title="Methodology",
        url_path="methodology",
    )

    page = st.navigation(
        [pages["home"], pages["demo"], pages["market"], pages["methodology"]],
        position="hidden",
    )
    render_app_sidebar(jobs, data_source, has_real_data, status, pages)
    page.run()


if __name__ == "__main__":
    main()
