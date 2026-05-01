from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.config import DATA_PATH, PROJECT_ROOT
from app.demo.sample_data import SYNTHETIC_JOBS
from app.runtime import live_jobs
from app.runtime import ml as runtime

runtime_artifact_status = runtime.artifact_status
artifacts_ready = runtime.artifacts_ready
cluster_position = runtime.cluster_position
encode_resume = runtime.encode_resume
feedback_terms = runtime.feedback_terms
hybrid_salary_band = runtime.hybrid_salary_band
load_cluster_artifacts = runtime.load_cluster_artifacts
load_job_embeddings = runtime.load_job_embeddings
load_real_jobs = runtime.load_jobs
load_occupation_router = runtime.load_occupation_router
load_public_assessment_artifacts = runtime.load_public_assessment_artifacts
load_quality_artifacts = runtime.load_quality_artifacts
load_retriever = runtime.load_retriever
load_salary_artifacts = runtime.load_salary_artifacts
load_wage_table = runtime.load_wage_table
learned_quality_signal = runtime.learned_quality_signal
public_resume_signals = runtime.public_resume_signals
retrieve_matches = runtime.retrieve_matches
salary_band_from_model = runtime.salary_band_from_model
salary_artifacts_ready = runtime.salary_artifacts_ready
apply_public_ats_fit = runtime.apply_public_ats_fit
validate_resume = runtime.validate_resume
build_live_job_query = live_jobs.build_live_job_query
exp_level_for_seniority = live_jobs.exp_level_for_seniority
linkedin_geo_id = live_jobs.linkedin_geo_id
rank_live_jobs = live_jobs.rank_live_jobs
serpdog_api_key = live_jobs.serpdog_api_key


@st.cache_data(show_spinner=False)
def load_jobs() -> tuple[pd.DataFrame, str, bool]:
    if DATA_PATH.exists():
        return (
            load_real_jobs(PROJECT_ROOT),
            str(DATA_PATH.relative_to(PROJECT_ROOT)),
            True,
        )

    return pd.DataFrame(SYNTHETIC_JOBS), "Sample role catalog", False


def artifact_status() -> list[dict[str, Any]]:
    return runtime_artifact_status(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_retriever_resource():
    return load_retriever(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_salary_resource():
    return load_salary_artifacts(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_quality_resource():
    return load_quality_artifacts(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_occupation_resource(_encoder):
    return load_occupation_router(PROJECT_ROOT, _encoder)


@st.cache_resource(show_spinner=False)
def load_wage_resource():
    return load_wage_table(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_cluster_resource():
    return load_cluster_artifacts(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_job_embedding_resource():
    return load_job_embeddings(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_public_assessment_resource():
    if not artifacts_ready(artifact_status(), "public_assessment"):
        return None
    return load_public_assessment_artifacts(PROJECT_ROOT)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_jobs_resource(
    query: str,
    exp_level: str | None,
    api_key: str,
) -> pd.DataFrame:
    return live_jobs.fetch_live_jobs(
        query,
        serpdog_key=api_key,
        geo_id=linkedin_geo_id(),
        exp_level=exp_level,
        sort_by="week",
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_linkedin_jobs_resource(
    query: str,
    geo_id: str,
    exp_level: str | None,
    sort_by: str,
    api_key: str,
) -> pd.DataFrame:
    return live_jobs.fetch_serpdog_linkedin_jobs(
        query,
        api_key=api_key,
        geo_id=geo_id,
        exp_level=exp_level,
        sort_by=sort_by,
    )
