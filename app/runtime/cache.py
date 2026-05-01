from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.config import DATA_PATH, PROJECT_ROOT
from app.demo.sample_data import SYNTHETIC_JOBS
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


def _data_path_signature() -> tuple[bool, int, int]:
    if not DATA_PATH.exists():
        return (False, 0, 0)
    stat = DATA_PATH.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


@st.cache_data(show_spinner=False)
def _load_jobs_cached(
    data_path_signature: tuple[bool, int, int],
) -> tuple[pd.DataFrame, str, bool]:
    data_exists = bool(data_path_signature[0])
    if data_exists and DATA_PATH.exists():
        return (
            load_real_jobs(PROJECT_ROOT),
            str(DATA_PATH.relative_to(PROJECT_ROOT)),
            True,
        )

    return pd.DataFrame(SYNTHETIC_JOBS), "Sample role catalog", False


def load_jobs() -> tuple[pd.DataFrame, str, bool]:
    return _load_jobs_cached(_data_path_signature())


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
