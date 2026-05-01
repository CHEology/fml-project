from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.config import PROJECT_ROOT
from app.demo.samples import linkedin_dataset_note
from app.runtime.artifacts import pipeline_readiness


def format_count(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def format_file_size(path: Path | None) -> str:
    if path is None or not path.exists():
        return "N/A"
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb >= 10:
        return f"{size_mb:.0f} MB"
    return f"{size_mb:.1f} MB"


def format_modified_date(path: Path | None) -> str:
    if path is None or not path.exists():
        return "N/A"
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return modified.strftime("%b %d, %Y")


def project_data_path(data_source: str) -> Path | None:
    source_path = Path(data_source)
    if source_path.suffix:
        return source_path if source_path.is_absolute() else PROJECT_ROOT / source_path
    return None


def artifact_readiness_summary(status: list[dict[str, Any]]) -> tuple[str, list[str]]:
    summary = pipeline_readiness(status)
    details: list[str] = []
    for group in summary["groups"]:
        line = f"{group['label']}: {group['ready_count']}/{group['total_count']} ready"
        if not group["ready"] and group.get("setup_command"):
            line = f"{line} | Next: {group['setup_command']}"
        details.append(line)

    important = summary.get("important_artifacts", [])
    if important:
        details.append("Important artifact timestamps:")
        details.extend(
            f"{item['label']}: {item.get('modified_label', 'N/A')}"
            for item in important[:8]
        )

    if summary["fully_established"]:
        return "Full pipeline established", details
    return "Pipeline needs setup", details


def render_data_source_card(
    jobs: pd.DataFrame,
    data_source: str,
    has_real_data: bool,
    status: list[dict[str, Any]],
    *,
    extra_class: str = "",
    show_artifact_expander: bool = True,
) -> None:
    data_path = project_data_path(data_source)
    source_label = (
        data_source if data_path is None else str(data_path.relative_to(PROJECT_ROOT))
    )
    salary_count = 0
    if "salary_annual" in jobs:
        salary_count = int(
            pd.to_numeric(jobs["salary_annual"], errors="coerce").notna().sum()
        )
    company_count = (
        int(jobs["company_name"].nunique(dropna=True)) if "company_name" in jobs else 0
    )
    location_count = (
        int(jobs["location"].nunique(dropna=True)) if "location" in jobs else 0
    )
    artifact_summary, artifact_details = artifact_readiness_summary(status)
    pipeline_ready = artifact_summary == "Full pipeline established"
    pipeline_class = "pipeline-ready" if pipeline_ready else "pipeline-missing"
    card_class = f"sidebar-info {extra_class}".strip()

    st.markdown(
        f"""
        <div class="{escape(card_class)}">
            <div class="info-title">Data source</div>
            <div class="source-line">
                <span class="info-source"><strong>{escape("LinkedIn job catalog" if has_real_data else "Sample role catalog")}</strong></span>
                <span class="sidebar-source-path">{escape(source_label)}</span>
            </div>
            <div class="sidebar-stat-grid">
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Postings</div>
                    <div class="sidebar-stat-value">{format_count(len(jobs))}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Salary rows</div>
                    <div class="sidebar-stat-value">{format_count(salary_count)}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Companies</div>
                    <div class="sidebar-stat-value">{format_count(company_count)}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Locations</div>
                    <div class="sidebar-stat-value">{format_count(location_count)}</div>
                </div>
            </div>
            <div class="data-meta-row">
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">File</div>
                    <div class="sidebar-stat-value">{escape(format_file_size(data_path))} · {escape(format_modified_date(data_path))}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Pipeline</div>
                    <div class="sidebar-stat-value {pipeline_class}">{escape(artifact_summary)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_artifact_expander:
        with st.expander("Artifact details"):
            for detail in artifact_details:
                st.caption(detail)


def render_app_sidebar(
    jobs: pd.DataFrame,
    data_source: str,
    has_real_data: bool,
    status: list[dict[str, Any]],
    pages: dict[str, Any],
) -> None:
    with st.sidebar:
        st.markdown("## ResuMatch")
        st.write("")
        render_data_source_card(jobs, data_source, has_real_data, status)
        st.page_link(pages["home"], label="Home", use_container_width=True)
        st.page_link(pages["demo"], label="Demo", use_container_width=True)
        st.page_link(
            pages["market"],
            label="Market Overview",
            use_container_width=True,
        )
        st.page_link(
            pages["methodology"],
            label="Methodology",
            use_container_width=True,
        )
        st.write("")
        st.caption(linkedin_dataset_note(has_real_data))
