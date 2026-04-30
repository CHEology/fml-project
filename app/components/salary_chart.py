"""
app/components/salary_chart.py

Reusable components for rendering salary band displays and Plotly fan charts.
Isolates all salary visualization logic from app.py.

Owner: @trp8625
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def fmt_money(value: float | int | None) -> str:
    """Format a numeric value as a USD dollar string."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${int(value):,}"


def render_cluster_salary_distribution(
    jobs: pd.DataFrame,
    assignments: Sequence[int] | None,
    cluster_labels: dict[str, dict[str, Any]] | None,
    cluster: dict[str, Any] | None,
    band: dict[str, Any] | None,
    *,
    job_embeddings: np.ndarray | None = None,
    resume_embedding: np.ndarray | None = None,
    sample_size: int = 900,
) -> None:
    """
    Render a 2D projection of K-Means market clusters with the user's
    predicted cluster overlaid.
    """
    if assignments is None:
        st.info(
            "Cluster salary distribution is unavailable until clustering artifacts are built."
        )
        return
    if not _PLOTLY_AVAILABLE:
        st.info("Cluster salary distribution requires Plotly to render.")
        return

    user_cluster_id = _cluster_value(cluster, "cluster_id")
    user_salary = None if band is None else band.get("q50")
    if user_cluster_id is None or user_salary is None or pd.isna(user_salary):
        st.info(
            "User salary position is unavailable until both salary and cluster predictions are available."
        )
        return

    if job_embeddings is None or resume_embedding is None:
        st.info("2D cluster map is unavailable until job embeddings are built.")
        return

    assignment_list = list(assignments)
    if len(assignment_list) != len(jobs):
        st.info(
            "Cluster salary distribution is unavailable because clustering assignments do not match the job catalog."
        )
        return

    projection = _embedding_projection(job_embeddings, resume_embedding, len(jobs))
    if projection is None:
        st.info(
            "2D cluster map is unavailable because embeddings do not match the job catalog."
        )
        return
    job_points, user_point = projection

    if "salary_annual" in jobs:
        salary_source = jobs["salary_annual"]
    else:
        salary_source = pd.Series([pd.NA] * len(jobs), index=jobs.index)
    salaries = pd.to_numeric(salary_source, errors="coerce")
    chart_frame = pd.DataFrame(
        {
            "salary_annual": salaries,
            "cluster_id": assignment_list,
            "component_1": job_points[:, 0],
            "component_2": job_points[:, 1],
        }
    )
    chart_frame = chart_frame.dropna(
        subset=["cluster_id", "component_1", "component_2"]
    ).copy()
    if chart_frame.empty:
        st.info(
            "2D cluster map is unavailable because no projected job points are available."
        )
        return

    chart_frame["cluster_id"] = chart_frame["cluster_id"].astype(int)
    chart_frame["cluster_label"] = chart_frame["cluster_id"].map(
        lambda value: _cluster_name(value, cluster_labels)
    )
    chart_frame["cluster_terms"] = chart_frame["cluster_id"].map(
        lambda value: _cluster_terms_summary(value, cluster_labels)
    )
    chart_frame["salary_label"] = chart_frame["salary_annual"].map(
        lambda value: fmt_money(value) if not pd.isna(value) else "N/A"
    )
    for column, fallback in (
        ("title", "Untitled role"),
        ("company_name", "Unknown company"),
        ("location", "Unknown location"),
        ("experience_level", "Experience TBD"),
        ("work_type", "Work type TBD"),
    ):
        if column in jobs:
            chart_frame[column] = jobs[column].fillna(fallback).astype(str)
        else:
            chart_frame[column] = fallback

    total_projected_jobs = len(chart_frame)
    chart_frame = _sample_chart_points(chart_frame, sample_size=sample_size)
    plotted_jobs = len(chart_frame)

    if chart_frame["salary_annual"].notna().sum() == 0:
        salary_context = "salary values are unavailable in this catalog"
    else:
        salary_context = "salary values are shown in hover details"
    sample_context = (
        f"shown as a deterministic random sample of {plotted_jobs:,} jobs "
        f"from {total_projected_jobs:,} projected postings for visualization purposes"
    )

    palette = [
        "#175cd3",
        "#079455",
        "#dc6803",
        "#7a5af8",
        "#c11574",
        "#0e9384",
        "#b42318",
        "#4e5ba6",
    ]
    color_by_cluster = {
        cluster_id: palette[index % len(palette)]
        for index, cluster_id in enumerate(sorted(chart_frame["cluster_id"].unique()))
    }
    user_label = _cluster_name(int(user_cluster_id), cluster_labels)
    user_terms = _cluster_terms_summary(int(user_cluster_id), cluster_labels)
    user_salary_label = fmt_money(float(user_salary))

    st.markdown(
        f"""
        <div class="section-label" style="margin-top:1rem;">2D market cluster map</div>
        <div class="evidence-line">
            Jobs are embedded as sentence-transformer vectors, grouped with K-Means,
            and projected into two PCA-reduced embedding components. Nearby dots
            are semantically similar postings, colors identify cluster names in the
            legend, and the diamond shows the resume in the same 2D space. The dots are
            {escape(sample_context)}; {escape(salary_context)}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    for cluster_id, group in chart_frame.groupby("cluster_id", sort=True):
        cluster_label = _cluster_name(int(cluster_id), cluster_labels)
        fig.add_trace(
            go.Scatter(
                x=group["component_1"],
                y=group["component_2"],
                mode="markers",
                name=cluster_label,
                marker=dict(
                    color=color_by_cluster[int(cluster_id)],
                    size=9,
                    opacity=0.72,
                    line=dict(color="#ffffff", width=0.8),
                ),
                customdata=group[
                    [
                        "title",
                        "company_name",
                        "location",
                        "experience_level",
                        "work_type",
                        "cluster_terms",
                        "cluster_label",
                        "salary_label",
                    ]
                ],
                hovertemplate=(
                    "Embedding component 1: %{x:.3f}<br>"
                    "Embedding component 2: %{y:.3f}<br>"
                    "Title: %{customdata[0]}<br>"
                    "Company: %{customdata[1]}<br>"
                    "Location: %{customdata[2]}<br>"
                    "Experience: %{customdata[3]}<br>"
                    "Work type: %{customdata[4]}<br>"
                    "Cluster: %{customdata[6]}<br>"
                    "Cluster evidence: %{customdata[5]}<br>"
                    "Salary: %{customdata[7]}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[float(user_point[0])],
            y=[float(user_point[1])],
            mode="markers+text",
            name="Your predicted position",
            marker=dict(
                color="#d92d20",
                size=16,
                symbol="diamond",
                line=dict(color="#ffffff", width=2),
            ),
            text=[fmt_money(float(user_salary))],
            textposition="top center",
            hovertemplate=(
                "Your predicted position<br>"
                f"Cluster: {user_label}<br>"
                "Embedding component 1: %{x:.3f}<br>"
                "Embedding component 2: %{y:.3f}<br>"
                f"Predicted salary: {user_salary_label}<br>"
                f"Cluster evidence: {user_terms}<extra></extra>"
            ),
            showlegend=True,
        )
    )

    fig.update_layout(
        yaxis=dict(
            title="Embedding component 2",
            gridcolor="rgba(0,0,0,0.06)",
            zerolinecolor="rgba(0,0,0,0.12)",
        ),
        xaxis=dict(
            title="Embedding component 1",
            gridcolor="rgba(0,0,0,0.06)",
            zerolinecolor="rgba(0,0,0,0.12)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=80),
        height=420,
        showlegend=True,
        legend=dict(title=dict(text="Cluster key")),
    )
    st.plotly_chart(fig, use_container_width=True)


def _sample_chart_points(
    frame: pd.DataFrame,
    *,
    sample_size: int,
) -> pd.DataFrame:
    if sample_size <= 0 or len(frame) <= sample_size:
        return frame
    return frame.sample(n=sample_size, random_state=42).sort_index()


def _embedding_projection(
    job_embeddings: np.ndarray,
    resume_embedding: np.ndarray,
    expected_jobs: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    jobs_matrix = np.asarray(job_embeddings, dtype=np.float32)
    resume_vector = np.asarray(resume_embedding, dtype=np.float32).reshape(-1)
    if jobs_matrix.ndim != 2 or len(jobs_matrix) != expected_jobs:
        return None
    if jobs_matrix.shape[1] != resume_vector.shape[0]:
        return None

    centered = jobs_matrix - jobs_matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or centered.shape[1] < 2:
        components = np.eye(jobs_matrix.shape[1], 2, dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].T.astype(np.float32, copy=False)
        if components.shape[1] < 2:
            components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))

    job_points = centered @ components
    user_point = (resume_vector - jobs_matrix.mean(axis=0)) @ components
    return np.asarray(job_points, dtype=np.float32), np.asarray(
        user_point, dtype=np.float32
    )


def _cluster_name(
    cluster_id: int,
    cluster_labels: dict[str, dict[str, Any]] | None,
) -> str:
    info = {} if cluster_labels is None else cluster_labels.get(str(cluster_id), {})
    label = (
        info if isinstance(info, str) else info.get("label", f"Cluster {cluster_id}")
    )
    return str(label)


def _cluster_terms_summary(
    cluster_id: int,
    cluster_labels: dict[str, dict[str, Any]] | None,
) -> str:
    terms = _cluster_terms(cluster_id, cluster_labels)
    if terms:
        return ", ".join(terms[:5])
    return "No top terms available"


def _cluster_terms(
    cluster_id: int,
    cluster_labels: dict[str, dict[str, Any]] | None,
) -> list[str]:
    info = {} if cluster_labels is None else cluster_labels.get(str(cluster_id), {})
    if not isinstance(info, dict):
        return []
    return [str(term) for term in info.get("top_terms", [])]


def _cluster_value(cluster: dict[str, Any] | None, key: str) -> Any:
    if isinstance(cluster, dict):
        return cluster.get(key)
    return None


def render_salary_band(band: dict[str, Any]) -> None:
    """
    Render the full salary band display: headline, quantile strip, and
    evidence footnote. Falls back gracefully when band is None.

    Args:
        band: Dict with keys q10, q25, q50, q75, q90, primary_source,
              confidence, and evidence sub-dict.
    """
    source_labels = {
        "retrieved_jobs": "Matched roles",
        "bls": "Occupation wage data",
        "neural_model": "Model estimate",
    }
    evidence = band.get("evidence", {})
    primary = source_labels.get(str(band.get("primary_source")), "Available evidence")
    confidence = str(band.get("confidence", "unknown")).title()
    low = fmt_money(band["q10"])
    midpoint = fmt_money(band["q50"])
    high = fmt_money(band["q90"])
    source_badge = escape(f"{primary} · {confidence}")

    st.markdown(
        f"""
        <div class="section-label">Matched-market salary range</div>
        <div class="salary-headline">
            <div>
                <div class="salary-main">{midpoint}</div>
                <div class="salary-range">{low} to {high} expected market range</div>
            </div>
            <div class="salary-source">{source_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="salary-band">
            <div class="salary-fill" style="width:100%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quantile_labels = {
        "q10": "Low",
        "q25": "Lower mid",
        "q50": "Median",
        "q75": "Upper mid",
        "q90": "High",
    }
    cells = "".join(
        '<div class="salary-step">'
        f'<div class="salary-step-label">{escape(quantile_labels[key])}</div>'
        f'<div class="salary-step-value">{escape(fmt_money(band[key]))}</div>'
        "</div>"
        for key in ("q10", "q25", "q50", "q75", "q90")
    )
    st.markdown(f'<div class="salary-strip">{cells}</div>', unsafe_allow_html=True)

    # Evidence footnote
    pieces = [f"Source: {primary}", f"Confidence: {confidence}"]
    salary_count = evidence.get("salary_count")
    if salary_count is not None:
        pieces.append(f"{int(salary_count)} roles with salary data")
    median_similarity = evidence.get("median_similarity")
    if median_similarity is not None and not pd.isna(median_similarity):
        pieces.append(f"{float(median_similarity) * 100:.0f}% median similarity")
    occupation_title = evidence.get("occupation_title")
    if occupation_title:
        pieces.append(str(occupation_title))
    if evidence.get("model_bls_disagreement"):
        pieces.append("supporting sources disagree")
    seniority_filter = evidence.get("seniority_filter")
    if seniority_filter:
        pieces.append(str(seniority_filter))

    evidence_html = escape(" · ".join(pieces))
    st.markdown(
        f'<div class="evidence-line">{evidence_html}</div>',
        unsafe_allow_html=True,
    )

    adjustment_notes = band.get("adjustment_notes") or []
    if adjustment_notes:
        notes_html = escape(" ".join(str(note) for note in adjustment_notes))
        st.markdown(
            f'<div class="evidence-line" style="color: var(--warning);">{notes_html}</div>',
            unsafe_allow_html=True,
        )


def render_salary_fan_chart(band: dict[str, Any]) -> None:
    """
    Render a Plotly fan/waterfall chart showing predicted salary quantiles.
    Falls back to render_salary_band if Plotly is not available.

    Args:
        band: Dict with keys q10, q25, q50, q75, q90.
    """
    if not _PLOTLY_AVAILABLE:
        render_salary_band(band)
        return

    quantiles = ["10th", "25th", "50th\n(Median)", "75th", "90th"]
    values = [
        band.get("q10", 0),
        band.get("q25", 0),
        band.get("q50", 0),
        band.get("q75", 0),
        band.get("q90", 0),
    ]

    fig = go.Figure()

    # Shaded range band (q10 to q90)
    fig.add_trace(
        go.Scatter(
            x=quantiles + quantiles[::-1],
            y=values + [values[0]] * len(values),
            fill="toself",
            fillcolor="rgba(23, 92, 211, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main line
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=values,
            mode="lines+markers",
            line=dict(color="#175cd3", width=2.5),
            marker=dict(size=8, color="#175cd3"),
            text=[fmt_money(v) for v in values],
            textposition="top center",
            hovertemplate="%{x}: %{text}<extra></extra>",
            showlegend=False,
        )
    )

    # Median highlight
    fig.add_trace(
        go.Scatter(
            x=["50th\n(Median)"],
            y=[band.get("q50", 0)],
            mode="markers+text",
            marker=dict(size=14, color="#175cd3", symbol="circle"),
            text=[fmt_money(band.get("q50", 0))],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        yaxis=dict(
            tickprefix="$",
            tickformat=",",
            gridcolor="rgba(0,0,0,0.06)",
            title="Annual Salary (USD)",
        ),
        xaxis=dict(title="Percentile"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Keep the evidence footnote
    evidence = band.get("evidence", {})
    source_labels = {
        "retrieved_jobs": "Matched roles",
        "bls": "Occupation wage data",
        "neural_model": "Model estimate",
    }
    primary = source_labels.get(str(band.get("primary_source")), "Available evidence")
    confidence = str(band.get("confidence", "unknown")).title()
    pieces = [f"Source: {primary}", f"Confidence: {confidence}"]
    salary_count = evidence.get("salary_count")
    if salary_count is not None:
        pieces.append(f"{int(salary_count)} roles with salary data")
    st.caption(" · ".join(pieces))
