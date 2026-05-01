from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from app.components.job_results import fmt_money, render_metric_card

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


@dataclass(frozen=True)
class ClusterSummary:
    frame: pd.DataFrame
    available: bool
    unavailable_reason: str
    top_label: str


MARKET_COLUMNS = (
    "title",
    "company_name",
    "location",
    "state",
    "experience_level",
    "work_type",
    "salary_annual",
    "text",
)


def normalize_market_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    frame = jobs.copy().reset_index(drop=True)
    defaults: dict[str, Any] = {
        "title": "Untitled role",
        "company_name": "Unknown company",
        "location": "Unknown",
        "state": "",
        "experience_level": "Unspecified",
        "work_type": "Unspecified",
        "salary_annual": pd.NA,
        "text": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    for column in (
        "title",
        "company_name",
        "location",
        "experience_level",
        "work_type",
    ):
        frame[column] = (
            frame[column]
            .fillna(defaults[column])
            .astype(str)
            .str.strip()
            .replace("", defaults[column])
        )
    frame["state"] = frame["state"].fillna("").astype(str).str.strip()
    frame["salary_annual"] = pd.to_numeric(frame["salary_annual"], errors="coerce")
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame.loc[
        :, [column for column in MARKET_COLUMNS if column in frame.columns]
    ]


def compute_market_summary(
    jobs: pd.DataFrame,
    *,
    has_real_data: bool,
    cluster_summary: ClusterSummary | None = None,
) -> dict[str, str]:
    frame = normalize_market_jobs(jobs)
    salaries = frame["salary_annual"].dropna()
    location_counts = frame["location"].replace("", "Unknown").value_counts()
    work_type_counts = frame["work_type"].replace("", "Unspecified").value_counts()
    top_work_type = _top_index(work_type_counts, fallback="Unspecified")

    return {
        "job_count": f"{len(frame):,}",
        "median_salary": fmt_money(float(salaries.median()))
        if len(salaries)
        else "N/A",
        "salary_iqr": _salary_iqr(salaries),
        "top_geography": _top_index(location_counts, fallback="Unknown"),
        "work_type_mix": f"{top_work_type} leads",
        "top_segment": (
            cluster_summary.top_label
            if cluster_summary is not None and cluster_summary.available
            else "Build clusters"
        ),
        "data_mode": "Real catalog" if has_real_data else "Sample catalog",
    }


def summarize_clusters(
    *,
    job_count: int,
    assignments: Sequence[int] | None,
    cluster_labels: dict[str, dict[str, Any]] | None,
) -> ClusterSummary:
    if assignments is None or not cluster_labels:
        return _unavailable_clusters("Clustering artifacts are unavailable.")

    assignment_list = list(assignments)
    if len(assignment_list) != job_count:
        return _unavailable_clusters(
            "Cluster assignments do not match the job catalog."
        )

    counts = Counter(int(value) for value in assignment_list)
    rows: list[dict[str, Any]] = []
    for cluster_id, info in cluster_labels.items():
        label_info = info if isinstance(info, dict) else {"label": str(info)}
        numeric_id = int(cluster_id)
        rows.append(
            {
                "cluster_id": numeric_id,
                "label": str(label_info.get("label", f"Cluster {cluster_id}")),
                "job_count": int(counts.get(numeric_id, 0)),
                "top_terms": ", ".join(
                    str(term) for term in label_info.get("top_terms", [])[:5]
                ),
                "common_titles": ", ".join(
                    str(title) for title in label_info.get("common_titles", [])[:3]
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["job_count", "cluster_id"], ascending=[False, True]
    )
    top_label = str(frame.iloc[0]["label"]) if not frame.empty else "Build clusters"
    return ClusterSummary(
        frame=frame.reset_index(drop=True),
        available=not frame.empty,
        unavailable_reason="",
        top_label=top_label,
    )


def build_salary_distribution_figure(jobs: pd.DataFrame):
    frame = normalize_market_jobs(jobs)
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required to build market overview figures.")

    salaries = frame.dropna(subset=["salary_annual"]).copy()
    fig = go.Figure()
    if salaries.empty:
        fig.add_annotation(
            text="No salary values available in this catalog.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    else:
        for work_type, group in salaries.groupby("work_type", sort=True):
            fig.add_trace(
                go.Box(
                    x=group["experience_level"],
                    y=group["salary_annual"],
                    name=str(work_type),
                    boxpoints="all",
                    jitter=0.32,
                    pointpos=-1.2,
                    hovertemplate=(
                        "Seniority: %{x}<br>"
                        "Salary: $%{y:,.0f}<br>"
                        f"Work type: {escape(str(work_type))}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        title="Salary bands by seniority",
        xaxis_title="Seniority signal from job postings",
        yaxis_title="Annual salary",
        yaxis_tickprefix="$",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=70),
        legend_title_text="Work type",
    )
    return fig


def build_market_mix_figure(jobs: pd.DataFrame):
    frame = normalize_market_jobs(jobs)
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required to build market overview figures.")

    location_counts = frame["location"].value_counts().head(8).sort_values()
    work_counts = frame["work_type"].value_counts()
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "domain"}]],
        subplot_titles=("Top locations", "Work type mix"),
        column_widths=[0.62, 0.38],
    )
    fig.add_trace(
        go.Bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation="h",
            marker_color="#2563EB",
            hovertemplate="%{y}<br>%{x:,} roles<extra></extra>",
            name="Locations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Pie(
            labels=work_counts.index,
            values=work_counts.values,
            hole=0.52,
            marker=dict(colors=["#7C3AED", "#0EA5E9", "#10B981", "#F59E0B"]),
            hovertemplate="%{label}<br>%{value:,} roles<extra></extra>",
            name="Work type",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Where the catalog concentrates",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=70, b=45),
        showlegend=False,
    )
    return fig


def build_cluster_distribution_figure(cluster_frame: pd.DataFrame):
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required to build market overview figures.")

    frame = cluster_frame.sort_values("job_count", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["job_count"],
            y=frame["label"],
            orientation="h",
            marker_color="#0F766E",
            customdata=frame[["top_terms", "common_titles"]],
            hovertemplate=(
                "%{y}<br>%{x:,} roles<br>"
                "Top terms: %{customdata[0]}<br>"
                "Common titles: %{customdata[1]}<extra></extra>"
            ),
            name="Segments",
        )
    )
    fig.update_layout(
        title="Semantic market segments",
        xaxis_title="Roles in catalog",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=50),
        showlegend=False,
    )
    return fig


def render_market_hero(data_source: str, has_real_data: bool) -> None:
    source = escape(data_source)
    mode = "real job catalog" if has_real_data else "sample catalog"
    st.markdown(
        f"""
        <div class="market-hero">
            <div class="eyebrow">Market Overview</div>
            <h1>The labor-market map behind the resume demo</h1>
            <p>
                This page explains why the ML demo behaves the way it does:
                which roles exist in the catalog, where salaries spread out,
                and how semantic segments become retrieval context, salary
                evidence, cluster positioning, and gap terms.
            </p>
            <div class="market-source">Using {escape(mode)} from <strong>{source}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(summary: dict[str, str]) -> None:
    cols = st.columns(5)
    metrics = (
        ("Jobs loaded", summary["job_count"], summary["data_mode"]),
        ("Median salary", summary["median_salary"], summary["salary_iqr"]),
        ("Top segment", summary["top_segment"], "from K-Means labels"),
        ("Top geography", summary["top_geography"], "catalog concentration"),
        ("Work type", summary["work_type_mix"], "remote/hybrid/on-site mix"),
    )
    for col, (label, value, helper) in zip(cols, metrics, strict=True):
        with col:
            render_metric_card(label, value, helper)


def render_explanation_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="market-explainer-card">
            <div class="market-explainer-title">{escape(title)}</div>
            <div class="market-explainer-copy">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_dependency_grid() -> None:
    cards = (
        (
            "Retrieval",
            "The matcher searches this catalog, so market concentration defines which roles can surface as strong neighbors.",
        ),
        (
            "Salary bands",
            "The demo reports q10/q25/q50/q75/q90 because compensation is a distribution, not a single deterministic label.",
        ),
        (
            "K-Means segments",
            "Clusters define role families; the candidate marker, segment label, and gap terms all depend on these neighborhoods.",
        ),
        (
            "Quality adjustments",
            "Resume quality and capability scores are interpreted against market evidence, so sparse or broad markets lower confidence.",
        ),
    )
    st.markdown('<div class="market-explainer-grid">', unsafe_allow_html=True)
    for title, body in cards:
        render_explanation_card(title, body)
    st.markdown("</div>", unsafe_allow_html=True)


def render_market_exemplars(jobs: pd.DataFrame) -> None:
    frame = normalize_market_jobs(jobs)
    display = frame.sort_values("salary_annual", ascending=False).head(12).copy()
    display["salary_annual"] = display["salary_annual"].map(
        lambda value: fmt_money(float(value)) if pd.notna(value) else "N/A"
    )
    display["text"] = display["text"].str.slice(0, 120)
    st.markdown("### Market exemplars")
    st.caption(
        "Representative postings show the raw evidence retrieval and salary models can draw from."
    )
    st.dataframe(
        display[
            [
                "title",
                "company_name",
                "location",
                "work_type",
                "experience_level",
                "salary_annual",
                "text",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def _salary_iqr(salaries: pd.Series) -> str:
    if salaries.empty:
        return "N/A"
    q25 = float(salaries.quantile(0.25))
    q75 = float(salaries.quantile(0.75))
    return f"{fmt_money(q25)} - {fmt_money(q75)}"


def _top_index(counts: pd.Series, *, fallback: str) -> str:
    if counts.empty:
        return fallback
    value = str(counts.index[0]).strip()
    return value or fallback


def _unavailable_clusters(reason: str) -> ClusterSummary:
    return ClusterSummary(
        frame=pd.DataFrame(
            columns=["cluster_id", "label", "job_count", "top_terms", "common_titles"]
        ),
        available=False,
        unavailable_reason=reason,
        top_label="Build clusters",
    )
