from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.market_overview import (
    build_cluster_distribution_figure,
    build_market_mix_figure,
    build_salary_distribution_figure,
    compute_market_summary,
    normalize_market_jobs,
    render_market_exemplars,
    render_market_hero,
    render_model_dependency_grid,
    render_summary_metrics,
    summarize_clusters,
)
from app.runtime.cache import artifacts_ready, load_cluster_resource


def render_market_overview_page(
    jobs: pd.DataFrame,
    data_source: str,
    has_real_data: bool,
    status: list[dict[str, str | bool]],
) -> None:
    display_jobs = normalize_market_jobs(jobs)
    cluster_summary = _load_cluster_summary(display_jobs, status)
    summary = compute_market_summary(
        display_jobs,
        has_real_data=has_real_data,
        cluster_summary=cluster_summary,
    )

    render_market_hero(data_source, has_real_data)
    render_summary_metrics(summary)

    st.markdown(
        """
        <div class="market-section-heading">
            <div class="section-heading-kicker">Compensation landscape</div>
            <h2>Salary is the distribution the model is trying to explain</h2>
            <p>
                The demo's salary output uses q10/q25/q50/q75/q90 bands because
                similar roles do not pay a single number. Seniority, work type,
                and catalog coverage shape the evidence before quality and
                capability adjustments are applied.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_salary_distribution_figure(display_jobs),
        width="stretch",
    )

    mix_col, segment_col = st.columns([0.52, 0.48], gap="large")
    with mix_col:
        st.markdown(
            """
            <div class="market-section-heading">
                <div class="section-heading-kicker">Catalog shape</div>
                <h2>Geography and work mode set the retrieval context</h2>
                <p>
                    If the catalog is concentrated in certain cities or work
                    modes, matched roles and salary evidence will reflect that
                    market mix. This is why the demo treats retrieved postings
                    as contextual evidence rather than universal truth.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_market_mix_figure(display_jobs), width="stretch")

    with segment_col:
        st.markdown(
            """
            <div class="market-section-heading">
                <div class="section-heading-kicker">Semantic structure</div>
                <h2>Clusters turn postings into role families</h2>
                <p>
                    K-Means clusters define the market neighborhoods used for
                    candidate positioning and gap terms. A resume lands near a
                    centroid, then the demo explains the segment with common
                    terms and titles from that neighborhood.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if cluster_summary.available:
            st.plotly_chart(
                build_cluster_distribution_figure(cluster_summary.frame),
                width="stretch",
            )
            _render_cluster_terms(cluster_summary.frame)
        else:
            st.info(cluster_summary.unavailable_reason)
            st.caption(
                "Build the clustering artifacts to enable segment counts, top terms, and centroid-based positioning."
            )

    st.markdown(
        """
        <div class="market-section-heading">
            <div class="section-heading-kicker">Model dependencies</div>
            <h2>How this page connects to the rest of the demo</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_model_dependency_grid()
    render_market_exemplars(display_jobs)


def _load_cluster_summary(
    jobs: pd.DataFrame,
    status: list[dict[str, str | bool]],
):
    assignments = None
    cluster_labels = None
    if artifacts_ready(status, "clustering"):
        try:
            _, assignments, cluster_labels = load_cluster_resource()
        except (FileNotFoundError, ValueError, OSError):
            assignments = None
            cluster_labels = None
    return summarize_clusters(
        job_count=len(jobs),
        assignments=assignments,
        cluster_labels=cluster_labels,
    )


def _render_cluster_terms(cluster_frame: pd.DataFrame) -> None:
    for _, row in cluster_frame.head(4).iterrows():
        st.markdown(
            f"""
            <div class="market-cluster-card">
                <div class="market-cluster-title">{row["label"]}</div>
                <div class="market-cluster-meta">{int(row["job_count"]):,} roles</div>
                <div class="market-cluster-copy">Top terms: {row["top_terms"] or "N/A"}</div>
                <div class="market-cluster-copy">Common titles: {row["common_titles"] or "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
