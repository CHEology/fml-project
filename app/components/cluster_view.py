"""
app/components/cluster_view.py

Reusable components for rendering job market cluster position cards,
cluster browsers, and missing-term gap displays.

Owner: @trp8625
"""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def render_cluster_position(cluster: dict[str, Any] | None) -> None:
    """
    Render the user's market segment card showing cluster ID, label,
    alignment score, and top terms.

    Args:
        cluster: Dict with keys cluster_id, label, distance, top_terms.
                 If None, renders a graceful unavailable state.
    """
    if cluster is None:
        st.markdown(
            """
            <div class="signal-card">
                <div class="signal-label">Market segment</div>
                <div class="signal-value">Unavailable</div>
                <div class="signal-copy">Build cluster artifacts to enable market positioning.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cluster_id = escape(str(cluster.get("cluster_id", "?")))
    label = escape(str(cluster.get("label", "Unknown segment")))
    distance = float(cluster.get("distance", 1.0))
    alignment = max(0, min(100, int(round(100 / (1 + distance)))))
    top_terms = cluster.get("top_terms", [])

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">Market segment</div>
            <div class="signal-value">Cluster {cluster_id}</div>
            <div class="signal-copy">{label} · {alignment}% alignment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if top_terms:
        st.markdown(
            '<div class="chip-cloud" style="margin-top:0.5rem;">'
            + "".join(
                f'<span class="mini-chip">{escape(str(term))}</span>'
                for term in top_terms[:8]
            )
            + "</div>",
            unsafe_allow_html=True,
        )


def render_missing_terms(missing_terms: list[str]) -> None:
    """
    Render the 'gaps to close' chip cloud showing skills and keywords
    present in matched roles but missing from the resume.

    Args:
        missing_terms: List of term strings to display as chips.
    """
    if not missing_terms:
        st.markdown(
            "The strongest matching roles are already well reflected in the resume text."
        )
        return

    st.markdown(
        '<div class="chip-cloud">'
        + "".join(
            f'<span class="mini-chip">Add stronger evidence for {escape(str(term))}</span>'
            for term in missing_terms
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def render_cluster_browser(
    cluster_labels: dict[str, dict[str, Any]],
    assignments: list[int] | None = None,
) -> None:
    """
    Render a browsable view of all market clusters with their labels,
    sizes, and top terms. Optionally highlights the user's cluster.

    Args:
        cluster_labels: Dict mapping cluster ID strings to cluster info dicts
                        (label, size, top_terms, common_titles).
        assignments: Optional list of cluster IDs for all jobs, used to
                     show cluster size distribution as a bar chart.
    """
    if not cluster_labels:
        st.info("No cluster labels available.")
        return

    st.markdown(
        '<div class="section-label">Job market segments</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2, gap="medium")
    for idx, (cluster_id, info) in enumerate(cluster_labels.items()):
        with cols[idx % 2]:
            label = escape(str(info.get("label", f"Cluster {cluster_id}")))
            size = int(info.get("size", 0))
            top_terms = info.get("top_terms", [])
            common_titles = info.get("common_titles", [])

            terms_html = "".join(
                f'<span class="mini-chip">{escape(str(term))}</span>'
                for term in top_terms[:5]
            )
            titles_html = ", ".join(escape(str(t)) for t in common_titles[:3])

            st.markdown(
                f"""
                <div class="signal-card" style="margin-bottom:0.75rem;">
                    <div class="signal-label">Cluster {escape(cluster_id)} · {size:,} jobs</div>
                    <div class="signal-value">{label}</div>
                    <div class="chip-cloud" style="margin-top:0.4rem;">{terms_html}</div>
                    <div class="signal-copy" style="margin-top:0.4rem;">
                        Common titles: {titles_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
