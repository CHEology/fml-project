"""
app/components/salary_chart.py

Reusable components for rendering salary band displays and Plotly fan charts.
Isolates all salary visualization logic from app.py.

Owner: @trp8625
"""

from __future__ import annotations

from html import escape
from typing import Any

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
    fig.add_trace(go.Scatter(
        x=quantiles + quantiles[::-1],
        y=values + [values[0]] * len(values),
        fill="toself",
        fillcolor="rgba(23, 92, 211, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=quantiles,
        y=values,
        mode="lines+markers",
        line=dict(color="#175cd3", width=2.5),
        marker=dict(size=8, color="#175cd3"),
        text=[fmt_money(v) for v in values],
        textposition="top center",
        hovertemplate="%{x}: %{text}<extra></extra>",
        showlegend=False,
    ))

    # Median highlight
    fig.add_trace(go.Scatter(
        x=["50th\n(Median)"],
        y=[band.get("q50", 0)],
        mode="markers+text",
        marker=dict(size=14, color="#175cd3", symbol="circle"),
        text=[fmt_money(band.get("q50", 0))],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
    ))

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
