"""
app/components/job_results.py

Reusable components for rendering job match cards and metric/signal cards.
All rendering logic for individual job results is isolated here so app.py
stays focused on orchestration.

Owner: @trp8625
"""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st


def fmt_money(value: float | int | None) -> str:
    """Format a numeric value as a USD dollar string."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${int(value):,}"


def render_job_card(row: pd.Series) -> None:
    """
    Render a single job match card with similarity score, title, company,
    location, salary, experience level, and a text snippet.

    Args:
        row: A DataFrame row with keys: title, company_name, location,
             work_type, experience_level, salary_annual, text, similarity,
             and optionally public_ats_score.
    """
    summary = escape(str(row.get("text", ""))[:190])
    similarity = row.get("similarity", np.nan)
    score_label = "Strong match"
    if not pd.isna(similarity):
        score_label = f"{float(similarity) * 100:.0f}% similarity"
    public_ats = row.get("public_ats_score", np.nan)
    if not pd.isna(public_ats):
        score_label += f" · {float(public_ats):.0f}% public fit"

    title = escape(str(row.get("title", "Untitled role")))
    company = escape(str(row.get("company_name", "Unknown company")))
    location = escape(str(row.get("location", "Unknown location")))
    work_type = escape(str(row.get("work_type", "Work type TBD")))
    experience = escape(str(row.get("experience_level", "Experience TBD")))
    salary = fmt_money(row.get("salary_annual"))

    st.markdown(
        f"""
        <div class="job-card">
            <div class="score-chip">{score_label}</div>
            <div class="job-title">{title}</div>
            <div class="job-meta">{company} · {location} · {work_type}</div>
            <div><strong>{salary}</strong> · {experience}</div>
            <div class="mono" style="margin-top:0.6rem;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_job_results(matches: pd.DataFrame) -> None:
    """
    Render a 2-column grid of job match cards from a DataFrame of results.

    Args:
        matches: DataFrame of job matches, each row passed to render_job_card.
    """
    if matches is None or matches.empty:
        st.info(
            "No matching roles surfaced. Try expanding the resume text with more domain terms."
        )
        return

    card_cols = st.columns(2, gap="medium")
    for index, (_, row) in enumerate(matches.iterrows()):
        with card_cols[index % 2]:
            render_job_card(row)


def render_metric_card(label: str, value: str, helper: str) -> None:
    """
    Render a single metric card with a large value and a helper caption.

    Args:
        label: Short uppercase label shown above the value.
        value: The primary metric value to display prominently.
        helper: Small muted helper text shown below the value.
    """
    label_html = escape(str(label))
    value_html = escape(str(value))
    helper_html = escape(str(helper))
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label_html}</div>
            <div class="metric-value">{value_html}</div>
            <div class="mono">{helper_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(label: str, value: str, copy: str) -> None:
    """
    Render a signal card with a label, prominent value, and explanatory copy.

    Args:
        label: Short uppercase label.
        value: Primary signal value (e.g. track name, seniority level).
        copy: One-sentence explanation shown below the value.
    """
    label_html = escape(str(label))
    value_html = escape(str(value))
    copy_html = escape(str(copy))
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">{label_html}</div>
            <div class="signal-value">{value_html}</div>
            <div class="signal-copy">{copy_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_banner(kicker: str, title: str, body: str) -> None:
    """
    Render a section header banner with a kicker, title, and body copy.

    Args:
        kicker: Small uppercase label above the title (currently unused in HTML
                but kept for API consistency with app.py).
        title: Section title rendered prominently.
        body: Supporting copy rendered in muted color below the title.
    """
    title_html = escape(str(title))
    body_html = escape(str(body))
    st.markdown(
        f"""
        <div class="panel-banner">
            <div class="panel-title">{title_html}</div>
            <div class="panel-copy">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
