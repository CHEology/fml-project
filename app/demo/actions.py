from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st
from app.components.salary_chart import fmt_money
from app.demo.components import render_demo_floating_nav, render_demo_section_header
from ml.resume_assessment.career_actions import (
    cluster_options,
    cluster_transition_advice,
    salary_growth_advice,
)

ACTION_OPTIONS = ["Improve my salary", "Move to a different cluster"]


def render_actions_page(
    jobs: pd.DataFrame,
    assessment: dict[str, Any] | None,
    *,
    restart_demo,
) -> None:
    if assessment is None:
        st.warning("Run profile analysis before choosing improvement actions.")
        render_demo_floating_nav(previous_stage="input", restart_demo=restart_demo)
        return

    render_demo_section_header(
        "Profile Actions",
        "Choose the career moves you want to explore from the current analysis.",
        (
            "This page uses local cluster assignments, salary evidence, job text, and "
            "resume signals. It recommends real experience to build and later document; "
            "it does not suggest fabricating resume content."
        ),
    )

    st.markdown(
        '<h2 class="action-choice-heading">Action to improve your profile</h2>',
        unsafe_allow_html=True,
    )
    selected_action = st.radio(
        "Action to improve your profile",
        ACTION_OPTIONS,
        key="demo_selected_action",
        horizontal=True,
        label_visibility="collapsed",
    )

    cluster = assessment.get("cluster") or {}
    cluster_id = _int_or_none(cluster.get("cluster_id"))
    assignments = assessment.get("cluster_assignments")
    labels = assessment.get("cluster_labels")
    resume_text = str(assessment.get("resume_text") or "")

    if not cluster or assignments is None or labels is None:
        st.warning(
            "Action recommendations need clustering artifacts. Build clusters, then rerun analysis."
        )
        render_demo_floating_nav(previous_stage="results", restart_demo=restart_demo)
        return

    if selected_action == "Improve my salary":
        advice = salary_growth_advice(
            jobs,
            assignments,
            cluster_labels=labels,
            current_cluster_id=cluster_id,
            resume_text=resume_text,
        )
        _render_advice(
            f"Improve salary within {str(cluster.get('label') or 'current cluster')}",
            (
                "High-salary evidence comes from q75+ postings inside "
                f"{escape(str(cluster.get('label') or 'the current cluster'))}."
            ),
            advice,
        )

    if selected_action == "Move to a different cluster":
        options = cluster_options(labels, cluster_id)
        if not options:
            st.warning("No alternate clusters are available for transition advice.")
        else:
            target_id = _target_cluster_id(options, cluster)
            option_ids = [int(option["cluster_id"]) for option in options]
            options_by_id = {int(option["cluster_id"]): option for option in options}
            default_index = option_ids.index(target_id)
            st.session_state.demo_target_cluster_id = target_id
            selected_target_id = st.selectbox(
                "Target cluster",
                option_ids,
                index=default_index,
                format_func=lambda option_id: _cluster_option_label(
                    option_id, options_by_id
                ),
                key="demo_target_cluster_id",
            )
            target_id = int(selected_target_id)
            selected_label = _cluster_option_label(target_id, options_by_id)
            advice = cluster_transition_advice(
                jobs,
                assignments,
                cluster_labels=labels,
                current_cluster_id=cluster_id,
                target_cluster_id=target_id,
                resume_text=resume_text,
            )
            _render_advice(
                "Move into a different cluster",
                (
                    "Transition evidence comes from postings in "
                    f"{escape(str(selected_label))}."
                ),
                advice,
            )

    render_demo_floating_nav(previous_stage="results", restart_demo=restart_demo)


def _target_cluster_id(options: list[dict[str, Any]], cluster: dict[str, Any]) -> int:
    option_ids = [int(option["cluster_id"]) for option in options]
    saved = _int_or_none(st.session_state.get("demo_target_cluster_id"))
    if saved in option_ids:
        return int(saved)
    next_best = _int_or_none(cluster.get("next_best_cluster_id"))
    if next_best in option_ids:
        return int(next_best)
    return option_ids[0]


def _cluster_option_label(
    cluster_id: int,
    options_by_id: dict[int, dict[str, Any]],
) -> str:
    option = options_by_id[int(cluster_id)]
    return f"{option['label']} (Cluster {int(cluster_id)})"


def _render_advice(title: str, intro: str, advice: dict[str, Any]) -> None:
    if not advice.get("available"):
        st.warning(str(advice.get("reason") or "Advice is unavailable."))
        return

    salary_line = ""
    if advice.get("salary_threshold") is not None:
        salary_line = (
            '<div class="action-summary-pill">High-salary cohort starts at '
            f"{escape(fmt_money(advice.get('salary_threshold')))}</div>"
        )
    else:
        salary_line = '<div class="action-summary-pill">Cluster transition</div>'
    st.markdown(
        f"""
        <div class="action-advice-panel">
            <div class="action-advice-header">
                <div>
                    <div class="section-heading-kicker">Action plan</div>
                    <h2>{escape(title)}</h2>
                    <p>{intro}</p>
                </div>
                {salary_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2, gap="small")
    with columns[0]:
        _render_list_card("Roles to build toward", advice.get("target_titles") or [])
        _render_list_card("Experience to add", advice.get("missing_terms") or [])
    with columns[1]:
        _render_list_card(
            "Education or credentials",
            advice.get("education_requirements")
            or ["No common credential signal found."],
        )
        _render_list_card(
            "Years-of-experience patterns",
            advice.get("experience_requirements")
            or ["No explicit years-of-experience pattern found."],
        )

    _render_jobs(advice.get("representative_jobs") or [])
    _render_list_card("Recommended actions", advice.get("career_actions") or [])


def _render_list_card(title: str, items: list[Any]) -> None:
    item_html = "".join(
        f"<li>{escape(str(item))}</li>" for item in items if str(item).strip()
    )
    if not item_html:
        item_html = "<li>No strong signal found in the current evidence.</li>"
    st.markdown(
        f"""
        <div class="action-list-card">
            <div class="snapshot-label">{escape(title)}</div>
            <ul>{item_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_jobs(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        return
    cards = ""
    for job in jobs[:5]:
        cards += (
            '<div class="action-job-row">'
            f"<strong>{escape(str(job.get('title') or 'Untitled role'))}</strong>"
            f"<span>{escape(fmt_money(job.get('salary_annual')))}</span>"
            "</div>"
        )
    st.markdown(
        f"""
        <div class="action-list-card">
            <div class="snapshot-label">Representative postings</div>
            <div class="action-job-list">{cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
