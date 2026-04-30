from __future__ import annotations

from collections.abc import Callable
from html import escape

import streamlit as st


def info_dot(text: str, *, extra_class: str = "") -> str:
    class_name = f"info-dot {extra_class}".strip()
    return (
        f'<span class="{escape(class_name)}" data-tooltip="{escape(text)}" '
        'aria-label="More information" role="button" tabindex="0">?</span>'
    )


def render_demo_section_header(title: str, body: str, explanation: str) -> None:
    st.markdown(
        f"""
        <div class="snapshot-hero result-section-header">
            <div class="snapshot-title-row">
                <h1 class="snapshot-title">{escape(title)}</h1>
            </div>
            <div class="snapshot-summary">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Read more about {title}"):
        st.markdown(explanation)


def render_demo_signal_card(
    label: str,
    value: str,
    copy: str,
    explanation: str,
) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">{escape(label)}{info_dot(explanation, extra_class="inline-info")}</div>
            <div class="signal-value">{escape(value)}</div>
            <div class="signal-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_floating_nav(
    *,
    previous_stage: str | None = None,
    restart_demo: Callable[[], None],
    next_label: str | None = None,
    next_stage: str | None = None,
) -> None:
    with st.container(key="demo-floating-nav"):
        if previous_stage is None and not (next_label and next_stage):
            if st.button("Start over with new resume/profile", width="stretch"):
                restart_demo()
                st.rerun()
            return

        widths = [0.25, 0.4, 0.35] if next_label and next_stage else [0.3, 0.45]
        nav_cols = st.columns(widths, gap="small")
        if previous_stage is not None:
            with nav_cols[0]:
                if st.button("← Previous", width="stretch"):
                    st.session_state.demo_stage = previous_stage
                    st.rerun()
        with nav_cols[1]:
            if st.button("Start over with new resume/profile", width="stretch"):
                restart_demo()
                st.rerun()
        if next_label and next_stage:
            with nav_cols[2]:
                if st.button(f"{next_label} →", type="primary", width="stretch"):
                    st.session_state.demo_stage = next_stage
                    st.rerun()
