from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from app.components.sidebar import render_data_source_card
from app.components.team import TEAM_MEMBERS, TEAM_NAME
from app.config import MASCOT_PATH


def encoded_image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_home_page(
    pages: dict[str, Any],
    jobs: pd.DataFrame,
    data_source: str,
    has_real_data: bool,
    status: list[dict[str, Any]],
) -> None:
    mascot_uri = encoded_image_data_uri(MASCOT_PATH)
    mascot_html = (
        f'<img src="{mascot_uri}" alt="ResuMatch hamster mascot wearing sunglasses and an NYU shirt while holding a money bag" />'
        if mascot_uri
        else ""
    )
    team_line = " | ".join(
        f"<strong>{escape(member['name'])}</strong> {escape(member['github'])}"
        for member in TEAM_MEMBERS
    )
    st.markdown(
        f"""
        <section class="home-hero">
            <div>
                <h1>ResuMatch</h1>
                <div class="home-subtitle">A machine learning project by the {escape(TEAM_NAME)}</div>
                <div class="home-team-line">{team_line}</div>
                <p class="home-lede">
                    Resume market intelligence for understanding role fit, salary range,
                    and market position from real job-posting evidence.
                </p>
            </div>
            <div class="home-mascot-frame">
                {mascot_html}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_data_source_card(
        jobs,
        data_source,
        has_real_data,
        status,
        extra_class="home-data-card",
        show_artifact_expander=False,
    )
    st.markdown(
        """
        <div class="home-cta-row">
            <a class="home-cta" href="demo" target="_self">Demo</a>
            <a class="home-cta secondary" href="market-overview" target="_self">Market Overview</a>
            <a class="home-cta secondary" href="methodology" target="_self">Methodology</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
