from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.methodology import render_methodology_page
from app.components.sidebar import render_app_sidebar
from app.demo.state import initialize_session_state
from app.pages.demo import render_demo_page
from app.pages.home import render_home_page
from app.pages.market import render_market_overview_page
from app.runtime.cache import artifact_status, load_jobs
from app.styles import inject_styles

st.set_page_config(
    page_title="ResuMatch",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    initialize_session_state()
    inject_styles(st.session_state.get("theme_name", "Lavender"))
    jobs, data_source, has_real_data = load_jobs()
    status = artifact_status()

    pages: dict[str, Any] = {}
    pages["home"] = st.Page(
        lambda: render_home_page(pages, jobs, data_source, has_real_data, status),
        title="Home",
        url_path="home",
        default=True,
    )
    pages["demo"] = st.Page(
        lambda: render_demo_page(jobs, has_real_data, status),
        title="Demo",
        url_path="demo",
    )
    pages["market"] = st.Page(
        lambda: render_market_overview_page(
            jobs,
            data_source,
            has_real_data,
            status,
        ),
        title="Market Overview",
        url_path="market-overview",
    )
    pages["methodology"] = st.Page(
        render_methodology_page,
        title="Methodology",
        url_path="methodology",
    )

    page = st.navigation(
        [pages["home"], pages["demo"], pages["market"], pages["methodology"]],
        position="hidden",
    )
    render_app_sidebar(jobs, data_source, has_real_data, status, pages)
    page.run()


if __name__ == "__main__":
    main()
