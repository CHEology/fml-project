from __future__ import annotations

import sys
from pathlib import Path


import streamlit as st

# Eager-load torch on the main thread BEFORE streamlit/numpy/pandas. On Windows,
# importing torch later from Streamlit's script-runner thread (after MKL/OpenMP
# have been pulled in by numpy/pandas) fails with WinError 1114 in c10.dll.
# Loading it here makes the subsequent thread-side import a cached no-op.
import torch  # noqa: E402, F401, I001

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Runtime and styles
from app.runtime.cache import artifact_status, load_jobs
from app.styles import inject_styles

# Component imports for the router
from app.pages.home import render_home_page
from app.pages.market import render_market_overview_page
from app.pages.demo import render_demo_page
from app.components.methodology import render_methodology_page
from app.components.sidebar import render_app_sidebar


def main() -> None:
    st.set_page_config(
        page_title="ResuMatch",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global styles from the modular style system
    inject_styles("Lavender")

    # Initialize session state for Demo and common components
    if "demo_stage" not in st.session_state:
        st.session_state.demo_stage = "input"
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "resume_source" not in st.session_state:
        st.session_state.resume_source = "Empty canvas"
    if "public_profile_url" not in st.session_state:
        st.session_state.public_profile_url = ""
    if "uploaded_resume_text" not in st.session_state:
        st.session_state.uploaded_resume_text = ""
    if "pasted_resume_text" not in st.session_state:
        st.session_state.pasted_resume_text = ""
    if "imported_profile_text" not in st.session_state:
        st.session_state.imported_profile_text = ""
    if "sample_resume_text" not in st.session_state:
        st.session_state.sample_resume_text = ""
    if "sample_resume_index" not in st.session_state:
        st.session_state.sample_resume_index = None
    if "assessment" not in st.session_state:
        st.session_state.assessment = None
    if "validation_override" not in st.session_state:
        st.session_state.validation_override = False
    if "demo_input_method" not in st.session_state:
        st.session_state.demo_input_method = "Upload a PDF or TXT resume"
    if "demo_selected_action" not in st.session_state:
        st.session_state.demo_selected_action = "Improve my salary"
    if "pending_analysis" not in st.session_state:
        st.session_state.pending_analysis = False
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "Lavender"
    if "demo_target_cluster_id" not in st.session_state:
        st.session_state.demo_target_cluster_id = None

    # Shared data loading with caching
    jobs, data_source, has_real_data = load_jobs()
    status = artifact_status()

    # Define Navigation Structure
    # Names match the hrefs in home.py for seamless transitions
    # Define Navigation Structure
    home_page = st.Page(
        lambda: render_home_page(pages, jobs, data_source, has_real_data, status),
        title="Home",
        icon="🏠",
        url_path="home",
        default=True,
    )
    demo_page = st.Page(
        lambda: render_demo_page(jobs, has_real_data, status),
        title="Demo",
        icon="⚡",
        url_path="demo",
    )
    market_page = st.Page(
        lambda: render_market_overview_page(jobs, data_source, has_real_data, status),
        title="Market Overview",
        icon="📈",
        url_path="market-overview",
    )
    methodology_page = st.Page(
        render_methodology_page,
        title="Methodology",
        icon="📚",
        url_path="methodology",
    )

    pages = {
        "home": home_page,
        "demo": demo_page,
        "market": market_page,
        "methodology": methodology_page,
    }

    # Use a flat navigation structure
    # position="hidden" allows us to render the links manually in our custom sidebar
    pg = st.navigation(
        [home_page, demo_page, market_page, methodology_page],
        position="hidden",
    )

    # Sidebar Global Components (matches the requested design)
    render_app_sidebar(jobs, data_source, has_real_data, status, pages)

    # Run the selected page
    pg.run()


if __name__ == "__main__":
    main()
