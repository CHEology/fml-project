from __future__ import annotations

import streamlit as st


def initialize_session_state() -> None:
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "resume_source" not in st.session_state:
        st.session_state.resume_source = "Empty canvas"
    if "uploaded_resume_text" not in st.session_state:
        st.session_state.uploaded_resume_text = ""
    if "uploaded_resume_source" not in st.session_state:
        st.session_state.uploaded_resume_source = "Uploaded resume"
    if "pasted_resume_text" not in st.session_state:
        st.session_state.pasted_resume_text = ""
    if "imported_profile_text" not in st.session_state:
        st.session_state.imported_profile_text = ""
    if "imported_profile_source" not in st.session_state:
        st.session_state.imported_profile_source = "Imported public webpage"
    if "sample_resume_text" not in st.session_state:
        st.session_state.sample_resume_text = ""
    if "sample_resume_source" not in st.session_state:
        st.session_state.sample_resume_source = "Sample resume"
    if "demo_input_method" not in st.session_state:
        st.session_state.demo_input_method = "Upload a PDF or TXT resume"
    if "public_profile_url" not in st.session_state:
        st.session_state.public_profile_url = ""
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "Lavender"
    if "assessment" not in st.session_state:
        st.session_state.assessment = None
    if "sample_resume_index" not in st.session_state:
        st.session_state.sample_resume_index = None
    if "pending_analysis" not in st.session_state:
        st.session_state.pending_analysis = False
    if "demo_stage" not in st.session_state:
        st.session_state.demo_stage = "input"
    if "demo_scroll_to_top" not in st.session_state:
        st.session_state.demo_scroll_to_top = False
    if "demo_selected_action" not in st.session_state:
        st.session_state.demo_selected_action = "Improve my salary"
    if "demo_target_cluster_id" not in st.session_state:
        st.session_state.demo_target_cluster_id = None
    if "validation_override" not in st.session_state:
        st.session_state.validation_override = False
