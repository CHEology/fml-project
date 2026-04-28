import re

with open("app/app.py", "r") as f:
    content = f.read()

main_code = """def main() -> None:
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "resume_source" not in st.session_state:
        st.session_state.resume_source = "Empty canvas"
    if "public_profile_url" not in st.session_state:
        st.session_state.public_profile_url = ""
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "Light"

    inject_styles(st.session_state.theme_name)
    jobs, data_source, has_real_data = load_jobs()
    status = artifact_status()

    with st.sidebar:
        st.markdown("## ResuMatch")
        st.caption("Resume market analysis and role matching")
        theme_choice = st.radio(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme_name == "Light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        if theme_choice != st.session_state.theme_name:
            st.session_state.theme_name = theme_choice
            st.rerun()

        source_path = Path(data_source)
        source_label = source_path.name if source_path.suffix else data_source
        source_parent = str(source_path.parent) if source_path.suffix else ""
        source_detail = (
            f'<div class="sidebar-source-path">{escape(source_parent)}</div>'
            if source_parent and source_parent != "."
            else ""
        )
        st.markdown(
            f'''
            <div class="info-card" style="padding: 0.8rem;">
                <div class="info-title" style="font-size: 0.9rem;">Data source</div>
                <div style="font-size: 0.85rem;"><strong>{escape(source_label)}</strong></div>
                {source_detail}
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.write("")

        with st.expander("Data readiness", expanded=False):
            for item in status:
                flag = "Ready" if item["ready"] else "Missing"
                st.write(f"{flag}: `{item['path']}`")

        st.caption(linkedin_dataset_note(has_real_data))

    st.markdown(
        '''
        <div class="hero">
            <div class="eyebrow">Resume market intelligence</div>
            <h1>Understand role fit, salary range, and market position.</h1>
            <p>
                Upload or paste a resume to compare it with salary-bearing LinkedIn roles,
                review relevant opportunities, and identify practical ways to strengthen
                the candidate profile.
            </p>
            <div class="pill-row">
                <span class="pill">Resume profile</span>
                <span class="pill">Role matching</span>
                <span class="pill">Salary range</span>
                <span class="pill">Market position</span>
                <span class="pill">Profile guidance</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Jobs loaded", f"{len(jobs):,}", "local catalog size")
    with metric_cols[1]:
        median_salary = pd.to_numeric(
            jobs.get("salary_annual"), errors="coerce"
        ).dropna()
        render_metric_card(
            "Median salary",
            fmt_money(median_salary.median() if len(median_salary) else None),
            "from current dataset",
        )
    with metric_cols[2]:
        ready_count = sum(item["ready"] for item in status)
        render_metric_card(
            "Data readiness", f"{ready_count}/{len(status)}", "available resources"
        )

    launchpad_tab, radar_tab, pipeline_tab = st.tabs(
        ["Resume Analysis", "Market Overview", "Setup"]
    )

    with launchpad_tab:
        left, right = st.columns([1.6, 1.0], gap="large")

        with left:
            st.markdown("## Analyze a candidate profile")
            st.caption("Add candidate information to review market positioning.")
            
            uploader = st.file_uploader(
                "Drag and drop resume here (PDF or TXT, up to 200MB)", type=["pdf", "txt"]
            )
            if uploader is not None:
                parsed = extract_uploaded_text(uploader)
                if parsed:
                    st.session_state.resume_text = parsed
                    st.session_state.resume_source = f"Uploaded file: {uploader.name}"
                else:
                    st.warning(
                        "Could not extract text from the uploaded file. Paste the resume text below instead."
                    )

            url_col, import_col = st.columns([0.76, 0.24], gap="small")
            with url_col:
                public_profile_url = st.text_input(
                    "Public profile or portfolio URL",
                    value=st.session_state.public_profile_url,
                    placeholder="https://portfolio.example.com/about",
                )
                st.session_state.public_profile_url = public_profile_url
            with import_col:
                st.write("")
                st.write("")
                import_clicked = st.button("Import page", width="stretch")

            if import_clicked:
                try:
                    with st.spinner("Importing public page text..."):
                        imported_text, imported_host = fetch_public_webpage_text(
                            st.session_state.public_profile_url
                        )
                    st.session_state.resume_text = imported_text
                    st.session_state.resume_source = (
                        f"Imported public webpage: {imported_host}"
                    )
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception:
                    st.warning(
                        "Could not import that page. Try another public URL or paste the resume text directly."
                    )

            st.caption(
                "Public webpage import is intended for generic portfolio or resume pages. LinkedIn pages are not imported here; paste the visible profile text or use an approved API flow instead."
            )

            st.session_state.resume_text = st.text_area(
                "Paste resume text",
                value=st.session_state.resume_text,
                height=280,
                placeholder="Paste a resume, portfolio bio, or achievement summary here...",
            )
            word_count = len(st.session_state.resume_text.split())
            st.caption(f"Word count: {word_count}")

            preview_text = st.session_state.resume_text.strip() or SAMPLE_RESUME
            preview_profile = detect_profile(preview_text)

            pref_a, pref_b, pref_c = st.columns([1, 1, 1], gap="medium")
            with pref_a:
                preferred_location = st.selectbox(
                    "Preferred location",
                    ["Anywhere", "NY", "CA", "TX", "WA", "MA", "IL"],
                )
            with pref_b:
                seniority_level = st.selectbox(
                    "Seniority level", list(SENIORITY_MULTIPLIER)
                )
            with pref_c:
                remote_only = st.toggle("Remote only", value=False)
            st.caption(
                f"Detected focus: {preview_profile['track']}. The app infers this from the resume instead of asking you to choose a track."
            )

            sec_a, sec_b = st.columns(2)
            with sec_a:
                if st.button("Load sample resume", width="stretch"):
                    st.session_state.resume_text = SAMPLE_RESUME
                    st.session_state.resume_source = "Built-in sample resume"
                    st.rerun()
            with sec_b:
                if st.button("Generate sample profile", width="stretch"):
                    st.session_state.resume_text = generate_sample_profile(
                        preview_profile["track"],
                        seniority_level,
                        preferred_location,
                        jobs,
                    )
                    st.session_state.resume_source = (
                        f"Generated {preview_profile['track']} sample profile"
                    )
                    st.rerun()

            st.write("")
            action_a, action_b = st.columns(2)
            with action_a:
                analyze_clicked = st.button(
                    "Analyze profile", type="primary", width="stretch"
                )
            with action_b:
                if st.button("Clear", width="stretch"):
                    st.session_state.resume_text = ""
                    st.session_state.resume_source = "Empty canvas"
                    st.rerun()

        with right:
            st.markdown("## Candidate snapshot")
            st.caption("Preview the profile signals.")
            
            if not st.session_state.resume_text.strip():
                st.info("Add a resume to generate a candidate snapshot.")
            else:
                preview_structure = resume_structure(preview_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    render_signal_card(
                        "Detected focus",
                        preview_profile["track"],
                        "Inferred from resume language and market evidence.",
                    )
                with col2:
                    render_signal_card(
                        "Seniority",
                        preview_profile["seniority"],
                        "Detected level from titles, wins, and tone.",
                    )
                col3, col4 = st.columns(2)
                with col3:
                    render_signal_card(
                        "Sections",
                        f"{len(preview_structure['found_sections'])}/{len(SECTION_ALIASES)}",
                        "Structured resumes score better.",
                    )
                with col4:
                    render_signal_card(
                        "Data mode",
                        "Live data" if has_real_data else "Sample data",
                        "Uses the local LinkedIn job catalog.",
                    )

                profile_track_html = escape(str(preview_profile["track"]))
                profile_confidence_html = escape(str(preview_profile["confidence"]))
                st.markdown(
                    f'<div class="metric-card" style="margin-top:0.75rem;"><div class="metric-label">Profile read</div><div class="signal-copy">The current resume reads as a <strong>{profile_track_html}</strong> profile with approximately <strong>{profile_confidence_html}%</strong> confidence. Location and seniority preferences guide the market comparison.</div></div>',
                    unsafe_allow_html=True,
                )

                resume_source_html = escape(str(st.session_state.resume_source))
                word_count_html = escape(str(preview_structure["word_count"]))
                bullet_count_html = escape(str(preview_structure["bullet_count"]))
                link_count_html = escape(str(preview_structure["link_count"]))
                st.markdown(
                    f'''
                    <div class="metric-card" style="margin-top:0.75rem;">
                        <div class="metric-label">Resume source</div>
                        <div class="signal-copy">
                            <strong>{resume_source_html}</strong><br/>
                            {word_count_html} words • {bullet_count_html} bullets • {link_count_html} links detected
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-label" style="margin-top:0.9rem;">Detected strengths</div>',
                    unsafe_allow_html=True,
                )
                present_skills = preview_profile["skills_present"] or [
                    "Generalist profile",
                    "Cross-functional communication",
                ]
                st.markdown(
                    '<div class="chip-cloud">'
                    + "".join(
                        f'<span class="mini-chip">{escape(str(skill))}</span>'
                        for skill in present_skills[:6]
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-label" style="margin-top:0.9rem;">Resume organization</div>',
                    unsafe_allow_html=True,
                )
                structure_chips = preview_structure["found_sections"] or [
                    "No formal sections detected"
                ]
                st.markdown(
                    '<div class="chip-cloud">'
                    + "".join(
                        f'<span class="mini-chip">{escape(str(section))}</span>'
                        for section in structure_chips
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if preview_structure["missing_sections"]:
                    st.caption(
                        "Missing sections: "
                        + ", ".join(preview_structure["missing_sections"])
                    )

                st.markdown(
                    '<div class="section-label" style="margin-top:0.9rem;">Market data</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'''
                    <span class="status-pill {"ready" if has_real_data else "missing"}">{"LinkedIn job catalog" if has_real_data else "Sample role catalog"}</span>
                    ''',
                    unsafe_allow_html=True,
                )
                st.caption(linkedin_dataset_note(has_real_data))
                
                st.markdown(
                    f'<div class="metric-card" style="margin-top:0.75rem;"><div class="metric-label">Next steps</div><div class="signal-copy">Click "Analyze profile" to see salary ranges and missing market signals.</div></div>',
                    unsafe_allow_html=True,
                )

        if analyze_clicked and st.session_state.resume_text.strip():
            if not has_real_data:
                st.error(
                    "The job catalog is not ready. Run preprocessing before using this analysis path."
                )
                return
            if not artifacts_ready(status, "retrieval"):
                st.error(
                    "Role-matching data is not ready. Build the job index and metadata first."
                )
                return

            try:
                with st.spinner("Matching resume to relevant roles..."):
                    retriever, encoder = load_retriever_resource()
                    resume_embedding = encode_resume(
                        encoder, st.session_state.resume_text
                    )
                    matches = retrieve_matches(
                        retriever,
                        jobs,
                        resume_embedding,
                        preferred_location=preferred_location,
                        remote_only=remote_only,
                        top_k=6,
                    )

                neural_band = None
                if salary_artifacts_ready(PROJECT_ROOT):
                    with st.spinner("Calculating salary reference..."):
                        salary_model, salary_scaler = load_salary_resource()
                        neural_band = salary_band_from_model(
                            salary_model, resume_embedding, salary_scaler
                        )

                occupation_match = None
                bls_band = None
                occupation_router = load_occupation_resource(encoder)
                wage_table = load_wage_resource()
                if occupation_router is not None:
                    soc_matches = occupation_router.route(resume_embedding, k=1)
                    if soc_matches:
                        occupation_match = soc_matches[0]
                        if wage_table is not None:
                            bls_band = wage_table.lookup(occupation_match.soc_code)

                band = hybrid_salary_band(
                    matches,
                    neural_band=neural_band,
                    bls_band=bls_band,
                    occupation_match=occupation_match,
                )

                cluster = None
                if artifacts_ready(status, "clustering"):
                    with st.spinner("Finding market segment..."):
                        kmeans_model, _, cluster_labels = load_cluster_resource()
                        cluster = cluster_position(
                            kmeans_model, cluster_labels, resume_embedding
                        )

                missing_terms = feedback_terms(
                    st.session_state.resume_text, matches, cluster
                )
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.error(f"Analysis failed: {exc}")
                return

            st.write("")
            render_panel_banner(
                "Market Readout",
                "Salary range",
                "This estimate is anchored to the salary data from the most relevant roles.",
            )
            with st.container(border=True):
                if band is not None:
                    render_salary_band(band)
                else:
                    st.warning(
                        "No salary evidence is available from retrieved jobs, BLS, or the neural model."
                    )

            st.write("")
            render_panel_banner(
                "Profile Signal",
                "Market segment and match evidence",
                "The app infers the closest market segment and shows the strength of the role matches in one row.",
            )
            with st.container(border=True):
                signal_cols = st.columns(4, gap="small")
                with signal_cols[0]:
                    if cluster is not None:
                        render_signal_card(
                            "Segment",
                            str(cluster["cluster_id"]),
                            cluster["label"],
                        )
                    else:
                        render_signal_card(
                            "Segment",
                            "Unavailable",
                            "Market segment data is not available.",
                        )
                with signal_cols[1]:
                    if cluster is not None:
                        alignment = max(
                            0, min(100, int(round(100 / (1 + cluster["distance"]))))
                        )
                        render_signal_card(
                            "Alignment",
                            f"{alignment}%",
                            "Relative closeness to this segment.",
                        )
                    else:
                        render_signal_card(
                            "Alignment", "N/A", "Build segment data first."
                        )
                with signal_cols[2]:
                    if matches.empty:
                        render_signal_card(
                            "Top similarity", "N/A", "No matching roles passed filters."
                        )
                    else:
                        render_signal_card(
                            "Top similarity",
                            f"{matches.iloc[0]['similarity'] * 100:.0f}%",
                            "Best role match after filters.",
                        )
                with signal_cols[3]:
                    render_signal_card(
                        "Retrieved roles",
                        f"{len(matches):,}",
                        "Roles shown after filters.",
                    )
                if cluster is not None:
                    st.markdown(
                        '<div class="chip-cloud">'
                        + "".join(
                            f'<span class="mini-chip">{escape(str(term))}</span>'
                            for term in cluster["top_terms"][:8]
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            st.write("")
            render_panel_banner(
                "Opportunity Lens",
                "Gaps to close",
                "Missing terms from the strongest matching roles and market segment.",
            )
            with st.container(border=True):
                if missing_terms:
                    st.markdown(
                        '<div class="chip-cloud">'
                        + "".join(
                            f'<span class="mini-chip">Add stronger evidence for {escape(str(item))}</span>'
                            for item in missing_terms
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "The strongest matching roles are already well reflected in the resume text."
                    )

            st.write("")
            render_panel_banner(
                "Match Board",
                "Top matching roles",
                "These roles are ordered by similarity to the resume after the selected filters.",
            )
            if matches.empty:
                st.info(
                    "No roles matched the selected filters. Try Anywhere or disable Remote only."
                )
            else:
                card_cols = st.columns(2, gap="medium")
                for index, (_, row) in enumerate(matches.iterrows()):
                    with card_cols[index % 2]:
                        render_job_card(row)
        elif analyze_clicked:
            st.warning(
                "Paste a resume or load the sample resume before running the analysis."
            )

    with radar_tab:
        render_panel_banner(
            "Market Radar",
            "Where the current feed is concentrated",
            "A quick view of geography, seniority, and salary distribution in the available job catalog.",
        )
        display_jobs = jobs.copy()
        display_jobs["salary_annual"] = pd.to_numeric(
            display_jobs.get("salary_annual"), errors="coerce"
        )

        left, right = st.columns([0.52, 0.48], gap="large")
        with left, st.container(border=True):
            st.markdown("**Top locations**")
            location_counts = (
                display_jobs["location"].fillna("Unknown").value_counts().head(8)
            )
            st.bar_chart(location_counts)

            st.markdown("**Experience mix**")
            exp_counts = (
                display_jobs["experience_level"]
                .fillna("Unknown")
                .value_counts()
                .head(8)
            )
            st.bar_chart(exp_counts)

            if artifacts_ready(status, "clustering"):
                _, assignments, cluster_labels = load_cluster_resource()
                cluster_names = [
                    cluster_labels.get(str(cluster_id), {}).get(
                        "label", f"Cluster {cluster_id}"
                    )
                    for cluster_id in assignments
                ]
                st.markdown("**Market segments**")
                st.bar_chart(pd.Series(cluster_names).value_counts())

        with right, st.container(border=True):
            st.markdown("**Salary sample**")
            salary_view = display_jobs[
                ["title", "company_name", "location", "salary_annual"]
            ].copy()
            salary_view = salary_view.sort_values(
                "salary_annual", ascending=False
            ).head(12)
            st.dataframe(salary_view, width="stretch", hide_index=True)

            st.markdown("**Dataset notes**")
            if has_real_data:
                st.success(f"Loaded real project data from `{data_source}`.")
            else:
                st.info("Using sample roles until the local job catalog is prepared.")

    with pipeline_tab:
        render_panel_banner(
            "Setup",
            "Data readiness",
            "This view shows which local data resources are available for the full analysis workflow.",
        )
        status_frame = pd.DataFrame(
            [
                {
                    "Resource": item["label"],
                    "Status": "Ready" if item["ready"] else "Missing",
                    "Path": item["path"],
                }
                for item in status
            ]
        )
        st.dataframe(status_frame, width="stretch", hide_index=True)

        st.write("")
        with st.container(border=True):
            st.markdown("**Recommended next commands**")
            st.code(
                "\\n".join(
                    [
                        "uv run python scripts/preprocess_data.py",
                        "uv run python scripts/build_index.py",
                        "uv run python scripts/train_salary_model.py --embeddings models/job_embeddings.npy --salaries data/processed/salaries.npy --output models/salary_model.pt",
                        "uv run python scripts/train_resume_salary_model.py --resumes data/eval/synthetic_resumes.parquet --out models/resume_salary_model.pt",
                        "uv run python scripts/load_onet_skills.py --download",
                        "uv run python scripts/load_bls_oews.py --download",
                        "uv run python scripts/build_clusters.py",
                        "uv run streamlit run app/app.py",
                    ]
                ),
                language="bash",
            )
            st.caption(
                "Retrieved salaries power the primary range; resume-side salary and BLS resources add fallback/reference evidence."
            )


if __name__ == "__main__":
    main()
"""

match = re.search(r'def main\(\) -> None:.*', content, re.DOTALL)
if match:
    new_content = content[:match.start()] + main_code
    with open("app/app.py", "w") as f:
        f.write(new_content)
    print("Replaced main function")
else:
    print("Could not find main function")
