from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
from app.components.job_results import render_job_results
from app.components.quality import render_profile_quality_section
from app.components.resume_revision import (
    render_resume_revision_panel,
    reset_revision_state,
)
from app.components.resume_upload import (
    extract_uploaded_text,
    fetch_public_webpage_text,
)
from app.components.salary_chart import (
    render_cluster_salary_distribution,
    render_salary_band,
)
from app.config import MASCOT_PATH, PROJECT_ROOT
from app.demo.components import (
    info_dot,
    render_demo_floating_nav,
    render_demo_section_header,
    render_demo_signal_card,
)
from app.demo.sample_data import SAMPLE_RESUME_SOURCE_HELP, SAMPLE_RESUME_SOURCE_SUMMARY
from app.demo.samples import random_premade_sample_resume
from app.demo.snapshot import (
    capability_evidence_html,
    encoded_image_data_uri,
    focus_evidence_html,
    render_scroll_to_top,
    seniority_evidence_html,
    seniority_ladder_html,
)
from app.runtime import ml as runtime
from app.runtime.cache import (
    apply_public_ats_fit,
    artifacts_ready,
    cluster_position,
    encode_resume,
    feedback_terms,
    hybrid_salary_band,
    learned_quality_signal,
    load_cluster_resource,
    load_job_embedding_resource,
    load_occupation_resource,
    load_public_assessment_resource,
    load_quality_resource,
    load_retriever_resource,
    load_salary_resource,
    load_wage_resource,
    public_resume_signals,
    retrieve_matches,
    salary_artifacts_ready,
    salary_band_from_model,
    validate_resume,
)
from ml.resume_assessment import (
    add_salary_evidence_note,
    apply_capability_adjustment,
    apply_quality_discount,
    assess_capability_tier,
    assess_quality,
    detect_profile,
    enhance_structure_with_public_sections,
    extract_work_history,
    resume_structure,
    score_projects,
    seniority_filtered_salary_matches,
)
from ml.resume_assessment.taxonomy import SECTION_ALIASES


def render_demo_page(
    jobs: pd.DataFrame,
    has_real_data: bool,
    status: list[dict[str, str | bool]],
) -> None:
    valid_stages = {"input", "results"}
    if st.session_state.demo_stage not in valid_stages:
        st.session_state.demo_stage = "input"

    def restart_demo() -> None:
        st.session_state.resume_text = ""
        st.session_state.resume_source = "Empty canvas"
        st.session_state.public_profile_url = ""
        st.session_state.uploaded_resume_text = ""
        st.session_state.uploaded_resume_source = "Uploaded resume"
        st.session_state.pasted_resume_text = ""
        st.session_state.imported_profile_text = ""
        st.session_state.imported_profile_source = "Imported public webpage"
        st.session_state.sample_resume_text = ""
        st.session_state.sample_resume_source = "Sample resume"
        st.session_state.demo_input_method = "Upload a PDF or TXT resume"
        st.session_state.assessment = None
        st.session_state.pending_analysis = False
        st.session_state.demo_scroll_to_top = False
        st.session_state.demo_stage = "input"
        reset_revision_state()

    current_text = st.session_state.resume_text.strip()
    assessment = st.session_state.get("assessment")
    assessment_ready = (
        assessment is not None
        and bool(current_text)
        and str(assessment.get("resume_text", "")).strip() == current_text
    )
    if st.session_state.demo_stage != "input" and not assessment_ready:
        st.session_state.demo_stage = "input"

    if st.session_state.demo_stage == "input":
        mascot_uri = encoded_image_data_uri(MASCOT_PATH)
        mascot_html = (
            f'<img src="{mascot_uri}" alt="ResuMatch mascot" />' if mascot_uri else ""
        )

        def render_method_status(title: str, copy: str, method_text: str) -> None:
            word_count = len(method_text.split())
            st.markdown(
                f"""
                <div class="demo-method-status">
                    <div><strong>{escape(title)}</strong><br/><span>{escape(copy)}</span></div>
                    <div class="demo-word-count">{word_count:,} words loaded</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        analyze_clicked = False

        def queue_profile_analysis(
            text_key: str, source_key: str | None = None
        ) -> None:
            st.session_state.resume_text = st.session_state.get(text_key, "")
            if source_key is None:
                st.session_state.resume_source = "Pasted resume/profile text"
            else:
                st.session_state.resume_source = st.session_state.get(source_key, "")
            st.session_state.pending_analysis = True

        st.markdown(
            f"""
            <div class="demo-intake-hero">
                <div>
                    <h1>Add a resume or profile</h1>
                    <p class="demo-intake-copy">
                        Choose between uploading a resume, pasting profile text,
                        importing a public portfolio page, or using a sample to see the analysis flow.
                    </p>
                </div>
                <div class="demo-intake-mascot">
                    {mascot_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container():
            input_methods = [
                "Upload a PDF or TXT resume",
                "Paste resume/profile text",
                "Import a public portfolio or resume page",
                "Use a random sample resume",
            ]
            st.markdown(
                """
                <div class="demo-accordion-intro">
                    <div>
                        <h2>Choose an input method</h2>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            selected_method = st.radio(
                "Input method",
                input_methods,
                key="demo_input_method",
                horizontal=True,
                label_visibility="collapsed",
            )

            with st.expander(selected_method, expanded=True):
                if selected_method == "Upload a PDF or TXT resume":
                    uploader = st.file_uploader(
                        "Upload resume file",
                        type=["pdf", "txt"],
                        accept_multiple_files=False,
                        help="PDF and TXT files are supported.",
                    )
                    if uploader is not None:
                        parsed = extract_uploaded_text(uploader)
                        if parsed:
                            st.session_state.uploaded_resume_text = parsed
                            st.session_state.uploaded_resume_source = (
                                f"Uploaded file: {uploader.name}"
                            )
                        else:
                            st.warning(
                                "Could not extract text from the uploaded file. Paste the resume text instead."
                            )
                    st.text_area(
                        "Uploaded resume text",
                        value=st.session_state.uploaded_resume_text,
                        height=260,
                        disabled=True,
                        placeholder="Upload a PDF or TXT resume to preview the extracted text here.",
                    )
                    current_text = st.session_state.uploaded_resume_text.strip()
                    render_method_status(
                        "Upload input",
                        "Upload a PDF or TXT file, then run analysis from this panel.",
                        st.session_state.uploaded_resume_text,
                    )
                    st.button(
                        "Run profile analysis",
                        type="primary",
                        width="stretch",
                        disabled=not bool(current_text),
                        key="analyze_upload_resume",
                        on_click=queue_profile_analysis,
                        args=("uploaded_resume_text", "uploaded_resume_source"),
                    )

                elif selected_method == "Paste resume/profile text":
                    st.session_state.pasted_resume_text = st.text_area(
                        "Paste or edit resume/profile text",
                        value=st.session_state.pasted_resume_text,
                        height=260,
                        placeholder="Paste a resume, portfolio bio, or achievement summary here...",
                    )
                    current_text = st.session_state.pasted_resume_text.strip()
                    render_method_status(
                        "Paste input",
                        "Only the text currently in this box will be used for analysis.",
                        st.session_state.pasted_resume_text,
                    )
                    st.button(
                        "Run profile analysis",
                        type="primary",
                        width="stretch",
                        disabled=not bool(current_text),
                        key="analyze_pasted_resume",
                        on_click=queue_profile_analysis,
                        args=("pasted_resume_text", None),
                    )

                elif selected_method == "Import a public portfolio or resume page":
                    st.markdown(
                        '<p class="demo-method-note">Import a public portfolio, personal site, or resume page. Private sites and LinkedIn pages are not imported.</p>',
                        unsafe_allow_html=True,
                    )
                    public_profile_url = st.text_input(
                        "Public profile or portfolio URL",
                        value=st.session_state.public_profile_url,
                        placeholder="https://portfolio.example.com/about",
                    )
                    st.session_state.public_profile_url = public_profile_url
                    import_clicked = st.button("Import page", width="stretch")

                    if import_clicked:
                        try:
                            with st.spinner("Importing public page text..."):
                                imported_text, imported_host = (
                                    fetch_public_webpage_text(
                                        st.session_state.public_profile_url
                                    )
                                )
                            st.session_state.imported_profile_text = imported_text
                            st.session_state.imported_profile_source = (
                                f"Imported public webpage: {imported_host}"
                            )
                            st.rerun()
                        except ValueError as exc:
                            st.warning(str(exc))
                        except Exception:
                            st.warning(
                                "Could not import that page. Try another public URL or paste the resume text directly."
                            )
                    st.text_area(
                        "Imported page text",
                        value=st.session_state.imported_profile_text,
                        height=260,
                        disabled=True,
                        placeholder="Import a public portfolio or resume page to preview the extracted text here.",
                    )
                    current_text = st.session_state.imported_profile_text.strip()
                    render_method_status(
                        "Imported page input",
                        "",
                        st.session_state.imported_profile_text,
                    )
                    st.button(
                        "Run profile analysis",
                        type="primary",
                        width="stretch",
                        disabled=not bool(current_text),
                        key="analyze_imported_profile",
                        on_click=queue_profile_analysis,
                        args=("imported_profile_text", "imported_profile_source"),
                    )

                else:
                    st.info(SAMPLE_RESUME_SOURCE_SUMMARY)
                    with st.expander("Read more about sample resume generation"):
                        st.markdown(SAMPLE_RESUME_SOURCE_HELP)
                    if st.button("Load random sample resume", width="stretch"):
                        sample_text, sample_source, sample_index = (
                            random_premade_sample_resume(
                                jobs,
                                st.session_state.sample_resume_index,
                            )
                        )
                        st.session_state.sample_resume_text = sample_text
                        st.session_state.sample_resume_source = sample_source
                        st.session_state.sample_resume_index = sample_index
                        st.session_state.resume_text = sample_text
                        st.session_state.resume_source = sample_source
                        st.session_state.assessment = None
                        reset_revision_state()
                        st.rerun()
                    st.text_area(
                        "Random sample resume text",
                        value=st.session_state.sample_resume_text,
                        height=260,
                        disabled=True,
                        placeholder="Load a random sample resume to preview it here.",
                    )
                    current_text = st.session_state.sample_resume_text.strip()
                    render_method_status(
                        "Sample resume input",
                        "",
                        st.session_state.sample_resume_text,
                    )
                    st.button(
                        "Run profile analysis",
                        type="primary",
                        width="stretch",
                        disabled=not bool(current_text),
                        key="analyze_sample_resume",
                        on_click=queue_profile_analysis,
                        args=("sample_resume_text", "sample_resume_source"),
                    )

        draft_text = current_text.strip()
        analyze_clicked = bool(st.session_state.get("pending_analysis", False))
        if analyze_clicked:
            st.session_state.pending_analysis = False
        current_text = st.session_state.resume_text.strip()

        with st.container(key="resume-revision-input-section"):
            render_demo_section_header(
                "Generate revised resume",
                "Draft a stronger full resume before or after analysis.",
                "This rewrite uses the current resume text and targets the weakest quality dimensions with clearer structure, stronger specificity, and placeholder metrics for you to replace with real results.",
            )
            render_resume_revision_panel(
                draft_text,
                st.session_state.get("assessment"),
                key_prefix="demo_input",
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

            resume_text_now = st.session_state.resume_text.strip()
            try:
                with st.spinner("Reviewing resume content..."):
                    public_models = load_public_assessment_resource()

                    # Validation check
                    validation = validate_resume(public_models, resume_text_now)
                    if not validation["is_resume"]:
                        reasons_str = ", ".join(validation["reasons"])
                        st.error(
                            f"This text does not appear to be a valid resume: {reasons_str}"
                        )
                        return

                    public_signals = public_resume_signals(
                        public_models, resume_text_now
                    )
                    structure = resume_structure(resume_text_now)
                    structure = enhance_structure_with_public_sections(
                        structure, public_signals
                    )
                    work_history = extract_work_history(resume_text_now)
                    projects = score_projects(resume_text_now)
                    profile = detect_profile(
                        resume_text_now, work_history, projects, structure
                    )
                    quality = assess_quality(
                        resume_text_now,
                        profile,
                        structure,
                        work_history,
                        projects,
                        public_signals,
                    )
                    capability = assess_capability_tier(
                        resume_text_now,
                        profile,
                        quality,
                        work_history,
                        projects,
                        public_signals,
                    )

                with st.spinner("Matching resume to relevant roles..."):
                    retriever, encoder = load_retriever_resource()
                    resume_embedding = encode_resume(encoder, resume_text_now)
                    matches = retrieve_matches(
                        retriever,
                        jobs,
                        resume_embedding,
                        target_seniority=profile["seniority"],
                        top_k=6,
                    )
                    matches = apply_public_ats_fit(
                        public_models,
                        resume_text_now,
                        matches,
                    )

                learned_quality = None
                if artifacts_ready(status, "quality"):
                    with st.spinner("Running learned quality cross-check..."):
                        quality_model, quality_scaler = load_quality_resource()
                        learned_quality = learned_quality_signal(
                            quality_model,
                            resume_embedding,
                            quality_scaler,
                        )

                neural_band = None
                if salary_artifacts_ready(PROJECT_ROOT):
                    with st.spinner("Calculating salary reference..."):
                        salary_model, salary_scaler, salary_feature_metadata = (
                            load_salary_resource()
                        )
                        neural_band = salary_band_from_model(
                            salary_model,
                            resume_embedding,
                            salary_scaler,
                            salary_feature_metadata,
                            resume_features={
                                "experience_level_ordinal": float(
                                    runtime.SENIORITY_RANKS.get(profile["seniority"], 2)
                                ),
                                "work_type_remote": (
                                    1.0 if "remote" in resume_text_now.lower() else 0.0
                                ),
                            },
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

                salary_matches, seniority_salary_note = (
                    seniority_filtered_salary_matches(matches)
                )
                band = hybrid_salary_band(
                    salary_matches,
                    neural_band=neural_band,
                    bls_band=bls_band,
                    occupation_match=occupation_match,
                )
                band = add_salary_evidence_note(band, seniority_salary_note)
                band = apply_quality_discount(band, quality)
                band = apply_capability_adjustment(band, capability)

                cluster = None
                cluster_assignments = None
                cluster_labels = None
                job_embeddings = None
                if artifacts_ready(status, "clustering"):
                    with st.spinner("Finding market segment..."):
                        kmeans_model, cluster_assignments, cluster_labels = (
                            load_cluster_resource()
                        )
                        cluster = cluster_position(
                            kmeans_model, cluster_labels, resume_embedding
                        )
                        if artifacts_ready(status, "retrieval"):
                            job_embeddings = load_job_embedding_resource()

                missing_terms = feedback_terms(resume_text_now, matches, cluster)
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.error(f"Analysis failed: {exc}")
                return

            st.session_state.assessment = {
                "resume_text": resume_text_now,
                "resume_source": st.session_state.resume_source,
                "profile": profile,
                "structure": structure,
                "work_history": work_history,
                "projects": projects,
                "quality": quality,
                "learned_quality": learned_quality,
                "capability": capability,
                "public_signals": public_signals,
                "matches": matches,
                "salary_matches": salary_matches,
                "band": band,
                "cluster": cluster,
                "cluster_assignments": cluster_assignments,
                "cluster_labels": cluster_labels,
                "job_embeddings": job_embeddings,
                "resume_embedding": resume_embedding,
                "missing_terms": missing_terms,
            }
            st.session_state.demo_stage = "results"
            st.session_state.demo_scroll_to_top = True
            st.rerun()
        elif analyze_clicked:
            st.warning(
                "Paste a resume or load a random sample resume before running the analysis."
            )
        return

    if st.session_state.demo_stage == "results":
        if st.session_state.get("demo_scroll_to_top", False):
            st.markdown(
                '<div id="candidate-snapshot-top"></div>', unsafe_allow_html=True
            )
            render_scroll_to_top()
            st.session_state.demo_scroll_to_top = False

        current_text = st.session_state.resume_text.strip()
        profile = assessment["profile"]
        structure = assessment["structure"]
        quality = assessment["quality"]
        capability = assessment.get("capability") or {}
        public_signals = assessment.get("public_signals")
        learned_quality = assessment.get("learned_quality")
        snapshot_info = (
            "Candidate Snapshot is computed in ml.resume_assessment from resume text parsing, "
            "detect_profile(), extract_work_history(), score_projects(), "
            "assess_quality(), and assess_capability_tier(). It is rule-based and "
            "uses resume wording, detected skills, section structure, job-title "
            "signals, quantified-impact language, and optional public-resume models "
            "from ml/public_assessment.py. Assumption: the uploaded or pasted text is "
            "a truthful candidate profile; this is an evidence summary, not a hiring decision."
        )
        focus_info = (
            "Detected focus comes from ml.resume_assessment.detect_profile(), which scores "
            "track keyword hits against TRACK_KEYWORDS and resume evidence such as "
            "skills, titles, projects, and market examples."
        )
        seniority_info = (
            "Seniority is inferred in ml.resume_assessment from extracted work-history "
            "spans, seniority words in titles, project evidence, and "
            "runtime.SENIORITY_RANKS. It is an estimate when dates or titles are missing."
        )
        capability_info = (
            "Capability tier is computed by ml.resume_assessment.assess_capability_tier(). "
            "It combines quality subscores, detected track-specific skills, project "
            "strength, public-model signals, and high-rigor employer/title/publication "
            "signals into a 0-100 within-level score used for salary adjustment."
        )
        market_info = (
            "Market positioning is built in app/ml_runtime.py and app/components/salary_chart.py. "
            "The salary range combines retrieved-role salary quantiles from FAISS/vector "
            "matching in ml/retrieval.py, optional neural quantile salary estimates from "
            "ml/salary_model.py, optional BLS/O*NET occupation wages via ml/occupation_router.py "
            "and ml/wage_bands.py, then ml.resume_assessment applies quality and capability adjustments. "
            "For the 2D cluster map, jobs are embedded as sentence-transformer vectors; "
            "K-Means groups job embeddings into 8 role-family clusters; cluster labels come from "
            "TF-IDF top terms and common titles; PCA-reduced embedding components are used as "
            "the x/y axes; and the resume is embedded and placed in the nearest centroid cluster. "
            "To keep the app responsive, the plotted job dots are a deterministic random visualization sample "
            "from the projected dataset rather than every posting. "
            "The plotted user marker uses the hybrid band q50 median estimate in hover text for "
            "the candidate's predicted salary and cluster position. "
            "Assumption: this is a matched-market reference, not a guaranteed compensation offer."
        )
        match_info = (
            "Gaps and matching roles combine clustering and retrieval. Market segment uses "
            "K-Means artifacts from scripts/build_clusters.py through ml/clustering.py and "
            "app/ml_runtime.py::cluster_position(). Missing terms come from "
            "app/ml_runtime.py::feedback_terms(), comparing resume language with top matched "
            "roles and cluster terms. Assumption: terms are directional signals to strengthen "
            "evidence, not mandatory requirements."
            "Top matching roles come from app/ml_runtime.py::retrieve_matches(), which encodes "
            "the resume with the loaded embedding model, searches the local FAISS index built "
            "by scripts/build_index.py, filters/scales by inferred seniority, and renders the "
            "top jobs in app/components/job_results.py. Similarity is cosine-like vector "
            "similarity from the local embedding space."
        )

        profile_track_html = escape(str(profile["track"]))
        profile_seniority_html = escape(str(profile["seniority"]))
        profile_confidence_html = escape(str(profile["confidence"]))
        capability_tier_html = escape(str(capability.get("tier", "Competitive")))
        capability_score_html = escape(str(capability.get("score", 0)))
        effect = float(capability.get("salary_effect_pct", 0.0) or 0.0)
        direction = "+" if effect > 0 else ""
        effect_html = escape(f"{direction}{effect:.1f}%")
        found_sections_count = len(structure["found_sections"])
        total_sections_count = len(SECTION_ALIASES)
        focus_evidence = focus_evidence_html(profile)
        seniority_evidence = seniority_evidence_html(
            str(profile["seniority"]), assessment["work_history"]
        )
        capability_evidence = capability_evidence_html(capability, quality)
        seniority_ladder = seniority_ladder_html(str(profile["seniority"]))
        with st.container(key="candidate-snapshot-section"):
            st.markdown(
                f"""
                <div class="snapshot-hero candidate-snapshot-hero">
                    <div class="snapshot-title-row">
                        <h1 class="snapshot-title">Candidate Snapshot</h1>
                    </div>
                    <div class="snapshot-summary">
                        This resume reads as a <strong>{profile_track_html}</strong> profile
                        at the <strong>{profile_seniority_html}</strong> level with about
                        <strong>{profile_confidence_html}%</strong> confidence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Read more about Candidate Snapshot"):
                st.markdown(snapshot_info)
            st.markdown(
                f"""
                <div class="snapshot-highlight-grid">
                    <div class="snapshot-card primary">
                        <div class="snapshot-label">Detected focus{info_dot(focus_info, extra_class="inline-info")}</div>
                        <div class="snapshot-value">{profile_track_html}</div>
                        {focus_evidence}
                    </div>
                    <div class="snapshot-card primary">
                        <div class="snapshot-label">Seniority{info_dot(seniority_info, extra_class="inline-info")}</div>
                        <div class="snapshot-value">{profile_seniority_html}</div>
                        {seniority_evidence}
                        {seniority_ladder}
                    </div>
                    <div class="snapshot-card primary">
                        <div class="snapshot-label">Capability tier{info_dot(capability_info, extra_class="inline-info")}</div>
                        <div class="snapshot-value">{capability_tier_html} ({capability_score_html}/100)</div>
                        <div class="snapshot-copy">Within-level strength; salary effect {effect_html}.</div>
                        {capability_evidence}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        present_skills = profile["skills_present"] or [
            "Generalist profile",
            "Cross-functional communication",
        ]
        structure_chips = structure["found_sections"] or ["No formal sections detected"]
        missing_sections = structure["missing_sections"]
        missing_terms = assessment.get("missing_terms") or []
        render_profile_quality_section(
            quality=quality,
            learned_quality=learned_quality,
            public_signals=public_signals,
            resume_stats={
                "word_count": int(structure["word_count"]),
                "bullet_count": int(structure["bullet_count"]),
                "link_count": int(structure["link_count"]),
                "found_sections_count": found_sections_count,
                "total_sections_count": total_sections_count,
            },
            strengths=[str(skill) for skill in present_skills],
            sections=[str(section) for section in structure_chips],
            missing_sections=[str(section) for section in missing_sections],
            missing_terms=[str(term) for term in missing_terms],
        )

        with st.container(key="resume-revision-results-section"):
            render_demo_section_header(
                "Generate revised resume",
                "Produce a targeted rewrite from the current profile.",
                "This rewrite keeps the current resume's direction but sharpens it around the weakest quality dimensions so you can compare the before/after version directly.",
            )
            render_resume_revision_panel(
                current_text,
                assessment,
                key_prefix="demo_results",
            )

        band = assessment.get("band")
        cluster = assessment.get("cluster")
        cluster_assignments = assessment.get("cluster_assignments")
        cluster_labels = assessment.get("cluster_labels")
        job_embeddings = assessment.get("job_embeddings")
        resume_embedding = assessment.get("resume_embedding")
        matches = assessment.get("matches")
        if matches is not None and not isinstance(matches, pd.DataFrame):
            matches = pd.DataFrame(matches)

        with st.container(key="market-positioning-section"):
            render_demo_section_header(
                "Market Positioning",
                "",
                market_info,
            )
            if band is not None:
                render_salary_band(band)
                render_cluster_salary_distribution(
                    jobs,
                    cluster_assignments,
                    cluster_labels,
                    cluster,
                    band,
                    job_embeddings=job_embeddings,
                    resume_embedding=resume_embedding,
                )
            else:
                st.warning(
                    "No salary evidence is available from retrieved jobs, BLS, or the neural model."
                )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            signal_cols = st.columns(4, gap="small")
            with signal_cols[0]:
                if cluster is not None:
                    render_demo_signal_card(
                        "Segment",
                        str(cluster["cluster_id"]),
                        cluster["label"],
                        "Segment is the nearest K-Means cluster from models/cluster artifacts, loaded through app/ml_runtime.py::load_cluster_artifacts() and assigned by cluster_position().",
                    )
                else:
                    render_demo_signal_card(
                        "Segment",
                        "Unavailable",
                        "Market segment data is not available.",
                        "Segment is unavailable when clustering artifacts from scripts/build_clusters.py are not present or not marked ready.",
                    )
            with signal_cols[1]:
                if cluster is not None:
                    alignment = max(
                        0, min(100, int(round(100 / (1 + cluster["distance"]))))
                    )
                    render_demo_signal_card(
                        "Alignment",
                        f"{alignment}%",
                        "Relative closeness to this segment.",
                        "Alignment is a display score derived from the distance to the nearest K-Means centroid: round(100 / (1 + distance)), clipped to 0-100.",
                    )
                else:
                    render_demo_signal_card(
                        "Alignment",
                        "N/A",
                        "Build segment data first.",
                        "Alignment requires cluster distance from app/ml_runtime.py::cluster_position().",
                    )
            with signal_cols[2]:
                if matches is None or matches.empty:
                    render_demo_signal_card(
                        "Top similarity",
                        "N/A",
                        "No matching roles surfaced.",
                        "Top similarity requires at least one retrieved role from the FAISS index.",
                    )
                else:
                    render_demo_signal_card(
                        "Top similarity",
                        f"{matches.iloc[0]['similarity'] * 100:.0f}%",
                        "Best role match for this resume.",
                        "Top similarity is the first retrieved role's embedding similarity returned by app/ml_runtime.py::retrieve_matches() using ml/retrieval.py and the local FAISS index.",
                    )
            with signal_cols[3]:
                count = 0 if matches is None else len(matches)
                render_demo_signal_card(
                    "Retrieved roles",
                    f"{count:,}",
                    "Roles surfaced for this resume.",
                    "Retrieved roles is the count of top-k matches retained after vector search, metadata join, and seniority-aware filtering in app/ml_runtime.py.",
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

        with st.container(key="top-matching-roles-section"):
            render_demo_section_header(
                "Top matching roles",
                "These roles are ordered by similarity to the resume.",
                match_info,
            )
            if matches is None or matches.empty:
                st.info(
                    "No matching roles surfaced for this resume. Try expanding the resume text with more domain terms."
                )
            else:
                render_job_results(matches)

        render_demo_floating_nav(
            restart_demo=restart_demo,
        )
