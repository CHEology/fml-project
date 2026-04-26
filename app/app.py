from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    runtime = importlib.import_module("ml_runtime")
except ModuleNotFoundError as err:
    spec = importlib.util.spec_from_file_location(
        "resumatch_ml_runtime", PROJECT_ROOT / "app" / "ml_runtime.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load app/ml_runtime.py") from err
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)

runtime_artifact_status = runtime.artifact_status
artifacts_ready = runtime.artifacts_ready
cluster_position = runtime.cluster_position
encode_resume = runtime.encode_resume
feedback_terms = runtime.feedback_terms
load_cluster_artifacts = runtime.load_cluster_artifacts
load_real_jobs = runtime.load_jobs
load_retriever = runtime.load_retriever
load_salary_artifacts = runtime.load_salary_artifacts
retrieve_matches = runtime.retrieve_matches
salary_band_from_model = runtime.salary_band_from_model

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"

SAMPLE_RESUME = """Alex Rivera
Senior Machine Learning Engineer

Experience:
- Built retrieval and ranking pipelines for job recommendations.
- Shipped PyTorch models for salary forecasting and churn prediction.
- Deployed Streamlit prototypes for internal stakeholders.
- Worked with Python, SQL, pandas, FAISS, AWS, Docker, and Airflow.

Skills:
machine learning, recommender systems, nlp, embeddings, python, pytorch,
sql, streamlit, analytics, product experimentation, data pipelines
"""

SYNTHETIC_JOBS = [
    {
        "job_id": 101,
        "title": "Senior Machine Learning Engineer",
        "company_name": "North Harbor AI",
        "salary_annual": 182000,
        "location": "New York, NY",
        "state": "NY",
        "experience_level": "Mid-Senior level",
        "work_type": "Remote",
        "text": "Build recommendation models, embeddings, and retrieval systems with Python, PyTorch, FAISS, and AWS.",
    },
    {
        "job_id": 102,
        "title": "Product Data Scientist",
        "company_name": "Metric Canvas",
        "salary_annual": 156000,
        "location": "San Francisco, CA",
        "state": "CA",
        "experience_level": "Associate",
        "work_type": "Hybrid",
        "text": "Own experimentation, forecasting, funnel analytics, and executive storytelling across product teams.",
    },
    {
        "job_id": 103,
        "title": "Applied NLP Engineer",
        "company_name": "Signal Foundry",
        "salary_annual": 171000,
        "location": "Boston, MA",
        "state": "MA",
        "experience_level": "Mid-Senior level",
        "work_type": "Remote",
        "text": "Train NLP systems, search relevance pipelines, and semantic retrieval stacks using Python and transformers.",
    },
    {
        "job_id": 104,
        "title": "Analytics Engineer",
        "company_name": "Cinder Labs",
        "salary_annual": 138000,
        "location": "Austin, TX",
        "state": "TX",
        "experience_level": "Associate",
        "work_type": "On-site",
        "text": "Design data marts, dbt models, dashboards, KPI reporting, and SQL-heavy analytics workflows.",
    },
    {
        "job_id": 105,
        "title": "Backend Software Engineer, Recommendations",
        "company_name": "Circuit North",
        "salary_annual": 164000,
        "location": "Seattle, WA",
        "state": "WA",
        "experience_level": "Mid-Senior level",
        "work_type": "Hybrid",
        "text": "Ship APIs, ranking services, experiment infrastructure, and production Python services on AWS.",
    },
    {
        "job_id": 106,
        "title": "Staff Product Analyst",
        "company_name": "Aster Point",
        "salary_annual": 149000,
        "location": "Chicago, IL",
        "state": "IL",
        "experience_level": "Director",
        "work_type": "Remote",
        "text": "Drive executive decision support, growth analytics, lifecycle reporting, and product strategy.",
    },
]

st.set_page_config(
    page_title="ResuMatch",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #f6efe4;
            --panel: rgba(255, 249, 241, 0.92);
            --ink: #1d1b18;
            --muted: #6b6257;
            --line: rgba(29, 27, 24, 0.08);
            --sun: #ff8d5d;
            --teal: #0c7c78;
            --gold: #d8a24c;
            --mint: #c7efe7;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 141, 93, 0.22), transparent 26%),
                radial-gradient(circle at top right, rgba(12, 124, 120, 0.18), transparent 24%),
                linear-gradient(180deg, #fcf6ed 0%, #f4ecdf 100%);
            color: var(--ink);
        }

        html, body, [class*="css"] {
            font-family: "Space Grotesk", "Avenir Next", sans-serif;
        }

        .mono {
            font-family: "IBM Plex Mono", monospace;
            color: var(--muted);
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,255,255,0.55));
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.5rem 1.6rem 1.3rem 1.6rem;
            box-shadow: 0 18px 48px rgba(70, 42, 18, 0.08);
            margin-bottom: 1rem;
        }

        .eyebrow {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: var(--teal);
            font-weight: 700;
        }

        .hero h1 {
            margin: 0.3rem 0 0.7rem 0;
            font-size: 3rem;
            line-height: 0.95;
        }

        .hero p {
            margin: 0;
            font-size: 1rem;
            color: var(--muted);
            max-width: 62rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .pill {
            border: 1px solid rgba(12,124,120,0.14);
            background: rgba(199, 239, 231, 0.65);
            color: #084744;
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-card, .info-card, .job-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(70, 42, 18, 0.05);
            height: 100%;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.05;
        }

        .info-title {
            font-size: 1.12rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .job-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .job-meta {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.55rem;
        }

        .score-chip {
            display: inline-block;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            background: rgba(255, 141, 93, 0.16);
            color: #9a4b1f;
            font-weight: 700;
            font-size: 0.8rem;
            margin-bottom: 0.6rem;
        }

        .salary-band {
            background: linear-gradient(90deg, rgba(12,124,120,0.18), rgba(255,141,93,0.18));
            border-radius: 999px;
            height: 14px;
            margin: 0.45rem 0 0.2rem 0;
            overflow: hidden;
        }

        .salary-fill {
            background: linear-gradient(90deg, #0c7c78, #ff8d5d);
            height: 14px;
            border-radius: 999px;
        }

        .section-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_jobs() -> tuple[pd.DataFrame, str, bool]:
    if DATA_PATH.exists():
        return (
            load_real_jobs(PROJECT_ROOT),
            str(DATA_PATH.relative_to(PROJECT_ROOT)),
            True,
        )

    return pd.DataFrame(SYNTHETIC_JOBS), "Synthetic demo dataset", False


def artifact_status() -> list[dict[str, Any]]:
    return runtime_artifact_status(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_retriever_resource():
    return load_retriever(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_salary_resource():
    return load_salary_artifacts(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_cluster_resource():
    return load_cluster_artifacts(PROJECT_ROOT)


def extract_uploaded_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            return ""

        pages: list[str] = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    return ""


def fmt_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${int(value):,}"


def render_metric_card(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="mono">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_job_card(row: pd.Series) -> None:
    summary = str(row.get("text", ""))[:190]
    similarity = row.get("similarity", np.nan)
    score_label = (
        f"Cosine {float(similarity):.3f}" if not pd.isna(similarity) else "FAISS match"
    )
    st.markdown(
        f"""
        <div class="job-card">
            <div class="score-chip">{score_label}</div>
            <div class="job-title">{row.get("title", "Untitled role")}</div>
            <div class="job-meta">{row.get("company_name", "Unknown company")} · {row.get("location", "Unknown location")} · {row.get("work_type", "Work type TBD")}</div>
            <div><strong>{fmt_money(row.get("salary_annual"))}</strong> · {row.get("experience_level", "Experience TBD")}</div>
            <div class="mono" style="margin-top:0.6rem;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_salary_band(band: dict[str, int]) -> None:
    st.markdown(
        '<div class="section-label">Projected salary corridor</div>',
        unsafe_allow_html=True,
    )
    width = max(
        12, min(100, int((band["q75"] - band["q10"]) / max(band["q90"], 1) * 100))
    )
    start = max(0, int(band["q10"] / max(band["q90"], 1) * 100))
    st.markdown(
        f"""
        <div class="salary-band">
            <div class="salary-fill" style="width:{width}%; margin-left:{start}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    for col, key in zip(cols, ("q10", "q25", "q50", "q75", "q90"), strict=True):
        col.metric(key.upper(), fmt_money(band[key]))


def main() -> None:
    inject_styles()
    jobs, data_source, has_real_data = load_jobs()
    status = artifact_status()

    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""

    with st.sidebar:
        st.markdown("## ResuMatch")
        st.caption("Local frontend shell for the NYU final project")
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Runtime mode</div>
                <div class="mono">{data_source}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Load sample resume", width="stretch"):
            st.session_state.resume_text = SAMPLE_RESUME
            st.rerun()

        st.markdown("### Artifact status")
        for item in status:
            flag = "Ready" if item["ready"] else "Missing"
            st.write(f"{flag}: `{item['path']}`")

        if not has_real_data:
            st.info(
                "No processed jobs file found. The app is running with a synthetic dataset so the frontend can still be demoed locally."
            )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Resume intelligence • artifact-backed ML</div>
            <h1>ResuMatch turns a raw resume into market direction.</h1>
            <p>
                Upload or paste a resume to query the FAISS job index, predict a PyTorch
                quantile-regression salary range, and locate the resume within the clustered
                LinkedIn job market.
            </p>
            <div class="pill-row">
                <span class="pill">Resume intake</span>
                <span class="pill">FAISS retrieval</span>
                <span class="pill">Quantile salary model</span>
                <span class="pill">KMeans market clusters</span>
            </div>
        </div>
        """,
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
            "Artifacts ready", f"{ready_count}/{len(status)}", "pipeline completeness"
        )

    launchpad_tab, radar_tab, pipeline_tab = st.tabs(
        ["Launchpad", "Job Radar", "Pipeline"]
    )

    with launchpad_tab:
        left, right = st.columns([1.15, 0.85], gap="large")

        with left:
            st.subheader("Resume input")
            uploader = st.file_uploader(
                "Upload a resume (.pdf or .txt)", type=["pdf", "txt"]
            )
            if uploader is not None:
                parsed = extract_uploaded_text(uploader)
                if parsed:
                    st.session_state.resume_text = parsed
                else:
                    st.warning(
                        "Could not extract text from the uploaded file. Paste the resume text below instead."
                    )

            st.session_state.resume_text = st.text_area(
                "Paste resume text",
                value=st.session_state.resume_text,
                height=260,
                placeholder="Paste a resume, portfolio bio, or achievement summary here...",
            )

            pref_a, pref_b, pref_c = st.columns(3)
            with pref_a:
                preferred_location = st.selectbox(
                    "Preferred location",
                    ["Anywhere", "NY", "CA", "TX", "WA", "MA", "IL"],
                )
            with pref_b:
                remote_only = st.toggle("Remote only", value=False)
            with pref_c:
                top_k = st.slider("Matches", min_value=3, max_value=10, value=6)

            analyze_clicked = st.button(
                "Run ML analysis", type="primary", width="stretch"
            )

        with right:
            st.markdown(
                """
                    <div class="info-card">
                    <div class="info-title">What this run does</div>
                    <div class="mono">
                        Encodes the resume with the same sentence-transformer pipeline used for job postings,
                        searches the FAISS index, predicts salary quantiles from the trained PyTorch model,
                        and assigns a KMeans job-market cluster.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                """
                    <div class="info-card">
                    <div class="info-title">Required artifacts</div>
                    <div class="mono">
                        Runtime analysis needs jobs.parquet, jobs.index, jobs_meta.parquet,
                        salary_model.pt, salary_model.scaler.json, kmeans_k8.pkl,
                        cluster_assignments.npy, and cluster_labels.json.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if analyze_clicked and st.session_state.resume_text.strip():
            if not has_real_data:
                st.error(
                    "Real processed data is required for ML analysis. Run preprocessing before using this path."
                )
            elif not artifacts_ready(status, "retrieval"):
                st.error(
                    "Retrieval artifacts are missing. Build `models/jobs.index`, `models/jobs_meta.parquet`, and `models/job_embeddings.npy` first."
                )
            else:
                try:
                    with st.spinner("Encoding resume and querying FAISS..."):
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
                            top_k=top_k,
                        )

                    band = None
                    if artifacts_ready(status, "salary"):
                        with st.spinner("Predicting salary quantiles..."):
                            salary_model, salary_scaler = load_salary_resource()
                            band = salary_band_from_model(
                                salary_model, resume_embedding, salary_scaler
                            )

                    cluster = None
                    if artifacts_ready(status, "clustering"):
                        with st.spinner("Assigning market cluster..."):
                            kmeans_model, _, cluster_labels = load_cluster_resource()
                            cluster = cluster_position(
                                kmeans_model, cluster_labels, resume_embedding
                            )

                    st.session_state.analysis = {
                        "resume_text": st.session_state.resume_text,
                        "matches": matches,
                        "salary_band": band,
                        "cluster": cluster,
                        "feedback_terms": feedback_terms(
                            st.session_state.resume_text, matches, cluster
                        ),
                    }
                except Exception as exc:  # pragma: no cover - UI guardrail
                    st.session_state.analysis = None
                    st.error(f"ML analysis failed: {exc}")

        analysis = st.session_state.get("analysis")
        if analysis and analysis.get("resume_text") == st.session_state.resume_text:
            matches = analysis["matches"]
            band = analysis["salary_band"]
            cluster = analysis["cluster"]
            missing_terms = analysis["feedback_terms"]

            st.write("")
            top_row = st.columns([0.65, 0.35], gap="large")
            with top_row[0]:
                st.subheader("Market readout")
                if band is not None:
                    render_salary_band(band)
                else:
                    st.warning(
                        "Salary model artifacts are missing, so quantile prediction is unavailable."
                    )
            with top_row[1]:
                if cluster is not None:
                    terms = ", ".join(cluster["top_terms"][:5]) or "No label terms"
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="info-title">{cluster["label"]}</div>
                            <div class="metric-value" style="font-size:1.7rem;">Cluster {cluster["cluster_id"]}</div>
                            <div class="mono">Nearest-centroid distance: {cluster["distance"]:.3f}</div>
                            <div class="mono" style="margin-top:0.45rem;">{terms}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        "Clustering artifacts are missing, so market position is unavailable."
                    )

            st.write("")
            insight_cols = st.columns(2, gap="large")
            with insight_cols[0]:
                st.subheader("Retrieved evidence")
                if matches.empty:
                    st.info("No matching roles passed the selected filters.")
                else:
                    st.metric(
                        "Top cosine similarity", f"{matches.iloc[0]['similarity']:.3f}"
                    )
                    st.metric("Retrieved roles", f"{len(matches):,}")
                    if cluster and cluster.get("size"):
                        st.metric("Cluster size", f"{int(cluster['size']):,} jobs")
            with insight_cols[1]:
                st.subheader("Resume gaps")
                if missing_terms:
                    for item in missing_terms:
                        st.markdown(f"- Add stronger evidence for **{item}**")
                else:
                    st.markdown(
                        "- Top retrieved roles and cluster labels are already reflected in the resume text."
                    )

            st.write("")
            st.subheader("Top matching roles")
            if matches.empty:
                st.info(
                    "No roles matched the selected filters. Try Anywhere or disable Remote only."
                )
            else:
                card_cols = st.columns(2, gap="medium")
                for index, (_, row) in enumerate(matches.iterrows()):
                    with card_cols[index % 2]:
                        render_job_card(row)
        elif analyze_clicked and not st.session_state.resume_text.strip():
            st.warning(
                "Paste a resume or load the sample resume before running the analysis."
            )

    with radar_tab:
        st.subheader("Job market radar")
        display_jobs = jobs.copy()
        display_jobs["salary_annual"] = pd.to_numeric(
            display_jobs.get("salary_annual"), errors="coerce"
        )

        left, right = st.columns([0.52, 0.48], gap="large")
        with left:
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
                st.markdown("**KMeans market clusters**")
                st.bar_chart(pd.Series(cluster_names).value_counts())

        with right:
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
                st.info(
                    "Using synthetic roles so the frontend can be reviewed before the real preprocessing pipeline is run."
                )

    with pipeline_tab:
        st.subheader("Pipeline readiness")
        pipeline_cols = st.columns(len(status))
        for col, item in zip(pipeline_cols, status, strict=True):
            with col:
                render_metric_card(
                    item["label"], "Ready" if item["ready"] else "Missing", item["path"]
                )

        st.write("")
        st.markdown("**Recommended next commands**")
        st.code(
            "\n".join(
                [
                    "uv run python scripts/preprocess_data.py",
                    "uv run python scripts/build_index.py",
                    "uv run python scripts/train_salary_model.py --embeddings models/job_embeddings.npy --salaries data/processed/salaries.npy --output models/salary_model.pt",
                    "uv run python scripts/build_clusters.py",
                    "uv run streamlit run app/app.py",
                ]
            ),
            language="bash",
        )
        st.caption(
            "The salary checkpoint is required for quantile predictions; the KMeans artifacts power market-position output."
        )


if __name__ == "__main__":
    main()
