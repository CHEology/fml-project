from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

TRACK_KEYWORDS = {
    "Machine Learning": ["machine learning", "ml", "pytorch", "tensorflow", "nlp", "embedding", "faiss"],
    "Data Science": ["data science", "experimentation", "statistics", "modeling", "python", "pandas"],
    "Software Engineering": ["software", "backend", "api", "distributed", "docker", "aws", "systems"],
    "Analytics": ["analytics", "dashboard", "sql", "bi", "reporting", "tableau"],
    "Product / Strategy": ["product", "roadmap", "market", "strategy", "stakeholder", "growth"],
}

SKILL_GROUPS = {
    "Python": ["python"],
    "SQL": ["sql", "postgres", "snowflake"],
    "ML Modeling": ["machine learning", "model", "xgboost", "pytorch", "tensorflow", "sklearn"],
    "NLP / Retrieval": ["nlp", "llm", "embedding", "retrieval", "faiss", "vector"],
    "Cloud / Ops": ["aws", "gcp", "docker", "kubernetes", "airflow"],
    "Product Sense": ["experiment", "stakeholder", "roadmap", "product", "growth"],
}

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

BASE_SALARY = {
    "Machine Learning": 165000,
    "Data Science": 148000,
    "Software Engineering": 155000,
    "Analytics": 128000,
    "Product / Strategy": 145000,
}

SENIORITY_MULTIPLIER = {
    "Intern / Entry": 0.72,
    "Associate": 0.9,
    "Mid": 1.0,
    "Senior": 1.18,
    "Lead / Executive": 1.35,
}


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
        frame = pd.read_parquet(DATA_PATH)
        for column in ("title", "company_name", "location", "experience_level", "text"):
            if column not in frame.columns:
                frame[column] = ""
        if "salary_annual" not in frame.columns:
            frame["salary_annual"] = np.nan
        if "state" not in frame.columns:
            frame["state"] = frame["location"].astype(str).str.split(",").str[-1].str.strip()
        if "work_type" not in frame.columns:
            frame["work_type"] = frame.get("formatted_work_type", "")
        return frame.copy(), str(DATA_PATH.relative_to(PROJECT_ROOT)), True

    return pd.DataFrame(SYNTHETIC_JOBS), "Synthetic demo dataset", False


def artifact_status() -> list[dict[str, Any]]:
    paths = [
        ("Processed jobs", PROJECT_ROOT / "data" / "processed" / "jobs.parquet"),
        ("Salary targets", PROJECT_ROOT / "data" / "processed" / "salaries.npy"),
        ("Job embeddings", PROJECT_ROOT / "models" / "job_embeddings.npy"),
        ("FAISS index", PROJECT_ROOT / "models" / "jobs.index"),
        ("Salary model", PROJECT_ROOT / "models" / "salary_model.pt"),
    ]
    return [
        {
            "label": label,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "ready": path.exists(),
        }
        for label, path in paths
    ]


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


def detect_profile(resume_text: str, preferred_track: str) -> dict[str, Any]:
    lowered = resume_text.lower()
    track_scores = {
        track: sum(lowered.count(keyword) for keyword in keywords)
        for track, keywords in TRACK_KEYWORDS.items()
    }
    detected_track = max(track_scores, key=track_scores.get)
    if track_scores[detected_track] == 0:
        detected_track = preferred_track

    seniority = "Mid"
    if any(token in lowered for token in ("chief", "director", "vice president", "vp", "staff", "principal", "lead ")):
        seniority = "Lead / Executive"
    elif any(token in lowered for token in ("senior", "sr.", "sr ", "mid-senior")):
        seniority = "Senior"
    elif any(token in lowered for token in ("entry", "junior", "new grad", "intern")):
        seniority = "Intern / Entry"
    elif "associate" in lowered:
        seniority = "Associate"

    skill_hits = {
        skill: any(keyword in lowered for keyword in keywords)
        for skill, keywords in SKILL_GROUPS.items()
    }
    present = [skill for skill, hit in skill_hits.items() if hit]
    missing = [skill for skill, hit in skill_hits.items() if not hit]

    confidence = 55 + min(35, sum(track_scores.values()) * 3 + len(present) * 4)
    confidence = min(confidence, 96)

    return {
        "track": detected_track,
        "seniority": seniority,
        "seniority_multiplier": SENIORITY_MULTIPLIER[seniority],
        "skills_present": present,
        "skills_missing": missing[:3],
        "confidence": confidence,
    }


def score_jobs(
    jobs: pd.DataFrame,
    resume_text: str,
    profile: dict[str, Any],
    preferred_location: str,
    remote_only: bool,
    top_k: int = 6,
) -> pd.DataFrame:
    lowered_resume = resume_text.lower()
    role_terms = TRACK_KEYWORDS[profile["track"]]
    skills = [skill.lower() for skill in profile["skills_present"]]

    ranked = jobs.copy()
    searchable = (
        ranked["title"].astype(str) + " "
        + ranked["company_name"].astype(str) + " "
        + ranked["location"].astype(str) + " "
        + ranked["experience_level"].astype(str) + " "
        + ranked["work_type"].astype(str) + " "
        + ranked["text"].astype(str)
    ).str.lower()

    score = pd.Series(0.0, index=ranked.index)
    for term in role_terms:
        score += searchable.str.count(term) * 4.0
    for term in skills:
        score += searchable.str.count(term) * 3.0

    score += ranked["title"].astype(str).str.lower().str.contains(profile["track"].split("/")[0].strip().lower(), regex=False).astype(float) * 2.0

    if preferred_location and preferred_location != "Anywhere":
        location_mask = ranked["location"].astype(str).str.contains(preferred_location, case=False, na=False)
        state_mask = ranked["state"].astype(str).str.fullmatch(preferred_location, case=False, na=False)
        score += (location_mask | state_mask).astype(float) * 4.5

    if remote_only:
        remote_mask = ranked["work_type"].astype(str).str.contains("remote", case=False, na=False)
        score += remote_mask.astype(float) * 3.0
        ranked = ranked[remote_mask | (ranked["work_type"].astype(str) == "")]
        score = score.loc[ranked.index]

    if profile["seniority"] != "Mid":
        seniority_mask = ranked["experience_level"].astype(str).str.contains(profile["seniority"].split("/")[0].strip(), case=False, na=False)
        score += seniority_mask.astype(float) * 2.0

    ranked["match_score"] = score.loc[ranked.index]
    ranked = ranked.sort_values(["match_score", "salary_annual"], ascending=[False, False])
    return ranked.head(top_k)


def salary_band(matches: pd.DataFrame, profile: dict[str, Any]) -> dict[str, int]:
    salaries = pd.to_numeric(matches.get("salary_annual"), errors="coerce").dropna()
    if len(salaries) >= 3:
        quantiles = np.percentile(salaries, [10, 25, 50, 75, 90]) * profile["seniority_multiplier"]
    else:
        base = BASE_SALARY[profile["track"]] * profile["seniority_multiplier"]
        quantiles = np.array([base * 0.78, base * 0.9, base, base * 1.12, base * 1.24])
    return {
        "q10": int(round(quantiles[0], -3)),
        "q25": int(round(quantiles[1], -3)),
        "q50": int(round(quantiles[2], -3)),
        "q75": int(round(quantiles[3], -3)),
        "q90": int(round(quantiles[4], -3)),
    }


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
    st.markdown(
        f"""
        <div class="job-card">
            <div class="score-chip">Match {float(row.get('match_score', 0.0)):.1f}</div>
            <div class="job-title">{row.get('title', 'Untitled role')}</div>
            <div class="job-meta">{row.get('company_name', 'Unknown company')} · {row.get('location', 'Unknown location')} · {row.get('work_type', 'Work type TBD')}</div>
            <div><strong>{fmt_money(row.get('salary_annual'))}</strong> · {row.get('experience_level', 'Experience TBD')}</div>
            <div class="mono" style="margin-top:0.6rem;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_salary_band(band: dict[str, int]) -> None:
    st.markdown('<div class="section-label">Projected salary corridor</div>', unsafe_allow_html=True)
    width = max(12, min(100, int((band["q75"] - band["q10"]) / max(band["q90"], 1) * 100)))
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
    for col, key in zip(cols, ("q10", "q25", "q50", "q75", "q90")):
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
        if st.button("Load sample resume", use_container_width=True):
            st.session_state.resume_text = SAMPLE_RESUME
            st.rerun()

        st.markdown("### Artifact status")
        for item in status:
            flag = "Ready" if item["ready"] else "Missing"
            st.write(f"{flag}: `{item['path']}`")

        if not has_real_data:
            st.info("No processed jobs file found. The app is running with a synthetic dataset so the frontend can still be demoed locally.")

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Resume intelligence • local preview</div>
            <h1>ResuMatch turns a raw resume into market direction.</h1>
            <p>
                This frontend is wired to run in two modes: real project artifacts when they exist,
                and a synthetic local demo when they do not. That lets you iterate on the app shell now,
                then plug in preprocessing, embeddings, retrieval, and salary modeling later.
            </p>
            <div class="pill-row">
                <span class="pill">Resume intake</span>
                <span class="pill">Match scoring</span>
                <span class="pill">Salary banding</span>
                <span class="pill">Pipeline visibility</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Jobs loaded", f"{len(jobs):,}", "local catalog size")
    with metric_cols[1]:
        median_salary = pd.to_numeric(jobs.get("salary_annual"), errors="coerce").dropna()
        render_metric_card("Median salary", fmt_money(median_salary.median() if len(median_salary) else None), "from current dataset")
    with metric_cols[2]:
        ready_count = sum(item["ready"] for item in status)
        render_metric_card("Artifacts ready", f"{ready_count}/{len(status)}", "pipeline completeness")

    launchpad_tab, radar_tab, pipeline_tab = st.tabs(["Launchpad", "Job Radar", "Pipeline"])

    with launchpad_tab:
        left, right = st.columns([1.15, 0.85], gap="large")

        with left:
            st.subheader("Resume input")
            uploader = st.file_uploader("Upload a resume (.pdf or .txt)", type=["pdf", "txt"])
            if uploader is not None:
                parsed = extract_uploaded_text(uploader)
                if parsed:
                    st.session_state.resume_text = parsed
                else:
                    st.warning("Could not extract text from the uploaded file. Paste the resume text below instead.")

            st.session_state.resume_text = st.text_area(
                "Paste resume text",
                value=st.session_state.resume_text,
                height=260,
                placeholder="Paste a resume, portfolio bio, or achievement summary here...",
            )

            pref_a, pref_b, pref_c = st.columns(3)
            with pref_a:
                preferred_track = st.selectbox("Focus track", list(TRACK_KEYWORDS))
            with pref_b:
                preferred_location = st.selectbox("Preferred location", ["Anywhere", "NY", "CA", "TX", "WA", "MA", "IL"])
            with pref_c:
                remote_only = st.toggle("Remote only", value=False)

            analyze_clicked = st.button("Run frontend demo", type="primary", use_container_width=True)

        with right:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-title">What this demo does</div>
                    <div class="mono">
                        Scores jobs from the local dataset, detects role track and seniority from resume text,
                        and estimates a market salary band. When no real project data exists, it falls back to
                        a curated in-memory jobs catalog so the UI is still interactive.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-title">What is still backend-dependent</div>
                    <div class="mono">
                        Semantic embeddings, FAISS retrieval, trained quantile salary prediction, and cluster views
                        are not auto-loaded yet because the repo currently has no local model artifacts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if analyze_clicked and st.session_state.resume_text.strip():
            profile = detect_profile(st.session_state.resume_text, preferred_track)
            matches = score_jobs(
                jobs,
                st.session_state.resume_text,
                profile,
                preferred_location,
                remote_only,
            )
            band = salary_band(matches, profile)

            st.write("")
            top_row = st.columns([0.65, 0.35], gap="large")
            with top_row[0]:
                st.subheader("Market readout")
                render_salary_band(band)
            with top_row[1]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-title">{profile['track']}</div>
                        <div class="metric-value" style="font-size:1.7rem;">{profile['seniority']}</div>
                        <div class="mono">Match confidence: {profile['confidence']}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            insight_cols = st.columns(2, gap="large")
            with insight_cols[0]:
                st.subheader("Skill coverage")
                for skill_name, keywords in SKILL_GROUPS.items():
                    hit = any(keyword in st.session_state.resume_text.lower() for keyword in keywords)
                    st.write(f"{'Strong' if hit else 'Thin'}: {skill_name}")
                    st.progress(100 if hit else 28)
            with insight_cols[1]:
                st.subheader("Gaps to close")
                if profile["skills_missing"]:
                    for item in profile["skills_missing"]:
                        st.markdown(f"- Add proof points for **{item}**")
                else:
                    st.markdown("- Coverage is broad enough for this demo profile.")

            st.write("")
            st.subheader("Top matching roles")
            card_cols = st.columns(2, gap="medium")
            for index, (_, row) in enumerate(matches.iterrows()):
                with card_cols[index % 2]:
                    render_job_card(row)
        elif analyze_clicked:
            st.warning("Paste a resume or load the sample resume before running the demo.")

    with radar_tab:
        st.subheader("Job market radar")
        display_jobs = jobs.copy()
        display_jobs["salary_annual"] = pd.to_numeric(display_jobs.get("salary_annual"), errors="coerce")

        left, right = st.columns([0.52, 0.48], gap="large")
        with left:
            st.markdown("**Top locations**")
            location_counts = display_jobs["location"].fillna("Unknown").value_counts().head(8)
            st.bar_chart(location_counts)

            st.markdown("**Experience mix**")
            exp_counts = display_jobs["experience_level"].fillna("Unknown").value_counts().head(8)
            st.bar_chart(exp_counts)

        with right:
            st.markdown("**Salary sample**")
            salary_view = display_jobs[["title", "company_name", "location", "salary_annual"]].copy()
            salary_view = salary_view.sort_values("salary_annual", ascending=False).head(12)
            st.dataframe(salary_view, use_container_width=True, hide_index=True)

            st.markdown("**Dataset notes**")
            if has_real_data:
                st.success(f"Loaded real project data from `{data_source}`.")
            else:
                st.info("Using synthetic roles so the frontend can be reviewed before the real preprocessing pipeline is run.")

    with pipeline_tab:
        st.subheader("Pipeline readiness")
        pipeline_cols = st.columns(len(status))
        for col, item in zip(pipeline_cols, status):
            with col:
                render_metric_card(item["label"], "Ready" if item["ready"] else "Missing", item["path"])

        st.write("")
        st.markdown("**Recommended next commands**")
        st.code(
            "\n".join(
                [
                    "python scripts/preprocess_data.py",
                    "python scripts/build_index.py --smoke",
                    "streamlit run app/app.py",
                ]
            ),
            language="bash",
        )
        st.caption("The first command becomes real once the Kaggle raw CSVs are present under `data/raw/`.")


if __name__ == "__main__":
    main()
