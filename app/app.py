from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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

TRACK_KEYWORDS = {
    "Machine Learning": [
        "machine learning",
        "ml",
        "pytorch",
        "tensorflow",
        "nlp",
        "embedding",
        "faiss",
    ],
    "Data Science": [
        "data science",
        "experimentation",
        "statistics",
        "modeling",
        "python",
        "pandas",
    ],
    "Software Engineering": [
        "software",
        "backend",
        "api",
        "distributed",
        "docker",
        "aws",
        "systems",
    ],
    "Analytics": ["analytics", "dashboard", "sql", "bi", "reporting", "tableau"],
    "Product / Strategy": [
        "product",
        "roadmap",
        "market",
        "strategy",
        "stakeholder",
        "growth",
    ],
}

SKILL_GROUPS = {
    "Python": ["python"],
    "SQL": ["sql", "postgres", "snowflake"],
    "ML Modeling": [
        "machine learning",
        "model",
        "xgboost",
        "pytorch",
        "tensorflow",
        "sklearn",
    ],
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

THEMES = {
    "Light": {
        "bg_start": "#fcf6ed",
        "bg_end": "#f4ecdf",
        "flare_a": "rgba(255, 141, 93, 0.22)",
        "flare_b": "rgba(12, 124, 120, 0.18)",
        "panel": "rgba(255, 249, 241, 0.92)",
        "ink": "#1d1b18",
        "muted": "#6b6257",
        "line": "rgba(29, 27, 24, 0.08)",
        "pill_bg": "rgba(199, 239, 231, 0.65)",
        "pill_ink": "#084744",
        "hero_a": "rgba(255,255,255,0.78)",
        "hero_b": "rgba(255,255,255,0.55)",
        "shadow": "rgba(70, 42, 18, 0.08)",
        "score_bg": "rgba(255, 141, 93, 0.16)",
        "score_ink": "#9a4b1f",
    },
    "Dark": {
        "bg_start": "#121416",
        "bg_end": "#1d2127",
        "flare_a": "rgba(255, 141, 93, 0.16)",
        "flare_b": "rgba(12, 124, 120, 0.16)",
        "panel": "rgba(28, 33, 39, 0.92)",
        "ink": "#eef2f4",
        "muted": "#a7b0b5",
        "line": "rgba(238, 242, 244, 0.09)",
        "pill_bg": "rgba(12, 124, 120, 0.28)",
        "pill_ink": "#c9f7f3",
        "hero_a": "rgba(36,41,49,0.92)",
        "hero_b": "rgba(26,30,36,0.86)",
        "shadow": "rgba(0, 0, 0, 0.28)",
        "score_bg": "rgba(255, 141, 93, 0.22)",
        "score_ink": "#ffd6c4",
    },
}

FAKE_NAMES = [
    "Jordan Kim",
    "Priya Shah",
    "Maya Hernandez",
    "Ethan Brooks",
    "Leila Hassan",
    "Noah Patel",
    "Ava Morales",
    "Lucas Chen",
    "Sofia Bennett",
    "Owen Park",
]
FAKE_COMPANIES = [
    "Helio Metrics",
    "North Harbor AI",
    "Aster Point",
    "Circuit North",
    "Signal Foundry",
    "Blue Summit Labs",
    "Lattice Grove",
    "Quarry Logic",
    "Cinder Labs",
    "Metric Canvas",
]
FAKE_SCHOOLS = [
    "NYU",
    "Georgia Tech",
    "UC Berkeley",
    "University of Michigan",
    "Northeastern",
    "Carnegie Mellon",
    "Columbia",
    "UT Austin",
]
TRACK_SUMMARIES = {
    "Machine Learning": [
        "Built retrieval and ranking pipelines for talent and content discovery.",
        "Deployed PyTorch models for forecasting, search relevance, and recommendation.",
        "Partnered with product and platform teams to productionize model experiments.",
    ],
    "Data Science": [
        "Owned experimentation frameworks and KPI reporting for product teams.",
        "Built predictive models and decision-support dashboards for growth strategy.",
        "Translated ambiguous business questions into analytical roadmaps and measurable outcomes.",
    ],
    "Software Engineering": [
        "Shipped backend APIs and event-driven services used by high-traffic applications.",
        "Improved reliability, observability, and deployment speed across production systems.",
        "Worked across Python services, cloud infrastructure, and CI/CD tooling.",
    ],
    "Analytics": [
        "Modeled warehouse data, built dashboards, and defined executive KPI layers.",
        "Standardized metrics and reporting workflows for product and operations teams.",
        "Partnered with stakeholders to turn noisy requests into repeatable insights.",
    ],
    "Product / Strategy": [
        "Led market analysis, roadmap shaping, and cross-functional decision reviews.",
        "Built strategy narratives using product, revenue, and customer behavior signals.",
        "Connected business goals to measurable experiments and planning frameworks.",
    ],
}
TRACK_SKILLS = {
    "Machine Learning": [
        "Python",
        "PyTorch",
        "FAISS",
        "NLP",
        "Embeddings",
        "SQL",
        "AWS",
        "Docker",
    ],
    "Data Science": [
        "Python",
        "pandas",
        "SQL",
        "Experimentation",
        "Forecasting",
        "A/B Testing",
        "Statistics",
    ],
    "Software Engineering": [
        "Python",
        "APIs",
        "Docker",
        "AWS",
        "Distributed Systems",
        "Postgres",
        "CI/CD",
    ],
    "Analytics": [
        "SQL",
        "dbt",
        "Looker",
        "Dashboarding",
        "KPIs",
        "Pandas",
        "Data Modeling",
    ],
    "Product / Strategy": [
        "Market Analysis",
        "Stakeholder Mgmt",
        "Roadmapping",
        "SQL",
        "Growth",
        "Presentation",
    ],
}
TRACK_TITLES = {
    "Machine Learning": [
        "Machine Learning Engineer",
        "Applied Scientist",
        "NLP Engineer",
    ],
    "Data Science": ["Data Scientist", "Product Data Scientist", "Decision Scientist"],
    "Software Engineering": [
        "Backend Software Engineer",
        "Platform Engineer",
        "Systems Engineer",
    ],
    "Analytics": [
        "Analytics Engineer",
        "Senior Analyst",
        "Business Intelligence Analyst",
    ],
    "Product / Strategy": [
        "Product Strategy Lead",
        "Growth Strategist",
        "Product Analyst",
    ],
}
SENIORITY_LABELS = {
    "Intern / Entry": "Entry level",
    "Associate": "Associate",
    "Mid": "Mid-level",
    "Senior": "Senior",
    "Lead / Executive": "Director",
}
TRACK_INITIATIVES = {
    "Machine Learning": [
        "retrieval ranking stack",
        "candidate matching pipeline",
        "semantic search service",
        "feature-rich recommendation workflow",
    ],
    "Data Science": [
        "experiment measurement layer",
        "forecasting and scenario planning workflow",
        "growth analytics operating model",
        "decision support dashboard suite",
    ],
    "Software Engineering": [
        "distributed service layer",
        "platform reliability program",
        "event-driven API stack",
        "deployment automation pipeline",
    ],
    "Analytics": [
        "executive KPI layer",
        "warehouse modeling program",
        "self-serve reporting surface",
        "business performance review workflow",
    ],
    "Product / Strategy": [
        "market expansion narrative",
        "cross-functional planning rhythm",
        "growth prioritization framework",
        "portfolio strategy review",
    ],
}
TRACK_PROJECTS = {
    "Machine Learning": [
        "search relevance simulator",
        "embedding quality scorecard",
        "model rollout control tower",
    ],
    "Data Science": [
        "retention diagnostic notebook",
        "experimentation intake rubric",
        "forecast variance monitor",
    ],
    "Software Engineering": [
        "API latency watchtower",
        "service dependency mapper",
        "release health dashboard",
    ],
    "Analytics": [
        "metric definition library",
        "reporting coverage audit",
        "dashboard adoption tracker",
    ],
    "Product / Strategy": [
        "market signal brief",
        "roadmap tradeoff memo series",
        "customer segment sizing model",
    ],
}
TRACK_METRICS = {
    "Machine Learning": "matching quality",
    "Data Science": "forecast accuracy",
    "Software Engineering": "service reliability",
    "Analytics": "report turnaround time",
    "Product / Strategy": "planning clarity",
}
SECTION_ALIASES = {
    "Summary": ["summary", "professional summary", "profile"],
    "Experience": ["experience", "professional experience", "work experience"],
    "Projects": ["projects", "selected projects"],
    "Education": ["education"],
    "Skills": ["skills", "core skills", "technical skills"],
}


st.set_page_config(
    page_title="ResuMatch",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles(theme_name: str) -> None:
    theme = THEMES[theme_name]
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --panel: __PANEL__;
            --ink: __INK__;
            --muted: __MUTED__;
            --line: __LINE__;
            --sun: #ff8d5d;
            --teal: #0c7c78;
            --gold: #d8a24c;
            --mint: #c7efe7;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, __FLARE_A__, transparent 26%),
                radial-gradient(circle at top right, __FLARE_B__, transparent 24%),
                linear-gradient(180deg, __BG_START__ 0%, __BG_END__ 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: __PANEL__;
            border-right: 1px solid __LINE__;
        }

        .stMarkdown, .stText, .stCaption, label, .stSelectbox, .stRadio {
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
            background: linear-gradient(135deg, __HERO_A__, __HERO_B__);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.5rem 1.6rem 1.3rem 1.6rem;
            box-shadow: 0 18px 48px __SHADOW__;
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
            background: __PILL_BG__;
            color: __PILL_INK__;
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
            box-shadow: 0 12px 28px __SHADOW__;
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
            background: __SCORE_BG__;
            color: __SCORE_INK__;
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

        .stTextArea textarea, .stTextInput input {
            background: rgba(255,255,255,0.04);
            color: var(--ink);
        }

        .panel-banner {
            margin-bottom: 0.75rem;
        }

        .panel-kicker {
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--teal);
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .panel-title {
            font-size: 1.28rem;
            font-weight: 700;
            margin-bottom: 0.24rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.45;
        }

        .signal-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 0.95rem;
            min-height: 124px;
        }

        .signal-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }

        .signal-value {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.22rem;
        }

        .signal-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.42;
        }

        .chip-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.35rem;
        }

        .mini-chip {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.78rem;
            color: var(--ink);
            background: rgba(255,255,255,0.04);
        }

        .callout {
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.04);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-top: 0.75rem;
        }

        .callout-title {
            font-size: 0.92rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }

        .callout-body {
            font-size: 0.98rem;
            line-height: 1.5;
        }

        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .status-pill.ready {
            background: rgba(12,124,120,0.18);
            color: #0c7c78;
        }

        .status-pill.missing {
            background: rgba(255,141,93,0.16);
            color: #b85d2d;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            margin-bottom: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.04);
            padding: 0.45rem 0.95rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(12,124,120,0.16);
            color: var(--ink);
        }
        </style>
    """
    for placeholder, value in {
        "__PANEL__": theme["panel"],
        "__INK__": theme["ink"],
        "__MUTED__": theme["muted"],
        "__LINE__": theme["line"],
        "__FLARE_A__": theme["flare_a"],
        "__FLARE_B__": theme["flare_b"],
        "__BG_START__": theme["bg_start"],
        "__BG_END__": theme["bg_end"],
        "__HERO_A__": theme["hero_a"],
        "__HERO_B__": theme["hero_b"],
        "__SHADOW__": theme["shadow"],
        "__PILL_BG__": theme["pill_bg"],
        "__PILL_INK__": theme["pill_ink"],
        "__SCORE_BG__": theme["score_bg"],
        "__SCORE_INK__": theme["score_ink"],
    }.items():
        css = css.replace(placeholder, value)
    st.markdown(
        css,
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


def track_job_subset(jobs: pd.DataFrame, track: str) -> pd.DataFrame:
    if jobs.empty:
        return jobs

    searchable = (
        jobs["title"].astype(str)
        + " "
        + jobs["company_name"].astype(str)
        + " "
        + jobs["text"].astype(str)
    ).str.lower()

    mask = pd.Series(False, index=jobs.index)
    for keyword in TRACK_KEYWORDS[track]:
        mask = mask | searchable.str.contains(keyword, regex=False, na=False)

    subset = jobs.loc[mask].copy()
    return subset if not subset.empty else jobs.copy()


def market_skill_stack(jobs: pd.DataFrame, track: str, limit: int = 8) -> list[str]:
    subset = track_job_subset(jobs, track)
    searchable = (
        subset["title"].astype(str) + " " + subset["text"].astype(str)
    ).str.lower()

    ranked_skills: list[tuple[int, str]] = []
    for skill in TRACK_SKILLS[track]:
        count = int(searchable.str.count(skill.lower()).sum())
        ranked_skills.append((count, skill))

    ranked_skills.sort(key=lambda item: (-item[0], item[1]))
    ordered = [skill for count, skill in ranked_skills if count > 0]
    for skill in TRACK_SKILLS[track]:
        if skill not in ordered:
            ordered.append(skill)
    return ordered[:limit]


def choose_market_examples(
    jobs: pd.DataFrame,
    track: str,
    preferred_location: str,
) -> pd.DataFrame:
    subset = track_job_subset(jobs, track)
    if preferred_location != "Anywhere":
        location_mask = subset["location"].astype(str).str.contains(
            preferred_location,
            case=False,
            na=False,
        ) | subset["state"].astype(str).str.fullmatch(
            preferred_location,
            case=False,
            na=False,
        )
        if location_mask.any():
            subset = subset.loc[location_mask].copy()

    columns = ["title", "company_name", "location", "text"]
    for column in columns:
        if column not in subset.columns:
            subset[column] = ""
    return subset[columns].drop_duplicates().head(6)


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".")
    return slug or "candidate"


def compose_headline(seniority: str, title: str) -> str:
    title_clean = re.sub(r"\s+", " ", title).strip()
    seniority_clean = seniority.strip()
    if title_clean.lower().startswith(seniority_clean.lower()):
        return title_clean
    return f"{seniority_clean} {title_clean}"


def resume_structure(text: str) -> dict[str, Any]:
    lowered = text.lower()
    found_sections = [
        label
        for label, aliases in SECTION_ALIASES.items()
        if any(alias in lowered for alias in aliases)
    ]
    missing_sections = [
        label for label in SECTION_ALIASES if label not in found_sections
    ]
    bullet_count = sum(
        1 for line in text.splitlines() if line.strip().startswith(("-", "*"))
    )
    link_count = len(re.findall(r"(https?://|linkedin\.com/|github\.com/)", lowered))
    return {
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "bullet_count": bullet_count,
        "word_count": len(text.split()),
        "link_count": link_count,
    }


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_public_webpage_text(url: str) -> tuple[str, str]:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid public http or https URL.")
    if "linkedin.com" in parsed.netloc.lower():
        raise ValueError(
            "LinkedIn pages are not imported here. Paste profile text directly or use an approved LinkedIn OAuth/API flow."
        )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=12) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise ValueError("That URL did not return an HTML page.")
        html = response.read(1_500_000).decode("utf-8", errors="ignore")

    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
    html = re.sub(
        r"(?i)</?(p|div|section|article|li|h1|h2|h3|h4|h5|h6|br)[^>]*>", "\n", html
    )
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    cleaned = text.strip()
    if len(cleaned) < 120:
        raise ValueError(
            "The page did not expose enough public text to use as a resume input."
        )
    return cleaned[:8000], parsed.netloc.lower()


def generate_fake_resume(
    track: str,
    seniority: str,
    preferred_location: str,
    jobs: pd.DataFrame,
) -> str:
    seed = abs(hash((track, seniority, preferred_location))) % (2**32)
    rng = np.random.default_rng(seed)

    name = rng.choice(FAKE_NAMES)
    school = rng.choice(FAKE_SCHOOLS)
    subset = choose_market_examples(jobs, track, preferred_location)
    companies = (
        subset["company_name"]
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(companies) < 2:
        fallback_companies = [
            company for company in FAKE_COMPANIES if company not in companies
        ]
        companies.extend(fallback_companies)
    company_a, company_b = companies[0], companies[1]

    titles = subset["title"].replace("", np.nan).dropna().astype(str).unique().tolist()
    title = titles[0] if titles else str(rng.choice(TRACK_TITLES[track]))
    secondary_title = (
        titles[1] if len(titles) > 1 else str(rng.choice(TRACK_TITLES[track]))
    )
    skills = market_skill_stack(jobs, track, limit=7)
    initiatives = rng.choice(TRACK_INITIATIVES[track], size=2, replace=False)
    project_name = str(rng.choice(TRACK_PROJECTS[track]))
    years = {
        "Intern / Entry": "0-2",
        "Associate": "2-4",
        "Mid": "4-6",
        "Senior": "6-9",
        "Lead / Executive": "9+",
    }[seniority]
    preferred_place = (
        "Remote / flexible" if preferred_location == "Anywhere" else preferred_location
    )
    contact_slug = slugify_name(name)
    headline = compose_headline(seniority, title)
    metric_name = TRACK_METRICS[track]
    impact_one = int(rng.integers(18, 42))
    impact_two = int(rng.integers(24, 57))
    stakeholder_count = int(rng.integers(4, 11))
    experiment_count = int(rng.integers(8, 22))

    return f"""{name}
{headline}
{preferred_place} | {contact_slug}@example.com | github.com/{contact_slug} | linkedin.com/in/{contact_slug}-demo

PROFESSIONAL SUMMARY
{track} operator with {years} years of experience turning ambiguous business goals into reliable systems, clear metrics, and execution plans. Strongest in {skills[0]}, {skills[1]}, and {skills[2]}, with a bias toward shipping practical results instead of isolated analysis.

CORE SKILLS
{" | ".join(skills)}

EXPERIENCE
{company_a} | {headline} | 2022 - Present
- Led a {initiatives[0]} program using {skills[0]}, {skills[1]}, and cross-functional partner input, improving {metric_name} by {impact_one}% while reducing delivery friction for {stakeholder_count} teams.
- Built operating reviews that connected product questions, delivery milestones, and measurable business outcomes across hiring, growth, and leadership stakeholders.

{company_b} | {secondary_title} | 2019 - 2022
- Owned the execution layer for {initiatives[1]}, translating messy requests into production-ready workflows and concise stakeholder narratives.
- Ran {experiment_count}+ scoped iterations across analytics, automation, and tooling, improving speed-to-insight by {impact_two}% and creating cleaner handoffs between technical and business teams.

SELECTED PROJECTS
- {project_name.title()}: Created a reusable demo asset that combined representative market data, narrative framing, and actionable reporting for internal reviews.
- Resume Market Mapper: Synthesized public role patterns into a structured candidate story and surfaced the missing proof points needed for stronger role alignment.

EDUCATION
- B.S. in Computer Science / Data Science, {school}

SKILLS
- {", ".join(skills)}
"""


def linkedin_dataset_note(has_real_data: bool) -> str:
    if has_real_data:
        return "Using the processed LinkedIn Job Postings Kaggle dataset from the local pipeline."
    return "Using a synthetic LinkedIn-style jobs catalog because no processed local dataset is available yet."


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
    if any(
        token in lowered
        for token in (
            "chief",
            "director",
            "vice president",
            "vp",
            "staff",
            "principal",
            "lead ",
        )
    ):
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
    role_terms = TRACK_KEYWORDS[profile["track"]]
    skills = [skill.lower() for skill in profile["skills_present"]]

    ranked = jobs.copy()
    searchable = (
        ranked["title"].astype(str)
        + " "
        + ranked["company_name"].astype(str)
        + " "
        + ranked["location"].astype(str)
        + " "
        + ranked["experience_level"].astype(str)
        + " "
        + ranked["work_type"].astype(str)
        + " "
        + ranked["text"].astype(str)
    ).str.lower()

    score = pd.Series(0.0, index=ranked.index)
    for term in role_terms:
        score += searchable.str.count(term) * 4.0
    for term in skills:
        score += searchable.str.count(term) * 3.0

    score += (
        ranked["title"]
        .astype(str)
        .str.lower()
        .str.contains(profile["track"].split("/")[0].strip().lower(), regex=False)
        .astype(float)
        * 2.0
    )

    if preferred_location and preferred_location != "Anywhere":
        location_mask = (
            ranked["location"]
            .astype(str)
            .str.contains(preferred_location, case=False, na=False)
        )
        state_mask = (
            ranked["state"]
            .astype(str)
            .str.fullmatch(preferred_location, case=False, na=False)
        )
        score += (location_mask | state_mask).astype(float) * 4.5

    if remote_only:
        remote_mask = (
            ranked["work_type"].astype(str).str.contains("remote", case=False, na=False)
        )
        score += remote_mask.astype(float) * 3.0
        ranked = ranked[remote_mask | (ranked["work_type"].astype(str) == "")]
        score = score.loc[ranked.index]

    if profile["seniority"] != "Mid":
        seniority_mask = (
            ranked["experience_level"]
            .astype(str)
            .str.contains(
                profile["seniority"].split("/")[0].strip(), case=False, na=False
            )
        )
        score += seniority_mask.astype(float) * 2.0

    ranked["match_score"] = score.loc[ranked.index]
    ranked = ranked.sort_values(
        ["match_score", "salary_annual"], ascending=[False, False]
    )
    return ranked.head(top_k)


def salary_band(matches: pd.DataFrame, profile: dict[str, Any]) -> dict[str, int]:
    salaries = pd.to_numeric(matches.get("salary_annual"), errors="coerce").dropna()
    if len(salaries) >= 3:
        quantiles = (
            np.percentile(salaries, [10, 25, 50, 75, 90])
            * profile["seniority_multiplier"]
        )
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


def render_panel_banner(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel-banner">
            <div class="panel-kicker">{kicker}</div>
            <div class="panel-title">{title}</div>
            <div class="panel-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">{label}</div>
            <div class="signal-value">{value}</div>
            <div class="signal-copy">{copy}</div>
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
        st.caption("Local frontend shell for the NYU final project")
        theme_choice = st.radio(
            "Background mode",
            ["Light", "Dark"],
            index=0 if st.session_state.theme_name == "Light" else 1,
        )
        if theme_choice != st.session_state.theme_name:
            st.session_state.theme_name = theme_choice
            st.rerun()

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
            st.session_state.resume_source = "Built-in sample resume"
            st.rerun()

        st.markdown("### Artifact status")
        for item in status:
            flag = "Ready" if item["ready"] else "Missing"
            st.write(f"{flag}: `{item['path']}`")

        st.info(linkedin_dataset_note(has_real_data))

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
                <span class="pill">Resume studio</span>
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
        left, right = st.columns(2, gap="large")

        with left, st.container(border=True):
            render_panel_banner(
                "Input Studio",
                "Build the candidate story",
                "Upload a real resume, paste a summary, or synthesize a fake profile for a fast local demo.",
            )
            uploader = st.file_uploader(
                "Upload a resume (.pdf or .txt)", type=["pdf", "txt"]
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

            pref_a, pref_b, pref_c, pref_d = st.columns(4)
            with pref_a:
                preferred_track = st.selectbox("Focus track", list(TRACK_KEYWORDS))
            with pref_b:
                preferred_location = st.selectbox(
                    "Preferred location",
                    ["Anywhere", "NY", "CA", "TX", "WA", "MA", "IL"],
                )
            with pref_c:
                fake_level = st.selectbox("Demo seniority", list(SENIORITY_MULTIPLIER))
            with pref_d:
                remote_only = st.toggle("Remote only", value=False)

            action_a, action_b = st.columns(2)
            with action_a:
                if st.button("Generate fake resume", width="stretch"):
                    st.session_state.resume_text = generate_fake_resume(
                        preferred_track,
                        fake_level,
                        preferred_location,
                        jobs,
                    )
                    st.session_state.resume_source = (
                        f"Generated {preferred_track} demo resume"
                    )
                    st.rerun()
            with action_b:
                analyze_clicked = st.button(
                    "Run ML analysis", type="primary", width="stretch"
                )

        with right, st.container(border=True):
            render_panel_banner(
                "Signal Deck",
                "See how the app will read the candidate",
                "The right panel mirrors the left panel's baseline so both surfaces stay visually locked while you edit inputs.",
            )
            preview_text = st.session_state.resume_text.strip() or SAMPLE_RESUME
            preview_profile = detect_profile(preview_text, preferred_track)
            preview_structure = resume_structure(preview_text)
            signal_cols = st.columns(4, gap="small")
            with signal_cols[0]:
                render_signal_card(
                    "Track",
                    preview_profile["track"],
                    "Dominant role direction from resume language.",
                )
            with signal_cols[1]:
                render_signal_card(
                    "Seniority",
                    preview_profile["seniority"],
                    "Detected level from titles, wins, and tone.",
                )
            with signal_cols[2]:
                render_signal_card(
                    "Sections",
                    f"{len(preview_structure['found_sections'])}/{len(SECTION_ALIASES)}",
                    "Structured resumes score better when summary, experience, projects, education, and skills are explicit.",
                )
            with signal_cols[3]:
                render_signal_card(
                    "Data mode",
                    "Real" if has_real_data else "Demo",
                    "Switches to local LinkedIn data when artifacts exist.",
                )

            st.markdown(
                '<div class="callout"><div class="callout-title">Resume read</div><div class="callout-body">The app currently sees a candidate oriented toward <strong>{}</strong> with approximately <strong>{}%</strong> confidence. Preferred location and demo seniority then bias the market output and fake-resume generator.</div></div>'.format(
                    preview_profile["track"], preview_profile["confidence"]
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="callout">
                    <div class="callout-title">Resume source</div>
                    <div class="callout-body">
                        <strong>{st.session_state.resume_source}</strong><br/>
                        {preview_structure["word_count"]} words • {preview_structure["bullet_count"]} bullets • {preview_structure["link_count"]} links detected
                    </div>
                </div>
                """,
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
                    f'<span class="mini-chip">{skill}</span>'
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
                    f'<span class="mini-chip">{section}</span>'
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
                '<div class="section-label" style="margin-top:0.9rem;">Data feed</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <span class="status-pill {"ready" if has_real_data else "missing"}">{"Processed LinkedIn parquet" if has_real_data else "Synthetic LinkedIn-style feed"}</span>
                """,
                unsafe_allow_html=True,
            )
            st.caption(linkedin_dataset_note(has_real_data))

        if analyze_clicked and st.session_state.resume_text.strip():
            if not has_real_data:
                st.error(
                    "Real processed data is required for ML analysis. Run preprocessing before using this path."
                )
                return
            if not artifacts_ready(status, "retrieval"):
                st.error(
                    "Retrieval artifacts are missing. Build `models/jobs.index`, `models/jobs_meta.parquet`, and `models/job_embeddings.npy` first."
                )
                return

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
                        top_k=6,
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

                missing_terms = feedback_terms(
                    st.session_state.resume_text, matches, cluster
                )
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.error(f"ML analysis failed: {exc}")
                return

            st.write("")
            top_row = st.columns([0.65, 0.35], gap="large")
            with top_row[0]:
                render_panel_banner(
                    "Market Readout",
                    "Salary corridor",
                    "This band comes from the trained raw-PyTorch quantile regression model.",
                )
                with st.container(border=True):
                    if band is not None:
                        render_salary_band(band)
                    else:
                        st.warning(
                            "Salary model artifacts are missing, so quantile prediction is unavailable."
                        )
            with top_row[1]:
                render_panel_banner(
                    "Profile Signal",
                    "Market cluster",
                    "A compact view of where the resume lands in the KMeans job-market segmentation.",
                )
                with st.container(border=True):
                    if cluster is not None:
                        signal_cols = st.columns(2, gap="small")
                        with signal_cols[0]:
                            render_signal_card(
                                "Cluster",
                                str(cluster["cluster_id"]),
                                cluster["label"],
                            )
                        with signal_cols[1]:
                            render_signal_card(
                                "Distance",
                                f"{cluster['distance']:.3f}",
                                "Nearest-centroid distance.",
                            )
                        st.markdown(
                            '<div class="chip-cloud">'
                            + "".join(
                                f'<span class="mini-chip">{term}</span>'
                                for term in cluster["top_terms"][:6]
                            )
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(
                            "Clustering artifacts are missing, so market position is unavailable."
                        )

            st.write("")
            insight_cols = st.columns(2, gap="large")
            with insight_cols[0]:
                render_panel_banner(
                    "Retrieved Evidence",
                    "FAISS match strength",
                    "These signals come from cosine similarity against the real job index.",
                )
                with st.container(border=True):
                    if matches.empty:
                        st.info("No matching roles passed the selected filters.")
                    else:
                        render_signal_card(
                            "Top similarity",
                            f"{matches.iloc[0]['similarity']:.3f}",
                            "Highest cosine score returned by FAISS.",
                        )
                        st.write("")
                        render_signal_card(
                            "Retrieved roles",
                            f"{len(matches):,}",
                            "Roles shown after filters.",
                        )
            with insight_cols[1]:
                render_panel_banner(
                    "Opportunity Lens",
                    "Gaps to close",
                    "Target missing terms from the assigned cluster and top retrieved jobs.",
                )
                with st.container(border=True):
                    if missing_terms:
                        for item in missing_terms:
                            st.markdown(f"- Add stronger evidence for **{item}**")
                    else:
                        st.markdown(
                            "- Top retrieved roles and cluster labels are already reflected in the resume text."
                        )

            st.write("")
            render_panel_banner(
                "Match Board",
                "Top matching roles",
                "These cards are ordered by FAISS cosine similarity against the real job index.",
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
            "A quick structural view of geography, seniority, and salary shape from the available job catalog.",
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
                st.markdown("**KMeans market clusters**")
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
                st.info(
                    "Using synthetic LinkedIn-style roles so the frontend can be reviewed before the real preprocessing pipeline is run."
                )

    with pipeline_tab:
        render_panel_banner(
            "Pipeline",
            "Artifact readiness",
            "This view shows which offline project outputs are already available to upgrade the app from demo mode to project mode.",
        )
        pipeline_cols = st.columns(len(status))
        for col, item in zip(pipeline_cols, status, strict=True):
            with col:
                render_metric_card(
                    item["label"], "Ready" if item["ready"] else "Missing", item["path"]
                )

        st.write("")
        with st.container(border=True):
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
