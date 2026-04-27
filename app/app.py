from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from html import escape, unescape
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
hybrid_salary_band = runtime.hybrid_salary_band
load_cluster_artifacts = runtime.load_cluster_artifacts
load_real_jobs = runtime.load_jobs
load_occupation_router = runtime.load_occupation_router
load_retriever = runtime.load_retriever
load_salary_artifacts = runtime.load_salary_artifacts
load_wage_table = runtime.load_wage_table
retrieve_matches = runtime.retrieve_matches
salary_band_from_model = runtime.salary_band_from_model
salary_artifacts_ready = runtime.salary_artifacts_ready

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"

SAMPLE_RESUME = """Alex Rivera
Senior Machine Learning Engineer

Experience:
- Built retrieval and ranking pipelines for job recommendations.
- Shipped PyTorch models for salary forecasting and churn prediction.
- Deployed Streamlit prototypes for internal stakeholders.
- Worked with Python, SQL, pandas, vector search, AWS, Docker, and Airflow.

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
    "Human Resources": [
        "human resources",
        "hr",
        "talent",
        "recruiting",
        "benefits",
        "employee relations",
        "people operations",
    ],
    "Finance / Accounting": [
        "finance",
        "accounting",
        "financial",
        "budget",
        "forecast",
        "audit",
        "fp&a",
    ],
    "Marketing": [
        "marketing",
        "campaign",
        "brand",
        "content",
        "demand generation",
        "seo",
        "crm",
    ],
    "Sales / Customer Success": [
        "sales",
        "account",
        "customer success",
        "client",
        "pipeline",
        "renewal",
        "quota",
    ],
    "Operations / Administration": [
        "operations",
        "office",
        "administrative",
        "process",
        "vendor",
        "scheduling",
        "coordination",
    ],
    "Healthcare / Clinical": [
        "nursing",
        "nurse",
        "clinical",
        "patient",
        "physician",
        "hospital",
        "medical",
        "pharmacy",
        "therapist",
        "diagnosis",
        "ehr",
        "epic",
        "cerner",
        "rn ",
        "ms n",
        "phlebotomy",
    ],
    "Education / Teaching": [
        "teaching",
        "teacher",
        "curriculum",
        "classroom",
        "professor",
        "lecturer",
        "instructional",
        "lesson plan",
        "k-12",
        "pedagogy",
        "student outcomes",
        "edtech",
        "tutor",
    ],
    "Legal / Compliance": [
        "attorney",
        "lawyer",
        "litigation",
        "paralegal",
        "contract review",
        "general counsel",
        "regulatory",
        "compliance",
        "legal research",
        "deposition",
        "law firm",
        "juris doctor",
    ],
    "Design / Creative": [
        "designer",
        "ux",
        "ui",
        "user experience",
        "user interface",
        "figma",
        "sketch",
        "photoshop",
        "illustrator",
        "branding",
        "wireframe",
        "prototype",
        "visual design",
        "art direction",
        "creative direction",
    ],
    "Engineering / Hardware": [
        "mechanical engineer",
        "civil engineer",
        "electrical engineer",
        "structural",
        "manufacturing",
        "cad",
        "solidworks",
        "autocad",
        "fea",
        "pcb",
        "embedded systems",
        "firmware",
        "hardware",
        "matlab",
    ],
    "Research / Academia": [
        "research",
        "publication",
        "peer-review",
        "peer review",
        "laboratory",
        "postdoc",
        "phd",
        "dissertation",
        "principal investigator",
        "grant",
        "conference",
        "abstract",
        "journal",
    ],
    "Public Sector / Policy": [
        "government",
        "federal agency",
        "policy analysis",
        "civic",
        "public administration",
        "nonprofit",
        "advocacy",
        "legislative",
        "constituent",
        "municipal",
        "public sector",
        "ngo",
    ],
    "Hospitality / Service": [
        "hospitality",
        "restaurant",
        "hotel",
        "concierge",
        "barista",
        "chef",
        "food service",
        "front of house",
        "back of house",
        "guest services",
        "front desk",
        "events coordinator",
    ],
}

SKILL_GROUPS = {
    "Python": ["python"],
    "SQL": ["sql", "postgres", "snowflake"],
    "Predictive Modeling": [
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
    "People Operations": ["hr", "talent", "recruiting", "benefits", "employee"],
    "Financial Planning": ["finance", "accounting", "budget", "forecast", "audit"],
    "Go-to-Market": ["marketing", "sales", "customer", "campaign", "account"],
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
        "text": "Build recommendation models, embeddings, and retrieval systems with Python, PyTorch, vector search, and AWS.",
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
    "Human Resources": 102000,
    "Finance / Accounting": 118000,
    "Marketing": 112000,
    "Sales / Customer Success": 125000,
    "Operations / Administration": 94000,
    "Healthcare / Clinical": 96000,
    "Education / Teaching": 68000,
    "Legal / Compliance": 132000,
    "Design / Creative": 102000,
    "Engineering / Hardware": 118000,
    "Research / Academia": 84000,
    "Public Sector / Policy": 86000,
    "Hospitality / Service": 58000,
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
        "bg_start": "#f6f8fb",
        "bg_end": "#f6f8fb",
        "flare_a": "transparent",
        "flare_b": "transparent",
        "panel": "#ffffff",
        "ink": "#111827",
        "muted": "#667085",
        "line": "#e5e7eb",
        "pill_bg": "#edf4ff",
        "pill_ink": "#175cd3",
        "hero_a": "#ffffff",
        "hero_b": "#ffffff",
        "shadow": "rgba(15, 23, 42, 0.04)",
        "score_bg": "#ecfdf3",
        "score_ink": "#027a48",
    },
    "Dark": {
        "bg_start": "#0f1115",
        "bg_end": "#0f1115",
        "flare_a": "transparent",
        "flare_b": "transparent",
        "panel": "#1b1d22",
        "ink": "#f2f4f7",
        "muted": "#a5adba",
        "line": "#374151",
        "pill_bg": "#182b45",
        "pill_ink": "#84caff",
        "hero_a": "#1b222b",
        "hero_b": "#1b222b",
        "shadow": "rgba(0, 0, 0, 0.24)",
        "score_bg": "#12372a",
        "score_ink": "#75e0a7",
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
    "Human Resources": [
        "Led recruiting, onboarding, and employee programs across growing teams.",
        "Improved people operations workflows with clearer policies and reporting.",
        "Partnered with managers on talent planning, retention, and employee relations.",
    ],
    "Finance / Accounting": [
        "Owned budgeting, forecasting, reconciliations, and financial reporting workflows.",
        "Built planning models that clarified spend, revenue, and operating tradeoffs.",
        "Partnered with leaders on controls, audits, and monthly business reviews.",
    ],
    "Marketing": [
        "Led campaign planning, lifecycle messaging, and brand performance reporting.",
        "Improved acquisition and engagement programs through structured experiments.",
        "Connected audience insight, content strategy, and measurable business outcomes.",
    ],
    "Sales / Customer Success": [
        "Managed account planning, pipeline reviews, customer health, and renewal motions.",
        "Built repeatable client workflows that improved retention and expansion signals.",
        "Translated customer needs into concise plans for product and revenue teams.",
    ],
    "Operations / Administration": [
        "Coordinated office operations, vendor processes, scheduling, and executive support.",
        "Improved administrative workflows with clearer ownership and operating cadence.",
        "Supported cross-functional teams through practical process and communication systems.",
    ],
}
TRACK_SKILLS = {
    "Machine Learning": [
        "Python",
        "PyTorch",
        "Vector Search",
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
    "Human Resources": [
        "Talent Acquisition",
        "Employee Relations",
        "Onboarding",
        "HRIS",
        "Benefits",
        "Workforce Planning",
    ],
    "Finance / Accounting": [
        "Budgeting",
        "Forecasting",
        "Accounting",
        "Financial Reporting",
        "Excel",
        "Audit Support",
    ],
    "Marketing": [
        "Campaign Strategy",
        "Content",
        "CRM",
        "SEO",
        "Lifecycle Marketing",
        "Analytics",
    ],
    "Sales / Customer Success": [
        "Account Management",
        "Pipeline Review",
        "Renewals",
        "Customer Health",
        "CRM",
        "Executive Communication",
    ],
    "Operations / Administration": [
        "Process Improvement",
        "Vendor Management",
        "Scheduling",
        "Office Operations",
        "Documentation",
        "Stakeholder Support",
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
    "Human Resources": [
        "People Operations Manager",
        "Talent Acquisition Partner",
        "HR Generalist",
    ],
    "Finance / Accounting": [
        "Financial Analyst",
        "Accounting Manager",
        "FP&A Analyst",
    ],
    "Marketing": [
        "Marketing Manager",
        "Growth Marketing Specialist",
        "Lifecycle Marketing Lead",
    ],
    "Sales / Customer Success": [
        "Customer Success Manager",
        "Account Executive",
        "Revenue Operations Analyst",
    ],
    "Operations / Administration": [
        "Operations Manager",
        "Administrative Coordinator",
        "Business Operations Associate",
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
    "Human Resources": [
        "structured hiring workflow",
        "onboarding and retention program",
        "employee relations intake process",
        "manager enablement cadence",
    ],
    "Finance / Accounting": [
        "monthly close workflow",
        "budget variance review",
        "forecast planning model",
        "audit readiness program",
    ],
    "Marketing": [
        "lifecycle campaign calendar",
        "brand performance review",
        "content measurement workflow",
        "demand generation program",
    ],
    "Sales / Customer Success": [
        "account health review",
        "renewal planning workflow",
        "pipeline inspection cadence",
        "customer expansion program",
    ],
    "Operations / Administration": [
        "vendor management process",
        "office operations rhythm",
        "executive scheduling workflow",
        "cross-team coordination system",
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
    "Human Resources": [
        "talent funnel scorecard",
        "onboarding journey map",
        "employee feedback tracker",
    ],
    "Finance / Accounting": [
        "budget variance model",
        "monthly close checklist",
        "cash planning dashboard",
    ],
    "Marketing": [
        "campaign performance brief",
        "content calendar scorecard",
        "audience segment review",
    ],
    "Sales / Customer Success": [
        "renewal risk dashboard",
        "account planning toolkit",
        "customer health review",
    ],
    "Operations / Administration": [
        "vendor tracker",
        "office workflow audit",
        "meeting cadence planner",
    ],
}
TRACK_METRICS = {
    "Machine Learning": "matching quality",
    "Data Science": "forecast accuracy",
    "Software Engineering": "service reliability",
    "Analytics": "report turnaround time",
    "Product / Strategy": "planning clarity",
    "Human Resources": "hiring cycle time",
    "Finance / Accounting": "forecast accuracy",
    "Marketing": "campaign conversion",
    "Sales / Customer Success": "renewal readiness",
    "Operations / Administration": "process turnaround time",
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --panel: __PANEL__;
            --ink: __INK__;
            --muted: __MUTED__;
            --line: __LINE__;
            --accent: #175cd3;
            --accent-soft: __PILL_BG__;
            --success: #027a48;
            --warning: #b54708;
        }

        .stApp {
            background: linear-gradient(180deg, __BG_START__ 0%, __BG_END__ 100%);
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
            font-family: "Inter", "Avenir Next", sans-serif;
        }

        .mono {
            font-family: "IBM Plex Mono", monospace;
            color: var(--muted);
        }

        .sidebar-source-path {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .hero {
            background: __HERO_A__;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.2rem 0.95rem 1.2rem;
            box-shadow: 0 8px 20px __SHADOW__;
            margin-bottom: 1rem;
        }

        .eyebrow {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: var(--accent);
            font-weight: 700;
        }

        .hero h1 {
            margin: 0.25rem 0 0.5rem 0;
            font-size: 1.7rem;
            line-height: 1.15;
            letter-spacing: 0;
        }

        .hero p {
            margin: 0;
            font-size: 0.95rem;
            color: var(--muted);
            max-width: 62rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.65rem;
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
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 12px __SHADOW__;
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
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.05;
        }

        
        .signal-value, .metric-value { word-break: normal; overflow-wrap: break-word; hyphens: none; }

        .info-title {
            font-size: 1.12rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .job-title {
            font-size: 1.02rem;
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
            font-weight: 600;
            font-size: 0.8rem;
            margin-bottom: 0.6rem;
        }

        .salary-band {
            background: var(--accent-soft);
            border-radius: 999px;
            height: 10px;
            margin: 0.8rem 0 0.65rem 0;
            overflow: hidden;
        }

        .salary-fill {
            background: var(--accent);
            height: 10px;
            border-radius: 999px;
        }

        .salary-headline {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.75rem;
            align-items: end;
            margin-bottom: 0.4rem;
        }

        .salary-main {
            font-size: 2.15rem;
            line-height: 1;
            font-weight: 700;
        }

        .salary-range {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .salary-source {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.35rem 0.6rem;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .salary-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(7.5rem, 1fr));
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(255,255,255,0.03);
            margin-top: 0.75rem;
        }

        .salary-step {
            padding: 0.68rem 0.75rem;
            border-left: 1px solid var(--line);
        }

        .salary-step:first-child {
            border-left: 0;
        }

        .salary-step-label {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            white-space: nowrap;
        }

        .salary-step-value {
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.18rem;
            white-space: nowrap;
        }

        .evidence-line {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 0.7rem;
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
            color: var(--accent);
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
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
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
            border-radius: 12px;
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
            background: __SCORE_BG__;
            color: __SCORE_INK__;
        }

        .status-pill.missing {
            background: rgba(181, 71, 8, 0.12);
            color: var(--warning);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            border: 0;
            background: transparent;
            padding: 0.6rem 0.25rem;
            margin-right: 1.5rem;
            font-weight: 500;
            color: var(--muted);
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--ink);
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent);
            border-bottom-color: var(--accent);
            font-weight: 600;
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        .field-label {
            font-size: 0.88rem;
            font-weight: 500;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }

        .sidebar-info {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.8rem;
            box-shadow: 0 4px 12px __SHADOW__;
        }

        .sidebar-info .info-title {
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .sidebar-info .info-source {
            font-size: 0.85rem;
            font-weight: 600;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(23, 92, 211, 0.04);
            border: 1.5px dashed var(--line);
            border-radius: 12px;
            transition: border-color 0.15s ease, background 0.15s ease;
        }

        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent);
            background: rgba(23, 92, 211, 0.07);
        }

        [data-testid="stFileUploader"] label {
            font-size: 0.88rem;
            font-weight: 500;
        }

        .next-steps-list {
            margin: 0.4rem 0 0 0;
            padding-left: 1.1rem;
            color: var(--ink);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .next-steps-list li {
            margin-bottom: 0.25rem;
        }

        @media (max-width: 900px) {
            .salary-headline {
                grid-template-columns: 1fr;
                align-items: start;
            }

            .salary-source {
                width: fit-content;
            }

            .salary-strip {
                grid-template-columns: repeat(2, minmax(8rem, 1fr));
            }

            .salary-step {
                border-left: 0;
                border-top: 1px solid var(--line);
            }

            .salary-step:first-child,
            .salary-step:nth-child(2) {
                border-top: 0;
            }

            .salary-step:nth-child(even) {
                border-left: 1px solid var(--line);
            }
        }

        @media (max-width: 520px) {
            .hero h1 {
                font-size: 1.8rem;
            }

            .salary-main {
                font-size: 1.75rem;
            }

            .salary-strip {
                grid-template-columns: 1fr;
            }

            .salary-step,
            .salary-step:nth-child(even) {
                border-left: 0;
            }

            .salary-step:first-child {
                border-top: 0;
            }
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

    return pd.DataFrame(SYNTHETIC_JOBS), "Sample role catalog", False


def artifact_status() -> list[dict[str, Any]]:
    return runtime_artifact_status(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_retriever_resource():
    return load_retriever(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_salary_resource():
    return load_salary_artifacts(PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def load_occupation_resource(_encoder):
    return load_occupation_router(PROJECT_ROOT, _encoder)


@st.cache_resource(show_spinner=False)
def load_wage_resource():
    return load_wage_table(PROJECT_ROOT)


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


FALLBACK_TRACK_SKILLS = [
    "Communication",
    "Project Management",
    "Stakeholder Engagement",
    "Process Improvement",
    "Documentation",
    "Problem Solving",
]


def market_skill_stack(jobs: pd.DataFrame, track: str, limit: int = 8) -> list[str]:
    subset = track_job_subset(jobs, track)
    searchable = (
        subset["title"].astype(str) + " " + subset["text"].astype(str)
    ).str.lower()

    track_skills = TRACK_SKILLS.get(track, FALLBACK_TRACK_SKILLS)
    ranked_skills: list[tuple[int, str]] = []
    for skill in track_skills:
        count = int(searchable.str.count(skill.lower()).sum())
        ranked_skills.append((count, skill))

    ranked_skills.sort(key=lambda item: (-item[0], item[1]))
    ordered = [skill for count, skill in ranked_skills if count > 0]
    for skill in track_skills:
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


def generate_sample_profile(
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
    track_titles = TRACK_TITLES.get(track, ["Specialist", "Coordinator", "Manager"])
    title = titles[0] if titles else str(rng.choice(track_titles))
    secondary_title = (
        titles[1] if len(titles) > 1 else str(rng.choice(track_titles))
    )
    skills = market_skill_stack(jobs, track, limit=7)
    track_initiatives = TRACK_INITIATIVES.get(
        track,
        [
            "delivery workflow program",
            "stakeholder review cadence",
            "operating improvement initiative",
        ],
    )
    initiatives = rng.choice(track_initiatives, size=2, replace=False)
    track_projects = TRACK_PROJECTS.get(
        track,
        ["operating playbook", "improvement roadmap", "stakeholder review brief"],
    )
    project_name = str(rng.choice(track_projects))
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
    metric_name = TRACK_METRICS.get(track, "operating quality")
    impact_one = int(rng.integers(18, 42))
    impact_two = int(rng.integers(24, 57))
    stakeholder_count = int(rng.integers(4, 11))
    experiment_count = int(rng.integers(8, 22))

    return f"""{name}
{headline}
{preferred_place} | {contact_slug}@example.com | github.com/{contact_slug} | linkedin.com/in/{contact_slug}

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
- {project_name.title()}: Created a reusable market review asset that combined representative data, narrative framing, and actionable reporting for internal reviews.
- Resume Market Mapper: Synthesized public role patterns into a structured candidate story and surfaced the missing proof points needed for stronger role alignment.

EDUCATION
- B.S. in Computer Science / Data Science, {school}

SKILLS
- {", ".join(skills)}
"""


def linkedin_dataset_note(has_real_data: bool) -> str:
    if has_real_data:
        return "Using the local LinkedIn job catalog."
    return "Using sample roles because the local job catalog is not available yet."


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


def detect_profile(resume_text: str) -> dict[str, Any]:
    lowered = resume_text.lower()
    track_scores = {
        track: sum(lowered.count(keyword) for keyword in keywords)
        for track, keywords in TRACK_KEYWORDS.items()
    }
    detected_track = max(track_scores, key=track_scores.get)

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
    label_html = escape(str(label))
    value_html = escape(str(value))
    helper_html = escape(str(helper))
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label_html}</div>
            <div class="metric-value">{value_html}</div>
            <div class="mono">{helper_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_banner(kicker: str, title: str, body: str) -> None:
    title_html = escape(str(title))
    body_html = escape(str(body))
    st.markdown(
        f"""
        <div class="panel-banner">
            <div class="panel-title">{title_html}</div>
            <div class="panel-copy">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(label: str, value: str, copy: str) -> None:
    label_html = escape(str(label))
    value_html = escape(str(value))
    copy_html = escape(str(copy))
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">{label_html}</div>
            <div class="signal-value">{value_html}</div>
            <div class="signal-copy">{copy_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_job_card(row: pd.Series) -> None:
    summary = escape(str(row.get("text", ""))[:190])
    similarity = row.get("similarity", np.nan)
    score_label = "Strong match"
    if not pd.isna(similarity):
        score_label = f"{float(similarity) * 100:.0f}% similarity"
    title = escape(str(row.get("title", "Untitled role")))
    company = escape(str(row.get("company_name", "Unknown company")))
    location = escape(str(row.get("location", "Unknown location")))
    work_type = escape(str(row.get("work_type", "Work type TBD")))
    experience = escape(str(row.get("experience_level", "Experience TBD")))
    salary = fmt_money(row.get("salary_annual"))
    st.markdown(
        f"""
        <div class="job-card">
            <div class="score-chip">{score_label}</div>
            <div class="job-title">{title}</div>
            <div class="job-meta">{company} · {location} · {work_type}</div>
            <div><strong>{salary}</strong> · {experience}</div>
            <div class="mono" style="margin-top:0.6rem;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_salary_band(band: dict[str, Any]) -> None:
    source_labels = {
        "retrieved_jobs": "Matched roles",
        "bls": "Occupation wage data",
        "neural_model": "Model estimate",
    }
    evidence = band.get("evidence", {})
    primary = source_labels.get(str(band.get("primary_source")), "Available evidence")
    confidence = str(band.get("confidence", "unknown")).title()
    low = fmt_money(band["q10"])
    midpoint = fmt_money(band["q50"])
    high = fmt_money(band["q90"])
    source_badge = escape(f"{primary} · {confidence}")
    st.markdown(
        f"""
        <div class="section-label">Matched-market salary range</div>
        <div class="salary-headline">
            <div>
                <div class="salary-main">{midpoint}</div>
                <div class="salary-range">{low} to {high} expected market range</div>
            </div>
            <div class="salary-source">{source_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="salary-band">
            <div class="salary-fill" style="width:100%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quantile_labels = {
        "q10": "Low",
        "q25": "Lower mid",
        "q50": "Median",
        "q75": "Upper mid",
        "q90": "High",
    }
    cells = "".join(
        '<div class="salary-step">'
        f'<div class="salary-step-label">{escape(quantile_labels[key])}</div>'
        f'<div class="salary-step-value">{escape(fmt_money(band[key]))}</div>'
        "</div>"
        for key in ("q10", "q25", "q50", "q75", "q90")
    )
    st.markdown(f'<div class="salary-strip">{cells}</div>', unsafe_allow_html=True)

    pieces = [
        f"Source: {primary}",
        f"Confidence: {confidence}",
    ]
    salary_count = evidence.get("salary_count")
    if salary_count is not None:
        pieces.append(f"{int(salary_count)} roles with salary data")
    median_similarity = evidence.get("median_similarity")
    if median_similarity is not None and not pd.isna(median_similarity):
        pieces.append(f"{float(median_similarity) * 100:.0f}% median similarity")
    occupation_title = evidence.get("occupation_title")
    if occupation_title:
        pieces.append(str(occupation_title))
    if evidence.get("model_bls_disagreement"):
        pieces.append("supporting sources disagree")
    evidence_html = escape(" · ".join(pieces))
    st.markdown(
        f'<div class="evidence-line">{evidence_html}</div>',
        unsafe_allow_html=True,
    )


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
            <div class="sidebar-info">
                <div class="info-title">Data source</div>
                <div class="info-source"><strong>{escape(source_label)}</strong></div>
                {source_detail}
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption(linkedin_dataset_note(has_real_data))

    st.markdown(
        '''
        <div class="hero">
            <div class="eyebrow">Resume market intelligence</div>
            <h1>Understand role fit, salary range, and market position.</h1>
            <p>Compare a resume with salary-bearing LinkedIn roles and identify ways to strengthen the profile.</p>
            <div class="pill-row">
                <span class="pill">Role matching</span>
                <span class="pill">Salary range</span>
                <span class="pill">Profile guidance</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(2)
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

    launchpad_tab, radar_tab = st.tabs(
        ["Resume Analysis", "Market Overview"]
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

            st.markdown(
                '<div class="field-label">Public profile or portfolio URL</div>',
                unsafe_allow_html=True,
            )
            url_col, import_col = st.columns([0.76, 0.24], gap="small")
            with url_col:
                public_profile_url = st.text_input(
                    "Public profile or portfolio URL",
                    value=st.session_state.public_profile_url,
                    placeholder="https://portfolio.example.com/about",
                    label_visibility="collapsed",
                )
                st.session_state.public_profile_url = public_profile_url
            with import_col:
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

            st.caption(
                f"Detected focus: {preview_profile['track']} · Seniority: {preview_profile['seniority']}. Both are inferred from the resume."
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
                        preview_profile["seniority"],
                        "Anywhere",
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
                profile_seniority_html = escape(str(preview_profile["seniority"]))
                profile_confidence_html = escape(str(preview_profile["confidence"]))
                st.markdown(
                    f'<div class="metric-card" style="margin-top:0.75rem;"><div class="metric-label">Profile read</div><div class="signal-copy">The current resume reads as a <strong>{profile_track_html}</strong> profile at the <strong>{profile_seniority_html}</strong> level with approximately <strong>{profile_confidence_html}%</strong> confidence.</div></div>',
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
                
                next_steps: list[str] = []
                for missing in preview_structure["missing_sections"]:
                    next_steps.append(f"Add a {missing.lower()} section.")
                if preview_structure["word_count"] < 150:
                    next_steps.append(
                        "Expand the resume — under 150 words limits matching signal."
                    )
                if preview_structure["bullet_count"] < 3:
                    next_steps.append(
                        "Use bulleted achievements to surface quantified wins."
                    )
                if int(preview_profile["confidence"]) < 70:
                    next_steps.append(
                        "Strengthen the seniority signal with leadership or scope keywords."
                    )
                for skill in preview_profile["skills_missing"][:2]:
                    next_steps.append(f"Consider adding evidence of {skill.lower()}.")
                if not next_steps:
                    next_steps.append(
                        'Click "Analyze profile" to see salary ranges and matching roles.'
                    )

                steps_html = "".join(
                    f"<li>{escape(step)}</li>" for step in next_steps[:5]
                )
                st.markdown(
                    f'<div class="metric-card" style="margin-top:0.75rem;"><div class="metric-label">Next steps</div><ul class="next-steps-list">{steps_html}</ul></div>',
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
                    "No matching roles surfaced for this resume. Try expanding the resume text with more domain terms."
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



if __name__ == "__main__":
    main()
