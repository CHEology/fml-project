from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from datetime import date
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
load_public_assessment_artifacts = runtime.load_public_assessment_artifacts
load_retriever = runtime.load_retriever
load_salary_artifacts = runtime.load_salary_artifacts
load_wage_table = runtime.load_wage_table
public_resume_signals = runtime.public_resume_signals
retrieve_matches = runtime.retrieve_matches
salary_band_from_model = runtime.salary_band_from_model
salary_artifacts_ready = runtime.salary_artifacts_ready
apply_public_ats_fit = runtime.apply_public_ats_fit

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"

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
SAMPLE_FIELD_SKILLS = {
    "Machine Learning": [
        "Python",
        "PyTorch",
        "Feature Stores",
        "Vector Search",
        "Experiment Design",
        "Model Monitoring",
        "AWS",
        "Docker",
    ],
    "Data Science": [
        "Python",
        "SQL",
        "Causal Inference",
        "A/B Testing",
        "Forecasting",
        "Tableau",
        "Stakeholder Storytelling",
    ],
    "Software Engineering": [
        "Python",
        "Go",
        "Postgres",
        "Kubernetes",
        "Distributed Systems",
        "Observability",
        "CI/CD",
    ],
    "Analytics": [
        "SQL",
        "dbt",
        "Looker",
        "Data Modeling",
        "Metric Governance",
        "Python",
        "Executive Reporting",
    ],
    "Product / Strategy": [
        "Roadmapping",
        "Market Sizing",
        "Customer Research",
        "SQL",
        "Pricing",
        "Executive Communication",
        "Experiment Planning",
    ],
    "Human Resources": [
        "Talent Acquisition",
        "Workday",
        "Employee Relations",
        "Compensation Planning",
        "DEI Programs",
        "Manager Coaching",
        "People Analytics",
    ],
    "Finance / Accounting": [
        "FP&A",
        "Budgeting",
        "Revenue Recognition",
        "Excel",
        "NetSuite",
        "Audit Readiness",
        "Board Reporting",
    ],
    "Marketing": [
        "Lifecycle Marketing",
        "HubSpot",
        "Paid Social",
        "SEO",
        "Content Strategy",
        "Attribution",
        "Customer Segmentation",
    ],
    "Sales / Customer Success": [
        "Salesforce",
        "MEDDICC",
        "Pipeline Forecasting",
        "Renewal Strategy",
        "Executive Business Reviews",
        "Customer Health",
        "Revenue Operations",
    ],
    "Operations / Administration": [
        "Vendor Management",
        "Process Improvement",
        "Executive Support",
        "Facilities",
        "Procurement",
        "Documentation",
        "Scheduling",
    ],
    "Healthcare / Clinical": [
        "Epic",
        "Care Coordination",
        "Patient Safety",
        "Clinical Documentation",
        "Quality Improvement",
        "HIPAA",
        "Interdisciplinary Rounds",
    ],
    "Education / Teaching": [
        "Curriculum Design",
        "Differentiated Instruction",
        "Student Assessment",
        "Learning Analytics",
        "Classroom Management",
        "IEP Support",
        "Instructional Coaching",
    ],
    "Legal / Compliance": [
        "Contract Review",
        "Regulatory Analysis",
        "Privacy Compliance",
        "Negotiation",
        "Policy Drafting",
        "Risk Assessment",
        "Legal Research",
    ],
    "Design / Creative": [
        "Figma",
        "Design Systems",
        "User Research",
        "Prototyping",
        "Accessibility",
        "Visual Design",
        "Content Strategy",
    ],
    "Engineering / Hardware": [
        "SolidWorks",
        "DFMEA",
        "Manufacturing Transfer",
        "Tolerance Analysis",
        "Test Fixtures",
        "Supplier Quality",
        "MATLAB",
    ],
    "Research / Academia": [
        "Grant Writing",
        "Statistical Modeling",
        "IRB Protocols",
        "Peer-Reviewed Publications",
        "R",
        "Experimental Design",
        "Conference Presentations",
    ],
    "Public Sector / Policy": [
        "Policy Analysis",
        "Program Evaluation",
        "Stakeholder Engagement",
        "Legislative Research",
        "Grant Reporting",
        "Public Dashboards",
        "Community Outreach",
    ],
    "Hospitality / Service": [
        "Guest Experience",
        "Team Scheduling",
        "Inventory Control",
        "POS Systems",
        "Event Operations",
        "Vendor Coordination",
        "Service Recovery",
    ],
}
SAMPLE_RESUME_SPECS = [
    ("Alex Rivera", "Machine Learning", "Senior Machine Learning Engineer", "Senior", "New York, NY", "marketplace ranking", "excellent"),
    ("Priya Shah", "Data Science", "Product Data Scientist", "Senior", "San Francisco, CA", "consumer growth", "excellent"),
    ("Maya Hernandez", "Software Engineering", "Backend Platform Engineer", "Mid", "Austin, TX", "payments infrastructure", "excellent"),
    ("Ethan Brooks", "Analytics", "Analytics Engineer", "Mid", "Chicago, IL", "subscription metrics", "excellent"),
    ("Leila Hassan", "Product / Strategy", "Product Strategy Lead", "Lead / Executive", "Seattle, WA", "B2B pricing", "excellent"),
    ("Noah Patel", "Human Resources", "People Operations Manager", "Senior", "Boston, MA", "distributed teams", "excellent"),
    ("Ava Morales", "Finance / Accounting", "FP&A Manager", "Senior", "Denver, CO", "SaaS planning", "excellent"),
    ("Lucas Chen", "Marketing", "Lifecycle Marketing Lead", "Senior", "Remote", "B2B retention", "excellent"),
    ("Sofia Bennett", "Sales / Customer Success", "Enterprise Customer Success Manager", "Senior", "Atlanta, GA", "cloud accounts", "excellent"),
    ("Owen Park", "Operations / Administration", "Business Operations Manager", "Mid", "Los Angeles, CA", "multi-site operations", "solid"),
    ("Nadia Okafor", "Healthcare / Clinical", "Clinical Quality Improvement Nurse", "Senior", "Philadelphia, PA", "cardiology care", "excellent"),
    ("Grace Lee", "Education / Teaching", "Instructional Coach", "Senior", "Queens, NY", "middle school math", "excellent"),
    ("Marcus Reed", "Legal / Compliance", "Privacy Compliance Counsel", "Senior", "Washington, DC", "health technology", "excellent"),
    ("Iris Wong", "Design / Creative", "Senior Product Designer", "Senior", "Portland, OR", "mobile onboarding", "excellent"),
    ("Caleb Morgan", "Engineering / Hardware", "Mechanical Design Engineer", "Mid", "Detroit, MI", "EV battery systems", "excellent"),
    ("Hannah Stein", "Research / Academia", "Research Scientist", "Senior", "Cambridge, MA", "computational biology", "excellent"),
    ("Diego Alvarez", "Public Sector / Policy", "Policy Analyst", "Mid", "Sacramento, CA", "housing programs", "solid"),
    ("Emma Johnson", "Hospitality / Service", "Hotel Operations Manager", "Mid", "Miami, FL", "luxury guest services", "solid"),
    ("Sam Taylor", "Machine Learning", "Computer Vision Engineer", "Mid", "Pittsburgh, PA", "manufacturing inspection", "solid"),
    ("Rina Mehta", "Data Science", "Decision Scientist", "Mid", "Remote", "risk analytics", "solid"),
    ("Theo Martin", "Software Engineering", "Full Stack Engineer", "Associate", "Raleigh, NC", "healthcare scheduling", "solid"),
    ("Camila Torres", "Analytics", "Business Intelligence Analyst", "Associate", "Dallas, TX", "retail inventory", "solid"),
    ("Jasper Nguyen", "Product / Strategy", "Associate Product Manager", "Associate", "San Jose, CA", "developer tools", "solid"),
    ("Mina Ali", "Human Resources", "Talent Acquisition Partner", "Mid", "Minneapolis, MN", "clinical hiring", "solid"),
    ("Ben Carter", "Finance / Accounting", "Senior Accountant", "Mid", "Charlotte, NC", "manufacturing close", "solid"),
    ("Talia Green", "Marketing", "Demand Generation Manager", "Mid", "Phoenix, AZ", "cybersecurity pipeline", "solid"),
    ("Andre Wilson", "Sales / Customer Success", "Account Executive", "Mid", "Nashville, TN", "mid-market sales", "solid"),
    ("Lena Brooks", "Operations / Administration", "Executive Assistant", "Associate", "New York, NY", "founder support", "solid"),
    ("Mei Lin", "Healthcare / Clinical", "Physical Therapist", "Mid", "San Diego, CA", "orthopedic rehab", "solid"),
    ("Patrick O'Neill", "Education / Teaching", "High School Science Teacher", "Mid", "Columbus, OH", "project-based learning", "solid"),
    ("Alina Petrova", "Legal / Compliance", "Contracts Manager", "Mid", "Remote", "vendor agreements", "solid"),
    ("Jonah Weiss", "Design / Creative", "Brand Designer", "Mid", "Brooklyn, NY", "consumer packaging", "solid"),
    ("Sasha Kim", "Engineering / Hardware", "Electrical Engineer", "Associate", "San Jose, CA", "sensor boards", "solid"),
    ("Omar Farouk", "Research / Academia", "Postdoctoral Researcher", "Mid", "Ann Arbor, MI", "urban mobility", "solid"),
    ("Claire Dubois", "Public Sector / Policy", "Program Evaluation Specialist", "Mid", "Boston, MA", "workforce grants", "solid"),
    ("Mateo Garcia", "Hospitality / Service", "Restaurant General Manager", "Mid", "Orlando, FL", "high-volume dining", "solid"),
    ("Yuki Tanaka", "Machine Learning", "Junior ML Engineer", "Associate", "Seattle, WA", "recommendation prototypes", "solid"),
    ("Fatima Khan", "Data Science", "Data Analyst", "Associate", "Houston, TX", "energy operations", "solid"),
    ("Mason Wright", "Software Engineering", "Software Engineering Intern", "Intern / Entry", "Madison, WI", "campus tools", "thin"),
    ("Elena Rossi", "Analytics", "Junior Analyst", "Intern / Entry", "Tampa, FL", "dashboard support", "thin"),
    ("Daniel Kim", "Product / Strategy", "Product Intern", "Intern / Entry", "Remote", "student marketplace", "thin"),
    ("Aisha Brown", "Human Resources", "HR Coordinator", "Associate", "Baltimore, MD", "onboarding operations", "thin"),
    ("Liam Evans", "Finance / Accounting", "Accounting Assistant", "Associate", "Cleveland, OH", "accounts payable", "thin"),
    ("Nora Murphy", "Marketing", "Social Media Coordinator", "Associate", "Salt Lake City, UT", "local campaigns", "thin"),
    ("Luis Romero", "Sales / Customer Success", "Sales Development Representative", "Associate", "San Antonio, TX", "outbound prospecting", "thin"),
    ("Zara Ahmed", "Operations / Administration", "Administrative Coordinator", "Associate", "Las Vegas, NV", "office scheduling", "thin"),
    ("Victor Chen", "Healthcare / Clinical", "Medical Assistant", "Associate", "Fresno, CA", "primary care", "thin"),
    ("Molly Adams", "Education / Teaching", "Teaching Assistant", "Intern / Entry", "Ithaca, NY", "undergraduate tutoring", "thin"),
    ("Rachel Cohen", "Legal / Compliance", "Paralegal", "Associate", "Newark, NJ", "litigation support", "thin"),
    ("Chris Miller", "Design / Creative", "UX Design Intern", "Intern / Entry", "Remote", "portfolio redesign", "thin"),
]
SECTION_ALIASES = {
    "Summary": ["summary", "professional summary", "profile"],
    "Experience": [
        "experience",
        "professional experience",
        "work experience",
        "employment history",
    ],
    "Projects": [
        "projects",
        "selected projects",
        "research publications",
        "publications",
    ],
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

        .quality-card {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            background: var(--panel);
            box-shadow: 0 4px 12px __SHADOW__;
            margin-top: 0.75rem;
        }

        .quality-headline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            margin-bottom: 0.6rem;
        }

        .quality-overall {
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1;
        }

        .quality-band-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .quality-band-Strong { background: __SCORE_BG__; color: __SCORE_INK__; }
        .quality-band-Mixed { background: __PILL_BG__; color: __PILL_INK__; }
        .quality-band-Weak { background: rgba(181, 71, 8, 0.14); color: var(--warning); }
        .quality-band-Thin { background: rgba(181, 71, 8, 0.22); color: var(--warning); }

        .quality-subscores {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem 0.9rem;
            margin: 0.4rem 0 0.7rem 0;
        }

        .quality-sub-label {
            font-size: 0.78rem;
            color: var(--muted);
            margin-bottom: 0.2rem;
        }

        .quality-bar-track {
            background: rgba(15, 23, 42, 0.08);
            border-radius: 999px;
            height: 6px;
            overflow: hidden;
        }

        .quality-bar-fill {
            background: var(--accent);
            height: 6px;
            border-radius: 999px;
        }

        .quality-section-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-top: 0;
            margin-bottom: 0.3rem;
        }

        .quality-feedback-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 0.75rem;
        }

        .quality-feedback-panel {
            border-top: 1px solid var(--line);
            padding-top: 0.7rem;
        }

        .quality-list {
            margin: 0;
            padding-left: 1.1rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .quality-list.flags li { color: var(--warning); }
        .quality-list.strengths li { color: var(--success); }

        @media (max-width: 900px) {
            .quality-feedback-grid {
                grid-template-columns: 1fr;
            }

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


@st.cache_resource(show_spinner=False)
def load_public_assessment_resource():
    if not artifacts_ready(artifact_status(), "public_assessment"):
        return None
    return load_public_assessment_artifacts(PROJECT_ROOT)


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


PUBLIC_SECTION_MAP = {
    "Sum": "Summary",
    "Exp": "Experience",
    "Edu": "Education",
    "Skill": "Skills",
}


def enhance_structure_with_public_sections(
    structure: dict[str, Any],
    public_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    if not public_signals or not public_signals.get("ready"):
        return structure
    counts = public_signals.get("sections", {}).get("counts", {})
    found = list(structure.get("found_sections", []))
    for public_label, app_label in PUBLIC_SECTION_MAP.items():
        if counts.get(public_label, 0) > 0 and app_label not in found:
            found.append(app_label)
    missing = [label for label in SECTION_ALIASES if label not in found]
    updated = dict(structure)
    updated["found_sections"] = found
    updated["missing_sections"] = missing
    return updated


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


def generate_premade_sample_resume(
    spec: tuple[str, str, str, str, str, str, str],
    jobs: pd.DataFrame,
) -> str:
    """Render one curated sample profile from the 50-profile sample library."""
    name, track, title, seniority, location, domain, quality = spec
    seed = sum((idx + 1) * ord(char) for idx, char in enumerate("|".join(spec)))
    rng = np.random.default_rng(seed)
    slug = slugify_name(name)
    school = str(rng.choice(FAKE_SCHOOLS))
    skills = SAMPLE_FIELD_SKILLS.get(track, TRACK_SKILLS.get(track, FALLBACK_TRACK_SKILLS))
    primary, secondary, tertiary = skills[:3]
    metric_name = TRACK_METRICS.get(track, "operating quality")
    market_examples = choose_market_examples(jobs, track, location)
    companies = (
        market_examples["company_name"]
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(companies) < 2:
        companies.extend(company for company in FAKE_COMPANIES if company not in companies)
    company_a, company_b = companies[:2]

    if quality == "excellent":
        impact_one = int(rng.integers(24, 48))
        impact_two = int(rng.integers(16, 34))
        impact_three = int(rng.integers(8, 21))
        scope_one = int(rng.integers(6, 18))
        scope_two = int(rng.integers(120, 900))
        current_start = "2021"
        prior_range = "2017 - 2021"
        earlier_range = "2014 - 2017"
        project_count = int(rng.integers(3, 8))
        return f"""{name}
{title}
{location} | {slug}@example.com | linkedin.com/in/{slug} | github.com/{slug}

PROFESSIONAL SUMMARY
{seniority} {track.lower()} professional focused on {domain}. Known for combining {primary}, {secondary}, and {tertiary} with crisp operating judgment, measurable delivery, and cross-functional leadership across technical and business teams.

CORE SKILLS
{" | ".join(skills)}

EXPERIENCE
{company_a} | {title} | {current_start} - Present
- Led the {domain} roadmap across {scope_one} teams, using {primary} and {secondary} to improve {metric_name} by {impact_one}% while reducing review cycles by {impact_three}%.
- Built a weekly executive scorecard covering {scope_two:,}+ records, surfacing risk, ownership, and next actions for product, finance, and operations leaders.
- Mentored 4 team members on evidence-backed problem framing, raising peer review quality and shortening stakeholder turnaround by {impact_two}%.
- Partnered with legal, finance, and customer-facing teams to convert ambiguous requests into scoped delivery plans with clear metrics, owners, and launch criteria.

{company_b} | {compose_headline("Mid", title.replace("Senior ", "").replace("Lead ", ""))} | {prior_range}
- Owned a high-priority {domain} initiative from discovery through rollout, increasing adoption by {impact_two}% across {scope_one - 1 if scope_one > 7 else scope_one + 2} business units.
- Rebuilt the reporting and documentation workflow in {tertiary}, cutting manual reconciliation from 9 hours to 3 hours per week.
- Presented monthly findings to VP-level stakeholders and turned feedback into a sequenced roadmap with measurable acceptance criteria.

{str(rng.choice(FAKE_COMPANIES))} | Associate {track.split('/')[0].strip()} Specialist | {earlier_range}
- Supported operating reviews, QA checks, and customer research for a {domain} portfolio serving {int(scope_two / 2):,}+ users or records.
- Created reusable templates that improved handoffs between analysts, operators, and managers.

SELECTED PROJECTS
- {domain.title()} Control Room: Built a reusable dashboard and decision log connecting leading indicators, owner actions, and post-launch impact.
- Evidence Quality Rubric: Created a {project_count}-part review framework that helped teams distinguish activity updates from measurable outcomes.
- Stakeholder Briefing Pack: Standardized monthly leadership narratives with metric definitions, risks, and recommended next steps.

EDUCATION
- B.S. in {track.split('/')[0].strip()} / Business Analytics, {school}

CERTIFICATIONS
- Advanced {primary} for Practitioners
- Cross-Functional Leadership Workshop
"""

    if quality == "solid":
        impact_one = int(rng.integers(10, 28))
        impact_two = int(rng.integers(7, 19))
        return f"""{name}
{title}
{location} | {slug}@example.com | linkedin.com/in/{slug}

SUMMARY
{track} professional with practical experience in {domain}, {primary}, and {secondary}. Comfortable working with managers and cross-functional partners to improve recurring workflows.

SKILLS
{" | ".join(skills[:6])}

EXPERIENCE
{company_a} | {title} | 2020 - Present
- Managed day-to-day {domain} work using {primary} and {secondary}, improving {metric_name} by {impact_one}% over two planning cycles.
- Created status reports and process documentation for 5 stakeholder groups.
- Coordinated issue triage, follow-up actions, and handoffs between internal teams.

{company_b} | Associate {track.split('/')[0].strip()} Specialist | 2018 - 2020
- Supported reporting, quality checks, and stakeholder requests for a busy {domain} team.
- Reduced recurring manual work by {impact_two}% by standardizing templates and review steps.

PROJECTS
- {domain.title()} Tracker: Built a shared tracker for open issues, owners, deadlines, and basic trend reporting.

EDUCATION
- B.A. in Business Administration, {school}
"""

    return f"""{name}
{title}
{location} | {slug}@example.com

Summary
Motivated {track.lower()} candidate interested in {domain}. I am organized, dependable, and eager to grow.

Experience
{company_a} | {title} | 2023 - Present
- Responsible for helping with {domain} tasks.
- Worked on reports and team requests.

{company_b} | Intern | Summer 2022
- Helped the team with research and coordination.

Education
- Coursework in business and technology, {school}

Skills
- {", ".join(skills[:4])}
"""


def random_premade_sample_resume(
    jobs: pd.DataFrame,
    previous_index: int | None = None,
) -> tuple[str, str, int]:
    rng = np.random.default_rng()
    choices = list(range(len(SAMPLE_RESUME_SPECS)))
    if previous_index in choices and len(choices) > 1:
        choices.remove(previous_index)
    index = int(rng.choice(choices))
    spec = SAMPLE_RESUME_SPECS[index]
    text = generate_premade_sample_resume(spec, jobs)
    source = f"Random sample resume: {spec[0]}, {spec[2]}"
    return text, source, index


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
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                if not text.strip():
                    text = page.extract_text() or ""
                pages.append(text)
        return clean_extracted_pdf_text("\n".join(pages))

    return ""


def clean_extracted_pdf_text(text: str) -> str:
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


PRESTIGIOUS_COMPANY_TOKENS = (
    "google", "alphabet", "apple", "microsoft", "meta", "facebook", "amazon",
    "netflix", "openai", "anthropic", "deepmind", "nvidia",
    "stripe", "databricks", "snowflake", "palantir", "spacex", "tesla",
    "airbnb", "uber", "linkedin", "tiktok", "bytedance",
    "mckinsey", "boston consulting", "bain & company", "bain and company",
    "goldman sachs", "morgan stanley", "j.p. morgan", "jpmorgan",
    "blackstone", "blackrock",
    "citadel", "jane street", "two sigma", "renaissance technologies", "de shaw",
    "jump trading", "hudson river trading", "point72",
    "mayo clinic", "johns hopkins", "cleveland clinic", "memorial sloan kettering",
    "massachusetts general", "stanford health",
    "cravath", "wachtell", "sullivan & cromwell", "latham & watkins",
    "skadden", "kirkland & ellis", "davis polk",
    "nasa", "lawrence livermore", "los alamos", "fermilab", "bell labs",
    "national institutes of health",
)
PRESTIGIOUS_EDUCATION_TOKENS = (
    "stanford university", "stanford",
    "massachusetts institute of technology", "mit",
    "harvard university", "harvard",
    "princeton university", "princeton",
    "yale university", "yale",
    "caltech", "california institute of technology",
    "uc berkeley", "university of california berkeley", "berkeley",
    "carnegie mellon", "cmu",
    "university of chicago",
    "columbia university", "columbia",
    "cornell university", "cornell",
    "university of pennsylvania", "upenn",
    "oxford", "university of oxford",
    "cambridge", "university of cambridge",
    "eth zurich",
    "tsinghua university", "tsinghua",
    "peking university",
)

RIGOROUS_TITLE_TOKENS = (
    "software engineer", "machine learning engineer", "research scientist",
    "data scientist", "research engineer", "applied scientist",
    "member of technical staff", "technical staff", "staff research scientist",
    "physician", "surgeon", "resident", "attending",
    "attorney", "lawyer", "associate attorney",
    "investment banker", "investment banking",
    "quantitative researcher", "quant trader", "quantitative analyst",
    "actuary", "tax lawyer",
    "principal", "staff engineer", "senior engineer", "senior scientist",
    "director", "chief", "vice president", "head of",
    "fellow", "professor", "postdoctoral", "postdoc",
)

LOW_RIGOR_TITLE_TOKENS = (
    "cashier", "server", "barista", "waiter", "waitress", "bartender",
    "retail associate", "sales associate", "store associate",
    "receptionist", "host", "hostess", "cleaner", "janitor",
    "tutor", "babysitter", "delivery driver", "lifeguard", "camp counselor",
    "front desk", "stocker", "dishwasher", "valet",
)


VAGUE_PHRASES = (
    "worked on",
    "helped with",
    "involved in",
    "responsible for",
    "various projects",
    "many projects",
    "assisted with",
    "participated in",
    "contributed to",
    "exposed to",
    "familiar with",
)

ACTION_VERBS = (
    "built",
    "shipped",
    "deployed",
    "launched",
    "led",
    "scaled",
    "architected",
    "optimized",
    "reduced",
    "increased",
    "automated",
    "migrated",
    "owned",
    "designed",
    "implemented",
    "delivered",
    "developed",
    "drove",
    "spearheaded",
    "authored",
)

SCHOOL_PROJECT_MARKERS = (
    "course project",
    "class project",
    "coursework",
    "capstone",
    "thesis",
    "hackathon",
    "university project",
    "academic project",
    "school project",
    "for cs",
    "csci",
    "cs10",
    "cs20",
    "cs30",
    "cs40",
    "as part of cs",
    "team project for",
    "homework",
)

IMPACT_REGEX = re.compile(
    r"(?:\$\s?\d[\d,\.]*\s?(?:k|m|million|billion|b)?\b"
    r"|\b\d+(?:\.\d+)?\s?%"
    r"|\b\d+(?:\.\d+)?\s?(?:x|×)\b"
    r"|\b\d{2,}\s?(?:users|customers|clients|requests|qps|ms|hours|days|weeks|months|years|reps|teams|engineers|orders|tickets|sessions|deals|contracts|patients|students)\b)",
    re.IGNORECASE,
)

_MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
)
_MONTH_INDEX = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
DATE_RANGE_REGEX = re.compile(
    r"(?:(?P<m1>" + "|".join(_MONTHS) + r")\s*)?"
    r"(?P<y1>20\d{2}|19\d{2})(?:[./](?P<n1>\d{1,2}))?\s*[-–—]+\s*"
    r"(?:\(?\s*(?P<present>present|current|now|today|expected)\s*\)?|"
    r"(?:(?P<m2>" + "|".join(_MONTHS) + r")\s*)?"
    r"(?P<y2>20\d{2}|19\d{2})(?:[./](?P<n2>\d{1,2}))?)",
    re.IGNORECASE,
)
WORK_SECTION_LABELS = {
    "experience",
    "professional experience",
    "work experience",
    "employment",
    "employment history",
    "career history",
    "internships",
}
NON_WORK_SECTION_LABELS = {
    "education",
    "honors",
    "honors and awards",
    "awards",
    "certifications",
    "licenses",
    "skills",
    "technical skills",
    "projects",
    "selected projects",
    "publications",
    "activities",
}
NON_WORK_DATE_TOKENS = (
    "honor",
    "dean's list",
    "deans list",
    "award",
    "scholarship",
    "education",
    "university",
    "college",
    "school",
    "b.s.",
    "b.a.",
    "m.s.",
    "m.a.",
    "ph.d",
    "gpa",
    "graduation",
    "expected",
    "coursework",
    "certification",
    "license",
)


def _section_label_from_line(line: str) -> str | None:
    label = re.sub(r"[^a-z/& ]+", "", line.lower()).strip(" :")
    label = re.sub(r"\s+", " ", label)
    if not label or len(label.split()) > 4:
        return None
    if label in WORK_SECTION_LABELS or label in NON_WORK_SECTION_LABELS:
        return label
    return None


def _date_context_looks_like_non_work(line: str, context: str, section: str | None) -> bool:
    lowered_line = line.lower()
    lowered_context = context.lower()
    if section in NON_WORK_SECTION_LABELS:
        return True
    if section in WORK_SECTION_LABELS:
        return False
    if any(token in lowered_line for token in NON_WORK_DATE_TOKENS):
        work_tokens = RIGOROUS_TITLE_TOKENS + LOW_RIGOR_TITLE_TOKENS + (
            "manager",
            "engineer",
            "analyst",
            "designer",
            "nurse",
            "teacher",
            "attorney",
            "coordinator",
            "specialist",
            "associate",
            "consultant",
            "intern",
        )
        return not any(token in lowered_context for token in work_tokens)
    return False


def academic_cv_signals(text: str) -> dict[str, Any]:
    lowered = text.lower()
    school_hits = [
        school
        for school in PRESTIGIOUS_EDUCATION_TOKENS
        if re.search(r"(?<![a-z0-9])" + re.escape(school) + r"(?![a-z0-9])", lowered)
    ]
    prestigious_education = [
        school
        for school in school_hits
        if not any(
            school != other and school in other and other in school_hits
            for other in school_hits
        )
    ]
    degree_hits = sum(
        1
        for token in (
            "ph.d",
            "phd",
            "doctor of philosophy",
            "m.sc",
            "m.s.",
            "b.sc",
            "b.s.",
        )
        if token in lowered
    )
    publication_count = len(
        re.findall(
            r"\b(?:journal articles|preprints|physical review letters|journal of high energy physics|conference|proceedings|preprint)\b",
            lowered,
        )
    )
    numbered_publications = len(re.findall(r"(?m)^\s*\d+\s+[A-Z][^,\n]+,", text))
    publication_count = max(publication_count, numbered_publications)
    award_count = len(
        re.findall(
            r"\b(?:fellowship|award|prize|honor|highest honor|presidential award|dean's list)\b",
            lowered,
        )
    )
    referee_count = len(re.findall(r"\b(?:referee|reviewer|review service)\b", lowered))
    return {
        "prestigious_education": sorted(set(prestigious_education)),
        "prestigious_education_count": len(set(prestigious_education)),
        "degree_hits": degree_hits,
        "publication_count": publication_count,
        "award_count": award_count,
        "referee_count": referee_count,
    }


def _months_between(y1: int, m1: int, y2: int, m2: int) -> int:
    return max(0, (y2 - y1) * 12 + (m2 - m1) + 1)


def _date_match_month(match: re.Match[str], named_month: str, numeric_month: str, default: int) -> int:
    month_name = match.group(named_month)
    if month_name:
        return _MONTH_INDEX.get(month_name.lower(), default)
    month_number = match.group(numeric_month)
    if month_number:
        return max(1, min(12, int(month_number)))
    return default


def extract_work_history(text: str) -> dict[str, Any]:
    """Parse date ranges from resume text and quantify employment.

    Discounts internships and academic roles vs full-time months.
    """
    if not text.strip():
        return {
            "total_ft_months": 0,
            "total_intern_months": 0,
            "weighted_ft_months": 0,
            "role_count": 0,
            "ft_role_count": 0,
            "internship_role_count": 0,
            "rigorous_role_count": 0,
            "low_rigor_role_count": 0,
            "prestigious_company_count": 0,
            "max_seniority_keyword": None,
            "has_progression": False,
            "spans": [],
        }

    today_y, today_m = date.today().year, date.today().month
    spans: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, int, int]] = set()

    lines = _resume_lines(text)
    current_section: str | None = None
    for idx, line in enumerate(lines):
        section_label = _section_label_from_line(line)
        if section_label is not None:
            current_section = section_label
            continue

        window = " ".join(lines[max(0, idx - 1) : min(len(lines), idx + 2)])
        if _date_context_looks_like_non_work(line, window, current_section):
            continue

        for match in DATE_RANGE_REGEX.finditer(line):
            y1 = int(match.group("y1"))
            m1 = _date_match_month(match, "m1", "n1", 1)
            if match.group("present"):
                y2, m2 = today_y, today_m
            else:
                y2 = int(match.group("y2"))
                m2 = _date_match_month(match, "m2", "n2", 12)
            if (y2, m2) < (y1, m1):
                continue
            key = (y1, m1, y2, m2)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            months = _months_between(y1, m1, y2, m2)
            if months > 240:
                continue

            context = window.lower()

            is_intern = any(
                token in context
                for token in (
                    "intern",
                    "internship",
                    "co-op",
                    "coop",
                    "practicum",
                    "summer associate",
                )
            )
            is_academic = any(
                token in context
                for token in (
                    "teaching assistant",
                    "research assistant",
                    " ta ",
                    " ra ",
                    "graduate student",
                    "phd student",
                    "undergraduate research",
                )
            )
            seniority_kw = None
            for kw, label in (
                ("chief", "Lead / Executive"),
                ("vice president", "Lead / Executive"),
                ("director", "Lead / Executive"),
                ("principal", "Lead / Executive"),
                ("staff ", "Lead / Executive"),
                ("lead ", "Senior"),
                ("senior", "Senior"),
                ("sr.", "Senior"),
                ("associate", "Associate"),
                ("junior", "Intern / Entry"),
                ("intern", "Intern / Entry"),
            ):
                if kw in context:
                    seniority_kw = label
                    break

            rigorous_title = any(token in context for token in RIGOROUS_TITLE_TOKENS)
            low_rigor_title = any(token in context for token in LOW_RIGOR_TITLE_TOKENS)
            prestigious_company = any(
                token in context for token in PRESTIGIOUS_COMPANY_TOKENS
            )

            # Per-span rigor weight in [0.2, 1.0]: thin "default" job at 0.55,
            # bumped up by rigorous-title or prestigious-company evidence,
            # pulled down by clearly low-rigor titles. Internships are scored
            # separately by the internship multiplier later.
            weight = 0.55
            if rigorous_title:
                weight += 0.25
            if prestigious_company:
                weight += 0.25
            if low_rigor_title:
                weight = min(weight, 0.3)
            weight = max(0.2, min(1.0, weight))

            spans.append(
                {
                    "months": months,
                    "weight": weight,
                    "is_intern": is_intern,
                    "is_academic": is_academic,
                    "rigorous_title": rigorous_title,
                    "low_rigor_title": low_rigor_title,
                    "prestigious_company": prestigious_company,
                    "seniority_kw": seniority_kw,
                    "context": context,
                    "line": line,
                    "section": current_section,
                }
            )

    ft_spans = [s for s in spans if not s["is_intern"] and not s["is_academic"]]
    intern_spans = [s for s in spans if s["is_intern"] or s["is_academic"]]

    ft_months = sum(s["months"] for s in ft_spans)
    intern_months = sum(s["months"] for s in intern_spans)
    ft_role_count = len(ft_spans)
    internship_role_count = len(intern_spans)

    # Weighted FT months: each span's months scaled by its rigor weight.
    # Internships get an additional flat 0.15 multiplier on top of their weight.
    weighted_ft_months = int(round(sum(s["months"] * s["weight"] for s in ft_spans)))
    weighted_intern_contribution = int(
        round(sum(s["months"] * s["weight"] * 0.15 for s in intern_spans))
    )

    rigorous_role_count = sum(
        1 for s in ft_spans if s["rigorous_title"] or s["prestigious_company"]
    )
    low_rigor_role_count = sum(1 for s in ft_spans if s["low_rigor_title"])
    prestigious_company_count = sum(1 for s in spans if s["prestigious_company"])

    progression_order = [
        "Intern / Entry",
        "Associate",
        "Mid",
        "Senior",
        "Lead / Executive",
    ]
    seen_levels = [s["seniority_kw"] for s in spans if s["seniority_kw"]]
    indices = [progression_order.index(lv) for lv in seen_levels if lv in progression_order]
    has_progression = len(indices) >= 2 and indices[-1] > indices[0]

    max_seniority_keyword = None
    if indices:
        max_seniority_keyword = progression_order[max(indices)]

    return {
        "total_ft_months": ft_months,
        "total_intern_months": intern_months,
        "weighted_ft_months": weighted_ft_months + weighted_intern_contribution,
        "role_count": len(spans),
        "ft_role_count": ft_role_count,
        "internship_role_count": internship_role_count,
        "rigorous_role_count": rigorous_role_count,
        "low_rigor_role_count": low_rigor_role_count,
        "prestigious_company_count": prestigious_company_count,
        "max_seniority_keyword": max_seniority_keyword,
        "has_progression": has_progression,
        "spans": spans,
    }


def _split_project_blocks(text: str) -> list[str]:
    """Heuristically pull project/experience bullet blocks for quality scoring."""
    if not text.strip():
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith(("-", "*", "•")):
            current.append(line.lstrip("-*• ").strip())
        else:
            if current:
                blocks.append(" ".join(current))
                current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def score_projects(text: str) -> dict[str, Any]:
    blocks = _split_project_blocks(text)
    if not blocks:
        return {
            "n_total": 0,
            "n_school": 0,
            "n_vague_dominant": 0,
            "vague_total": 0,
            "action_total": 0,
            "impact_total": 0,
            "mean_score": 0,
        }

    n_school = 0
    n_vague_dominant = 0
    vague_total = 0
    action_total = 0
    impact_total = 0
    block_scores: list[int] = []

    for block in blocks:
        lower = block.lower()
        is_school = any(marker in lower for marker in SCHOOL_PROJECT_MARKERS)
        vague_count = sum(1 for phrase in VAGUE_PHRASES if phrase in lower)
        action_count = sum(
            1 for verb in ACTION_VERBS if re.search(r"\b" + verb + r"\b", lower)
        )
        impact_count = len(IMPACT_REGEX.findall(block))

        score = 0
        if is_school:
            score -= 25
        score -= vague_count * 12
        score += action_count * 10
        score += impact_count * 18
        score = max(-100, min(100, score))

        block_scores.append(score)
        if is_school:
            n_school += 1
        if vague_count >= 2 and impact_count == 0:
            n_vague_dominant += 1
        vague_total += vague_count
        action_total += action_count
        impact_total += impact_count

    mean_score = int(sum(block_scores) / len(block_scores)) if block_scores else 0
    return {
        "n_total": len(blocks),
        "n_school": n_school,
        "n_vague_dominant": n_vague_dominant,
        "vague_total": vague_total,
        "action_total": action_total,
        "impact_total": impact_total,
        "mean_score": mean_score,
    }


def _resume_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]


def _quote_resume_line(line: str, max_chars: int = 120) -> str:
    cleaned = line.lstrip("-*• ").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return f'"{cleaned}"'


def _find_resume_line(
    text: str,
    predicate,
    *,
    default: str = "",
) -> str:
    for line in _resume_lines(text):
        if predicate(line):
            return _quote_resume_line(line)
    return default


def _find_first_bullet(text: str) -> str:
    return _find_resume_line(
        text,
        lambda line: line.startswith(("-", "*", "•")),
        default="",
    )


def _work_history_date_example(work_history: dict[str, Any]) -> str:
    spans = work_history.get("spans") or []
    for span in spans:
        if span.get("is_intern") or span.get("is_academic"):
            continue
        line = str(span.get("line") or "").strip()
        if line:
            return _quote_resume_line(line)
    for span in spans:
        line = str(span.get("line") or "").strip()
        if line:
            return _quote_resume_line(line)
    return ""


def _select_feedback_items(
    items: list[str],
    overall: int,
    *,
    positive: bool,
) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)

    if positive:
        limit = 5 if overall >= 85 else 4 if overall >= 70 else 3 if overall >= 50 else 2
    else:
        limit = 1 if overall >= 85 else 2 if overall >= 70 else 3 if overall >= 50 else 4
        if overall < 30:
            limit = 5

    if not unique:
        return []
    return unique[: max(1, min(5, limit))]


def _format_school_name(name: str) -> str:
    overrides = {
        "uc berkeley": "UC Berkeley",
        "mit": "MIT",
        "cmu": "CMU",
        "eth zurich": "ETH Zurich",
        "upenn": "UPenn",
    }
    lowered = name.lower()
    return overrides.get(lowered, name.title())


def _seniority_from_ft_months(months: int) -> str:
    """Strict, prevalent seniority bands keyed off rigor-weighted FT months.

    Roughly: Entry <1y, Associate 1–3y, Mid 3–7y, Senior 7–12y, Lead 12y+.
    These are intentionally tighter than the original thresholds so that
    low-rigor or short-tenure resumes can't claim Senior on tenure alone.
    """
    if months < 12:
        return "Intern / Entry"
    if months < 36:
        return "Associate"
    if months < 84:
        return "Mid"
    if months < 144:
        return "Senior"
    return "Lead / Executive"


def assess_quality(
    text: str,
    profile: dict[str, Any],
    structure: dict[str, Any],
    work_history: dict[str, Any],
    projects: dict[str, Any],
    public_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute HR-style quality sub-scores, red flags, and strengths."""
    word_count = int(structure.get("word_count", 0))
    bullet_count = int(structure.get("bullet_count", 0))
    sections_found = len(structure.get("found_sections", []))
    sections_total = len(SECTION_ALIASES)
    ft_months = int(work_history.get("total_ft_months", 0))
    intern_months = int(work_history.get("total_intern_months", 0))
    weighted_months = int(work_history.get("weighted_ft_months", 0))
    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    low_rigor_role_count = int(work_history.get("low_rigor_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))
    academic = academic_cv_signals(text)
    publication_count = int(academic["publication_count"])
    award_count = int(academic["award_count"])
    prestigious_education_count = int(academic["prestigious_education_count"])
    is_research_cv = publication_count >= 3 or (
        prestigious_education_count > 0 and int(academic["degree_hits"]) > 0
    )

    # Experience sub-score: dominated by rigor-weighted FT months, with a
    # progression bonus and a prestige bonus.
    if ft_months <= 0 and intern_months <= 0:
        experience_score = 5 if word_count > 0 else 0
    else:
        experience_score = min(100, int(weighted_months * 1.0 + intern_months * 0.2))
        if work_history.get("has_progression"):
            experience_score = min(100, experience_score + 8)
        if prestigious_company_count > 0:
            experience_score = min(100, experience_score + 6)
    if is_research_cv:
        academic_experience_bonus = (
            prestigious_education_count * 8
            + min(publication_count, 12) * 2
            + min(award_count, 4) * 3
            + int(academic["degree_hits"]) * 4
        )
        experience_score = min(100, experience_score + min(40, academic_experience_bonus))

    # Impact sub-score: quantified outcomes density.
    impact_total = int(projects.get("impact_total", 0))
    if bullet_count == 0:
        impact_score = 0
    else:
        density = impact_total / max(1, bullet_count)
        impact_score = min(100, int(density * 220))
    if publication_count >= 3:
        research_impact_score = min(
            100,
            45
            + publication_count * 4
            + min(award_count, 4) * 5
            + min(int(academic["referee_count"]), 4) * 3,
        )
        impact_score = max(impact_score, research_impact_score)

    # Specificity sub-score: action verbs vs vague phrases.
    action_total = int(projects.get("action_total", 0))
    vague_total = int(projects.get("vague_total", 0))
    raw_specificity = action_total * 12 - vague_total * 18 + 30
    if publication_count >= 3:
        raw_specificity += 25
    if prestigious_education_count > 0:
        raw_specificity += 8
    specificity_score = max(0, min(100, raw_specificity))

    # Structure sub-score: sections present + bullets + reasonable length.
    structure_score = 0
    if sections_total:
        structure_score += int(60 * sections_found / sections_total)
    if bullet_count >= 6:
        structure_score += 25
    elif bullet_count >= 3:
        structure_score += 15
    if 350 <= word_count <= 1200:
        structure_score += 15
    elif word_count >= 200:
        structure_score += 8
    if is_research_cv and publication_count >= 3:
        structure_score += 12
    structure_score = max(0, min(100, structure_score))

    overall = int(
        experience_score * 0.45
        + impact_score * 0.20
        + specificity_score * 0.20
        + structure_score * 0.15
    )

    # School-project penalty applied to overall.
    n_total = int(projects.get("n_total", 0))
    n_school = int(projects.get("n_school", 0))
    if n_total and n_school / n_total > 0.5:
        overall = max(0, overall - 12)
    if is_research_cv:
        academic_overall_bonus = min(
            12,
            prestigious_education_count * 2
            + min(publication_count, 10) // 2
            + min(award_count, 3),
        )
        overall = min(100, overall + academic_overall_bonus)

    if overall >= 75:
        band_label = "Strong"
    elif overall >= 55:
        band_label = "Mixed"
    elif overall >= 30:
        band_label = "Weak"
    else:
        band_label = "Thin"

    red_flags: list[str] = []
    strengths: list[str] = []
    date_example = _work_history_date_example(work_history)
    quantified_example = _find_resume_line(
        text,
        lambda line: IMPACT_REGEX.search(line) is not None,
    )
    action_example = _find_resume_line(
        text,
        lambda line: any(
            re.search(r"\b" + verb + r"\b", line.lower()) for verb in ACTION_VERBS
        ),
    )
    vague_example = _find_resume_line(
        text,
        lambda line: any(phrase in line.lower() for phrase in VAGUE_PHRASES),
    )
    first_bullet = _find_first_bullet(text)
    skills_example = _find_resume_line(
        text,
        lambda line: "skill" in line.lower()
        or any(skill.lower() in line.lower() for skill in SAMPLE_FIELD_SKILLS.get(profile["track"], [])[:4]),
    )

    if ft_months == 0 and intern_months == 0:
        red_flags.append(
            "No parseable employment date ranges were detected, so seniority defaults to Entry."
        )
    elif ft_months < 6 and intern_months > 0:
        red_flags.append(
            "Experience reads as internships only; full-time evidence is limited"
            + (f" despite date evidence like {date_example}." if date_example else ".")
        )

    if n_total and n_school / max(1, n_total) > 0.5:
        red_flags.append(
            "Most listed work appears to be coursework or class projects, which weakens job-level evidence."
        )
    if low_rigor_role_count > 0 and rigorous_role_count == 0:
        red_flags.append(
            "Listed roles read as service or retail positions, so the resume needs stronger role-scope evidence."
        )
    if vague_total >= 3 and impact_total == 0:
        red_flags.append(
            "Frequent vague phrasing appears without measurable outcomes"
            + (f", for example {vague_example}." if vague_example else ".")
        )
    if impact_total == 0 and bullet_count >= 3 and not is_research_cv:
        red_flags.append(
            "No quantified outcomes were detected across the bullet list"
            + (f"; add scale to lines like {first_bullet}." if first_bullet else ".")
        )
    if word_count < 150:
        red_flags.append(
            f"Resume is short at {word_count} words, which limits evidence depth."
        )
    missing_sections = structure.get("missing_sections") or []
    if len(missing_sections) >= 3:
        red_flags.append(
            "Several standard sections missing: " + ", ".join(missing_sections[:3]) + "."
        )
    elif missing_sections and structure_score < 70:
        red_flags.append(
            "Resume organization could be clearer; missing "
            + ", ".join(missing_sections[:2])
            + "."
        )
    if impact_score < 45 and impact_total > 0 and bullet_count >= 6 and not is_research_cv:
        red_flags.append(
            "Only a small share of bullets are quantified; add more numbers beyond "
            + quantified_example
            + "."
        )
    if specificity_score < 45 and vague_total > 0:
        red_flags.append(
            "Some bullets still rely on broad activity language instead of concrete ownership."
        )
    if experience_score < 45 and ft_months > 0:
        red_flags.append(
            "Work-history depth is still limited relative to the claimed level."
        )

    if weighted_months >= 24 and date_example:
        strengths.append(
            "Multi-year work history is supported by date evidence"
            + (f" such as {date_example}." if date_example else ".")
        )
    if weighted_months >= 60 and int(work_history.get("ft_role_count", 0)) >= 2:
        strengths.append("Experience spans multiple full-time roles with enough tenure to support level calibration.")
    if prestigious_company_count > 0:
        strengths.append("History includes recognized, selective employers or labs.")
    if prestigious_education_count > 0:
        schools = ", ".join(
            _format_school_name(str(school))
            for school in academic["prestigious_education"][:3]
        )
        strengths.append(
            f"Educational background includes highly selective institutions: {schools}."
        )
    if publication_count >= 5:
        strengths.append(
            f"Research output is unusually strong, with {publication_count}+ publications or preprints detected."
        )
    elif publication_count >= 3:
        strengths.append(
            f"Research output is substantial, with {publication_count}+ publications or preprints detected."
        )
    if award_count >= 2:
        strengths.append(
            f"Awards and fellowships add external validation, with {award_count} honor signals detected."
        )
    if rigorous_role_count >= 2:
        strengths.append("Multiple rigorous, high-bar roles in the work history.")
    if work_history.get("has_progression"):
        strengths.append("Title progression visible across roles.")
    if impact_total >= 3:
        strengths.append(
            "Several bullets contain quantified impact"
            + (f", including {quantified_example}." if quantified_example else ".")
        )
    elif impact_total >= 1:
        strengths.append(
            "At least one bullet shows quantified impact"
            + (f": {quantified_example}." if quantified_example else ".")
        )
    if action_total >= 5:
        strengths.append(
            "Action verbs create a strong ownership signal"
            + (f", as in {action_example}." if action_example else ".")
        )
    if specificity_score >= 70 and first_bullet:
        strengths.append(
            f"Bullet phrasing is reasonably concrete about work performed, for example {first_bullet}."
        )
    if sections_found == sections_total:
        strengths.append("All standard resume sections present.")
    elif sections_found >= 3:
        strengths.append(
            "The resume has a usable structure with sections including "
            + ", ".join(structure.get("found_sections", [])[:3])
            + "."
        )
    if 350 <= word_count <= 1200:
        strengths.append(
            f"The resume has enough depth for review at {word_count} words without becoming bloated."
        )

    if public_signals and public_signals.get("ready"):
        public_domain = public_signals.get("domain", {})
        domain_confidence = float(public_domain.get("confidence", 0.0) or 0.0)
        if domain_confidence >= 0.18:
            strengths.append(
                "Public resume-domain model independently recognizes this as a "
                f"{str(public_domain.get('label', 'professional')).title()} profile."
            )
        public_sections = public_signals.get("sections", {}).get("counts", {})
        if public_sections.get("Exp", 0) and public_sections.get("Edu", 0):
            strengths.append(
                "Public section model finds separate experience and education evidence."
            )
        public_entities = public_signals.get("entities", {}).get("counts", {})
        entity_evidence = [
            label
            for label in ("Companies worked at", "College Name", "Degree", "Designation", "Skills")
            if int(public_entities.get(label, 0)) > 0
        ]
        if len(entity_evidence) >= 2:
            strengths.append(
                "Public entity model detects structured resume evidence: "
                + ", ".join(entity_evidence[:3])
                + "."
            )

    if not strengths:
        if skills_example:
            strengths.append(
                f"The profile has at least one concrete domain signal in {skills_example}."
            )
        elif profile.get("track"):
            strengths.append(
                f"The language is specific enough to infer a {profile['track']} focus."
            )
        else:
            strengths.append("The resume provides an initial base for evaluation.")

    if not red_flags:
        weakest = min(
            (
                ("experience", experience_score),
                ("quantified impact", impact_score),
                ("specificity", specificity_score),
                ("structure", structure_score),
            ),
            key=lambda item: item[1],
        )[0]
        if weakest == "quantified impact" and first_bullet:
            red_flags.append(
                f"The next improvement is sharper quantified impact; add outcome scale to {first_bullet}."
            )
        elif weakest == "structure":
            red_flags.append(
                "Structure is the main remaining constraint; add or clarify standard sections."
            )
        elif weakest == "specificity" and first_bullet:
            red_flags.append(
                f"Some phrasing could still be more specific about ownership and scope, starting with {first_bullet}."
            )
        else:
            red_flags.append(
                "The resume is strong overall, but it can still add more context on scope, collaborators, and business outcome."
            )

    return {
        "overall": overall,
        "band_label": band_label,
        "experience_score": experience_score,
        "impact_score": impact_score,
        "specificity_score": specificity_score,
        "structure_score": structure_score,
        "red_flags": _select_feedback_items(red_flags, overall, positive=False),
        "strengths": _select_feedback_items(strengths, overall, positive=True),
    }


def apply_quality_discount(
    band: dict[str, Any] | None, quality: dict[str, Any]
) -> dict[str, Any] | None:
    """Apply a multiplicative haircut to all salary quantiles when evidence is thin.

    Surfaces qualitative reasoning notes (no coefficients) that the renderer
    can append to the existing evidence line.
    """
    if band is None:
        return None
    multiplier = 1.0
    notes: list[str] = []
    if quality.get("experience_score", 100) < 40:
        multiplier *= 0.90
        notes.append("Adjusted downward — limited verified employment history.")
    if quality.get("impact_score", 100) < 30:
        multiplier *= 0.92
        notes.append("Adjusted downward — projects lack quantified outcomes.")
    if quality.get("specificity_score", 100) < 40:
        multiplier *= 0.95
        notes.append("Adjusted downward — descriptions are vague, hard to verify scope.")
    if multiplier >= 0.999:
        return band

    adjusted = dict(band)
    for key in ("q10", "q25", "q50", "q75", "q90"):
        value = adjusted.get(key)
        if value is None:
            continue
        try:
            adjusted[key] = int(round(float(value) * multiplier))
        except (TypeError, ValueError):
            continue
    adjusted["adjustment_notes"] = notes[:2]
    return adjusted


def assess_capability_tier(
    text: str,
    profile: dict[str, Any],
    quality: dict[str, Any],
    work_history: dict[str, Any],
    projects: dict[str, Any],
    public_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate within-level capability strength for salary calibration.

    This is intentionally separate from seniority. Seniority answers "what
    level is supported?" Capability answers "within that level, how strong is
    the evidence?"
    """
    lowered = text.lower()
    track = str(profile.get("track", ""))
    track_skills = SAMPLE_FIELD_SKILLS.get(track, TRACK_SKILLS.get(track, []))
    skill_hits = [
        skill for skill in track_skills if skill.lower().replace("/", " ") in lowered
    ]
    skill_depth = min(100.0, len(skill_hits) / max(4, len(track_skills[:8])) * 100.0)

    academic = academic_cv_signals(text)
    publication_count = int(academic["publication_count"])
    award_count = int(academic["award_count"])
    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))
    prestige_score = min(
        100.0,
        rigorous_role_count * 22
        + prestigious_company_count * 18
        + publication_count * 4
        + award_count * 5,
    )

    public_score = 0.0
    if public_signals and public_signals.get("ready"):
        domain_conf = float(public_signals.get("domain", {}).get("confidence", 0.0) or 0.0)
        entity_counts = public_signals.get("entities", {}).get("counts", {})
        public_score = min(
            100.0,
            domain_conf * 40.0
            + int(entity_counts.get("Skills", 0)) * 8
            + int(entity_counts.get("Designation", 0)) * 8
            + int(entity_counts.get("Companies worked at", 0)) * 6
            + int(entity_counts.get("Degree", 0)) * 5,
        )

    project_score = max(
        0.0,
        min(
            100.0,
            float(projects.get("mean_score", 0)) + 45.0,
        ),
    )
    score = (
        float(quality.get("impact_score", 0)) * 0.24
        + float(quality.get("specificity_score", 0)) * 0.18
        + float(quality.get("experience_score", 0)) * 0.16
        + skill_depth * 0.16
        + prestige_score * 0.12
        + project_score * 0.08
        + public_score * 0.06
    )
    score = max(0.0, min(100.0, score))

    if score >= 78:
        tier = "Standout"
        salary_multiplier = 1.10
        summary = "top-of-level evidence"
    elif score >= 55:
        tier = "Competitive"
        salary_multiplier = 1.00
        summary = "market-level evidence"
    else:
        tier = "Developing"
        salary_multiplier = 0.93
        summary = "thin within-level evidence"

    notes: list[str] = []
    if skill_hits:
        notes.append("Track-specific skills detected: " + ", ".join(skill_hits[:4]) + ".")
    if float(quality.get("impact_score", 0)) >= 70:
        notes.append("Impact evidence is strong for the claimed level.")
    elif float(quality.get("impact_score", 0)) < 40:
        notes.append("Impact evidence is light relative to similar-level candidates.")
    if prestige_score >= 55:
        notes.append("High-rigor employers, titles, publications, or awards lift the tier.")
    if public_score >= 35:
        notes.append("Public-data models find corroborating skills, roles, or credentials.")
    if not notes:
        notes.append("Capability tier is driven by the resume's specificity and experience evidence.")

    return {
        "score": round(score, 1),
        "tier": tier,
        "summary": summary,
        "salary_multiplier": salary_multiplier,
        "salary_effect_pct": round((salary_multiplier - 1.0) * 100.0, 1),
        "skill_hits": skill_hits[:6],
        "notes": notes[:3],
    }


def apply_capability_adjustment(
    band: dict[str, Any] | None,
    capability: dict[str, Any],
) -> dict[str, Any] | None:
    if band is None:
        return None
    multiplier = float(capability.get("salary_multiplier", 1.0))
    if abs(multiplier - 1.0) < 0.001:
        adjusted = dict(band)
    else:
        adjusted = dict(band)
        for key in ("q10", "q25", "q50", "q75", "q90"):
            value = adjusted.get(key)
            if value is None:
                continue
            try:
                adjusted[key] = int(round(float(value) * multiplier, -3))
            except (TypeError, ValueError):
                continue

    effect = float(capability.get("salary_effect_pct", 0.0))
    if abs(effect) >= 0.1:
        direction = "+" if effect > 0 else ""
        note = (
            f"Capability tier adjustment: {capability.get('tier', 'Competitive')} "
            f"({direction}{effect:.1f}% within level)."
        )
        adjusted["adjustment_notes"] = [
            *(adjusted.get("adjustment_notes") or []),
            note,
        ][:3]
    adjusted["capability_tier"] = capability
    return adjusted


def seniority_filtered_salary_matches(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    if matches.empty or "salary_eligible" not in matches.columns:
        return matches, None
    eligible_mask = matches["salary_eligible"].fillna(True).astype(bool)
    eligible = matches[eligible_mask].copy()
    excluded = matches[~eligible_mask]
    if excluded.empty:
        return eligible, None

    below = int(
        excluded["salary_eligibility_note"]
        .astype(str)
        .str.contains("below candidate level", case=False, na=False)
        .sum()
    )
    above = int(len(excluded) - below)
    parts = []
    if below:
        parts.append(f"{below} lower-seniority role{'s' if below != 1 else ''}")
    if above:
        parts.append(f"{above} over-level role{'s' if above != 1 else ''}")
    note = "Salary evidence excludes " + " and ".join(parts) + "."
    return eligible, note


def add_salary_evidence_note(
    band: dict[str, Any] | None,
    note: str | None,
) -> dict[str, Any] | None:
    if band is None or not note:
        return band
    updated = dict(band)
    evidence = dict(updated.get("evidence", {}))
    evidence["seniority_filter"] = note
    updated["evidence"] = evidence
    return updated


def detect_profile(
    resume_text: str,
    work_history: dict[str, Any] | None = None,
    projects: dict[str, Any] | None = None,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect track, seniority, and confidence using HR-style evidence weighting.

    work_history / projects / structure are optional — if omitted, this acts
    like the legacy keyword-only behavior (used by inline button handlers
    that just need a track for sample generation).
    """
    lowered = resume_text.lower()
    track_scores = {
        track: sum(lowered.count(keyword) for keyword in keywords)
        for track, keywords in TRACK_KEYWORDS.items()
    }
    detected_track = max(track_scores, key=track_scores.get)

    skill_hits = {
        skill: any(keyword in lowered for keyword in keywords)
        for skill, keywords in SKILL_GROUPS.items()
    }
    present = [skill for skill, hit in skill_hits.items() if hit]
    missing = [skill for skill, hit in skill_hits.items() if not hit]

    legacy_mode = work_history is None
    if legacy_mode:
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
        confidence = 55 + min(35, sum(track_scores.values()) * 3 + len(present) * 4)
        confidence = min(confidence, 96)
        return {
            "track": detected_track,
            "seniority": seniority,
            "seniority_multiplier": SENIORITY_MULTIPLIER[seniority],
            "skills_present": present,
            "skills_missing": missing[:3],
            "confidence": confidence,
            "seniority_reason": "",
        }

    ft_months = int(work_history.get("total_ft_months", 0))
    intern_months = int(work_history.get("total_intern_months", 0))
    # weighted_ft_months is already rigor-weighted (low-rigor jobs count less,
    # prestigious / rigorous-titled jobs count more, internships count tiny).
    weighted_months = int(work_history.get("weighted_ft_months", 0))
    floor_seniority = _seniority_from_ft_months(weighted_months)

    title_cap = work_history.get("max_seniority_keyword")
    progression_order = [
        "Intern / Entry",
        "Associate",
        "Mid",
        "Senior",
        "Lead / Executive",
    ]
    seniority_reason = ""

    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    low_rigor_role_count = int(work_history.get("low_rigor_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))
    ft_role_count = int(work_history.get("ft_role_count", 0))

    if work_history.get("role_count", 0) == 0:
        seniority = "Intern / Entry"
        seniority_reason = "Defaulted to Entry — no employment dates were detected."
    elif ft_role_count == 0 and intern_months > 0:
        seniority = "Intern / Entry"
        seniority_reason = (
            "Capped at Entry — only internship or academic roles were detected."
        )
    elif title_cap is not None and progression_order.index(title_cap) < progression_order.index(floor_seniority):
        seniority = title_cap
        seniority_reason = (
            f"Capped at {title_cap} — titles do not support a higher level."
        )
    elif (
        low_rigor_role_count > 0
        and rigorous_role_count == 0
        and weighted_months < ft_months
    ):
        seniority = floor_seniority
        seniority_reason = (
            "Tenure weighted down — listed roles are common entry-tier positions "
            "without rigorous-title or recognized-employer evidence."
        )
    else:
        seniority = floor_seniority

    academic = academic_cv_signals(resume_text)
    academic_pub_count = int(academic["publication_count"])
    academic_degree_hits = int(academic["degree_hits"])
    academic_prestige_count = int(academic["prestigious_education_count"])
    academic_floor = None
    if academic_pub_count >= 10 and academic_prestige_count > 0 and rigorous_role_count > 0:
        academic_floor = "Senior"
    elif academic_pub_count >= 5 and academic_degree_hits > 0:
        academic_floor = "Mid"
    elif academic_prestige_count > 0 and academic_degree_hits > 0:
        academic_floor = "Associate"

    if academic_floor and progression_order.index(academic_floor) > progression_order.index(seniority):
        seniority = academic_floor
        seniority_reason = (
            f"Raised to {academic_floor} — publication record and elite academic background support a higher research level."
        )

    if (
        title_cap == "Lead / Executive"
        and academic_pub_count >= 10
        and rigorous_role_count >= 2
        and prestigious_company_count > 0
    ):
        seniority = "Lead / Executive"
        seniority_reason = (
            "Raised to Lead / Executive — senior/staff research title is backed by publications and elite employers."
        )

    structure = structure or {}
    projects = projects or {}
    word_count = int(structure.get("word_count", 0))
    sections_found = len(structure.get("found_sections", []))
    sections_total = len(SECTION_ALIASES)
    impact_total = int(projects.get("impact_total", 0))
    action_total = int(projects.get("action_total", 0))
    n_total = int(projects.get("n_total", 0))
    n_school = int(projects.get("n_school", 0))
    vague_total = int(projects.get("vague_total", 0))

    confidence = 0
    if ft_months > 0:
        confidence += 25
        confidence += min(30, int(ft_months / 12 * 10))
    elif intern_months > 0:
        confidence += 10
    if impact_total >= 1:
        confidence += 15
    if action_total >= 3:
        confidence += 10
    if sections_total and sections_found == sections_total:
        confidence += 10
    if word_count >= 350:
        confidence += 5
    if academic_pub_count >= 5:
        confidence += 15
    if academic_prestige_count > 0:
        confidence += 8
    if n_total and n_school > n_total / 2:
        confidence -= 15
    if vague_total >= 3 and impact_total == 0:
        confidence -= 15
    if word_count < 150:
        confidence -= 10
    confidence = max(5, min(95, confidence))

    return {
        "track": detected_track,
        "seniority": seniority,
        "seniority_multiplier": SENIORITY_MULTIPLIER[seniority],
        "skills_present": present,
        "skills_missing": missing[:3],
        "confidence": confidence,
        "seniority_reason": seniority_reason,
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
    public_ats = row.get("public_ats_score", np.nan)
    if not pd.isna(public_ats):
        score_label += f" · {float(public_ats):.0f}% public fit"
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


def render_quality_scorecard(quality: dict[str, Any]) -> None:
    overall = int(quality.get("overall", 0))
    band_label = str(quality.get("band_label", "Mixed"))
    sub_pairs = [
        ("Experience", int(quality.get("experience_score", 0))),
        ("Quantified impact", int(quality.get("impact_score", 0))),
        ("Specificity", int(quality.get("specificity_score", 0))),
        ("Structure", int(quality.get("structure_score", 0))),
    ]
    sub_html = "".join(
        f'<div><div class="quality-sub-label">{escape(label)}</div>'
        f'<div class="quality-bar-track"><div class="quality-bar-fill" '
        f'style="width:{max(0, min(100, value))}%;"></div></div></div>'
        for label, value in sub_pairs
    )
    flags = quality.get("red_flags") or []
    strengths = quality.get("strengths") or []
    strengths_html = (
        '<div class="quality-feedback-panel">'
        '<div class="quality-section-label">What stood out positively</div>'
        '<ul class="quality-list strengths">'
        + "".join(f"<li>{escape(str(s))}</li>" for s in strengths[:5])
        + "</ul></div>"
    )
    flags_html = (
        '<div class="quality-feedback-panel">'
        '<div class="quality-section-label">What needs work</div>'
        '<ul class="quality-list flags">'
        + "".join(f"<li>{escape(str(f))}</li>" for f in flags[:5])
        + "</ul></div>"
    )
    band_label_safe = escape(band_label)
    overall_html = (
        f'<div class="quality-overall">{overall}'
        f'<span style="font-size:0.9rem;color:var(--muted);font-weight:500;"> / 100</span>'
        f'</div>'
    )
    headline = (
        '<div class="quality-headline">'
        f'<div><div class="metric-label">Resume quality</div>{overall_html}</div>'
        f'<span class="quality-band-pill quality-band-{band_label_safe}">{band_label_safe}</span>'
        '</div>'
    )
    body = (
        '<div class="quality-card">'
        + headline
        + f'<div class="quality-subscores">{sub_html}</div>'
        + f'<div class="quality-feedback-grid">{strengths_html}{flags_html}</div>'
        + '</div>'
    )
    st.markdown(body, unsafe_allow_html=True)


def render_public_model_card(public_signals: dict[str, Any] | None) -> None:
    if not public_signals or not public_signals.get("ready"):
        return
    domain = public_signals.get("domain", {})
    domain_label = str(domain.get("label", "Unknown")).title()
    domain_confidence = float(domain.get("confidence", 0.0) or 0.0)
    sections = public_signals.get("sections", {}).get("counts", {})
    entities = public_signals.get("entities", {}).get("counts", {})
    chips = [
        f"Domain: {domain_label} ({domain_confidence * 100:.0f}%)",
        "Sections: "
        + ", ".join(
            f"{label}:{count}"
            for label, count in sections.items()
            if label in {"Exp", "Edu", "Skill", "Sum"}
        ),
        "Entities: "
        + ", ".join(
            f"{label}:{count}"
            for label, count in entities.items()
            if label in {"Companies worked at", "College Name", "Degree", "Designation", "Skills"}
        ),
    ]
    chips = [chip for chip in chips if not chip.endswith(": ")]
    if not chips:
        return
    st.markdown(
        '<div class="section-label" style="margin-top:0.9rem;">Public-data model checks</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chip-cloud">'
        + "".join(f'<span class="mini-chip">{escape(chip)}</span>' for chip in chips[:5])
        + "</div>",
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
    seniority_filter = evidence.get("seniority_filter")
    if seniority_filter:
        pieces.append(str(seniority_filter))
    evidence_html = escape(" · ".join(pieces))
    st.markdown(
        f'<div class="evidence-line">{evidence_html}</div>',
        unsafe_allow_html=True,
    )
    adjustment_notes = band.get("adjustment_notes") or []
    if adjustment_notes:
        notes_html = escape(" ".join(str(note) for note in adjustment_notes))
        st.markdown(
            f'<div class="evidence-line" style="color: var(--warning);">{notes_html}</div>',
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
    if "assessment" not in st.session_state:
        st.session_state.assessment = None
    if "sample_resume_index" not in st.session_state:
        st.session_state.sample_resume_index = None

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
        left = st.container()
        right = st.container()

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
                height=340,
                placeholder="Paste a resume, portfolio bio, or achievement summary here...",
            )
            word_count = len(st.session_state.resume_text.split())
            st.caption(f"{word_count} words · click Analyze to evaluate.")

            if st.button("Load random sample resume", width="stretch"):
                sample_text, sample_source, sample_index = random_premade_sample_resume(
                    jobs,
                    st.session_state.sample_resume_index,
                )
                st.session_state.resume_text = sample_text
                st.session_state.resume_source = sample_source
                st.session_state.sample_resume_index = sample_index
                st.session_state.assessment = None
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
                    st.session_state.assessment = None
                    st.rerun()

        with right:
            assessment = st.session_state.get("assessment")
            current_text = st.session_state.resume_text.strip()
            if assessment is not None and current_text:
                st.markdown("## Candidate snapshot")
                if assessment.get("resume_text", "") != current_text:
                    st.warning(
                        "Resume text changed since the last analysis. Click Analyze profile to refresh."
                    )

                profile = assessment["profile"]
                structure = assessment["structure"]
                quality = assessment["quality"]
                capability = assessment.get("capability") or {}
                public_signals = assessment.get("public_signals")

                render_quality_scorecard(quality)
                render_public_model_card(public_signals)

                col1, col2 = st.columns(2)
                with col1:
                    render_signal_card(
                        "Detected focus",
                        profile["track"],
                        "Inferred from resume language and market evidence.",
                    )
                with col2:
                    render_signal_card(
                        "Seniority",
                        profile["seniority"],
                        profile.get("seniority_reason")
                        or "Derived from parsed employment history and titles.",
                    )
                col3, col4, col5 = st.columns(3)
                with col3:
                    effect = float(capability.get("salary_effect_pct", 0.0) or 0.0)
                    direction = "+" if effect > 0 else ""
                    render_signal_card(
                        "Capability tier",
                        f"{capability.get('tier', 'Competitive')} ({capability.get('score', 0)}/100)",
                        f"Within-level strength; salary effect {direction}{effect:.1f}%.",
                    )
                with col4:
                    render_signal_card(
                        "Sections",
                        f"{len(structure['found_sections'])}/{len(SECTION_ALIASES)}",
                        "Structured resumes score better.",
                    )
                with col5:
                    render_signal_card(
                        "Data mode",
                        "Live data" if has_real_data else "Sample data",
                        "Uses the local LinkedIn job catalog.",
                    )

                profile_track_html = escape(str(profile["track"]))
                profile_seniority_html = escape(str(profile["seniority"]))
                profile_confidence_html = escape(str(profile["confidence"]))
                st.markdown(
                    f'<div class="metric-card" style="margin-top:0.75rem;"><div class="metric-label">Profile read</div><div class="signal-copy">This resume reads as a <strong>{profile_track_html}</strong> profile at the <strong>{profile_seniority_html}</strong> level with about <strong>{profile_confidence_html}%</strong> confidence.</div></div>',
                    unsafe_allow_html=True,
                )

                resume_source_html = escape(str(assessment.get("resume_source", "")))
                word_count_html = escape(str(structure["word_count"]))
                bullet_count_html = escape(str(structure["bullet_count"]))
                link_count_html = escape(str(structure["link_count"]))
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
                present_skills = profile["skills_present"] or [
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
                structure_chips = structure["found_sections"] or [
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
                if structure["missing_sections"]:
                    st.caption(
                        "Missing sections: "
                        + ", ".join(structure["missing_sections"])
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

            resume_text_now = st.session_state.resume_text
            try:
                with st.spinner("Reviewing resume content..."):
                    public_models = load_public_assessment_resource()
                    public_signals = public_resume_signals(public_models, resume_text_now)
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

                salary_matches, seniority_salary_note = seniority_filtered_salary_matches(
                    matches
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
                if artifacts_ready(status, "clustering"):
                    with st.spinner("Finding market segment..."):
                        kmeans_model, _, cluster_labels = load_cluster_resource()
                        cluster = cluster_position(
                            kmeans_model, cluster_labels, resume_embedding
                        )

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
                "capability": capability,
                "public_signals": public_signals,
                "matches": matches,
                "salary_matches": salary_matches,
                "band": band,
                "cluster": cluster,
                "missing_terms": missing_terms,
            }
            st.rerun()
        elif analyze_clicked:
            st.warning(
                "Paste a resume or load a random sample resume before running the analysis."
            )

        assessment = st.session_state.get("assessment")
        if assessment is not None:
            band = assessment.get("band")
            cluster = assessment.get("cluster")
            matches = assessment.get("matches")
            missing_terms = assessment.get("missing_terms") or []

            st.write("")
            render_panel_banner(
                "Market Readout",
                "Salary range",
                "This estimate is anchored to the salary data from the most relevant roles, with a haircut when the resume's evidence is thin.",
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
                    if matches is None or matches.empty:
                        render_signal_card(
                            "Top similarity", "N/A", "No matching roles surfaced."
                        )
                    else:
                        render_signal_card(
                            "Top similarity",
                            f"{matches.iloc[0]['similarity'] * 100:.0f}%",
                            "Best role match for this resume.",
                        )
                with signal_cols[3]:
                    count = 0 if matches is None else len(matches)
                    render_signal_card(
                        "Retrieved roles",
                        f"{count:,}",
                        "Roles surfaced for this resume.",
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
                "These roles are ordered by similarity to the resume.",
            )
            if matches is None or matches.empty:
                st.info(
                    "No matching roles surfaced for this resume. Try expanding the resume text with more domain terms."
                )
            else:
                card_cols = st.columns(2, gap="medium")
                for index, (_, row) in enumerate(matches.iterrows()):
                    with card_cols[index % 2]:
                        render_job_card(row)

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
