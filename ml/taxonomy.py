from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleProfile:
    title: str
    family: str
    aliases: tuple[str, ...]
    core_skills: tuple[str, ...]
    nice_to_have: tuple[str, ...]
    project_templates: tuple[str, ...]


ROLE_PROFILES = (
    RoleProfile(
        title="Data Scientist",
        family="data",
        aliases=("Applied Analytics Specialist", "Predictive Modeling Analyst"),
        core_skills=("Python", "SQL", "pandas", "statistics", "machine learning"),
        nice_to_have=("A/B testing", "Spark", "Tableau", "NLP", "forecasting"),
        project_templates=(
            "built retention models that improved outreach precision by {metric}%",
            "created experiment readouts for {metric} product decisions",
            "trained classifiers over {metric}K customer records",
        ),
    ),
    RoleProfile(
        title="Machine Learning Engineer",
        family="ml",
        aliases=("ML Systems Engineer", "Model Platform Engineer"),
        core_skills=(
            "Python",
            "PyTorch",
            "JAX",
            "CUDA",
            "model serving",
            "Docker",
            "MLOps",
        ),
        nice_to_have=(
            "Kubernetes",
            "feature stores",
            "AWS",
            "CI/CD",
            "monitoring",
            "NLP",
        ),
        project_templates=(
            "shipped recommendation services with p95 latency below {metric} ms",
            "rebuilt training workflows and reduced retraining time by {metric}%",
            "added drift monitoring across {metric} production models",
        ),
    ),
    RoleProfile(
        title="Data Analyst",
        family="analytics",
        aliases=("Business Insights Analyst", "Reporting Analyst"),
        core_skills=("SQL", "Excel", "Tableau", "data cleaning", "business metrics"),
        nice_to_have=(
            "Python",
            "Looker",
            "dbt",
            "A/B testing",
            "stakeholder reporting",
        ),
        project_templates=(
            "created dashboards used by {metric} weekly stakeholders",
            "automated reporting and saved {metric} analyst hours monthly",
            "analyzed funnels and identified {metric}% conversion lift",
        ),
    ),
    RoleProfile(
        title="Backend Software Engineer",
        family="software",
        aliases=("API Engineer", "Platform Software Engineer"),
        core_skills=("Python", "REST APIs", "PostgreSQL", "testing", "system design"),
        nice_to_have=("FastAPI", "Docker", "Redis", "AWS", "observability"),
        project_templates=(
            "built services handling {metric}K requests per day",
            "improved database query performance by {metric}%",
            "raised automated coverage across {metric} service modules",
        ),
    ),
    RoleProfile(
        title="Marketing Coordinator",
        family="marketing",
        aliases=("Growth Marketing Associate", "Campaign Coordinator"),
        core_skills=("campaign planning", "copywriting", "analytics", "events", "CRM"),
        nice_to_have=(
            "Adobe Creative Cloud",
            "email marketing",
            "SEO",
            "social media",
            "vendor management",
        ),
        project_templates=(
            "coordinated campaigns that lifted engagement by {metric}%",
            "managed event logistics for {metric} field programs",
            "organized CRM lists supporting {metric}K customer contacts",
        ),
    ),
)

PROFILE_BY_FAMILY = {profile.family: profile for profile in ROLE_PROFILES}

MULTI_WORD_SKILLS = (
    "machine learning",
    "system design",
    "data cleaning",
    "stakeholder reporting",
    "stakeholder management",
    "model serving",
    "process improvement",
    "feature stores",
    "campaign planning",
    "vendor management",
    "email marketing",
    "social media",
    "business metrics",
    "deep learning",
    "data engineering",
    "natural language processing",
    "a/b testing",
    "rest apis",
    "ci/cd",
    "transformer architecture",
    "attention mechanism",
    "distributed training",
    "model quantization",
    "feature engineering",
    "vector database",
)


def quality_label_from_score(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "medium"
    return "weak"
