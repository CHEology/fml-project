from __future__ import annotations

import pandas as pd
from app.components import market_overview
from streamlit.testing.v1 import AppTest


def test_market_summary_handles_normal_catalog() -> None:
    jobs = pd.DataFrame(
        {
            "title": ["ML Engineer", "Data Analyst", "Backend Engineer"],
            "company_name": ["A", "B", "C"],
            "location": ["New York, NY", "Austin, TX", "New York, NY"],
            "state": ["NY", "TX", "NY"],
            "experience_level": ["Senior", "Associate", "Senior"],
            "work_type": ["Remote", "Hybrid", "Remote"],
            "salary_annual": [180_000, 120_000, 160_000],
            "text": ["embeddings retrieval", "sql dashboards", "apis systems"],
        }
    )

    summary = market_overview.compute_market_summary(jobs, has_real_data=True)

    assert summary["job_count"] == "3"
    assert summary["median_salary"] == "$160,000"
    assert summary["salary_iqr"] == "$140,000 - $170,000"
    assert summary["top_geography"] == "New York, NY"
    assert summary["work_type_mix"] == "Remote leads"
    assert summary["data_mode"] == "Real catalog"


def test_market_summary_handles_missing_salaries_and_sample_data() -> None:
    jobs = pd.DataFrame(
        {
            "title": ["Role A", "Role B"],
            "location": ["", None],
            "work_type": [None, ""],
            "salary_annual": [None, "not available"],
        }
    )

    summary = market_overview.compute_market_summary(jobs, has_real_data=False)

    assert summary["job_count"] == "2"
    assert summary["median_salary"] == "N/A"
    assert summary["salary_iqr"] == "N/A"
    assert summary["top_geography"] == "Unknown"
    assert summary["work_type_mix"] == "Unspecified leads"
    assert summary["data_mode"] == "Sample catalog"


def test_cluster_summary_uses_labels_and_assignments() -> None:
    clusters = market_overview.summarize_clusters(
        job_count=4,
        assignments=[0, 1, 1, 0],
        cluster_labels={
            "0": {
                "label": "Software / Engineering",
                "top_terms": ["python", "systems"],
                "common_titles": ["Software Engineer"],
            },
            "1": {
                "label": "Business / Data Analysis",
                "top_terms": ["sql", "reporting"],
                "common_titles": ["Data Analyst"],
            },
        },
    )

    assert clusters.available is True
    assert clusters.unavailable_reason == ""
    assert list(clusters.frame["label"]) == [
        "Software / Engineering",
        "Business / Data Analysis",
    ]
    assert list(clusters.frame["job_count"]) == [2, 2]
    assert clusters.top_label == "Software / Engineering"


def test_cluster_summary_handles_unavailable_and_mismatched_artifacts() -> None:
    missing = market_overview.summarize_clusters(
        job_count=2,
        assignments=None,
        cluster_labels={},
    )
    mismatched = market_overview.summarize_clusters(
        job_count=2,
        assignments=[0],
        cluster_labels={"0": {"label": "Software"}},
    )

    assert missing.available is False
    assert missing.unavailable_reason == "Clustering artifacts are unavailable."
    assert mismatched.available is False
    assert (
        mismatched.unavailable_reason
        == "Cluster assignments do not match the job catalog."
    )


def test_market_figures_return_plotly_objects() -> None:
    jobs = market_overview.normalize_market_jobs(
        pd.DataFrame(
            {
                "title": ["Role A", "Role B", "Role C"],
                "company_name": ["A", "B", "C"],
                "location": ["NY", "SF", "NY"],
                "experience_level": ["Entry", "Senior", "Senior"],
                "work_type": ["Remote", "Hybrid", "Remote"],
                "salary_annual": [100_000, 180_000, 160_000],
                "text": ["alpha", "beta", "gamma"],
            }
        )
    )
    clusters = market_overview.summarize_clusters(
        job_count=3,
        assignments=[0, 0, 1],
        cluster_labels={
            "0": {"label": "Software", "top_terms": ["python"]},
            "1": {"label": "Analytics", "top_terms": ["sql"]},
        },
    )

    salary_fig = market_overview.build_salary_distribution_figure(jobs)
    mix_fig = market_overview.build_market_mix_figure(jobs)
    cluster_fig = market_overview.build_cluster_distribution_figure(clusters.frame)

    assert salary_fig.data
    assert salary_fig.layout.title.text == "Salary bands by seniority"
    assert mix_fig.data
    assert mix_fig.layout.title.text == "Where the catalog concentrates"
    assert cluster_fig.data
    assert cluster_fig.layout.title.text == "Semantic market segments"


def test_market_overview_page_renders_explanatory_context() -> None:
    script = """
import pandas as pd
from app.pages.market import render_market_overview_page
from app.styles import inject_styles

inject_styles("Lavender")
jobs = pd.DataFrame(
    [
        {
            "title": "Machine Learning Engineer",
            "company_name": "North Harbor AI",
            "location": "New York, NY",
            "state": "NY",
            "experience_level": "Senior",
            "work_type": "Remote",
            "salary_annual": 180000,
            "text": "Build embeddings and retrieval models.",
        },
        {
            "title": "Analytics Engineer",
            "company_name": "Cinder Labs",
            "location": "Austin, TX",
            "state": "TX",
            "experience_level": "Associate",
            "work_type": "Hybrid",
            "salary_annual": 130000,
            "text": "Build dashboards and warehouse models.",
        },
    ]
)
status = [{"label": "KMeans model", "path": "", "ready": False, "required_for": "clustering"}]
render_market_overview_page(jobs, "Sample role catalog", False, status)
"""

    at = AppTest.from_string(script, default_timeout=10).run()

    assert not at.error
    assert not at.exception
    page_text = " ".join(markdown.value for markdown in at.markdown)
    assert "Market Overview" in page_text
    assert "why the ML demo behaves the way it does" in page_text
    assert "q10/q25/q50/q75/q90" in page_text
    assert "Market exemplars" in page_text
