from __future__ import annotations

import numpy as np
import pandas as pd
from ml.resume_assessment.career_actions import (
    cluster_options,
    cluster_transition_advice,
    salary_growth_advice,
)


def test_cluster_options_excludes_current_cluster() -> None:
    options = cluster_options(
        {
            "0": {"label": "Software / Engineering", "top_terms": ["python"]},
            "1": {"label": "Business / Data Analysis", "top_terms": ["sql"]},
        },
        current_cluster_id=0,
    )

    assert options == [
        {
            "cluster_id": 1,
            "label": "Business / Data Analysis",
            "top_terms": ["sql"],
            "size": None,
        }
    ]


def test_salary_growth_advice_uses_current_cluster_high_salary_cohort() -> None:
    jobs = pd.DataFrame(
        {
            "title": [
                "Associate ML Engineer",
                "ML Engineer",
                "Senior ML Engineer",
                "Principal ML Engineer",
                "Sales Director",
            ],
            "salary_annual": [100_000, 200_000, 300_000, 400_000, 500_000],
            "text": [
                "python model support",
                "python model delivery",
                "python distributed systems",
                "python kubernetes platform ownership ms degree 8+ years of machine learning experience",
                "quota sales pipeline account management",
            ],
        }
    )
    assignments = np.array([0, 0, 0, 0, 1])

    advice = salary_growth_advice(
        jobs,
        assignments,
        cluster_labels={"0": {"label": "ML", "top_terms": ["python", "kubernetes"]}},
        current_cluster_id=0,
        resume_text="Python model delivery",
    )

    assert advice["salary_threshold"] == 325_000
    assert advice["target_titles"] == ["Principal ML Engineer"]
    assert "Sales Director" not in advice["target_titles"]
    assert "python" not in advice["missing_terms"]
    assert "kubernetes" in advice["missing_terms"]
    assert advice["education_requirements"] == ["ms degree"]
    assert advice["experience_requirements"] == [
        "8+ years of machine learning experience"
    ]


def test_cluster_transition_advice_uses_selected_target_cluster() -> None:
    jobs = pd.DataFrame(
        {
            "title": [
                "Software Engineer",
                "Customer Success Manager",
                "Revenue Operations Analyst",
            ],
            "salary_annual": [140_000, 120_000, 150_000],
            "text": [
                "python apis postgres",
                "salesforce renewal strategy mba 5+ years of customer success experience",
                "salesforce pipeline forecasting sql account management",
            ],
        }
    )
    assignments = np.array([0, 1, 1])

    advice = cluster_transition_advice(
        jobs,
        assignments,
        cluster_labels={
            "0": {"label": "Software / Engineering", "top_terms": ["python"]},
            "1": {
                "label": "Sales / Customer Growth",
                "top_terms": ["salesforce", "renewal"],
            },
        },
        current_cluster_id=0,
        target_cluster_id=1,
        resume_text="Software engineer with Python APIs and Postgres",
    )

    assert advice["target_cluster_id"] == 1
    assert advice["target_cluster_label"] == "Sales / Customer Growth"
    assert "Software Engineer" not in advice["target_titles"]
    assert "salesforce" in advice["missing_terms"]
    assert "renewal" in advice["missing_terms"]
    assert advice["education_requirements"] == ["mba"]
    assert advice["experience_requirements"] == [
        "5+ years of customer success experience"
    ]
