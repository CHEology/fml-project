from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_synthetic_resumes import (  # noqa: E402
    DEGREES_BY_FAMILY,
    generate_paired_synthetic_resumes,
    generate_synthetic_resumes,
    write_synthetic_resumes,
)


EXPECTED_COLUMNS = {
    "resume_id",
    "source_job_id",
    "source_title",
    "source_company_name",
    "target_title",
    "role_family",
    "level",
    "persona",
    "writing_style",
    "years_experience",
    "location",
    "skills",
    "missing_core_skills",
    "project_count",
    "has_metrics",
    "typo_count",
    "education",
    "quality_score",
    "quality_label",
    "hard_negative_job_id",
    "hard_negative_title",
    "hard_negative_role_family",
    "hard_negative_reason",
    "generation_notes",
    "resume_text",
}


def test_generate_synthetic_resumes_schema_and_ranges() -> None:
    df = generate_synthetic_resumes(12, seed=7)

    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 12
    assert df["resume_id"].is_unique
    assert df["quality_score"].between(0, 100).all()
    assert set(df["quality_label"]).issubset({"weak", "medium", "strong"})
    assert df["resume_text"].str.len().gt(100).all()
    assert df["resume_text"].str.contains("Skills|Tools|Primary strengths").all()
    assert set(df["persona"]).issubset({
        "direct_match",
        "under_qualified",
        "over_qualified",
        "career_switcher",
    })
    assert set(df["writing_style"]).issubset({
        "concise_bullets",
        "first_person_bullets",
        "third_person_summary",
        "abbreviated_keywords",
    })


def test_generate_synthetic_resumes_is_deterministic() -> None:
    first = generate_synthetic_resumes(8, seed=99)
    second = generate_synthetic_resumes(8, seed=99)

    pd.testing.assert_frame_equal(first, second)


def test_direct_match_education_follows_role_family() -> None:
    df = generate_synthetic_resumes(12, seed=11)
    direct_matches = df[df["persona"] == "direct_match"]

    assert not direct_matches.empty
    for _, row in direct_matches.iterrows():
        assert row["education"] in DEGREES_BY_FAMILY[row["role_family"]]


def test_generate_synthetic_resumes_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        generate_synthetic_resumes(-1)


def test_generate_paired_synthetic_resumes_adds_source_and_hard_negative() -> None:
    jobs = pd.DataFrame(
        {
            "job_id": [101, 102, 103, 104],
            "title": [
                "Data Scientist",
                "Senior Data Scientist",
                "Machine Learning Engineer",
                "Marketing Coordinator",
            ],
            "company_name": ["A", "B", "C", "D"],
            "location": ["New York, NY", "Remote, US", "Austin, TX", "Chicago, IL"],
            "experience_level": ["Entry level", "Mid-Senior level", "Mid-Senior level", "Associate"],
            "experience_level_ordinal": [1.0, 3.0, 3.0, 2.0],
            "skills_desc": [
                "Python, SQL, statistics",
                "Python, SQL, forecasting",
                "Python, PyTorch, Docker",
                "campaign planning, CRM, analytics",
            ],
            "text": [
                "data scientist python sql statistics experiments",
                "senior data scientist python sql forecasting",
                "machine learning engineer pytorch docker model serving",
                "marketing coordinator campaign crm analytics",
            ],
        }
    )

    df = generate_paired_synthetic_resumes(jobs, n=8, seed=5)

    assert len(df) == 8
    assert df["source_job_id"].notna().all()
    assert df["hard_negative_job_id"].notna().all()
    assert (df["source_job_id"] != df["hard_negative_job_id"]).all()
    assert df["resume_text"].str.contains("Job description").sum() == 0
    assert df["generation_notes"].str.contains("No JD sentences copied").all()


def test_generate_paired_synthetic_resumes_rejects_empty_jobs() -> None:
    with pytest.raises(ValueError, match="jobs must contain at least one row"):
        generate_paired_synthetic_resumes(pd.DataFrame(), n=1)


def test_write_synthetic_resumes_csv_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    df = generate_synthetic_resumes(5, seed=3)
    calls = {}

    def fake_to_csv(path: Path, index: bool) -> None:
        calls["path"] = path
        calls["index"] = index

    monkeypatch.setattr(df, "to_csv", fake_to_csv)
    out = write_synthetic_resumes(df, Path("synthetic_resumes_test.csv"))

    assert out == Path("synthetic_resumes_test.csv")
    assert calls == {"path": Path("synthetic_resumes_test.csv"), "index": False}


def test_write_synthetic_resumes_rejects_unknown_extension() -> None:
    df = generate_synthetic_resumes(1, seed=3)

    with pytest.raises(ValueError, match="Unsupported output extension"):
        write_synthetic_resumes(df, Path("synthetic_resumes_test.txt"))
