from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_synthetic_resumes import (  # noqa: E402
    DEGREES_BY_FAMILY,
    PERSONA_SALARY_RANGES,
    generate_paired_synthetic_resumes,
    generate_synthetic_resumes,
    quality_label_from_score,
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
    "experience_level_ordinal",
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
    "source_salary_annual",
    "source_salary_min",
    "source_salary_max",
    "expected_salary_annual",
    "hard_negative_job_id",
    "hard_negative_title",
    "hard_negative_role_family",
    "hard_negative_reason",
    "hard_negative_job_ids",
    "generation_notes",
    "resume_text",
}


def _paired_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [101, 102, 103, 104, 105],
            "title": [
                "Data Scientist",
                "Senior Data Scientist",
                "Machine Learning Engineer",
                "Marketing Coordinator",
                "Backend Software Engineer",
            ],
            "company_name": ["A", "B", "C", "D", "E"],
            "location": [
                "New York, NY",
                "Remote, US",
                "Austin, TX",
                "Chicago, IL",
                "San Francisco, CA",
            ],
            "experience_level": [
                "Entry level",
                "Mid-Senior level",
                "Mid-Senior level",
                "Associate",
                "Mid-Senior level",
            ],
            "experience_level_ordinal": [1.0, 4.0, 3.0, 2.0, 3.0],
            "skills_desc": [
                "Python, SQL, statistics, machine learning",
                "Python, SQL, forecasting, A/B testing",
                "Python, PyTorch, Docker, model serving",
                "campaign planning, CRM, analytics",
                "Python, REST APIs, PostgreSQL, system design",
            ],
            "salary_annual": [110_000.0, 165_000.0, 145_000.0, 70_000.0, 130_000.0],
            "min_salary": [95_000.0, 150_000.0, 130_000.0, 60_000.0, 115_000.0],
            "max_salary": [125_000.0, 185_000.0, 160_000.0, 80_000.0, 145_000.0],
            "text": [
                "data scientist python sql statistics machine learning",
                "senior data scientist python sql forecasting a/b testing",
                "machine learning engineer pytorch docker model serving",
                "marketing coordinator campaign crm analytics",
                "backend software engineer python rest apis postgresql system design",
            ],
        }
    )


def test_generate_synthetic_resumes_schema_and_ranges() -> None:
    df = generate_synthetic_resumes(12, seed=7)

    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 12
    assert df["resume_id"].is_unique
    assert df["quality_score"].between(0, 100).all()
    assert set(df["quality_label"]).issubset({"weak", "medium", "strong"})
    assert df["resume_text"].str.len().gt(100).all()
    assert df["resume_text"].str.contains("Skills|Tools|Primary strengths").all()
    assert set(df["persona"]).issubset(
        {
            "direct_match",
            "under_qualified",
            "over_qualified",
            "career_switcher",
        }
    )
    assert set(df["writing_style"]).issubset(
        {
            "concise_bullets",
            "first_person_bullets",
            "third_person_summary",
            "abbreviated_keywords",
        }
    )


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
    df = generate_paired_synthetic_resumes(_paired_jobs(), n=8, seed=5)

    assert len(df) == 8
    assert df["source_job_id"].notna().all()
    assert df["hard_negative_job_id"].notna().all()
    assert (df["source_job_id"] != df["hard_negative_job_id"]).all()
    assert df["resume_text"].str.contains("Job description").sum() == 0
    assert df["generation_notes"].str.contains("No JD sentences copied").all()


def test_generate_paired_synthetic_resumes_rejects_empty_jobs() -> None:
    with pytest.raises(ValueError, match="jobs must contain at least one row"):
        generate_paired_synthetic_resumes(pd.DataFrame(), n=1)


def test_generate_paired_resumes_carries_salary_and_experience_columns() -> None:
    jobs = _paired_jobs()
    df = generate_paired_synthetic_resumes(jobs, n=12, seed=13)

    assert df["source_salary_annual"].notna().all()
    assert df["source_salary_min"].notna().all()
    assert df["source_salary_max"].notna().all()
    assert df["expected_salary_annual"].notna().all()
    assert (df["source_salary_annual"] > 0).all()
    assert (df["expected_salary_annual"] > 0).all()
    assert df["experience_level_ordinal"].between(1, 5).all()


def test_persona_salary_ranges_reflect_persona_order() -> None:
    # Across enough samples, the persona-multiplier ordering holds on average.
    jobs = _paired_jobs()
    df = generate_paired_synthetic_resumes(jobs, n=80, seed=21)

    by_persona = df.groupby("persona")["expected_salary_annual"].mean()

    assert by_persona["under_qualified"] < by_persona["direct_match"]
    assert by_persona["direct_match"] < by_persona["over_qualified"]
    # Each individual ratio is inside the configured range (allow small slack
    # for rounding).
    ratios = df["expected_salary_annual"] / df["source_salary_annual"]
    for persona, (low, high) in PERSONA_SALARY_RANGES.items():
        persona_ratios = ratios[df["persona"] == persona]
        if persona_ratios.empty:
            continue
        assert persona_ratios.min() >= low - 1e-6
        assert persona_ratios.max() <= high + 1e-6


def test_multi_hard_negatives_returns_ranked_distinct_list() -> None:
    jobs = _paired_jobs()
    df = generate_paired_synthetic_resumes(jobs, n=8, seed=3, n_hard_negatives=3)

    assert "hard_negative_job_ids" in df.columns
    for _, row in df.iterrows():
        ids = list(row["hard_negative_job_ids"])
        assert ids, "expected at least one hard negative when jobs are present"
        assert len(ids) <= 3
        assert row["source_job_id"] not in ids
        assert len(set(ids)) == len(ids)
        # Legacy scalar column always matches the head of the list.
        assert ids[0] == row["hard_negative_job_id"]


def test_quality_score_separates_strong_and_weak_personas() -> None:
    df = generate_paired_synthetic_resumes(_paired_jobs(), n=80, seed=37)

    means = df.groupby("persona")["quality_score"].mean()
    assert means["under_qualified"] < means["direct_match"]
    assert means["under_qualified"] < means["over_qualified"]


def test_typo_injection_actually_mutates_resume_text() -> None:
    df = generate_paired_synthetic_resumes(_paired_jobs(), n=40, seed=44)

    typed = df[df["typo_count"] > 0]
    assert not typed.empty
    typo_markers = (
        "experiance",
        "analysys",
        "modle",
        "databse",
        "pipline",
        "campagin",
        "stakehldr",
    )
    text_blob = "\n".join(typed["resume_text"])
    assert any(marker in text_blob for marker in typo_markers)


def test_quality_label_from_score_thresholds() -> None:
    assert quality_label_from_score(80) == "strong"
    assert quality_label_from_score(75) == "strong"
    assert quality_label_from_score(74) == "medium"
    assert quality_label_from_score(50) == "medium"
    assert quality_label_from_score(49) == "weak"


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
