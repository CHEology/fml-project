from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.load_real_resumes import load_real_resumes  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_real_resumes.csv"


def test_load_real_resumes_from_fixture_normalises_columns() -> None:
    df = load_real_resumes(FIXTURE)
    assert set(
        ["resume_id", "resume_text", "source_path", "n_chars", "truncated"]
    ).issubset(df.columns)
    assert len(df) == 5
    assert df["resume_id"].is_unique
    assert (df["n_chars"] > 0).all()
    assert df["truncated"].dtype == bool


def test_load_real_resumes_redacts_pii_by_default(tmp_path: Path) -> None:
    file_ = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "resume_id": ["x"],
            "resume_text": [
                "Senior engineer with 6 years of experience.\n"
                "Contact: hire@example.com or (212) 555-0143.\n"
                "Skills: Python, SQL, machine learning, Docker, AWS, CI/CD\n"
                "- Built systems handling 80K requests per day.\n"
                "- Reduced latency by 35% across services.\n"
                "Education: M.S. Computer Science.\n"
            ],
        }
    ).to_csv(file_, index=False)

    df = load_real_resumes(file_)
    text = df.iloc[0]["resume_text"]
    assert "hire@example.com" not in text
    assert "[email]" in text
    assert "[phone]" in text


def test_load_real_resumes_directory_input(tmp_path: Path) -> None:
    body = (
        "Senior backend engineer with 7 years of experience.\n"
        "Skills: Python, REST APIs, PostgreSQL, Docker, AWS, system design\n"
        "- Designed a payments service handling 40K requests per second.\n"
        "- Reduced latency by 38% across the hot path.\n"
        "Education: B.S. Computer Science\n"
    )
    (tmp_path / "a.txt").write_text(body, encoding="utf-8")
    (tmp_path / "b.md").write_text(body.replace("Senior", "Staff"), encoding="utf-8")

    df = load_real_resumes(tmp_path)
    assert len(df) == 2
    assert set(df["resume_id"]) == {"a", "b"}


def test_load_real_resumes_picks_up_category_when_present(tmp_path: Path) -> None:
    file_ = tmp_path / "with_category.csv"
    pd.DataFrame(
        {
            "resume_id": ["c1", "c2"],
            "resume_text": [
                "Senior engineer with 6 years of experience and "
                "Python, SQL, machine learning, Docker skills. "
                "Built systems handling 80K requests per day. "
                "Reduced latency by 35% across services. "
                "M.S. Computer Science.",
                "Junior data analyst with 2 years experience. "
                "SQL, Excel, Tableau, business metrics, dbt. "
                "Built dashboards used by 60 weekly stakeholders. "
                "B.A. Economics.",
            ],
            "Category": ["Engineering", "Analytics"],
        }
    ).to_csv(file_, index=False)

    df = load_real_resumes(file_)
    assert df["category"].tolist() == ["Engineering", "Analytics"]


def test_load_real_resumes_raises_on_empty_input(tmp_path: Path) -> None:
    file_ = tmp_path / "empty.csv"
    pd.DataFrame({"resume_text": []}).to_csv(file_, index=False)
    with pytest.raises(ValueError, match="no usable resumes"):
        load_real_resumes(file_)
