from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.retrieval import JobMatch  # noqa: E402
from scripts.validate_on_real_resumes import _spearman, validate  # noqa: E402


def _fixture_resumes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resume_id": ["r1", "r2", "r3"],
            "resume_text": [
                # Strong: skills, experience, projects, metrics, no typos
                "Senior backend engineer with 7 years of experience.\n"
                "Skills: Python, REST APIs, PostgreSQL, Docker, AWS, system design, CI/CD\n"
                "- Built services handling 80K requests per day.\n"
                "- Reduced latency by 35% across services.\n"
                "- Mentored 4 junior engineers.\n"
                "Education: B.S. Computer Science\n",
                # Medium: some skills, fewer details
                "Mid data analyst with 3 years experience.\n"
                "SQL, Tableau, Excel\n"
                "- Built dashboards for the marketing team.\n"
                "B.A. Economics\n",
                # Weak: minimal text, no metrics
                "Junior candidate looking for a role. Excited to learn.\n"
                "Some experience with spreadsheets.\n",
            ],
        }
    )


class _StubRetriever:
    def __init__(self, salaries: list[float]):
        self._salaries = salaries

    def search(self, _resume_text: str, k: int) -> list[JobMatch]:
        return [
            JobMatch(
                row_id=i,
                job_id=1000 + i,
                title=f"Job {i}",
                company_name=f"Co{i}",
                salary_annual=self._salaries[i % len(self._salaries)],
                location="NY",
                experience_level="mid",
                similarity=1.0 - i * 0.01,
            )
            for i in range(min(k, 5))
        ]


def _stub_salary_predictor(_embedding: np.ndarray) -> dict[str, float]:
    return {
        "q10": 80_000.0,
        "q25": 95_000.0,
        "q50": 110_000.0,
        "q75": 130_000.0,
        "q90": 150_000.0,
    }


def _stub_quality_predictor(_embedding: np.ndarray) -> dict[str, float | str]:
    return {"score": 65.0, "label": "medium"}


def test_validate_rule_only_returns_summary_and_per_row() -> None:
    df = _fixture_resumes()
    embeddings = np.zeros((len(df), 8), dtype=np.float32)

    summary, per_row = validate(df, embeddings)

    assert summary["n"] == 3
    assert {"mean", "std", "min", "max"} <= summary["rule_score"].keys()
    assert set(summary["rule_label_counts"].keys()).issubset(
        {"weak", "medium", "strong"}
    )
    assert per_row["rule_score"].iloc[0] > per_row["rule_score"].iloc[2]
    assert per_row["rule_label"].iloc[0] in {"medium", "strong"}
    assert per_row["rule_label"].iloc[2] == "weak"


def test_validate_with_retriever_and_salary_records_self_consistency() -> None:
    df = _fixture_resumes()
    embeddings = np.zeros((len(df), 8), dtype=np.float32)
    retriever = _StubRetriever(salaries=[100_000.0, 110_000.0, 105_000.0])

    summary, per_row = validate(
        df,
        embeddings,
        retriever=retriever,
        salary_predictor=_stub_salary_predictor,
        k=3,
    )

    assert "self_consistency_mae" in summary
    assert "self_consistency_in_band_rate" in summary
    # Predicted q50 is 110k; retrieved median sits inside [80k, 150k] for all rows.
    assert summary["self_consistency_in_band_rate"] == pytest.approx(1.0)
    assert "retrieved_median_salary" in per_row.columns
    assert "pred_q50" in per_row.columns


def test_validate_reports_rule_vs_learned_spearman_when_available() -> None:
    df = _fixture_resumes()
    embeddings = np.zeros((len(df), 8), dtype=np.float32)

    # Make the learned predictor agree perfectly with rule order: r1 highest, r3 lowest.
    monotone_scores = iter([90.0, 60.0, 30.0])

    def predictor(_embedding: np.ndarray) -> dict[str, float | str]:
        return {"score": next(monotone_scores), "label": "medium"}

    summary, per_row = validate(df, embeddings, quality_predictor=predictor)

    assert "rule_vs_learned_spearman" in summary
    assert summary["rule_vs_learned_spearman"] == pytest.approx(1.0)
    assert per_row["learned_score"].iloc[0] == 90.0


def test_validate_rejects_misaligned_inputs() -> None:
    df = _fixture_resumes()
    with pytest.raises(ValueError, match="must match embeddings"):
        validate(df, np.zeros((len(df) + 1, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="resume_text"):
        validate(pd.DataFrame({"x": [1, 2]}), np.zeros((2, 8), dtype=np.float32))


def test_spearman_helper() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert _spearman(a, b) == pytest.approx(1.0)
    assert _spearman(a, -b) == pytest.approx(-1.0)
