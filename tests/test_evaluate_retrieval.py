from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.retrieval import JobMatch  # noqa: E402
from scripts.evaluate_retrieval import evaluate_retrieval  # noqa: E402


class StubSearcher:
    def __init__(self, rankings: dict[str, list[int]]):
        self.rankings = rankings

    def search(self, resume_text: str, k: int) -> list[JobMatch]:
        ids = self.rankings[resume_text][:k]
        return [
            JobMatch(
                row_id=rank - 1,
                job_id=job_id,
                title=f"Job {job_id}",
                company_name="Co",
                salary_annual=100000.0,
                location="Remote",
                experience_level="mid",
                similarity=1.0 / rank,
            )
            for rank, job_id in enumerate(ids, start=1)
        ]


def test_evaluate_retrieval_metrics_and_per_row_ranks() -> None:
    eval_df = pd.DataFrame(
        {
            "resume_id": ["a", "b", "c"],
            "resume_text": ["resume-a", "resume-b", "resume-c"],
            "source_job_id": [10, 20, 30],
            "hard_negative_job_id": [11, 21, 31],
        }
    )
    searcher = StubSearcher(
        {
            "resume-a": [10, 11, 99],
            "resume-b": [21, 20, 99],
            "resume-c": [31, 99, 98],
        }
    )

    metrics, per_row = evaluate_retrieval(eval_df, searcher, k=3)

    assert metrics["n"] == 3
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@3"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert metrics["discriminative_accuracy"] == pytest.approx(1 / 3)
    assert per_row["source_rank"].iloc[:2].tolist() == [1.0, 2.0]
    assert pd.isna(per_row["source_rank"].iloc[2])
    assert per_row["hard_negative_rank"].tolist() == [2, 1, 1]
    assert per_row["ndcg_at_k"].between(0, 1).all()


def test_evaluate_retrieval_requires_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_retrieval(pd.DataFrame({"resume_text": ["x"]}), StubSearcher({}), k=3)


def test_evaluate_retrieval_rejects_non_positive_k() -> None:
    eval_df = pd.DataFrame(
        {
            "resume_text": ["x"],
            "source_job_id": [1],
            "hard_negative_job_id": [2],
        }
    )
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_retrieval(eval_df, StubSearcher({"x": []}), k=0)
