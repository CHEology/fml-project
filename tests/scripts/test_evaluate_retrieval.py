from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def test_evaluate_retrieval_k_sweep_includes_each_cutoff() -> None:
    eval_df = pd.DataFrame(
        {
            "resume_id": ["a", "b"],
            "resume_text": ["r1", "r2"],
            "source_job_id": [10, 20],
            "hard_negative_job_id": [11, 21],
        }
    )
    searcher = StubSearcher(
        {
            "r1": [99, 98, 97, 96, 10],  # source ranks at #5
            "r2": [20, 19, 18, 17, 16],  # source ranks at #1
        }
    )

    metrics, _ = evaluate_retrieval(eval_df, searcher, k=10, recall_ks=(1, 5, 10))

    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)


def test_evaluate_retrieval_uses_multi_hard_negative_list() -> None:
    eval_df = pd.DataFrame(
        {
            "resume_id": ["a"],
            "resume_text": ["r1"],
            "source_job_id": [10],
            "hard_negative_job_ids": [[11, 12, 13]],
        }
    )
    searcher = StubSearcher({"r1": [11, 10, 12, 99]})

    metrics, per_row = evaluate_retrieval(eval_df, searcher, k=4)

    # Source is ranked #2; closest negative ("11") at #1 means we lose
    # discrimination on this row.
    assert per_row["source_rank"].iloc[0] == 2
    assert per_row["best_hard_negative_rank"].iloc[0] == 1
    assert metrics["discriminative_accuracy"] == pytest.approx(0.0)
    assert per_row["ndcg_at_k"].iloc[0] > 0


def test_evaluate_retrieval_emits_per_persona_breakdown() -> None:
    eval_df = pd.DataFrame(
        {
            "resume_id": ["a", "b", "c", "d"],
            "resume_text": ["r1", "r2", "r3", "r4"],
            "source_job_id": [10, 20, 30, 40],
            "hard_negative_job_id": [11, 21, 31, 41],
            "persona": [
                "direct_match",
                "direct_match",
                "under_qualified",
                "under_qualified",
            ],
        }
    )
    searcher = StubSearcher(
        {
            "r1": [10, 11],  # source #1
            "r2": [11, 20],  # source #2
            "r3": [99, 98],  # source missing from top-k
            "r4": [40, 41],  # source #1
        }
    )

    metrics, _ = evaluate_retrieval(eval_df, searcher, k=2)

    assert "by_persona" in metrics
    direct = metrics["by_persona"]["direct_match"]
    weak = metrics["by_persona"]["under_qualified"]
    assert direct["recall@1"] == pytest.approx(0.5)
    assert direct["recall@2"] == pytest.approx(1.0)
    assert weak["recall@1"] == pytest.approx(0.5)
    assert weak["recall@2"] == pytest.approx(0.5)


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
