"""
Evaluate resume-to-job retrieval on synthetic resume/JD pairs.

Expected input:
    data/eval/synthetic_resumes.parquet with columns:
      resume_text, source_job_id, hard_negative_job_id

Expected retrieval artifacts:
    models/jobs.index and models/jobs_meta.parquet from scripts/build_index.py

Outputs:
    data/eval/retrieval_metrics.json
    data/eval/retrieval_errors.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.retrieval import JobMatch, Retriever  # noqa: E402

DEFAULT_EVAL = PROJECT_ROOT / "data" / "eval" / "synthetic_resumes.parquet"
DEFAULT_INDEX = PROJECT_ROOT / "models" / "jobs.index"
DEFAULT_META = PROJECT_ROOT / "models" / "jobs_meta.parquet"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "data" / "eval" / "retrieval_metrics.json"
DEFAULT_ERRORS_OUT = PROJECT_ROOT / "data" / "eval" / "retrieval_errors.csv"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 20


class Searcher(Protocol):
    def search(self, resume_text: str, k: int) -> list[JobMatch]:
        """Return ranked retrieval results for a resume."""


@dataclass(frozen=True)
class EvaluationRow:
    resume_id: str
    source_job_id: int | None
    hard_negative_job_id: int | None
    source_rank: int | None
    hard_negative_rank: int | None
    reciprocal_rank: float
    ndcg_at_k: float
    discriminative_correct: bool
    top_job_id: int | None
    top_similarity: float | None


def evaluate_retrieval(
    eval_df: pd.DataFrame,
    searcher: Searcher,
    *,
    k: int = DEFAULT_K,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a retriever on paired resume/JD rows."""
    required = {"resume_text", "source_job_id", "hard_negative_job_id"}
    missing = [column for column in required if column not in eval_df.columns]
    if missing:
        raise ValueError(f"eval_df is missing required columns: {missing}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    rows: list[EvaluationRow] = []
    for idx, row in eval_df.reset_index(drop=True).iterrows():
        resume_id = str(row.get("resume_id", f"row-{idx}"))
        source_job_id = _optional_int(row["source_job_id"])
        hard_negative_job_id = _optional_int(row["hard_negative_job_id"])
        results = searcher.search(str(row["resume_text"]), k=k)

        ranked_ids = [match.job_id for match in results]
        source_rank = _rank_of(ranked_ids, source_job_id)
        hard_negative_rank = _rank_of(ranked_ids, hard_negative_job_id)
        reciprocal_rank = 0.0 if source_rank is None else 1.0 / source_rank
        ndcg = _ndcg_at_k(ranked_ids, source_job_id, hard_negative_job_id, k)
        discriminative_correct = source_rank is not None and (
            hard_negative_rank is None or source_rank < hard_negative_rank
        )

        top = results[0] if results else None
        rows.append(
            EvaluationRow(
                resume_id=resume_id,
                source_job_id=source_job_id,
                hard_negative_job_id=hard_negative_job_id,
                source_rank=source_rank,
                hard_negative_rank=hard_negative_rank,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                discriminative_correct=discriminative_correct,
                top_job_id=None if top is None else top.job_id,
                top_similarity=None if top is None else float(top.similarity),
            )
        )

    per_row = pd.DataFrame([asdict(row) for row in rows])
    metrics = _aggregate_metrics(per_row, k=k)
    return metrics, per_row


def write_evaluation_outputs(
    metrics: dict[str, float],
    per_row: pd.DataFrame,
    metrics_out: str | Path,
    errors_out: str | Path,
) -> None:
    """Write aggregate metrics JSON and row-level CSV."""
    metrics_path = Path(metrics_out)
    errors_path = Path(errors_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    per_row.to_csv(errors_path, index=False)


def load_retriever(
    *,
    index_path: str | Path,
    meta_path: str | Path,
    model_name: str = DEFAULT_MODEL,
) -> Retriever:
    """Load the FAISS index, metadata, and project encoder."""
    import faiss

    try:
        from ml.embeddings import Encoder
    except ImportError as exc:
        raise RuntimeError(
            "ml.embeddings.Encoder is required to run retrieval evaluation. "
            "Build the embedding module first, then rebuild models/jobs.index."
        ) from exc

    index = _read_faiss_index(Path(index_path), faiss)
    metadata = pd.read_parquet(meta_path)
    encoder = Encoder(model_name=model_name)
    return Retriever(encoder, index, metadata)


def _aggregate_metrics(per_row: pd.DataFrame, *, k: int) -> dict[str, float]:
    n = len(per_row)
    if n == 0:
        return {
            "n": 0,
            "k": k,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            f"recall@{k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
            "discriminative_accuracy": 0.0,
        }

    source_rank = per_row["source_rank"]

    def recall_at(limit: int) -> float:
        return float(source_rank.notna().where(source_rank <= limit, False).mean())

    return {
        "n": int(n),
        "k": int(k),
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        f"recall@{k}": recall_at(k),
        "mrr": float(per_row["reciprocal_rank"].mean()),
        f"ndcg@{k}": float(per_row["ndcg_at_k"].mean()),
        "discriminative_accuracy": float(per_row["discriminative_correct"].mean()),
    }


def _ndcg_at_k(
    ranked_ids: list[int | None],
    source_job_id: int | None,
    hard_negative_job_id: int | None,
    k: int,
) -> float:
    relevance = []
    for job_id in ranked_ids[:k]:
        if source_job_id is not None and job_id == source_job_id:
            relevance.append(2)
        elif hard_negative_job_id is not None and job_id == hard_negative_job_id:
            relevance.append(1)
        else:
            relevance.append(0)

    dcg = _dcg(relevance)
    ideal_relevance = [2]
    if hard_negative_job_id is not None:
        ideal_relevance.append(1)
    idcg = _dcg(ideal_relevance[:k])
    return 0.0 if idcg == 0 else float(dcg / idcg)


def _dcg(relevance: list[int]) -> float:
    return float(
        sum(
            (2**rel - 1) / np.log2(rank + 1)
            for rank, rel in enumerate(relevance, start=1)
        )
    )


def _rank_of(ranked_ids: list[int | None], target: int | None) -> int | None:
    if target is None:
        return None
    for idx, job_id in enumerate(ranked_ids, start=1):
        if job_id == target:
            return idx
    return None


def _optional_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _read_faiss_index(path: Path, faiss):
    """Read FAISS index robustly on Windows paths with non-ASCII characters."""
    try:
        return faiss.read_index(str(path))
    except RuntimeError:
        buf = path.read_bytes()
        return faiss.deserialize_index(np.frombuffer(buf, dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume retrieval")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--errors-out", type=Path, default=DEFAULT_ERRORS_OUT)
    args = parser.parse_args()

    eval_df = pd.read_parquet(args.eval)
    retriever = load_retriever(
        index_path=args.index,
        meta_path=args.meta,
        model_name=args.model,
    )
    metrics, per_row = evaluate_retrieval(eval_df, retriever, k=args.k)
    write_evaluation_outputs(metrics, per_row, args.metrics_out, args.errors_out)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote row-level results to {args.errors_out}")


if __name__ == "__main__":
    main()
