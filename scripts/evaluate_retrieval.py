"""
Evaluate resume-to-job retrieval on synthetic resume/JD pairs.

Expected input:
    data/eval/synthetic_resumes.parquet with columns:
      resume_text, source_job_id, hard_negative_job_id (optional),
      hard_negative_job_ids (optional list, preferred when present)

Expected retrieval artifacts:
    models/jobs.index and models/jobs_meta.parquet from scripts/build_index.py

Outputs:
    data/eval/retrieval_metrics.json
    data/eval/retrieval_errors.csv

The evaluator gracefully accepts either the legacy single hard-negative
column or the new list column, and slices metrics per persona / quality
label when those columns are present.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
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
DEFAULT_RECALL_KS: tuple[int, ...] = (1, 5, 10, 20)
SLICE_COLUMNS: tuple[str, ...] = ("persona", "quality_label")


class Searcher(Protocol):
    def search(self, resume_text: str, k: int) -> list[JobMatch]:
        """Return ranked retrieval results for a resume."""


@dataclass(frozen=True)
class EvaluationRow:
    resume_id: str
    source_job_id: int | None
    hard_negative_job_id: int | None
    hard_negative_job_ids: tuple[int, ...] = field(default_factory=tuple)
    source_rank: int | None = None
    best_hard_negative_rank: int | None = None
    hard_negative_rank: int | None = None  # legacy alias for the first negative
    reciprocal_rank: float = 0.0
    ndcg_at_k: float = 0.0
    discriminative_correct: bool = False
    top_job_id: int | None = None
    top_similarity: float | None = None
    persona: str | None = None
    quality_label: str | None = None


def evaluate_retrieval(
    eval_df: pd.DataFrame,
    searcher: Searcher,
    *,
    k: int = DEFAULT_K,
    recall_ks: Iterable[int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a retriever on paired resume/JD rows.

    `recall_ks` controls which recall cutoffs appear in the aggregate
    metrics dict. When unset, defaults to (1, 5, 10, 20) intersected
    with `k` so we always include the search depth.
    """
    required = {"resume_text", "source_job_id"}
    missing = [column for column in required if column not in eval_df.columns]
    if missing:
        raise ValueError(f"eval_df is missing required columns: {missing}")
    if (
        "hard_negative_job_id" not in eval_df.columns
        and "hard_negative_job_ids" not in eval_df.columns
    ):
        raise ValueError(
            "eval_df is missing required columns: ['hard_negative_job_id'] or ['hard_negative_job_ids']"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    cutoffs = _normalise_recall_ks(recall_ks, k)

    rows: list[EvaluationRow] = []
    for idx, row in eval_df.reset_index(drop=True).iterrows():
        resume_id = str(row.get("resume_id", f"row-{idx}"))
        source_job_id = _optional_int(row.get("source_job_id"))
        negatives = _resolve_negatives(row)
        results = searcher.search(str(row["resume_text"]), k=k)

        ranked_ids = [match.job_id for match in results]
        source_rank = _rank_of(ranked_ids, source_job_id)
        negative_ranks = [_rank_of(ranked_ids, neg_id) for neg_id in negatives]
        ranked_negatives = [r for r in negative_ranks if r is not None]
        best_neg_rank = min(ranked_negatives) if ranked_negatives else None
        first_neg_rank = negative_ranks[0] if negative_ranks else None

        reciprocal_rank = 0.0 if source_rank is None else 1.0 / source_rank
        ndcg = _ndcg_at_k(ranked_ids, source_job_id, negatives, k)
        discriminative_correct = source_rank is not None and (
            best_neg_rank is None or source_rank < best_neg_rank
        )

        top = results[0] if results else None
        rows.append(
            EvaluationRow(
                resume_id=resume_id,
                source_job_id=source_job_id,
                hard_negative_job_id=negatives[0] if negatives else None,
                hard_negative_job_ids=tuple(negatives),
                source_rank=source_rank,
                best_hard_negative_rank=best_neg_rank,
                hard_negative_rank=first_neg_rank,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                discriminative_correct=discriminative_correct,
                top_job_id=None if top is None else top.job_id,
                top_similarity=None if top is None else float(top.similarity),
                persona=_optional_str(row.get("persona")),
                quality_label=_optional_str(row.get("quality_label")),
            )
        )

    per_row = pd.DataFrame([asdict(row) for row in rows])
    metrics = _aggregate_metrics(per_row, k=k, recall_ks=cutoffs)
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


def _aggregate_metrics(
    per_row: pd.DataFrame,
    *,
    k: int,
    recall_ks: tuple[int, ...],
) -> dict[str, float]:
    if per_row.empty:
        empty = {
            "n": 0,
            "k": k,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
            "discriminative_accuracy": 0.0,
        }
        for cutoff in recall_ks:
            empty[f"recall@{cutoff}"] = 0.0
        return empty

    overall = _metric_block(per_row, k=k, recall_ks=recall_ks)
    overall["k"] = int(k)

    by_slice: dict[str, dict[str, dict[str, float]]] = {}
    for column in SLICE_COLUMNS:
        if column not in per_row.columns or per_row[column].isna().all():
            continue
        slice_metrics: dict[str, dict[str, float]] = {}
        for value, group in per_row.groupby(column):
            if pd.isna(value):
                continue
            slice_metrics[str(value)] = _metric_block(group, k=k, recall_ks=recall_ks)
        if slice_metrics:
            by_slice[f"by_{column}"] = slice_metrics

    return {**overall, **by_slice}


def _metric_block(
    per_row: pd.DataFrame,
    *,
    k: int,
    recall_ks: tuple[int, ...],
) -> dict[str, float]:
    source_rank = per_row["source_rank"]
    n = int(len(per_row))

    def recall_at(limit: int) -> float:
        return float(source_rank.notna().where(source_rank <= limit, False).mean())

    block: dict[str, float] = {"n": n}
    for cutoff in recall_ks:
        block[f"recall@{cutoff}"] = recall_at(cutoff)

    block["mrr"] = float(per_row["reciprocal_rank"].mean())
    block[f"ndcg@{k}"] = float(per_row["ndcg_at_k"].mean())
    block["discriminative_accuracy"] = float(per_row["discriminative_correct"].mean())
    return block


def _ndcg_at_k(
    ranked_ids: list[int | None],
    source_job_id: int | None,
    hard_negative_job_ids: list[int],
    k: int,
) -> float:
    """NDCG@k where source counts as rel=2 and any listed negative as rel=1."""
    negative_set = {neg for neg in hard_negative_job_ids if neg is not None}
    relevance: list[int] = []
    for job_id in ranked_ids[:k]:
        if source_job_id is not None and job_id == source_job_id:
            relevance.append(2)
        elif job_id in negative_set:
            relevance.append(1)
        else:
            relevance.append(0)

    dcg = _dcg(relevance)
    ideal_relevance = [2] + [1] * len(negative_set)
    idcg = _dcg(ideal_relevance[:k])
    return 0.0 if idcg == 0 else float(dcg / idcg)


def _resolve_negatives(row: pd.Series) -> list[int]:
    """Prefer the multi-negative list column; fall back to the legacy scalar."""
    negatives: list[int] = []
    raw_list = (
        row.get("hard_negative_job_ids") if "hard_negative_job_ids" in row else None
    )
    if raw_list is not None and not _is_missing(raw_list):
        for value in raw_list:
            cleaned = _optional_int(value)
            if cleaned is not None and cleaned not in negatives:
                negatives.append(cleaned)
    if not negatives:
        scalar = _optional_int(row.get("hard_negative_job_id"))
        if scalar is not None:
            negatives.append(scalar)
    return negatives


def _normalise_recall_ks(
    recall_ks: Iterable[int] | None,
    k: int,
) -> tuple[int, ...]:
    if recall_ks is None:
        cutoffs = list(DEFAULT_RECALL_KS)
    else:
        cutoffs = [int(c) for c in recall_ks if int(c) > 0]
    cutoffs.append(int(k))
    cutoffs = sorted({c for c in cutoffs if c <= k})
    return tuple(cutoffs)


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_str(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


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
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated recall cutoffs (e.g. '1,5,10,20,50'). Each must be <= --k.",
    )
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--errors-out", type=Path, default=DEFAULT_ERRORS_OUT)
    args = parser.parse_args()

    eval_df = pd.read_parquet(args.eval)
    retriever = load_retriever(
        index_path=args.index,
        meta_path=args.meta,
        model_name=args.model,
    )

    recall_ks: tuple[int, ...] | None = None
    if args.k_sweep:
        recall_ks = tuple(int(part) for part in args.k_sweep.split(",") if part.strip())

    metrics, per_row = evaluate_retrieval(
        eval_df, retriever, k=args.k, recall_ks=recall_ks
    )
    write_evaluation_outputs(metrics, per_row, args.metrics_out, args.errors_out)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote row-level results to {args.errors_out}")


if __name__ == "__main__":
    main()
