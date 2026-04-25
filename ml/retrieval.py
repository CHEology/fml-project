"""
Cosine-similarity retrieval over a FAISS index.

Encodes a resume (or any text) with an injected encoder, queries a
FAISS IndexFlatIP built on L2-normalized job embeddings, and returns
the top-k matching jobs joined with their metadata.

Note:
    The module-level convenience `search()` and a `load_default_retriever`
    factory documented in plan.md §2.4 are intentionally not exposed yet —
    they require `ml/embeddings.py` (Task 2.1, @ohortig) to land first.
    In the meantime, callers (tests, app) construct a `Retriever` directly
    with their own encoder + index + metadata.

Grading constraint: pure inner-product on L2-normalized vectors —
no learned re-ranker, no sklearn dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K = 10

META_COLUMNS = (
    "row_id",
    "job_id",
    "title",
    "company_name",
    "salary_annual",
    "location",
    "experience_level",
)


# ---------------------------------------------------------------------------
# JobMatch — typed return shape
# ---------------------------------------------------------------------------

@dataclass
class JobMatch:
    """One ranked job result returned by `Retriever.search`.

    `similarity` is the inner product of the (L2-normalized) query and
    job vectors — equivalent to cosine similarity, in [-1.0, 1.0].
    """
    row_id: int
    job_id: int | None
    title: str
    company_name: str
    salary_annual: float | None
    location: str
    experience_level: str | None
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict — casts numpy scalars to native Python types."""
        d = asdict(self)
        for key, val in d.items():
            if isinstance(val, np.generic):
                d[key] = val.item()
        return d


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Cosine-similarity retriever with dependency-injected components.

    Args:
        encoder:  Any object exposing `.encode(texts: list[str]) -> np.ndarray`
                  returning L2-normalized float32 vectors of shape (N, dim).
        index:    Any object exposing `.search(q: np.ndarray, k: int) -> (D, I)`
                  and `.ntotal: int`. A FAISS `IndexFlatIP` is the default;
                  duck-typed for testability.
        metadata: DataFrame whose row order matches the index. Must contain
                  every name in `META_COLUMNS` (extra columns are allowed
                  and ignored).
    """

    def __init__(
        self,
        encoder: Any,
        index: Any,
        metadata: pd.DataFrame,
    ):
        missing = [c for c in META_COLUMNS if c not in metadata.columns]
        if missing:
            raise ValueError(
                f"metadata is missing required columns: {missing}. "
                f"Required: {list(META_COLUMNS)}"
            )

        self.encoder = encoder
        self.index = index
        self.metadata = metadata.reset_index(drop=True)

    def search(self, resume_text: str, k: int = DEFAULT_K) -> list[JobMatch]:
        """Encode `resume_text` and return the top-`k` matching jobs."""
        query_vec = self.encoder.encode([resume_text])
        query_vec = np.asarray(query_vec, dtype=np.float32)
        return self.search_by_vector(query_vec, k=k)

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        k: int = DEFAULT_K,
    ) -> list[JobMatch]:
        """Query the index directly with a pre-computed (already-normalized) vector."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        elif q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(
                f"query_vec must be shape (dim,) or (1, dim); got {q.shape}"
            )

        ntotal = getattr(self.index, "ntotal", len(self.metadata))
        k_eff = min(k, ntotal)
        if k_eff == 0:
            return []

        sims, idxs = self.index.search(q, k_eff)
        sims_row = np.asarray(sims[0])
        idxs_row = np.asarray(idxs[0])

        matches: list[JobMatch] = []
        for raw_idx, raw_sim in zip(idxs_row, sims_row):
            idx = int(raw_idx)
            if idx < 0:                  # FAISS returns -1 in unused slots
                continue
            row = self.metadata.iloc[idx]
            matches.append(_row_to_jobmatch(row, similarity=float(raw_sim)))
        return matches


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_jobmatch(row: pd.Series, *, similarity: float) -> JobMatch:
    """Build a JobMatch from a metadata row, normalizing optional fields."""
    return JobMatch(
        row_id=int(row["row_id"]),
        job_id=_optional_int(row["job_id"]),
        title=str(row["title"]),
        company_name=str(row["company_name"]),
        salary_annual=_optional_float(row["salary_annual"]),
        location=str(row["location"]),
        experience_level=_optional_str(row["experience_level"]),
        similarity=similarity,
    )


def _optional_int(v: Any) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return int(v)


def _optional_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return float(v)


def _optional_str(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return str(v)
