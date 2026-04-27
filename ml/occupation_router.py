"""
Route a resume to its most likely SOC occupation(s).

Embeds the O*NET occupation titles once at construction time and runs a
cosine-similarity nearest-neighbor search against the resume embedding.
Combined with `ml.wage_bands.WageBandTable.lookup`, this gives every
resume an actual federal wage band — a much broader signal than the
Phase 4 LinkedIn-trained model can offer for non-tech roles.

Public API:
    `OccupationRouter` — encoder-driven router.
    `OccupationRouter.from_onet_skills(path, encoder)` — build from the
        parquet emitted by `scripts/load_onet_skills.py`.
    `OccupationRouter.from_titles(titles, encoder)` — direct construction.
    `route(resume_text|embedding, k=3)` -> list of `(soc_code, similarity, title)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OccupationMatch:
    soc_code: str
    occupation_title: str
    similarity: float


class OccupationRouter:
    """Cosine-similarity router from a resume vector to SOC occupations."""

    def __init__(
        self,
        soc_codes: list[str],
        titles: list[str],
        title_embeddings: np.ndarray,
        *,
        encoder: Any | None = None,
    ):
        if not (len(soc_codes) == len(titles) == len(title_embeddings)):
            raise ValueError("soc_codes, titles, and title_embeddings must align")
        self.soc_codes = list(soc_codes)
        self.titles = list(titles)
        self.encoder = encoder
        # Pre-normalize so cosine reduces to a dot product.
        self._title_emb = _l2_normalize(title_embeddings.astype(np.float32))

    @classmethod
    def from_onet_skills(
        cls,
        skills_parquet: str | Path,
        encoder: Any,
    ) -> OccupationRouter:
        """Build from the parquet emitted by `scripts/load_onet_skills.py`.

        Each (soc_code, occupation_title) appears once even if the skills
        file lists many rows per occupation.
        """
        df = pd.read_parquet(skills_parquet)
        if "soc_code" not in df.columns or "occupation_title" not in df.columns:
            raise ValueError(
                "skills parquet must have 'soc_code' and 'occupation_title' columns"
            )
        unique = (
            df.dropna(subset=["soc_code", "occupation_title"])
            .drop_duplicates(subset=["soc_code"])
            .reset_index(drop=True)
        )
        unique = unique[unique["occupation_title"].astype(str).str.len() > 0]
        return cls.from_titles(
            unique["soc_code"].astype(str).tolist(),
            unique["occupation_title"].astype(str).tolist(),
            encoder=encoder,
        )

    @classmethod
    def from_titles(
        cls,
        soc_codes: list[str],
        titles: list[str],
        *,
        encoder: Any,
    ) -> OccupationRouter:
        embeddings = np.asarray(encoder.encode(titles), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(titles):
            raise ValueError(
                f"encoder returned shape {embeddings.shape}; expected (N, dim)"
            )
        return cls(soc_codes, titles, embeddings, encoder=encoder)

    def route(
        self,
        query: str | np.ndarray,
        *,
        k: int = 3,
    ) -> list[OccupationMatch]:
        """Top-`k` SOC matches for a single resume text or embedding."""
        if isinstance(query, str):
            if self.encoder is None:
                raise ValueError("routing text requires an encoder")
            encoded = np.asarray(self.encoder.encode([query]), dtype=np.float32)
            query = encoded[0] if encoded.ndim == 2 else encoded

        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 2 and q.shape[0] == 1:
            q = q[0]
        if q.ndim != 1:
            raise ValueError(f"query must be 1-D or (1, dim); got {q.shape}")
        q = _l2_normalize(q)

        sims = self._title_emb @ q
        k_eff = min(int(k), len(sims))
        if k_eff <= 0:
            return []
        top = np.argpartition(-sims, k_eff - 1)[:k_eff]
        top = top[np.argsort(-sims[top])]
        return [
            OccupationMatch(
                soc_code=self.soc_codes[i],
                occupation_title=self.titles[i],
                similarity=float(sims[i]),
            )
            for i in top
        ]

    def __len__(self) -> int:
        return len(self.soc_codes)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x if norm < 1e-12 else x / norm
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms


__all__ = ["OccupationMatch", "OccupationRouter"]
