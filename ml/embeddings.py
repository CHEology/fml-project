"""Text embedding helpers for resumes and job postings.

The retrieval pipeline expects all vectors to be float32 and L2-normalized so
that FAISS inner product search is equivalent to cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
EPSILON = 1e-12


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalized float32 matrix.

    Zero vectors are left as zeros rather than producing NaNs.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"vectors must be a 1D or 2D array; got shape {matrix.shape}")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < EPSILON] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


class Encoder:
    """Sentence-transformer encoder used by indexing and retrieval.

    Args:
        model_name: Hugging Face / sentence-transformers model name.
        device: Optional torch device forwarded to SentenceTransformer.
        batch_size: Default batch size used by ``encode``.
        model: Optional injected model for tests. It must expose
            ``encode(texts, ...)`` and should expose either
            ``get_sentence_embedding_dimension()`` or ``dim``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = (
            model if model is not None else self._load_model(model_name, device)
        )
        self.dim = self._infer_dim(self.model)

    @staticmethod
    def _load_model(model_name: str, device: str | None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Encoder. "
                "Install dependencies with `uv sync`."
            ) from e

        return SentenceTransformer(model_name, device=device)

    @staticmethod
    def _infer_dim(model: Any) -> int:
        if hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
        else:
            dim = getattr(model, "dim", None)

        if dim is None:
            raise ValueError(
                "Could not infer embedding dimension from model. "
                "Provide a model with get_sentence_embedding_dimension() or dim."
            )
        return int(dim)

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode text into L2-normalized float32 vectors.

        A single string is treated as one text. Empty input returns an empty
        matrix with shape ``(0, self.dim)``.
        """
        normalized_texts = self._coerce_texts(texts)
        if not normalized_texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {effective_batch_size}")

        raw_vectors = self._encode_with_model(
            normalized_texts,
            batch_size=effective_batch_size,
            show_progress_bar=show_progress_bar,
        )
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape != (len(normalized_texts), self.dim):
            raise ValueError(
                "Encoder model returned unexpected shape: "
                f"expected {(len(normalized_texts), self.dim)}, got {vectors.shape}"
            )
        return l2_normalize(vectors)

    def _encode_with_model(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        try:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except TypeError:
            # Keeps injected test doubles simple while the production
            # SentenceTransformer path still uses the full API above.
            return self.model.encode(texts)

    @staticmethod
    def _coerce_texts(texts: str | Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            return [texts]
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a string or a sequence of strings")

        coerced = list(texts)
        invalid = [
            type(value).__name__ for value in coerced if not isinstance(value, str)
        ]
        if invalid:
            raise TypeError(f"all texts must be strings; found {invalid[0]}")
        return coerced
