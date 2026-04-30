from __future__ import annotations

import numpy as np
import pytest
from ml.embeddings import DEFAULT_MODEL_NAME, Encoder, l2_normalize


class FakeSentenceTransformer:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict[str, object]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts,
        *,
        batch_size=None,
        show_progress_bar=None,
        convert_to_numpy=None,
        normalize_embeddings=None,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )

        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            if text:
                encoded = text.encode("utf-8")
                for col in range(self.dim):
                    vectors[row, col] = sum(encoded[col :: self.dim]) + col + 1
        return vectors


class MinimalFakeModel:
    dim = 3

    def encode(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)


def test_l2_normalize_returns_float32_unit_rows() -> None:
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64)

    normalized = l2_normalize(vectors)

    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_l2_normalize_keeps_zero_rows_zero() -> None:
    normalized = l2_normalize(np.zeros((2, 3), dtype=np.float32))

    assert np.array_equal(normalized, np.zeros((2, 3), dtype=np.float32))
    assert not np.isnan(normalized).any()


def test_encoder_metadata_defaults() -> None:
    model = FakeSentenceTransformer(dim=6)
    encoder = Encoder(model=model)

    assert encoder.model_name == DEFAULT_MODEL_NAME
    assert encoder.dim == 6
    assert encoder.batch_size == 32


def test_encode_returns_expected_shape_dtype_and_normalization() -> None:
    encoder = Encoder(model=FakeSentenceTransformer(dim=5))

    vectors = encoder.encode(["machine learning engineer", "data scientist"])

    assert vectors.shape == (2, 5)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_encode_accepts_single_string_as_one_text() -> None:
    encoder = Encoder(model=FakeSentenceTransformer(dim=4))

    vectors = encoder.encode("resume text")

    assert vectors.shape == (1, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_encode_empty_list_returns_empty_matrix_with_dim() -> None:
    encoder = Encoder(model=FakeSentenceTransformer(dim=7))

    vectors = encoder.encode([])

    assert vectors.shape == (0, 7)
    assert vectors.dtype == np.float32


def test_encode_empty_string_does_not_crash_or_nan() -> None:
    encoder = Encoder(model=FakeSentenceTransformer(dim=4))

    vectors = encoder.encode([""])

    assert vectors.shape == (1, 4)
    assert not np.isnan(vectors).any()


def test_encode_forwards_batch_size_and_progress_flag() -> None:
    model = FakeSentenceTransformer(dim=4)
    encoder = Encoder(model=model, batch_size=16)

    encoder.encode(["a", "b", "c"], batch_size=2, show_progress_bar=True)

    assert model.calls[-1]["texts"] == ["a", "b", "c"]
    assert model.calls[-1]["batch_size"] == 2
    assert model.calls[-1]["show_progress_bar"] is True
    assert model.calls[-1]["convert_to_numpy"] is True
    assert model.calls[-1]["normalize_embeddings"] is False


def test_encode_supports_minimal_test_double() -> None:
    encoder = Encoder(model=MinimalFakeModel())

    vectors = encoder.encode(["a", "b"])

    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_encode_rejects_non_string_items() -> None:
    encoder = Encoder(model=FakeSentenceTransformer())

    with pytest.raises(TypeError, match="all texts must be strings"):
        encoder.encode(["valid", 123])


def test_encode_rejects_bad_batch_size() -> None:
    encoder = Encoder(model=FakeSentenceTransformer())

    with pytest.raises(ValueError, match="batch_size must be positive"):
        encoder.encode(["valid"], batch_size=0)


def test_encode_validates_model_shape() -> None:
    class BadShapeModel(FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), self.dim + 1), dtype=np.float32)

    encoder = Encoder(model=BadShapeModel(dim=4))

    with pytest.raises(ValueError, match="unexpected shape"):
        encoder.encode(["text"])
