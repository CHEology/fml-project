"""
Tests for ml/retrieval.py — uses synthetic vectors and a fake encoder.

Real-data integration testing is deferred until Phase 1 lands. The two
TestIntegrationFromFixture tests exercise the full read-from-disk path
using the small committed fixture under tests/fixtures/.

Run:  pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

faiss = pytest.importorskip("faiss")

from ml.retrieval import (
    DEFAULT_K,
    META_COLUMNS,
    JobMatch,
    Retriever,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMB_DIM = 16
N_JOBS = 50
SEED = 0

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _read_faiss_index(path: Path):
    """Read a FAISS index via in-memory bytes.

    Works around a faiss C++ I/O bug on Windows when the absolute path
    contains non-ASCII characters (e.g. the project lives under a path
    with Chinese characters that mojibake through the underlying fopen).
    """
    buf = path.read_bytes()
    return faiss.deserialize_index(np.frombuffer(buf, dtype=np.uint8))


def _l2(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, safe against zero rows."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (matrix / norms).astype(np.float32)


class FakeEncoder:
    """Deterministic encoder: hash text -> seed -> normalized vector."""

    def __init__(self, dim: int = EMB_DIM, salt: str = ""):
        self.dim = dim
        self.salt = salt

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256((self.salt + t).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return _l2(out)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_embeddings(rng):
    return _l2(rng.standard_normal((N_JOBS, EMB_DIM)).astype(np.float32))


@pytest.fixture
def synthetic_metadata(rng):
    return pd.DataFrame({
        "row_id":           np.arange(N_JOBS, dtype=np.int64),
        "job_id":           np.arange(1000, 1000 + N_JOBS, dtype=np.int64),
        "title":            [f"Job {i}" for i in range(N_JOBS)],
        "company_name":     [f"Co {i % 7}" for i in range(N_JOBS)],
        "salary_annual":    rng.uniform(60_000, 200_000, N_JOBS),
        "location":         ["NYC, NY"] * N_JOBS,
        "experience_level": ["mid"] * N_JOBS,
    })


@pytest.fixture
def fake_encoder():
    return FakeEncoder(dim=EMB_DIM)


@pytest.fixture
def faiss_index(synthetic_embeddings):
    idx = faiss.IndexFlatIP(EMB_DIM)
    idx.add(synthetic_embeddings)
    return idx


@pytest.fixture
def retriever(fake_encoder, faiss_index, synthetic_metadata):
    return Retriever(fake_encoder, faiss_index, synthetic_metadata)


# ---------------------------------------------------------------------------
# JobMatch tests
# ---------------------------------------------------------------------------

class TestJobMatch:
    def test_to_dict_keys(self):
        m = JobMatch(
            row_id=0, job_id=10, title="t", company_name="c",
            salary_annual=100.0, location="l", experience_level="mid",
            similarity=0.9,
        )
        expected = {
            "row_id", "job_id", "title", "company_name",
            "salary_annual", "location", "experience_level", "similarity",
        }
        assert set(m.to_dict().keys()) == expected

    def test_to_dict_types_are_native(self):
        """numpy scalars should be cast to native Python types."""
        m = JobMatch(
            row_id=np.int64(5), job_id=np.int64(11), title="t",
            company_name="c", salary_annual=np.float32(120_000.0),
            location="l", experience_level="mid",
            similarity=float(np.float32(0.7)),
        )
        d = m.to_dict()
        assert isinstance(d["row_id"], int)
        assert isinstance(d["job_id"], int)
        assert isinstance(d["salary_annual"], float)
        assert isinstance(d["similarity"], float)

    def test_dataclass_equality(self):
        a = JobMatch(0, 1, "t", "c", 100.0, "l", "mid", 0.5)
        b = JobMatch(0, 1, "t", "c", 100.0, "l", "mid", 0.5)
        c = JobMatch(0, 1, "t", "c", 100.0, "l", "mid", 0.6)
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# Retriever — shape tests
# ---------------------------------------------------------------------------

class TestRetrieverShape:
    def test_search_returns_list_of_jobmatch(self, retriever):
        results = retriever.search("anything", k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, JobMatch) for r in results)

    def test_search_returns_k_results(self, retriever):
        assert len(retriever.search("anything", k=5)) == 5
        assert len(retriever.search("anything", k=15)) == 15

    def test_search_default_k(self, retriever):
        results = retriever.search("anything")
        assert len(results) == DEFAULT_K == 10

    def test_search_by_vector_accepts_1d_and_2d(self, retriever, rng):
        v = _l2(rng.standard_normal((1, EMB_DIM)).astype(np.float32))
        r1 = retriever.search_by_vector(v[0], k=3)
        r2 = retriever.search_by_vector(v, k=3)
        assert len(r1) == len(r2) == 3
        assert [m.row_id for m in r1] == [m.row_id for m in r2]


# ---------------------------------------------------------------------------
# Retriever — ordering / similarity correctness
# ---------------------------------------------------------------------------

class TestRetrieverOrdering:
    def test_results_sorted_by_similarity_descending(self, retriever):
        results = retriever.search("query text", k=10)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_top1_is_query_when_present(
        self, fake_encoder, synthetic_metadata, rng,
    ):
        """If we insert the query vector itself as a job, it should rank #1."""
        query_text = "exact match"
        q = fake_encoder.encode([query_text])  # (1, dim), normalized

        n_other = 30
        other = _l2(rng.standard_normal((n_other, EMB_DIM)).astype(np.float32))
        all_vecs = np.vstack([q, other])

        idx = faiss.IndexFlatIP(EMB_DIM)
        idx.add(all_vecs)

        meta = pd.DataFrame({
            "row_id":           np.arange(len(all_vecs), dtype=np.int64),
            "job_id":           np.arange(2000, 2000 + len(all_vecs), dtype=np.int64),
            "title":            [f"J{i}" for i in range(len(all_vecs))],
            "company_name":     ["X"] * len(all_vecs),
            "salary_annual":    [100_000.0] * len(all_vecs),
            "location":         ["NYC"] * len(all_vecs),
            "experience_level": ["mid"] * len(all_vecs),
        })

        r = Retriever(fake_encoder, idx, meta)
        top = r.search(query_text, k=1)[0]
        assert top.row_id == 0

    def test_similarity_monotone(self, retriever):
        results = retriever.search("anything", k=20)
        sims = [r.similarity for r in results]
        for a, b in zip(sims, sims[1:]):
            assert a >= b - 1e-6

    def test_similarity_in_unit_range(self, retriever):
        for r in retriever.search("anything", k=10):
            assert -1.0001 <= r.similarity <= 1.0001

    def test_self_match_similarity_is_one(
        self, fake_encoder, synthetic_metadata, rng,
    ):
        """Inserting query vector as job 0 should give similarity ~1.0 at top."""
        q = fake_encoder.encode(["match"])
        other = _l2(rng.standard_normal((20, EMB_DIM)).astype(np.float32))
        idx = faiss.IndexFlatIP(EMB_DIM)
        idx.add(np.vstack([q, other]))

        n = 21
        meta = pd.DataFrame({
            "row_id":           np.arange(n, dtype=np.int64),
            "job_id":           np.arange(3000, 3000 + n, dtype=np.int64),
            "title":            [f"J{i}" for i in range(n)],
            "company_name":     ["X"] * n,
            "salary_annual":    [1.0] * n,
            "location":         ["NYC"] * n,
            "experience_level": ["mid"] * n,
        })
        r = Retriever(fake_encoder, idx, meta)
        top = r.search("match", k=1)[0]
        assert top.similarity == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Retriever — edge cases
# ---------------------------------------------------------------------------

class TestRetrieverEdgeCases:
    def test_k_larger_than_n_jobs(self, retriever):
        """k > N must be clamped to N rather than padding with -1 rows."""
        results = retriever.search("anything", k=N_JOBS + 100)
        assert len(results) == N_JOBS

    def test_k_equals_one(self, retriever):
        assert len(retriever.search("anything", k=1)) == 1

    def test_k_zero_returns_empty(self, retriever):
        assert retriever.search("anything", k=0) == []

    def test_empty_resume_text_does_not_crash(self, retriever):
        """Empty string should still produce k results without raising."""
        results = retriever.search("", k=3)
        assert len(results) == 3

    def test_metadata_index_alignment(self, retriever, synthetic_metadata):
        """Each result's row_id must round-trip through the metadata frame."""
        for r in retriever.search("anything", k=5):
            row = synthetic_metadata.iloc[r.row_id]
            assert r.title == row["title"]
            assert r.job_id == int(row["job_id"])


# ---------------------------------------------------------------------------
# Retriever — dependency injection
# ---------------------------------------------------------------------------

class TestRetrieverDependencyInjection:
    def test_swap_encoder_changes_results(
        self, faiss_index, synthetic_metadata,
    ):
        enc_a = FakeEncoder(dim=EMB_DIM, salt="A")
        enc_b = FakeEncoder(dim=EMB_DIM, salt="B")
        r_a = Retriever(enc_a, faiss_index, synthetic_metadata)
        r_b = Retriever(enc_b, faiss_index, synthetic_metadata)
        ids_a = [m.row_id for m in r_a.search("query", k=10)]
        ids_b = [m.row_id for m in r_b.search("query", k=10)]
        assert ids_a != ids_b

    def test_custom_index_duck_type(
        self, fake_encoder, synthetic_embeddings, synthetic_metadata,
    ):
        """A hand-rolled object with .search and .ntotal must work."""
        class NumpyIndex:
            def __init__(self, vecs):
                self.vecs = vecs
                self.ntotal = vecs.shape[0]
            def search(self, q, k):
                sims = (q @ self.vecs.T)[0]
                top = np.argsort(-sims)[:k]
                return sims[top].reshape(1, -1), top.reshape(1, -1)

        idx = NumpyIndex(synthetic_embeddings)
        r = Retriever(fake_encoder, idx, synthetic_metadata)
        results = r.search("anything", k=4)
        assert len(results) == 4
        # ordering monotone
        sims = [m.similarity for m in results]
        assert sims == sorted(sims, reverse=True)

    def test_metadata_can_have_extra_columns(
        self, fake_encoder, faiss_index, synthetic_metadata,
    ):
        meta_plus = synthetic_metadata.copy()
        meta_plus["description"] = "extra text"
        meta_plus["skills_desc"] = "Python, ML"
        r = Retriever(fake_encoder, faiss_index, meta_plus)
        results = r.search("anything", k=3)
        assert len(results) == 3
        # JobMatch only surfaces META_COLUMNS + similarity
        assert "description" not in results[0].to_dict()

    def test_metadata_missing_required_column_raises(
        self, fake_encoder, faiss_index, synthetic_metadata,
    ):
        bad_meta = synthetic_metadata.drop(columns=["salary_annual"])
        with pytest.raises(ValueError, match="salary_annual"):
            Retriever(fake_encoder, faiss_index, bad_meta)


# ---------------------------------------------------------------------------
# Integration — load committed fixture from disk
# ---------------------------------------------------------------------------

class TestIntegrationFromFixture:
    def test_full_pipeline_from_fixture(self, fake_encoder):
        """Read .index + .parquet from disk; query works end-to-end."""
        idx = _read_faiss_index(FIXTURE_DIR / "synthetic_jobs.index")
        meta = pd.read_parquet(FIXTURE_DIR / "synthetic_jobs_meta.parquet")

        # encoder dim must match fixture dim (16)
        enc = FakeEncoder(dim=idx.d)
        r = Retriever(enc, idx, meta)
        results = r.search("software engineer", k=3)

        assert len(results) == 3
        assert all(isinstance(m, JobMatch) for m in results)
        assert all(set(META_COLUMNS).issubset(m.to_dict().keys()) for m in results)
        sims = [m.similarity for m in results]
        assert sims == sorted(sims, reverse=True)

    def test_reproducibility_from_fixture(self):
        """Same query + same seeded encoder → same top-k row ids."""
        idx = _read_faiss_index(FIXTURE_DIR / "synthetic_jobs.index")
        meta = pd.read_parquet(FIXTURE_DIR / "synthetic_jobs_meta.parquet")
        enc = FakeEncoder(dim=idx.d)

        r1 = Retriever(enc, idx, meta)
        r2 = Retriever(FakeEncoder(dim=idx.d), idx, meta)
        ids1 = [m.row_id for m in r1.search("data scientist", k=5)]
        ids2 = [m.row_id for m in r2.search("data scientist", k=5)]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_top_level_imports(self):
        from ml import JobMatch as JM, Retriever as RT  # noqa: F401
        assert JM is JobMatch
        assert RT is Retriever

    def test_jobmatch_to_dict_roundtrip_via_export(self):
        from ml import JobMatch as JM
        m = JM(0, 1, "t", "c", 100.0, "l", "mid", 0.42)
        d = m.to_dict()
        assert d["similarity"] == 0.42
        assert d["title"] == "t"
