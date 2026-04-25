"""
Build a FAISS retrieval index from preprocessed job postings.

Usage:
    python scripts/build_index.py \
        --input data/processed/jobs.parquet \
        --embeddings-out models/job_embeddings.npy \
        --index-out      models/jobs.index \
        --meta-out       models/jobs_meta.parquet

Reads the preprocessed jobs parquet (Phase 1 output), encodes the `text`
column with the project Encoder (Task 2.1), builds an IndexFlatIP over
L2-normalized vectors, and writes the embeddings, index, and a small
metadata parquet for downstream retrieval.

`--smoke` mode bypasses both the parquet load and the encoder. It generates
deterministic random vectors and a synthetic metadata frame, runs the full
build/write path, and exits. Intended for sanity-checking the script and
generating the test fixture under `tests/fixtures/` before Phase 1 lands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so `ml.*` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "input":           str(PROJECT_ROOT / "data" / "processed" / "jobs.parquet"),
    "model":           "all-MiniLM-L6-v2",
    "batch_size":      256,
    "text_column":     "text",
    "id_column":       "job_id",
    "embeddings_out":  str(PROJECT_ROOT / "models" / "job_embeddings.npy"),
    "index_out":       str(PROJECT_ROOT / "models" / "jobs.index"),
    "meta_out":        str(PROJECT_ROOT / "models" / "jobs_meta.parquet"),
    "seed":            42,
    "smoke_n":         30,
    "smoke_dim":       16,
}

META_COLUMNS_REQUIRED = (
    "title",
    "company_name",
    "salary_annual",
    "location",
    "experience_level",
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray):
    """Build a FAISS IndexFlatIP over L2-normalized vectors.

    Args:
        embeddings: (N, dim) float32 array, L2-normalized row-wise.
    Returns:
        A populated `faiss.IndexFlatIP`.
    """
    import faiss  # lazy import so module can be imported without faiss installed

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    assert index.ntotal == n, "FAISS index size mismatch"
    return index


def encode_jobs(
    texts: list[str],
    encoder,
    *,
    batch_size: int = DEFAULTS["batch_size"],
) -> np.ndarray:
    """Encode a list of texts with the project Encoder, in batches.

    Wraps the encoder's `.encode` so we get progress logs and a single
    concatenated float32 matrix back, even for very large inputs.
    """
    if len(texts) == 0:
        return np.zeros((0, getattr(encoder, "dim", 0)), dtype=np.float32)

    chunks: list[np.ndarray] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_vecs = encoder.encode(texts[start:end])
        chunks.append(np.asarray(batch_vecs, dtype=np.float32))
        print(f"  encoded {end}/{n} texts", flush=True)
    return np.concatenate(chunks, axis=0)


def write_meta(df, out_path: str | Path) -> None:
    """Subset a job DataFrame to META_COLUMNS and write parquet.

    Adds a `row_id` column matching the FAISS index row order so retrieval
    can join back without relying on DataFrame index.
    """
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [c for c in META_COLUMNS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Required: {list(META_COLUMNS_REQUIRED)}"
        )

    job_id = df["job_id"] if "job_id" in df.columns else pd.Series(
        np.arange(len(df), dtype=np.int64), name="job_id"
    )

    meta = pd.DataFrame({
        "row_id":           np.arange(len(df), dtype=np.int64),
        "job_id":           np.asarray(job_id),
        "title":            df["title"].astype(str).to_numpy(),
        "company_name":     df["company_name"].astype(str).to_numpy(),
        "salary_annual":    df["salary_annual"].astype(float).to_numpy(),
        "location":         df["location"].astype(str).to_numpy(),
        "experience_level": df["experience_level"].astype(str).to_numpy(),
    })
    meta.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------------
# Smoke-mode synthetic data
# ---------------------------------------------------------------------------

def _make_smoke_data(n: int, dim: int, seed: int):
    """Generate deterministic L2-normalized vectors + matching metadata."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    vecs = vecs / norms

    df = pd.DataFrame({
        "job_id":           np.arange(1000, 1000 + n, dtype=np.int64),
        "title":            [f"Synthetic Job {i}" for i in range(n)],
        "company_name":     [f"SynthCo {i % 7}" for i in range(n)],
        "salary_annual":    rng.uniform(60_000, 200_000, n).astype(float),
        "location":         ["NYC, NY"] * n,
        "experience_level": ["mid"] * n,
        "text":             [f"Synthetic job description {i}" for i in range(n)],
    })
    return vecs, df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index")
    parser.add_argument("--input", type=str, default=DEFAULTS["input"],
                        help="Path to preprocessed jobs parquet")
    parser.add_argument("--model", type=str, default=DEFAULTS["model"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--text-column", type=str, default=DEFAULTS["text_column"])
    parser.add_argument("--embeddings-out", type=str,
                        default=DEFAULTS["embeddings_out"])
    parser.add_argument("--index-out", type=str, default=DEFAULTS["index_out"])
    parser.add_argument("--meta-out", type=str, default=DEFAULTS["meta_out"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--smoke", action="store_true",
                        help="Bypass parquet/encoder; generate synthetic data")
    parser.add_argument("--smoke-n", type=int, default=DEFAULTS["smoke_n"],
                        help="Number of synthetic rows in --smoke mode")
    parser.add_argument("--smoke-dim", type=int, default=DEFAULTS["smoke_dim"],
                        help="Vector dimension in --smoke mode")
    args = parser.parse_args()

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    if args.smoke:
        print(f"--smoke: generating {args.smoke_n} synthetic rows "
              f"of dim {args.smoke_dim}")
        embeddings, df = _make_smoke_data(args.smoke_n, args.smoke_dim, args.seed)
        text_column = "text"
    else:
        import pandas as pd
        print(f"Loading jobs parquet: {args.input}")
        df = pd.read_parquet(args.input)
        text_column = args.text_column

        if text_column not in df.columns:
            raise ValueError(
                f"Input parquet has no column '{text_column}'. "
                f"Available: {list(df.columns)}"
            )

        try:
            from ml.embeddings import Encoder
        except ImportError as e:
            raise RuntimeError(
                "ml/embeddings.Encoder not yet implemented "
                "(Task 2.1, @ohortig). Run with --smoke for synthetic data."
            ) from e

        encoder = Encoder(model_name=args.model)
        print(f"Encoder: {args.model} (dim={getattr(encoder, 'dim', '?')})")
        print(f"Encoding {len(df)} texts in batches of {args.batch_size}...")
        embeddings = encode_jobs(
            df[text_column].astype(str).tolist(),
            encoder,
            batch_size=args.batch_size,
        )

    n, dim = embeddings.shape
    print(f"Embeddings shape: ({n}, {dim}) dtype={embeddings.dtype}")

    Path(args.embeddings_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_out, embeddings)
    print(f"Saved embeddings: {args.embeddings_out}")

    index = build_index(embeddings)

    import faiss
    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.index_out)
    print(f"Saved index:      {args.index_out}  (ntotal={index.ntotal})")

    write_meta(df, args.meta_out)
    print(f"Saved meta:       {args.meta_out}  ({len(df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
