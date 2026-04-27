"""
Validate the ResuMatch pipeline against real (or real-style) resumes.

Real resumes carry no salary or quality ground truth, so this harness
relies on three independent signals that *together* tell us whether
the pipeline behaves sensibly on real input:

1. **Quality**: rule-based `score_resume_quality(text)` over the full
   corpus (real-resume-safe by construction). When the learned MLP
   checkpoint is available, we also report it and the Spearman rank
   correlation with the rule-based score — a sanity-check that the
   learned proxy is at least monotone in the same direction.

2. **Retrieval**: top-k FAISS results per resume. We don't have a
   ground-truth job for a real candidate, so we only report
   summary statistics (mean similarity, distinct-companies coverage).

3. **Salary self-consistency**: predicted q50 vs. the *median salary
   of the top-k retrieved jobs*. This is the closest thing to ground
   truth without salary labels — the actual product use case is "given
   this resume, what would the matching jobs pay?", so its predicted
   q50 should land near the retrieved median, and the retrieved median
   should sit inside the predicted [q10, q90].

Each section degrades gracefully:

* No `ml.embeddings.Encoder` → use deterministic random embeddings
  (`--smoke`). Quality scores still computed; retrieval / salary
  sections become uninformative but do not crash.
* No FAISS index / metadata → skip retrieval + self-consistency.
* No salary checkpoint → skip salary + self-consistency.
* No quality MLP checkpoint → only report rule-based scores.

Usage:
    python scripts/validate_on_real_resumes.py \\
        --resumes data/eval/real_resumes.parquet \\
        --index   models/jobs.index \\
        --meta    models/jobs_meta.parquet \\
        --salary-model  models/resume_salary_model.pt \\
        --salary-scaler models/resume_salary_model.scaler.json \\
        --quality-model models/quality_model.pt \\
        --out data/eval/real_resume_validation.json

    # Smoke run (no encoder / no checkpoints required):
    python scripts/validate_on_real_resumes.py \\
        --resumes tests/fixtures/sample_real_resumes.csv \\
        --smoke --out .tmp/real_resume_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.quality import score_resume_quality  # noqa: E402

DEFAULT_RESUMES = PROJECT_ROOT / "data" / "eval" / "real_resumes.parquet"
DEFAULT_INDEX = PROJECT_ROOT / "models" / "jobs.index"
DEFAULT_META = PROJECT_ROOT / "models" / "jobs_meta.parquet"
DEFAULT_ONET_SKILLS = PROJECT_ROOT / "data" / "external" / "onet_skills.parquet"
DEFAULT_BLS_WAGES = PROJECT_ROOT / "data" / "external" / "bls_wages.parquet"
DEFAULT_SALARY_MODEL = PROJECT_ROOT / "models" / "resume_salary_model.pt"
DEFAULT_SALARY_SCALER = PROJECT_ROOT / "models" / "resume_salary_model.scaler.json"
DEFAULT_QUALITY_MODEL = PROJECT_ROOT / "models" / "quality_model.pt"
DEFAULT_QUALITY_SCALER = PROJECT_ROOT / "models" / "quality_model.scaler.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "eval" / "real_resume_validation.json"
DEFAULT_PER_ROW = PROJECT_ROOT / "data" / "eval" / "real_resume_validation_rows.csv"


def validate(
    resumes_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    retriever: Any | None = None,
    salary_predictor: Any | None = None,
    quality_predictor: Any | None = None,
    occupation_router: Any | None = None,
    wage_table: Any | None = None,
    k: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the validation pipeline and return (summary, per-row DataFrame)."""
    if "resume_text" not in resumes_df.columns:
        raise ValueError("resumes_df must have a 'resume_text' column")
    if len(resumes_df) != len(embeddings):
        raise ValueError(
            f"resumes_df rows ({len(resumes_df)}) must match embeddings ({len(embeddings)})"
        )

    rows: list[dict[str, Any]] = []
    for idx, row in resumes_df.reset_index(drop=True).iterrows():
        resume_text = str(row["resume_text"])
        embedding = np.asarray(embeddings[idx], dtype=np.float32)
        rule = score_resume_quality(resume_text)

        record: dict[str, Any] = {
            "resume_id": str(row.get("resume_id", f"row-{idx}")),
            "category": row.get("category"),
            "rule_score": rule["score"],
            "rule_label": rule["label"],
            "rule_weakest_dim": rule["weakest_dim"],
        }

        if quality_predictor is not None:
            try:
                learned = quality_predictor(embedding)
                record["learned_score"] = float(learned.get("score", float("nan")))
                record["learned_label"] = learned.get("label")
            except Exception as exc:  # pragma: no cover - defensive
                record["learned_score"] = float("nan")
                record["learned_error"] = str(exc)

        if occupation_router is not None:
            try:
                matches = occupation_router.route(embedding, k=1)
                if matches:
                    match = matches[0]
                    record["soc_code"] = match.soc_code
                    record["occupation_title"] = match.occupation_title
                    record["soc_similarity"] = float(match.similarity)
                    if wage_table is not None:
                        band = wage_table.lookup(match.soc_code)
                        if band is not None:
                            record.update(
                                {
                                    "bls_p10": band.p10,
                                    "bls_p25": band.p25,
                                    "bls_p50": band.p50,
                                    "bls_p75": band.p75,
                                    "bls_p90": band.p90,
                                }
                            )
            except Exception as exc:  # pragma: no cover - defensive
                record["occupation_error"] = str(exc)

        retrieved_salaries: list[float] = []
        if retriever is not None:
            results = retriever.search(resume_text, k=k)
            sims = [float(r.similarity) for r in results]
            retrieved_salaries = [
                float(r.salary_annual)
                for r in results
                if r.salary_annual is not None and r.salary_annual > 0
            ]
            record["retrieved_n"] = len(results)
            record["retrieved_mean_similarity"] = float(np.mean(sims)) if sims else 0.0
            record["retrieved_distinct_companies"] = len(
                {r.company_name for r in results if r.company_name}
            )
            record["retrieved_median_salary"] = (
                float(np.median(retrieved_salaries))
                if retrieved_salaries
                else float("nan")
            )

        if salary_predictor is not None:
            try:
                quants = salary_predictor(embedding)
                for key, value in quants.items():
                    record[f"pred_{key}"] = float(value)
                if retrieved_salaries:
                    median_retrieved = float(np.median(retrieved_salaries))
                    record["self_consistency_abs_err"] = abs(
                        float(quants["q50"]) - median_retrieved
                    )
                    record["self_consistency_in_band"] = (
                        float(quants["q10"]) <= median_retrieved <= float(quants["q90"])
                    )
            except Exception as exc:  # pragma: no cover - defensive
                record["salary_error"] = str(exc)

        rows.append(record)

    per_row = pd.DataFrame(rows)
    summary = _aggregate(per_row)
    return summary, per_row


def _aggregate(per_row: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": int(len(per_row))}
    if per_row.empty:
        return summary

    summary["rule_score"] = {
        "mean": float(per_row["rule_score"].mean()),
        "std": float(per_row["rule_score"].std(ddof=0)),
        "min": float(per_row["rule_score"].min()),
        "max": float(per_row["rule_score"].max()),
    }
    summary["rule_label_counts"] = {
        str(k): int(v) for k, v in per_row["rule_label"].value_counts().items()
    }
    if "category" in per_row.columns and per_row["category"].notna().any():
        category_summary: dict[str, Any] = {}
        for category, group in per_row.dropna(subset=["category"]).groupby("category"):
            category_summary[str(category)] = {
                "n": int(len(group)),
                "rule_score_mean": float(group["rule_score"].mean()),
                "rule_label_counts": {
                    str(k): int(v)
                    for k, v in group["rule_label"].value_counts().items()
                },
            }
        summary["category_quality"] = category_summary

    if "learned_score" in per_row.columns and per_row["learned_score"].notna().any():
        learned = per_row["learned_score"].dropna()
        summary["learned_score"] = {
            "mean": float(learned.mean()),
            "std": float(learned.std(ddof=0)),
        }
        # Spearman correlation between rule and learned scores.
        valid = per_row[["rule_score", "learned_score"]].dropna()
        if len(valid) >= 3:
            summary["rule_vs_learned_spearman"] = float(
                _spearman(
                    valid["rule_score"].to_numpy(),
                    valid["learned_score"].to_numpy(),
                )
            )

    if "retrieved_mean_similarity" in per_row.columns:
        summary["retrieved_mean_similarity"] = float(
            per_row["retrieved_mean_similarity"].mean()
        )
        valid_sal = per_row["retrieved_median_salary"].dropna()
        if len(valid_sal):
            summary["retrieved_median_salary"] = {
                "mean": float(valid_sal.mean()),
                "median": float(valid_sal.median()),
            }

    if "pred_q50" in per_row.columns and per_row["pred_q50"].notna().any():
        summary["pred_q50"] = {
            "mean": float(per_row["pred_q50"].mean()),
            "median": float(per_row["pred_q50"].median()),
        }
    if "bls_p50" in per_row.columns and per_row["bls_p50"].notna().any():
        valid = per_row["bls_p50"].dropna()
        summary["bls_wage_band"] = {
            "matched": int(len(valid)),
            "coverage": float(len(valid) / len(per_row)),
            "p50_mean": float(valid.mean()),
            "p50_median": float(valid.median()),
        }
    if "self_consistency_abs_err" in per_row.columns:
        valid = per_row["self_consistency_abs_err"].dropna()
        if len(valid):
            summary["self_consistency_mae"] = float(valid.mean())
        in_band = per_row.get("self_consistency_in_band")
        if in_band is not None:
            valid = in_band.dropna()
            if len(valid):
                summary["self_consistency_in_band_rate"] = float(valid.mean())

    return summary


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ddof=0."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


# ---------------------------------------------------------------------------
# Loaders (each piece is optional)
# ---------------------------------------------------------------------------


def _load_embeddings(
    df: pd.DataFrame,
    *,
    smoke: bool,
    embedding_dim: int,
    seed: int,
    encoder_name: str,
) -> tuple[np.ndarray, int]:
    if smoke:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((len(df), embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return vecs / norms, embedding_dim

    try:
        from ml.embeddings import Encoder
    except ImportError as exc:
        raise RuntimeError(
            "ml.embeddings.Encoder is required for non-smoke validation. "
            "Land Task 2.1 first or pass --smoke."
        ) from exc
    encoder = Encoder(model_name=encoder_name)
    embeddings = np.asarray(
        encoder.encode(df["resume_text"].astype(str).tolist()), dtype=np.float32
    )
    return embeddings, int(embeddings.shape[1])


def _load_retriever(index_path: Path, meta_path: Path, encoder_name: str):
    if not (index_path.exists() and meta_path.exists()):
        return None
    try:
        import faiss
        from ml.embeddings import Encoder
        from ml.retrieval import Retriever
    except ImportError:
        return None

    try:
        index = faiss.read_index(str(index_path))
    except RuntimeError:
        index = faiss.deserialize_index(
            np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        )
    metadata = pd.read_parquet(meta_path)
    return Retriever(Encoder(model_name=encoder_name), index, metadata)


def _load_salary_predictor(model_path: Path, scaler_path: Path, embedding_dim: int):
    if not model_path.exists():
        return None

    from ml.salary_model import SalaryScaler, load_model, predict_salary

    model = load_model(str(model_path), embedding_dim=embedding_dim, n_extra_features=0)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, encoding="utf-8") as f:
            scaler = SalaryScaler.from_state_dict(json.load(f))

    def predictor(embedding: np.ndarray) -> dict[str, float]:
        return predict_salary(model, embedding, scaler=scaler)

    return predictor


def _load_quality_predictor(model_path: Path, scaler_path: Path, embedding_dim: int):
    if not model_path.exists():
        return None

    from ml.quality import QualityScaler, load_model, predict_quality

    model = load_model(str(model_path), embedding_dim=embedding_dim)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, encoding="utf-8") as f:
            scaler = QualityScaler.from_state_dict(json.load(f))

    def predictor(embedding: np.ndarray) -> dict[str, float | str]:
        return predict_quality(model, embedding, scaler=scaler)

    return predictor


def _load_occupation_router(skills_path: Path, encoder_name: str):
    if not skills_path.exists():
        return None
    try:
        from ml.embeddings import Encoder
        from ml.occupation_router import OccupationRouter
    except ImportError:
        return None
    return OccupationRouter.from_onet_skills(
        skills_path, Encoder(model_name=encoder_name)
    )


def _load_wage_table(wages_path: Path):
    if not wages_path.exists():
        return None
    from ml.wage_bands import WageBandTable

    return WageBandTable.from_parquet(wages_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported input extension: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ResuMatch on real resumes")
    parser.add_argument("--resumes", type=Path, default=DEFAULT_RESUMES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--onet-skills", type=Path, default=DEFAULT_ONET_SKILLS)
    parser.add_argument("--bls-wages", type=Path, default=DEFAULT_BLS_WAGES)
    parser.add_argument("--salary-model", type=Path, default=DEFAULT_SALARY_MODEL)
    parser.add_argument("--salary-scaler", type=Path, default=DEFAULT_SALARY_SCALER)
    parser.add_argument("--quality-model", type=Path, default=DEFAULT_QUALITY_MODEL)
    parser.add_argument("--quality-scaler", type=Path, default=DEFAULT_QUALITY_SCALER)
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-row-out", type=Path, default=DEFAULT_PER_ROW)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use random embeddings + skip retrieval / salary if artifacts are absent.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading resumes from {args.resumes}")
    df = _read_input(args.resumes)
    if "resume_text" not in df.columns:
        raise ValueError("input must have a 'resume_text' column")

    embeddings, dim = _load_embeddings(
        df,
        smoke=args.smoke,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        encoder_name=args.encoder,
    )

    retriever = None
    if not args.smoke:
        retriever = _load_retriever(args.index, args.meta, args.encoder)
        if retriever is None:
            print("  retriever skipped (index / metadata / Encoder missing)")

    salary_predictor = _load_salary_predictor(
        args.salary_model, args.salary_scaler, dim
    )
    if salary_predictor is None:
        print(f"  salary predictor skipped (no checkpoint at {args.salary_model})")

    quality_predictor = _load_quality_predictor(
        args.quality_model, args.quality_scaler, dim
    )
    if quality_predictor is None:
        print(
            "  learned-quality predictor skipped (no checkpoint at "
            f"{args.quality_model})"
        )

    occupation_router = None
    if not args.smoke:
        occupation_router = _load_occupation_router(args.onet_skills, args.encoder)
        if occupation_router is None:
            print(
                f"  occupation router skipped (no O*NET skills at {args.onet_skills})"
            )

    wage_table = _load_wage_table(args.bls_wages)
    if wage_table is None:
        print(f"  BLS wage bands skipped (no wage table at {args.bls_wages})")

    summary, per_row = validate(
        df,
        embeddings,
        retriever=retriever,
        salary_predictor=salary_predictor,
        quality_predictor=quality_predictor,
        occupation_router=occupation_router,
        wage_table=wage_table,
        k=args.k,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.per_row_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    per_row.to_csv(args.per_row_out, index=False)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote per-row results to {args.per_row_out}")


if __name__ == "__main__":
    main()
