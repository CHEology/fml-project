"""
Evaluate the salary quantile model against synthetic resume/JD pairs.

Each synthetic resume row produced by `scripts/generate_synthetic_resumes.py`
now carries the source job's `salary_annual`. This script encodes the resume
text, asks `ml.salary_model.predict_salary` for a 5-quantile prediction, and
reports:

  * median MAE (q50 vs `source_salary_annual`)
  * pinball loss across all 5 quantiles
  * coverage of [q10, q90] and [q25, q75] vs nominal 80% / 50% — calibration
    targets defined in plan.md (±5pp)
  * the same metrics broken down per persona

Honest-eval caveat: until Phase 1 lands real Kaggle data and Phase 2.1
provides the project encoder, both sides are synthetic; treat numbers as
sanity-checks rather than absolute claims.

Usage:
    python scripts/evaluate_salary.py \
        --resumes data/eval/synthetic_resumes.parquet \
        --model models/salary_model.pt \
        --scaler models/salary_model.scaler.json \
        --out data/eval/salary_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.salary_features import (  # noqa: E402
    build_resume_salary_features,
    load_salary_feature_metadata,
)
from ml.salary_model import QUANTILES  # noqa: E402

DEFAULT_RESUMES = PROJECT_ROOT / "data" / "eval" / "synthetic_resumes.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "resume_salary_model.pt"
DEFAULT_SCALER = PROJECT_ROOT / "models" / "resume_salary_model.scaler.json"
DEFAULT_FEATURES = PROJECT_ROOT / "models" / "resume_salary_model.features.json"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "data" / "eval" / "salary_metrics.json"
DEFAULT_ERRORS_OUT = PROJECT_ROOT / "data" / "eval" / "salary_errors.csv"
DEFAULT_ENCODER = "all-MiniLM-L6-v2"

QUANTILE_KEYS = tuple(f"q{int(q * 100)}" for q in QUANTILES)


class SalaryPredictor(Protocol):
    def __call__(
        self, embedding: np.ndarray, extra_features: np.ndarray | None = None
    ) -> dict[str, float]:
        """Map a single embedding to a quantile dict."""


def evaluate_salary(
    eval_df: pd.DataFrame,
    embeddings: np.ndarray,
    predictor: SalaryPredictor,
    extra_features: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute calibration / accuracy metrics for `predictor` on a paired set.

    `eval_df` must contain `source_salary_annual` (target) and `persona`.
    `embeddings` is an (N, dim) array aligned to `eval_df`. `predictor` is
    any callable mapping a 1-D embedding to a 5-quantile dict (matches
    `ml.salary_model.predict_salary` after `functools.partial` over the
    model + scaler).
    """
    if "source_salary_annual" not in eval_df.columns:
        raise ValueError("eval_df is missing 'source_salary_annual'")
    if len(eval_df) != len(embeddings):
        raise ValueError(
            f"eval_df rows ({len(eval_df)}) must match embeddings ({len(embeddings)})"
        )
    if extra_features is not None and len(extra_features) != len(eval_df):
        raise ValueError(
            "extra_features rows must match eval_df rows: "
            f"{len(extra_features)} != {len(eval_df)}"
        )

    df = eval_df.reset_index(drop=True).copy()
    targets = df["source_salary_annual"].astype(float)
    valid_mask = targets.notna() & (targets > 0)
    if valid_mask.sum() == 0:
        raise ValueError("no rows with a positive source_salary_annual to evaluate")

    rows = []
    for idx, row in df.iterrows():
        target = row["source_salary_annual"]
        if pd.isna(target) or target <= 0:
            continue
        feature_row = None
        if extra_features is not None:
            feature_row = np.asarray(extra_features[idx], dtype=np.float32)
        prediction = predictor(
            np.asarray(embeddings[idx], dtype=np.float32), feature_row
        )
        sorted_quants = np.sort(np.array([prediction[k] for k in QUANTILE_KEYS]))
        rows.append(
            {
                "resume_id": str(row.get("resume_id", f"row-{idx}")),
                "persona": str(row.get("persona", "unknown")),
                "source_salary_annual": float(target),
                **{
                    k: float(v)
                    for k, v in zip(QUANTILE_KEYS, sorted_quants, strict=True)
                },
            }
        )

    per_row = pd.DataFrame(rows)
    metrics = _aggregate(per_row)
    return metrics, per_row


def _aggregate(per_row: pd.DataFrame) -> dict[str, Any]:
    if per_row.empty:
        return {"n": 0}

    targets = per_row["source_salary_annual"].to_numpy(dtype=np.float64)
    quant_matrix = per_row[list(QUANTILE_KEYS)].to_numpy(dtype=np.float64)

    overall = _metric_block(targets, quant_matrix)
    overall["n"] = int(len(per_row))

    per_persona: dict[str, dict[str, float]] = {}
    for persona, group in per_row.groupby("persona"):
        g_targets = group["source_salary_annual"].to_numpy(dtype=np.float64)
        g_quants = group[list(QUANTILE_KEYS)].to_numpy(dtype=np.float64)
        block = _metric_block(g_targets, g_quants)
        block["n"] = int(len(group))
        per_persona[str(persona)] = block

    return {**overall, "per_persona": per_persona}


def _metric_block(targets: np.ndarray, quants: np.ndarray) -> dict[str, float]:
    median_idx = QUANTILE_KEYS.index("q50")
    q10_idx = QUANTILE_KEYS.index("q10")
    q25_idx = QUANTILE_KEYS.index("q25")
    q75_idx = QUANTILE_KEYS.index("q75")
    q90_idx = QUANTILE_KEYS.index("q90")

    median_pred = quants[:, median_idx]
    median_mae = float(np.mean(np.abs(targets - median_pred)))

    pinball_components = []
    for j, tau in enumerate(QUANTILES):
        diff = targets - quants[:, j]
        pinball_components.append(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
    pinball_mean = float(np.mean(pinball_components))

    coverage_80 = float(
        np.mean((targets >= quants[:, q10_idx]) & (targets <= quants[:, q90_idx]))
    )
    coverage_50 = float(
        np.mean((targets >= quants[:, q25_idx]) & (targets <= quants[:, q75_idx]))
    )

    calibration = {
        f"calibration_{key}": float(np.mean(targets <= quants[:, j]))
        for j, key in enumerate(QUANTILE_KEYS)
    }

    return {
        "median_mae": median_mae,
        "pinball_loss": pinball_mean,
        "coverage_80": coverage_80,
        "coverage_50": coverage_50,
        **calibration,
    }


def write_outputs(
    metrics: dict[str, Any],
    per_row: pd.DataFrame,
    metrics_out: str | Path,
    errors_out: str | Path,
) -> None:
    metrics_path = Path(metrics_out)
    errors_path = Path(errors_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    per_row.to_csv(errors_path, index=False)


def _load_predictor(
    model_path: Path,
    scaler_path: Path | None,
    embedding_dim: int,
    n_extra_features: int = 0,
) -> SalaryPredictor:
    """Load Alan's model and return a closure compatible with `SalaryPredictor`."""
    import torch
    from ml.salary_model import SalaryScaler, load_model, predict_salary

    model = load_model(
        str(model_path),
        embedding_dim=embedding_dim,
        n_extra_features=n_extra_features,
    )
    scaler = None
    if scaler_path is not None and scaler_path.exists():
        with open(scaler_path, encoding="utf-8") as f:
            scaler = SalaryScaler.from_state_dict(json.load(f))

    def predictor(
        embedding: np.ndarray, extra_features: np.ndarray | None = None
    ) -> dict[str, float]:
        # `predict_salary` enforces monotonicity and inverse-scales for us.
        return predict_salary(model, embedding, extra_features, scaler=scaler)

    # Reference torch only to satisfy the typing import — the real torch use
    # happens inside `load_model`.
    _ = torch
    return predictor


def _load_embeddings(
    df: pd.DataFrame,
    *,
    encoder_name: str,
    smoke: bool,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    if smoke:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((len(df), embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return vecs / norms

    try:
        from ml.embeddings import Encoder
    except ImportError as exc:
        raise RuntimeError(
            "ml.embeddings.Encoder is required for non-smoke evaluation. "
            "Build the embedding module first or pass --smoke."
        ) from exc

    encoder = Encoder(model_name=encoder_name)
    return np.asarray(
        encoder.encode(df["resume_text"].astype(str).tolist()), dtype=np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate salary quantile model")
    parser.add_argument("--resumes", type=Path, default=DEFAULT_RESUMES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--n-extra-features", type=int, default=0)
    parser.add_argument("--features-metadata", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--errors-out", type=Path, default=DEFAULT_ERRORS_OUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use random L2-normalized embeddings in place of ml.embeddings.Encoder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.resumes)
    if "source_salary_annual" not in df.columns:
        raise ValueError(
            "synthetic resumes parquet has no 'source_salary_annual'. "
            "Regenerate with the latest scripts/generate_synthetic_resumes.py."
        )

    embeddings = _load_embeddings(
        df,
        encoder_name=args.encoder,
        smoke=args.smoke,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    extra_features = None
    n_extra_features = args.n_extra_features
    if args.features_metadata.exists():
        metadata = load_salary_feature_metadata(args.features_metadata)
        extra_features = build_resume_salary_features(df, metadata)
        n_extra_features = int(extra_features.shape[1])
    elif args.n_extra_features > 0:
        print(
            f"Warning: feature metadata not found at {args.features_metadata}; "
            "falling back to embeddings-only inference."
        )
    predictor = _load_predictor(
        args.model, args.scaler, args.embedding_dim, n_extra_features
    )

    metrics, per_row = evaluate_salary(df, embeddings, predictor, extra_features)
    write_outputs(metrics, per_row, args.metrics_out, args.errors_out)
    print(
        json.dumps({k: v for k, v in metrics.items() if k != "per_persona"}, indent=2)
    )
    print(f"Wrote per-row results to {args.errors_out}")


if __name__ == "__main__":
    main()
