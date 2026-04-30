from __future__ import annotations

import argparse
from pathlib import Path

from scripts.synthetic_resumes.generator import (
    DEFAULT_JOBS,
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    generate_paired_synthetic_resumes,
    generate_synthetic_resumes,
    load_jobs,
    write_synthetic_resumes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic resume/JD eval pairs"
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=DEFAULT_JOBS,
        help="Processed jobs parquet. If missing, falls back to profile-only resumes.",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of synthetic resumes to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path ending in .parquet, .csv, or .jsonl",
    )
    parser.add_argument(
        "--n-hard-negatives",
        type=int,
        default=1,
        help="Number of ranked hard negatives to attach per resume (>= 1)",
    )
    args = parser.parse_args()

    if args.jobs.exists():
        jobs = load_jobs(args.jobs)
        df = generate_paired_synthetic_resumes(
            jobs,
            n=args.n,
            seed=args.seed,
            n_hard_negatives=args.n_hard_negatives,
        )
        mode = f"paired to {args.jobs}"
    else:
        df = generate_synthetic_resumes(args.n, seed=args.seed)
        mode = "profile-only fallback; no processed jobs file found"

    out_path = write_synthetic_resumes(df, args.out)
    print(f"Wrote {len(df)} synthetic resumes to {out_path} ({mode})")
    if len(df) > 0:
        columns = [
            "resume_id",
            "source_job_id",
            "hard_negative_job_id",
            "persona",
            "writing_style",
            "quality_label",
        ]
        print(df[[column for column in columns if column in df.columns]].head())
