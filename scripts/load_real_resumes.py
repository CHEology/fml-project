"""
Process a user-provided real-resume corpus into a normalised parquet.

Why this script exists
----------------------
We can't ship a real-resume dataset with the repo (license + PII). The
recommended source is Kaggle's "Updated Resume Dataset" by Snehaanbhawal
(category-labeled, ~960 resumes). Once a user downloads any tabular or
directory-based corpus into `data/raw/resumes/`, this script normalises
it into `data/eval/real_resumes.parquet` with stable columns:

    resume_id (str), resume_text (str), source_path (str),
    n_chars (int), truncated (bool), category (str | None)

That output is what `scripts/validate_on_real_resumes.py` consumes.

Usage
-----
    # Tabular input (CSV / parquet / JSONL)
    python scripts/load_real_resumes.py \\
        --input data/raw/resumes/UpdatedResumeDataSet.csv \\
        --out   data/eval/real_resumes.parquet

    # Directory of PDFs / TXTs
    python scripts/load_real_resumes.py \\
        --input data/raw/resumes/ \\
        --out   data/eval/real_resumes.parquet

    # Smoke run against the committed sample fixture
    python scripts/load_real_resumes.py \\
        --input tests/fixtures/sample_real_resumes.csv \\
        --out   data/eval/real_resumes.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.resume_loader import (  # noqa: E402
    LoadedResume,
    load_resume_dir,
    load_resume_table,
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "eval" / "real_resumes.parquet"
CATEGORY_CANDIDATES = ("category", "Category", "label", "Category_Label", "job_role")


def load_real_resumes(
    input_path: str | Path,
    *,
    text_column: str = "resume_text",
    id_column: str = "resume_id",
    redact_pii: bool = True,
) -> pd.DataFrame:
    """Load a real-resume corpus and return a normalised DataFrame.

    Auto-detects directory vs. tabular input. Carries a `category`
    column through when present (the Kaggle dataset uses "Category").
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input path not found: {path}")

    if path.is_dir():
        resumes = load_resume_dir(path, redact_pii=redact_pii)
        category_lookup: dict[str, str] = {}
    else:
        resumes = load_resume_table(
            path,
            text_column=text_column,
            id_column=id_column,
            redact_pii=redact_pii,
        )
        category_lookup = _maybe_load_categories(path, id_column)

    if not resumes:
        raise ValueError(f"no usable resumes found in {path}")

    rows = [_resume_to_row(r, category_lookup) for r in resumes]
    return pd.DataFrame(rows)


def _resume_to_row(
    resume: LoadedResume,
    category_lookup: dict[str, str],
) -> dict:
    return {
        "resume_id": resume.resume_id,
        "resume_text": resume.text,
        "source_path": resume.source_path,
        "n_chars": resume.n_chars,
        "truncated": resume.truncated,
        "category": category_lookup.get(resume.resume_id),
    }


def _maybe_load_categories(path: Path, id_column: str) -> dict[str, str]:
    """Best-effort: read the same tabular file again to pick up a category column."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".jsonl", ".ndjson"}:
            df = pd.read_json(path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(path)
        else:
            return {}
    except (ValueError, OSError):
        return {}

    column = next((c for c in CATEGORY_CANDIDATES if c in df.columns), None)
    if column is None:
        return {}
    if id_column in df.columns:
        keys = df[id_column].astype(str)
    elif "id" in df.columns:
        keys = df["id"].astype(str)
    else:
        keys = pd.Series([f"{path.stem}-{i:05d}" for i in range(len(df))])
    return {
        str(k): str(v)
        for k, v in zip(keys, df[column].astype(str), strict=True)
        if pd.notna(v) and str(v).strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise a real-resume corpus")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a CSV / parquet / JSONL file or a directory of PDFs/TXTs.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output parquet path."
    )
    parser.add_argument("--text-column", type=str, default="resume_text")
    parser.add_argument("--id-column", type=str, default="resume_id")
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable PII redaction. Default is to redact emails / phones / URLs.",
    )
    args = parser.parse_args()

    df = load_real_resumes(
        args.input,
        text_column=args.text_column,
        id_column=args.id_column,
        redact_pii=not args.no_redact,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df)} resumes to {args.out}")
    if "category" in df.columns and df["category"].notna().any():
        counts = df["category"].value_counts()
        print("Top categories:")
        for cat, n in counts.head(10).items():
            print(f"  {cat}: {n}")
    print(f"  Mean length: {df['n_chars'].mean():.0f} chars")
    print(f"  Truncated:   {df['truncated'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
