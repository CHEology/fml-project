"""
Preprocess LinkedIn job-postings CSVs into a clean Parquet file.

Usage:
    python scripts/preprocess_data.py \
        --raw-dir      data/raw \
        --jobs-out     data/processed/jobs.parquet \
        --salaries-out data/processed/salaries.npy

The script looks for the Kaggle raw CSVs recursively under `data/raw/`,
joins job postings with company information, normalizes salary fields to an
annual target, cleans text for embeddings, and saves the processed dataset
used by downstream retrieval and salary-model steps.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_JOBS_OUT = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"
DEFAULT_SALARIES_OUT = PROJECT_ROOT / "data" / "processed" / "salaries.npy"

FILE_ALIASES = {
    "job_postings": ("job_postings.csv", "postings.csv"),
    "companies": ("companies.csv",),
    "benefits": ("benefits.csv",),
    "employee_counts": ("employee_counts.csv",),
}

PAY_PERIOD_MULTIPLIERS = {
    "hourly": 2080.0,
    "daily": 260.0,
    "weekly": 52.0,
    "biweekly": 26.0,
    "monthly": 12.0,
    "quarterly": 4.0,
    "yearly": 1.0,
    "annually": 1.0,
    "annual": 1.0,
}

MIN_REASONABLE_ANNUAL_SALARY = 10_000.0
MAX_REASONABLE_ANNUAL_SALARY = 1_000_000.0

EXPERIENCE_PATTERNS = (
    ("intern", 0),
    ("entry", 1),
    ("junior", 1),
    ("associate", 2),
    ("mid-senior", 3),
    ("mid senior", 3),
    ("senior", 3),
    ("lead", 4),
    ("director", 4),
    ("principal", 4),
    ("executive", 5),
    ("vp", 5),
    ("vice president", 5),
    ("chief", 5),
)

BENEFIT_TEXT_COLUMNS = (
    "type",
    "benefit_type",
    "benefit",
    "text",
    "description",
    "inferred_type",
)


def snake_case(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"_+", "_", name).strip("_").lower()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={column: snake_case(column) for column in frame.columns}
    )


def discover_input_files(raw_dir: Path) -> dict[str, Path | None]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    by_name: dict[str, Path] = {}
    for path in raw_dir.rglob("*.csv"):
        key = path.name.lower()
        by_name.setdefault(key, path)

    discovered: dict[str, Path | None] = {}
    for key, aliases in FILE_ALIASES.items():
        discovered[key] = next(
            (
                by_name.get(alias.lower())
                for alias in aliases
                if alias.lower() in by_name
            ),
            None,
        )

    missing_required = [
        key for key in ("job_postings", "companies") if discovered[key] is None
    ]
    if missing_required:
        raise FileNotFoundError(
            "Missing required raw CSVs: "
            f"{missing_required}. Looked recursively under {raw_dir}."
        )

    return discovered


def load_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return normalize_columns(pd.read_csv(path))


def select_latest_employee_counts(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty or "company_id" not in frame.columns:
        return None

    counts = frame.copy()
    if "time_recorded" in counts.columns:
        counts["time_recorded"] = pd.to_numeric(
            counts["time_recorded"], errors="coerce"
        )
        counts = counts.sort_values(["company_id", "time_recorded"], kind="stable")
    return counts.drop_duplicates(subset=["company_id"], keep="last")


def aggregate_benefits(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty or "job_id" not in frame.columns:
        return None

    benefits = frame.copy()
    text_column = next(
        (column for column in BENEFIT_TEXT_COLUMNS if column in benefits.columns), None
    )

    def _join_unique(values: pd.Series) -> str:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values.fillna(""):
            text = normalize_whitespace(str(value)).lower()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
        return "; ".join(ordered)

    grouped = benefits.groupby("job_id", dropna=False)
    aggregated = grouped.size().rename("benefit_count").reset_index()
    if text_column is not None:
        benefit_text = (
            grouped[text_column]
            .apply(_join_unique)
            .rename("benefits_text")
            .reset_index()
        )
        aggregated = aggregated.merge(benefit_text, on="job_id", how="left")
    return aggregated


def normalize_whitespace(value: Any) -> str:
    text = html.unescape("" if value is None else str(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_html(value: Any) -> str:
    text = normalize_whitespace(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def clean_skills(value: Any) -> str:
    text = strip_html(value).lower()
    if not text:
        return ""

    parts = [normalize_whitespace(part) for part in re.split(r"[|,;/\n]+", text)]
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            deduped.append(part)
    return " ".join(deduped)


def make_embedding_text(title: Any, description: Any, skills_desc: Any) -> str:
    pieces = [
        normalize_whitespace(title).lower(),
        strip_html(description).lower(),
        clean_skills(skills_desc),
    ]
    return normalize_whitespace(" ".join(piece for piece in pieces if piece))


def annualize_salaries(frame: pd.DataFrame) -> pd.Series:
    def _numeric(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    min_salary = _numeric("min_salary")
    med_salary = _numeric("med_salary")
    max_salary = _numeric("max_salary")
    midpoint = (min_salary + max_salary) / 2.0

    base_salary = med_salary.fillna(midpoint).fillna(min_salary).fillna(max_salary)
    pay_period = (
        frame.get("pay_period", pd.Series("", index=frame.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    multiplier = pay_period.map(PAY_PERIOD_MULTIPLIERS)

    annual_salary = base_salary * multiplier
    normalized_salary = _numeric("normalized_salary")
    normalized_salary = normalized_salary.where(
        np.isfinite(normalized_salary) & (normalized_salary > 0)
    )
    annual_salary = annual_salary.fillna(normalized_salary)
    annual_salary = annual_salary.where(
        np.isfinite(annual_salary) & (annual_salary > 0)
    )
    annual_salary = annual_salary.where(
        annual_salary.between(
            MIN_REASONABLE_ANNUAL_SALARY,
            MAX_REASONABLE_ANNUAL_SALARY,
            inclusive="both",
        )
    )
    return annual_salary


def map_experience_level(value: Any) -> float:
    text = normalize_whitespace(value).lower()
    if not text:
        return np.nan

    for pattern, ordinal in EXPERIENCE_PATTERNS:
        if pattern in text:
            return float(ordinal)
    return np.nan


def truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def extract_state(location: Any) -> str | None:
    text = normalize_whitespace(location)
    if not text:
        return None

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None

    candidate = parts[-1]
    if len(candidate) == 2 and candidate.isalpha():
        return candidate.upper()
    if candidate.lower() in {"united states", "usa", "us"} and len(parts) >= 2:
        prior = parts[-2]
        if len(prior) == 2 and prior.isalpha():
            return prior.upper()
    return candidate


def prepare_companies(frame: pd.DataFrame) -> pd.DataFrame:
    companies = frame.copy()
    rename_map = {
        "name": "company_name",
        "description": "company_description",
        "state": "company_state",
        "city": "company_city",
        "country": "company_country",
    }
    companies = companies.rename(
        columns={
            key: value for key, value in rename_map.items() if key in companies.columns
        }
    )
    if "company_id" not in companies.columns:
        raise KeyError("companies.csv is missing required column 'company_id'")
    return companies.drop_duplicates(subset=["company_id"], keep="first")


def preprocess_jobs(raw_dir: Path) -> pd.DataFrame:
    paths = discover_input_files(raw_dir)

    job_postings = load_csv(paths["job_postings"])
    companies = prepare_companies(load_csv(paths["companies"]))
    benefits = aggregate_benefits(load_csv(paths["benefits"]))
    employee_counts = select_latest_employee_counts(load_csv(paths["employee_counts"]))

    if job_postings is None:
        raise FileNotFoundError("Job postings CSV could not be loaded.")
    if "company_id" not in job_postings.columns:
        raise KeyError("job_postings.csv is missing required column 'company_id'")

    jobs = job_postings.merge(
        companies,
        on="company_id",
        how="inner",
        suffixes=("", "_company"),
    )
    if benefits is not None:
        jobs = jobs.merge(benefits, on="job_id", how="left")
    if employee_counts is not None:
        jobs = jobs.merge(employee_counts, on="company_id", how="left")

    if "company_name_company" in jobs.columns:
        if "company_name" in jobs.columns:
            jobs["company_name"] = jobs["company_name"].fillna(
                jobs["company_name_company"]
            )
        else:
            jobs["company_name"] = jobs["company_name_company"]

    jobs["title"] = jobs.get("title", pd.Series("", index=jobs.index)).map(
        normalize_whitespace
    )
    jobs["company_name"] = jobs.get(
        "company_name", pd.Series("", index=jobs.index)
    ).map(normalize_whitespace)
    jobs["location"] = jobs.get("location", pd.Series("", index=jobs.index)).map(
        normalize_whitespace
    )
    jobs["experience_level"] = jobs.get(
        "formatted_experience_level",
        jobs.get("experience_level", pd.Series("", index=jobs.index)),
    ).map(normalize_whitespace)
    jobs["description_clean"] = jobs.get(
        "description",
        pd.Series("", index=jobs.index),
    ).map(strip_html)
    jobs["skills_desc_clean"] = jobs.get(
        "skills_desc",
        pd.Series("", index=jobs.index),
    ).map(clean_skills)
    jobs["text"] = [
        make_embedding_text(title, description, skills)
        for title, description, skills in zip(
            jobs["title"],
            jobs.get("description", pd.Series("", index=jobs.index)),
            jobs.get("skills_desc", pd.Series("", index=jobs.index)),
            strict=True,
        )
    ]

    jobs["salary_annual"] = annualize_salaries(jobs)
    jobs["experience_level_ordinal"] = jobs["experience_level"].map(
        map_experience_level
    )

    work_type_source = (
        jobs.get(
            "work_type",
            pd.Series("", index=jobs.index),
        )
        .astype(str)
        .str.strip()
        .str.lower()
    )
    remote_allowed = jobs.get("remote_allowed", pd.Series(False, index=jobs.index)).map(
        truthy
    )

    jobs["work_type_remote"] = (
        (work_type_source.str.contains("remote", na=False)) | remote_allowed
    ).astype(np.int8)
    jobs["work_type_hybrid"] = work_type_source.str.contains("hybrid", na=False).astype(
        np.int8
    )
    jobs["work_type_onsite"] = work_type_source.str.contains(
        "on-site|onsite|on site", na=False
    ).astype(np.int8)

    jobs["state"] = jobs["location"].map(extract_state)
    if "company_state" in jobs.columns:
        jobs["state"] = jobs["state"].fillna(
            jobs["company_state"].map(normalize_whitespace)
        )

    jobs = (
        jobs.drop_duplicates(subset=["job_id"], keep="first")
        if "job_id" in jobs.columns
        else jobs.drop_duplicates()
    )
    jobs = jobs[jobs["salary_annual"].notna()]
    jobs = jobs[jobs["text"].str.len() > 0]
    jobs = jobs[jobs["company_name"].str.len() > 0]
    jobs = jobs.reset_index(drop=True)

    preferred_columns = [
        "job_id",
        "company_id",
        "title",
        "company_name",
        "salary_annual",
        "location",
        "state",
        "experience_level",
        "experience_level_ordinal",
        "formatted_work_type",
        "work_type",
        "work_type_remote",
        "work_type_hybrid",
        "work_type_onsite",
        "currency",
        "description_clean",
        "skills_desc_clean",
        "text",
        "benefit_count",
        "benefits_text",
        "company_size",
        "employee_count",
        "follower_count",
    ]
    existing_columns = [
        column for column in preferred_columns if column in jobs.columns
    ]
    remaining_columns = [
        column for column in jobs.columns if column not in existing_columns
    ]
    return jobs[existing_columns + remaining_columns]


def write_outputs(
    frame: pd.DataFrame,
    jobs_out: Path,
    salaries_out: Path | None = None,
) -> None:
    jobs_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(jobs_out, index=False)

    if salaries_out is not None:
        salaries_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(salaries_out, frame["salary_annual"].to_numpy(dtype=np.float32))


def print_summary(frame: pd.DataFrame, jobs_out: Path) -> None:
    salary = frame["salary_annual"]
    null_rates = (
        frame[["salary_annual", "text", "company_name"]].isna().mean().to_dict()
    )
    print(f"Saved processed jobs to {jobs_out}")
    print(f"Rows: {len(frame)}")
    print(
        "Salary annual: "
        f"min={salary.min():.2f}, median={salary.median():.2f}, max={salary.max():.2f}"
    )
    print(f"Null rates: {null_rates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LinkedIn job postings")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing the Kaggle raw CSV files",
    )
    parser.add_argument(
        "--jobs-out",
        type=Path,
        default=DEFAULT_JOBS_OUT,
        help="Output path for the processed jobs parquet",
    )
    parser.add_argument(
        "--salaries-out",
        type=Path,
        default=DEFAULT_SALARIES_OUT,
        help="Optional output path for salary targets (.npy)",
    )
    args = parser.parse_args()

    jobs = preprocess_jobs(args.raw_dir)
    write_outputs(jobs, args.jobs_out, args.salaries_out)
    print_summary(jobs, args.jobs_out)


if __name__ == "__main__":
    main()
