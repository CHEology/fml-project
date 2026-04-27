"""
Load BLS Occupational Employment & Wage Statistics (OEWS) into a wage table.

For every SOC occupation the BLS publishes annual wages at the 10th, 25th,
50th, 75th, and 90th percentiles. These are real federal wage statistics
covering every US occupation, not crowdsourced posts — the right anchor
for salary bands on resumes outside the tech/business slice that
Phase 4's LinkedIn-trained model fits well.

Public domain data — no auth required, direct HTTPS download.

Usage:
    # Auto-fetch the current national release into data/external/bls/
    python scripts/load_bls_oews.py --download

    # Or point at a pre-downloaded national xlsx (or csv with the same columns)
    python scripts/load_bls_oews.py --input data/external/bls/national_M2024_dl.xlsx

Output:
    data/external/bls_wages.parquet
        columns: soc_code, occupation_title, p10, p25, p50, p75, p90, mean
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_YEAR = 2024
DEFAULT_URL = (
    f"https://www.bls.gov/oes/special-requests/oesm{DEFAULT_YEAR % 100:02d}nat.zip"
)
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "bls"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "bls_wages.parquet"

# Column aliases — BLS occasionally tweaks naming between releases.
SOC_CANDIDATES = ("OCC_CODE", "occ_code")
TITLE_CANDIDATES = ("OCC_TITLE", "occ_title")
P10_CANDIDATES = ("A_PCT10", "a_pct10")
P25_CANDIDATES = ("A_PCT25", "a_pct25")
P50_CANDIDATES = ("A_MEDIAN", "a_median")
P75_CANDIDATES = ("A_PCT75", "a_pct75")
P90_CANDIDATES = ("A_PCT90", "a_pct90")
MEAN_CANDIDATES = ("A_MEAN", "a_mean")
GROUP_CANDIDATES = ("O_GROUP", "o_group")


def download_oews(url: str, dest_dir: Path) -> Path:
    """Download and unzip the OEWS national release. Returns the .xlsx path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BLS OEWS national release: {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - public, vetted URL
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    xlsx_files = sorted(dest_dir.rglob("*.xlsx"))
    if not xlsx_files:
        raise RuntimeError(f"no .xlsx file found in {dest_dir} after extraction")
    print(f"Extracted to {xlsx_files[0]}")
    return xlsx_files[0]


def build_wage_table(input_path: Path) -> pd.DataFrame:
    """Read the OEWS file and return one row per detailed SOC occupation."""
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"unsupported BLS extension '{suffix}'. Use .xlsx or .csv.")

    soc_col = _pick(df, SOC_CANDIDATES)
    title_col = _pick(df, TITLE_CANDIDATES)
    p10 = _pick(df, P10_CANDIDATES)
    p25 = _pick(df, P25_CANDIDATES)
    p50 = _pick(df, P50_CANDIDATES)
    p75 = _pick(df, P75_CANDIDATES)
    p90 = _pick(df, P90_CANDIDATES)
    mean_col = _pick(df, MEAN_CANDIDATES)
    group_col = _pick(df, GROUP_CANDIDATES)

    if not all([soc_col, title_col, p10, p25, p50, p75, p90]):
        raise ValueError(
            f"BLS file is missing expected columns. Found: {list(df.columns)[:15]}..."
        )

    if group_col is not None:
        # "detailed" rows are individual occupations; aggregates we drop.
        df = df[df[group_col].astype(str).str.lower().eq("detailed")].copy()

    out = pd.DataFrame(
        {
            "soc_code": df[soc_col].astype(str).str.strip(),
            "occupation_title": df[title_col].astype(str).str.strip(),
            "p10": _to_numeric(df[p10]),
            "p25": _to_numeric(df[p25]),
            "p50": _to_numeric(df[p50]),
            "p75": _to_numeric(df[p75]),
            "p90": _to_numeric(df[p90]),
            "mean": _to_numeric(df[mean_col]) if mean_col else None,
        }
    )
    out = out.dropna(subset=["soc_code", "p50"]).drop_duplicates("soc_code")
    out = out[(out["p50"] > 0) & (out["p10"] > 0)].reset_index(drop=True)
    return out


def _pick(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    """OEWS uses '#', '*', '**' as suppression markers; coerce to NaN."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").replace({"#": "", "*": "", "**": ""}),
        errors="coerce",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BLS OEWS wage table")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch the national OEWS release from bls.gov.",
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a pre-downloaded BLS xlsx or csv file.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.download:
        input_path = download_oews(args.url, DEFAULT_INPUT_DIR)
    elif args.input:
        input_path = args.input
    else:
        raise SystemExit("Pass --download to fetch, or --input PATH for a local copy.")

    df = build_wage_table(input_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df):,} occupations to {args.out}")
    print(f"  median p50 across occupations: ${df['p50'].median():,.0f}")
    print(f"  p10 range: ${df['p10'].min():,.0f} - ${df['p10'].max():,.0f}")
    print(f"  p90 range: ${df['p90'].min():,.0f} - ${df['p90'].max():,.0f}")


if __name__ == "__main__":
    main()
