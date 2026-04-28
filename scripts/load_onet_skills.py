"""
Build a skill lexicon from O*NET (US Dept of Labor occupational database).

O*NET ships per-occupation skill ratings and ~17K technology examples. The
existing rule-based quality scorer in `ml/quality.py` ships a ~50-entry
lexicon over 5 role families; this script extends it to ~thousands of
domain skills covering every US occupation, so resumes outside tech /
business (e.g., nursing, law, trades) score sensibly.

Public domain data — no auth required, direct HTTPS download.

Usage:
    # Auto-fetch the current O*NET text release into data/external/onet/
    python scripts/load_onet_skills.py --download

    # Or point at a pre-extracted O*NET text directory
    python scripts/load_onet_skills.py --input data/external/onet/db_30_2_text

Output:
    data/external/onet_skills.parquet
        columns: skill (str), source (str — "skill"/"technology"/"tool"),
                 soc_code (str), occupation_title (str)
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

# O*NET 30.2 is the current database release as of April 2026.
DEFAULT_RELEASE = "30_2"
DEFAULT_URL = (
    f"https://www.onetcenter.org/dl_files/database/db_{DEFAULT_RELEASE}_text.zip"
)
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "onet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external" / "onet_skills.parquet"

# Files we extract from the O*NET text bundle. All tab-separated.
SKILLS_FILE = "Skills.txt"
TECH_FILE = "Technology Skills.txt"
TOOLS_FILE = "Tools Used.txt"
OCCUPATIONS_FILE = "Occupation Data.txt"


def download_onet(url: str, dest_dir: Path) -> Path:
    """Download and unzip the O*NET text bundle into `dest_dir`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading O*NET text bundle: {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - public, vetted URL
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    # The zip extracts into a versioned subdirectory like db_30_2_text/.
    expected = dest_dir / f"db_{DEFAULT_RELEASE}_text"
    extracted = (
        expected
        if expected.exists()
        else next(p for p in dest_dir.iterdir() if p.is_dir())
    )
    print(f"Extracted to {extracted}")
    return extracted


def build_skill_lexicon(onet_dir: Path) -> pd.DataFrame:
    """Parse the O*NET text files and return a deduplicated skill-list DataFrame."""
    occupations = _read_occupations(onet_dir)

    pieces: list[pd.DataFrame] = []
    skills_path = onet_dir / SKILLS_FILE
    if skills_path.exists():
        skills = pd.read_csv(skills_path, sep="\t", dtype=str)
        # The Skills.txt rows are SOC × element; element_name is the skill.
        col = _pick_column(skills, ("Element Name", "Skill"))
        soc_col = _pick_column(skills, ("O*NET-SOC Code", "SOC Code"))
        if col and soc_col:
            pieces.append(
                pd.DataFrame(
                    {
                        "skill": skills[col].astype(str).str.strip(),
                        "soc_code": skills[soc_col].astype(str).str.strip(),
                        "source": "skill",
                    }
                )
            )

    tech_path = onet_dir / TECH_FILE
    if tech_path.exists():
        tech = pd.read_csv(tech_path, sep="\t", dtype=str)
        ex_col = _pick_column(tech, ("Example", "Technology Example"))
        soc_col = _pick_column(tech, ("O*NET-SOC Code", "SOC Code"))
        if ex_col and soc_col:
            pieces.append(
                pd.DataFrame(
                    {
                        "skill": tech[ex_col].astype(str).str.strip(),
                        "soc_code": tech[soc_col].astype(str).str.strip(),
                        "source": "technology",
                    }
                )
            )

    tools_path = onet_dir / TOOLS_FILE
    if tools_path.exists():
        tools = pd.read_csv(tools_path, sep="\t", dtype=str)
        ex_col = _pick_column(tools, ("Example", "Tool Example"))
        soc_col = _pick_column(tools, ("O*NET-SOC Code", "SOC Code"))
        if ex_col and soc_col:
            pieces.append(
                pd.DataFrame(
                    {
                        "skill": tools[ex_col].astype(str).str.strip(),
                        "soc_code": tools[soc_col].astype(str).str.strip(),
                        "source": "tool",
                    }
                )
            )

    if not pieces:
        raise ValueError(
            f"no usable skill / technology / tool files found in {onet_dir}. "
            "Expected Skills.txt, Technology Skills.txt, or Tools Used.txt."
        )

    df = pd.concat(pieces, ignore_index=True)
    df = df[df["skill"].str.len().between(2, 80)].copy()
    df["occupation_title"] = df["soc_code"].map(occupations).fillna("")

    # Deduplicate (skill, soc_code, source) — same skill can show up from
    # multiple sources / occupations and that's fine, but exact dupes are noise.
    df = df.drop_duplicates(subset=["skill", "soc_code", "source"]).reset_index(
        drop=True
    )
    return df


def _read_occupations(onet_dir: Path) -> dict[str, str]:
    path = onet_dir / OCCUPATIONS_FILE
    if not path.exists():
        return {}
    occ = pd.read_csv(path, sep="\t", dtype=str)
    soc_col = _pick_column(occ, ("O*NET-SOC Code", "SOC Code"))
    title_col = _pick_column(occ, ("Title", "Occupation Title"))
    if not (soc_col and title_col):
        return {}
    return dict(zip(occ[soc_col].astype(str), occ[title_col].astype(str), strict=True))


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build O*NET skill lexicon")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the O*NET text bundle from onetcenter.org.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"O*NET zip URL (default: release {DEFAULT_RELEASE}).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a pre-extracted O*NET text directory.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.download:
        onet_dir = download_onet(args.url, DEFAULT_INPUT_DIR)
    elif args.input:
        onet_dir = args.input
    else:
        raise SystemExit("Pass --download to fetch, or --input PATH for a local copy.")

    df = build_skill_lexicon(onet_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df):,} skill rows to {args.out}")
    print(f"  unique skills: {df['skill'].nunique():,}")
    print(f"  unique SOCs:   {df['soc_code'].nunique():,}")
    if "source" in df.columns:
        print(f"  by source:     {df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
