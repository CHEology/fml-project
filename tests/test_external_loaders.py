from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.load_bls_oews import build_wage_table  # noqa: E402
from scripts.load_onet_skills import build_skill_lexicon  # noqa: E402


def _scratch_dir(name: str) -> Path:
    path = Path(".tmp_smoke") / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_skill_lexicon_parses_onet_text_files() -> None:
    tmp_path = _scratch_dir("onet-loader")
    pd.DataFrame(
        {
            "O*NET-SOC Code": ["15-1252.00", "29-1141.00"],
            "Title": ["Software Developers", "Registered Nurses"],
        }
    ).to_csv(tmp_path / "Occupation Data.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "O*NET-SOC Code": ["15-1252.00", "29-1141.00"],
            "Element Name": ["Programming", "Monitoring"],
        }
    ).to_csv(tmp_path / "Skills.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "O*NET-SOC Code": ["15-1252.00"],
            "Example": ["Python"],
        }
    ).to_csv(tmp_path / "Technology Skills.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "O*NET-SOC Code": ["29-1141.00"],
            "Example": ["Blood pressure cuffs"],
        }
    ).to_csv(tmp_path / "Tools Used.txt", sep="\t", index=False)

    out = build_skill_lexicon(tmp_path)

    assert set(out["source"]) == {"skill", "technology", "tool"}
    assert {"skill", "soc_code", "occupation_title"} <= set(out.columns)
    assert "Software Developers" in set(out["occupation_title"])


def test_build_wage_table_parses_bls_csv_and_drops_aggregates() -> None:
    tmp_path = _scratch_dir("bls-loader")
    csv_path = tmp_path / "oews.csv"
    pd.DataFrame(
        {
            "OCC_CODE": ["15-0000", "15-1252", "29-1141"],
            "OCC_TITLE": [
                "Computer and Mathematical Occupations",
                "Software Developers",
                "Registered Nurses",
            ],
            "O_GROUP": ["major", "detailed", "detailed"],
            "A_PCT10": ["60,000", "70,000", "62,000"],
            "A_PCT25": ["80,000", "90,000", "76,000"],
            "A_MEDIAN": ["100,000", "120,000", "92,000"],
            "A_PCT75": ["130,000", "155,000", "112,000"],
            "A_PCT90": ["160,000", "190,000", "132,000"],
            "A_MEAN": ["110,000", "130,000", "96,000"],
        }
    ).to_csv(csv_path, index=False)

    out = build_wage_table(csv_path)

    assert out["soc_code"].tolist() == ["15-1252", "29-1141"]
    assert out.loc[out["soc_code"] == "15-1252", "p50"].iloc[0] == 120_000.0
