from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.wage_bands import WageBandTable  # noqa: E402


def _wages() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "soc_code": ["15-1252.00", "15-1253.00", "29-1141.00"],
            "occupation_title": [
                "Software Developers",
                "Software Quality Assurance Analysts and Testers",
                "Registered Nurses",
            ],
            "p10": [70_000.0, 65_000.0, 62_000.0],
            "p25": [90_000.0, 82_000.0, 76_000.0],
            "p50": [120_000.0, 105_000.0, 92_000.0],
            "p75": [155_000.0, 135_000.0, 112_000.0],
            "p90": [190_000.0, 165_000.0, 132_000.0],
            "mean": [130_000.0, 112_000.0, 96_000.0],
        }
    )


def test_wage_band_table_exact_and_prefix_lookup() -> None:
    table = WageBandTable.from_dataframe(_wages())

    exact = table.lookup("15-1252.00")
    assert exact is not None
    assert exact.p50 == 120_000.0

    family = table.lookup("15-1252.01")
    assert family is not None
    assert family.soc_code == "15-1252"
    assert family.p50 == 120_000.0

    major = table.lookup("15-9999.00")
    assert major is not None
    assert major.soc_code == "15-0000"
    assert major.p50 == pytest.approx((120_000.0 + 105_000.0) / 2)


def test_wage_band_table_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        WageBandTable.from_dataframe(pd.DataFrame({"soc_code": ["15-1252.00"]}))
