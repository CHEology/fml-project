"""
BLS-anchored wage-band lookup.

Loads the parquet produced by `scripts/load_bls_oews.py` and exposes a
`WageBandTable` that maps SOC codes (and aggregated SOC prefixes) to
(p10, p25, p50, p75, p90) annual wages. Used as a complementary salary
signal to Phase 4's `SalaryQuantileNet` — for non-tech occupations
where the LinkedIn-trained model is out of distribution, BLS is the
trustworthy anchor.

Public API:
    `WageBand` — dataclass with `soc_code`, `occupation_title`,
        `p10` … `p90`, optional `mean`.
    `WageBandTable.from_parquet(path)` — load.
    `WageBandTable.lookup(soc_code)` — exact match, then fall back to
        the 6-digit family ("15-1252.00" -> "15-1252") and the 2-digit
        major group ("15-0000") so partial matches still return a band.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WageBand:
    """Five-percentile annual wage band for one SOC occupation."""

    soc_code: str
    occupation_title: str
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    mean: float | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "soc_code": self.soc_code,
            "occupation_title": self.occupation_title,
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "mean": self.mean,
        }


class WageBandTable:
    """SOC -> WageBand lookup with hierarchical fallback."""

    def __init__(self, bands: dict[str, WageBand]):
        if not bands:
            raise ValueError("WageBandTable requires at least one row")
        self._bands = bands
        # Build a 6-digit prefix map (drops the ".XX" detail suffix) and a
        # 2-digit major-group map by averaging children. Both are pre-computed
        # so lookup is O(1).
        self._six = self._aggregate_by_prefix(6)
        self._two = self._aggregate_by_prefix(2)

    @classmethod
    def from_parquet(cls, path: str | Path) -> WageBandTable:
        return cls.from_dataframe(pd.read_parquet(path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> WageBandTable:
        required = {"soc_code", "p10", "p25", "p50", "p75", "p90"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"wage table is missing columns: {sorted(missing)}")

        bands: dict[str, WageBand] = {}
        for row in df.itertuples(index=False):
            soc = str(row.soc_code).strip()
            if not soc:
                continue
            bands[soc] = WageBand(
                soc_code=soc,
                occupation_title=str(getattr(row, "occupation_title", "")),
                p10=float(row.p10),
                p25=float(row.p25),
                p50=float(row.p50),
                p75=float(row.p75),
                p90=float(row.p90),
                mean=(
                    float(row.mean)
                    if "mean" in df.columns and pd.notna(row.mean)
                    else None
                ),
            )
        return cls(bands)

    def lookup(self, soc_code: str | None) -> WageBand | None:
        """Exact match, then 6-digit family, then 2-digit major group."""
        if not soc_code:
            return None
        soc = str(soc_code).strip()
        if soc in self._bands:
            return self._bands[soc]
        six = soc.split(".")[0]
        if six in self._six:
            return self._six[six]
        if "-" in six:
            major = six.split("-")[0] + "-0000"
            if major in self._two:
                return self._two[major]
        return None

    def __len__(self) -> int:
        return len(self._bands)

    def __contains__(self, soc_code: str) -> bool:
        return soc_code in self._bands

    def _aggregate_by_prefix(self, digits: int) -> dict[str, WageBand]:
        """Average percentiles across all detailed SOCs sharing a prefix."""
        groups: dict[str, list[WageBand]] = {}
        for soc, band in self._bands.items():
            key = self._prefix_key(soc, digits)
            if key:
                groups.setdefault(key, []).append(band)
        return {key: _average_band(key, members) for key, members in groups.items()}

    @staticmethod
    def _prefix_key(soc_code: str, digits: int) -> str | None:
        if "-" not in soc_code:
            return None
        major, _, rest = soc_code.partition("-")
        body = rest.split(".")[0]
        if digits == 6:
            return f"{major}-{body[:4].zfill(4)}"
        if digits == 2:
            return f"{major}-0000"
        return None


def _average_band(key: str, members: list[WageBand]) -> WageBand:
    n = len(members)
    return WageBand(
        soc_code=key,
        occupation_title=members[0].occupation_title
        if n == 1
        else f"{key} (aggregate)",
        p10=sum(b.p10 for b in members) / n,
        p25=sum(b.p25 for b in members) / n,
        p50=sum(b.p50 for b in members) / n,
        p75=sum(b.p75 for b in members) / n,
        p90=sum(b.p90 for b in members) / n,
        mean=None,
    )


__all__ = ["WageBand", "WageBandTable"]
