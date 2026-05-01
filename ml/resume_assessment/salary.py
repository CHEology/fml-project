from __future__ import annotations

from typing import Any

import pandas as pd


def apply_quality_discount(
    band: dict[str, Any] | None, quality: dict[str, Any]
) -> dict[str, Any] | None:
    """Apply a multiplicative haircut to all salary quantiles when evidence is thin.

    Surfaces qualitative reasoning notes (no coefficients) that the renderer
    can append to the existing evidence line.
    """
    if band is None:
        return None
    multiplier = 1.0
    notes: list[str] = []
    if quality.get("experience_score", 100) < 40:
        multiplier *= 0.90
        notes.append("Adjusted downward — limited verified employment history.")
    if quality.get("impact_score", 100) < 30:
        multiplier *= 0.92
        notes.append("Adjusted downward — projects lack quantified outcomes.")
    if quality.get("specificity_score", 100) < 40:
        multiplier *= 0.95
        notes.append(
            "Adjusted downward — descriptions are vague, hard to verify scope."
        )
    if multiplier >= 0.999:
        return band

    adjusted = dict(band)
    for key in ("q10", "q25", "q50", "q75", "q90"):
        value = adjusted.get(key)
        if value is None:
            continue
        try:
            adjusted[key] = int(round(float(value) * multiplier))
        except (TypeError, ValueError):
            continue
    adjusted["adjustment_notes"] = notes[:2]
    return adjusted


def apply_capability_adjustment(
    band: dict[str, Any] | None,
    capability: dict[str, Any],
) -> dict[str, Any] | None:
    if band is None:
        return None
    multiplier = float(capability.get("salary_multiplier", 1.0))
    if abs(multiplier - 1.0) < 0.001:
        adjusted = dict(band)
    else:
        adjusted = dict(band)
        for key in ("q10", "q25", "q50", "q75", "q90"):
            value = adjusted.get(key)
            if value is None:
                continue
            try:
                adjusted[key] = int(round(float(value) * multiplier, -3))
            except (TypeError, ValueError):
                continue

    effect = float(capability.get("salary_effect_pct", 0.0))
    if abs(effect) >= 0.1:
        direction = "+" if effect > 0 else ""
        note = (
            f"Capability tier adjustment: {capability.get('tier', 'Competitive')} "
            f"({direction}{effect:.1f}% within level)."
        )
        adjusted["adjustment_notes"] = [
            *(adjusted.get("adjustment_notes") or []),
            note,
        ][:3]
    adjusted["capability_tier"] = capability
    return adjusted


def seniority_filtered_salary_matches(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    if matches.empty or "salary_eligible" not in matches.columns:
        return matches, None
    eligible_mask = matches["salary_eligible"].fillna(True).astype(bool)
    eligible = matches[eligible_mask].copy()
    excluded = matches[~eligible_mask]
    if excluded.empty:
        return eligible, None

    below = int(
        excluded["salary_eligibility_note"]
        .astype(str)
        .str.contains("below candidate level", case=False, na=False)
        .sum()
    )
    above = int(len(excluded) - below)
    parts = []
    if below:
        parts.append(f"{below} lower-seniority role{'s' if below != 1 else ''}")
    if above:
        parts.append(f"{above} over-level role{'s' if above != 1 else ''}")
    note = "Salary evidence excludes " + " and ".join(parts) + "."
    return eligible, note


def add_salary_evidence_note(
    band: dict[str, Any] | None,
    note: str | None,
) -> dict[str, Any] | None:
    if band is None or not note:
        return band
    updated = dict(band)
    evidence = dict(updated.get("evidence", {}))
    evidence["seniority_filter"] = note
    updated["evidence"] = evidence
    return updated
