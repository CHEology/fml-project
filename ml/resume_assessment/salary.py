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


_BAND_ANCHORS = (
    (10.0, "q10"),
    (25.0, "q25"),
    (50.0, "q50"),
    (75.0, "q75"),
    (90.0, "q90"),
)


def _interp_band_at_percentile(band: dict[str, Any], target_pct: float) -> float | None:
    """Linearly interpolate a salary band at an arbitrary percentile.

    Outside [10, 90] we extrapolate from the nearest two anchors so that very
    strong / very weak candidates can land above q90 or below q10 — capped
    later by the new band's own q10/q90.
    """
    points: list[tuple[float, float]] = []
    for pct, key in _BAND_ANCHORS:
        value = band.get(key)
        if value is None:
            continue
        try:
            points.append((pct, float(value)))
        except (TypeError, ValueError):
            continue
    if len(points) < 2:
        return None
    target_pct = max(0.0, min(100.0, float(target_pct)))
    pairs = list(zip(points, points[1:], strict=False))
    if target_pct <= points[0][0]:
        lower, upper = points[0], points[1]
    elif target_pct >= points[-1][0]:
        lower, upper = points[-2], points[-1]
    else:
        lower, upper = next(
            (lo, hi) for lo, hi in pairs if lo[0] <= target_pct <= hi[0]
        )
    p0, v0 = lower
    p1, v1 = upper
    if p1 == p0:
        return v0
    return v0 + (v1 - v0) * (target_pct - p0) / (p1 - p0)


def apply_capability_adjustment(
    band: dict[str, Any] | None,
    capability: dict[str, Any],
    learned_quality: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Shift the role-band by the candidate's percentile within the band.

    Conditional-quantile calibration: instead of multiplying the role's
    median by ±10%, we use the capability score (0-100) to pick the
    candidate's percentile inside the retrieved role band. Strong evidence
    moves the candidate toward q75-q90; thin evidence pulls them down to
    q10-q25. The new band is narrowed (~±12 percentile points around the
    target) to reflect the tighter within-candidate uncertainty.

    When the learned quality model (trained end-to-end on resume quality
    labels) is available, we blend its 0-100 score with the rule-based
    capability score. The neural signal is more robust to resumes whose
    work-history dates the regex parser cannot extract — e.g. dense CVs
    where ft_months is computed as 0 and capability collapses to
    "Developing" regardless of the actual content.
    """
    if band is None:
        return None

    rule_score = float(capability.get("score", 50.0))
    rule_score = max(0.0, min(100.0, rule_score))
    if learned_quality is not None:
        try:
            learned = float(learned_quality.get("score", rule_score))
        except (TypeError, ValueError):
            learned = rule_score
        learned = max(0.0, min(100.0, learned))
        # Trust the neural quality model more when it disagrees strongly
        # with the rule scorer — the rule scorer is brittle to date and
        # work-history parsing failures.
        score = 0.65 * learned + 0.35 * rule_score
    else:
        score = rule_score
    score = max(0.0, min(100.0, score))
    # Map capability score -> target percentile inside the role band.
    # 0 -> p10, 50 -> p50, 100 -> p90 (linear).
    target_pct = 10.0 + (score / 100.0) * 80.0

    new_q50 = _interp_band_at_percentile(band, target_pct)
    if new_q50 is None:
        return band

    spread = (
        ("q10", target_pct - 25.0),
        ("q25", target_pct - 12.5),
        ("q50", target_pct),
        ("q75", target_pct + 12.5),
        ("q90", target_pct + 25.0),
    )
    adjusted = dict(band)
    for key, pct in spread:
        value = _interp_band_at_percentile(band, pct)
        if value is None:
            continue
        adjusted[key] = int(round(float(value), -3))

    # Enforce monotonic q10 <= q25 <= q50 <= q75 <= q90 after rounding.
    keys = ["q10", "q25", "q50", "q75", "q90"]
    last = -float("inf")
    for key in keys:
        v = adjusted.get(key)
        if v is None:
            continue
        if v < last:
            adjusted[key] = int(last)
        last = adjusted[key]

    tier = capability.get("tier", "Competitive")
    delta_pct = target_pct - 50.0
    direction = "+" if delta_pct > 0 else ""
    note = (
        f"Capability tier {tier}: positioned at p{target_pct:.0f} of role band "
        f"({direction}{delta_pct:.0f} percentile vs median)."
    )
    adjusted["adjustment_notes"] = [
        *(adjusted.get("adjustment_notes") or []),
        note,
    ][:3]
    adjusted["capability_tier"] = capability
    adjusted["capability_target_percentile"] = round(target_pct, 1)
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
