from __future__ import annotations

from typing import Any

from ml.resume_assessment.taxonomy import (
    SECTION_ALIASES,
    SENIORITY_MULTIPLIER,
    SKILL_GROUPS,
    TRACK_KEYWORDS,
)
from ml.resume_assessment.work_history import academic_cv_signals


def _seniority_from_ft_months(months: int) -> str:
    """Strict, prevalent seniority bands keyed off rigor-weighted FT months.

    Roughly: Entry <1y, Associate 1–3y, Mid 3–7y, Senior 7–12y, Lead 12y+.
    These are intentionally tighter than the original thresholds so that
    low-rigor or short-tenure resumes can't claim Senior on tenure alone.
    """
    if months < 12:
        return "Intern / Entry"
    if months < 36:
        return "Associate"
    if months < 84:
        return "Mid"
    if months < 144:
        return "Senior"
    return "Lead / Executive"


def detect_profile(
    resume_text: str,
    work_history: dict[str, Any] | None = None,
    projects: dict[str, Any] | None = None,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect track, seniority, and confidence using HR-style evidence weighting.

    work_history / projects / structure are optional — if omitted, this acts
    like the legacy keyword-only behavior (used by inline button handlers
    that just need a track for sample generation).
    """
    lowered = resume_text.lower()
    track_scores = {
        track: sum(lowered.count(keyword) for keyword in keywords)
        for track, keywords in TRACK_KEYWORDS.items()
    }
    detected_track = max(track_scores, key=track_scores.get)

    skill_hits = {
        skill: any(keyword in lowered for keyword in keywords)
        for skill, keywords in SKILL_GROUPS.items()
    }
    present = [skill for skill, hit in skill_hits.items() if hit]
    missing = [skill for skill, hit in skill_hits.items() if not hit]

    legacy_mode = work_history is None
    if legacy_mode:
        seniority = "Mid"
        if any(
            token in lowered
            for token in (
                "chief",
                "director",
                "vice president",
                "vp",
                "staff",
                "principal",
                "lead ",
            )
        ):
            seniority = "Lead / Executive"
        elif any(token in lowered for token in ("senior", "sr.", "sr ", "mid-senior")):
            seniority = "Senior"
        elif any(
            token in lowered for token in ("entry", "junior", "new grad", "intern")
        ):
            seniority = "Intern / Entry"
        elif "associate" in lowered:
            seniority = "Associate"
        confidence = 55 + min(35, sum(track_scores.values()) * 3 + len(present) * 4)
        confidence = min(confidence, 96)
        return {
            "track": detected_track,
            "seniority": seniority,
            "seniority_multiplier": SENIORITY_MULTIPLIER[seniority],
            "skills_present": present,
            "skills_missing": missing[:3],
            "confidence": confidence,
            "seniority_reason": "",
        }

    ft_months = int(work_history.get("total_ft_months", 0))
    intern_months = int(work_history.get("total_intern_months", 0))
    # weighted_ft_months is already rigor-weighted (low-rigor jobs count less,
    # prestigious / rigorous-titled jobs count more, internships count tiny).
    weighted_months = int(work_history.get("weighted_ft_months", 0))
    floor_seniority = _seniority_from_ft_months(weighted_months)

    title_cap = work_history.get("max_seniority_keyword")
    progression_order = [
        "Intern / Entry",
        "Associate",
        "Mid",
        "Senior",
        "Lead / Executive",
    ]
    seniority_reason = ""

    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    low_rigor_role_count = int(work_history.get("low_rigor_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))
    ft_role_count = int(work_history.get("ft_role_count", 0))

    if work_history.get("role_count", 0) == 0:
        seniority = "Intern / Entry"
        seniority_reason = "Defaulted to Entry — no employment dates were detected."
    elif ft_role_count == 0 and intern_months > 0:
        seniority = "Intern / Entry"
        seniority_reason = (
            "Capped at Entry — only internship or academic roles were detected."
        )
    elif title_cap is not None and progression_order.index(
        title_cap
    ) < progression_order.index(floor_seniority):
        seniority = title_cap
        seniority_reason = (
            f"Capped at {title_cap} — titles do not support a higher level."
        )
    elif (
        low_rigor_role_count > 0
        and rigorous_role_count == 0
        and weighted_months < ft_months
    ):
        seniority = floor_seniority
        seniority_reason = (
            "Tenure weighted down — listed roles are common entry-tier positions "
            "without rigorous-title or recognized-employer evidence."
        )
    else:
        seniority = floor_seniority

    academic = academic_cv_signals(resume_text)
    academic_pub_count = int(academic["publication_count"])
    academic_degree_hits = int(academic["degree_hits"])
    academic_prestige_count = int(academic["prestigious_education_count"])
    academic_floor = None
    if (
        academic_pub_count >= 10
        and academic_prestige_count > 0
        and rigorous_role_count > 0
    ):
        academic_floor = "Senior"
    elif academic_pub_count >= 5 and academic_degree_hits > 0:
        academic_floor = "Mid"
    elif academic_prestige_count > 0 and academic_degree_hits > 0:
        academic_floor = "Associate"

    if academic_floor and progression_order.index(
        academic_floor
    ) > progression_order.index(seniority):
        seniority = academic_floor
        seniority_reason = f"Raised to {academic_floor} — publication record and elite academic background support a higher research level."

    if (
        title_cap == "Lead / Executive"
        and academic_pub_count >= 10
        and rigorous_role_count >= 2
        and prestigious_company_count > 0
    ):
        seniority = "Lead / Executive"
        seniority_reason = "Raised to Lead / Executive — senior/staff research title is backed by publications and elite employers."

    structure = structure or {}
    projects = projects or {}
    word_count = int(structure.get("word_count", 0))
    sections_found = len(structure.get("found_sections", []))
    sections_total = len(SECTION_ALIASES)
    impact_total = int(projects.get("impact_total", 0))
    action_total = int(projects.get("action_total", 0))
    n_total = int(projects.get("n_total", 0))
    n_school = int(projects.get("n_school", 0))
    vague_total = int(projects.get("vague_total", 0))

    confidence = 0
    if ft_months > 0:
        confidence += 25
        confidence += min(30, int(ft_months / 12 * 10))
    elif intern_months > 0:
        confidence += 10
    if impact_total >= 1:
        confidence += 15
    if action_total >= 3:
        confidence += 10
    if sections_total and sections_found == sections_total:
        confidence += 10
    if word_count >= 350:
        confidence += 5
    if academic_pub_count >= 5:
        confidence += 15
    if academic_prestige_count > 0:
        confidence += 8
    if n_total and n_school > n_total / 2:
        confidence -= 15
    if vague_total >= 3 and impact_total == 0:
        confidence -= 15
    if word_count < 150:
        confidence -= 10
    confidence = max(5, min(95, confidence))

    return {
        "track": detected_track,
        "seniority": seniority,
        "seniority_multiplier": SENIORITY_MULTIPLIER[seniority],
        "skills_present": present,
        "skills_missing": missing[:3],
        "confidence": confidence,
        "seniority_reason": seniority_reason,
    }
