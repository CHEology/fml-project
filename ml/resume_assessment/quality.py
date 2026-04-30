from __future__ import annotations

import re
from typing import Any

from ml.resume_assessment.taxonomy import (
    ACTION_VERBS,
    IMPACT_REGEX,
    SAMPLE_FIELD_SKILLS,
    SECTION_ALIASES,
    TRACK_SKILLS,
    VAGUE_PHRASES,
)
from ml.resume_assessment.text_utils import find_first_bullet, find_resume_line
from ml.resume_assessment.work_history import (
    _work_history_date_example as work_history_date_example,
)
from ml.resume_assessment.work_history import (
    academic_cv_signals,
)


def _select_feedback_items(
    items: list[str],
    overall: int,
    *,
    positive: bool,
) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)

    if positive:
        limit = (
            5 if overall >= 85 else 4 if overall >= 70 else 3 if overall >= 50 else 2
        )
    else:
        limit = (
            1 if overall >= 85 else 2 if overall >= 70 else 3 if overall >= 50 else 4
        )
        if overall < 30:
            limit = 5

    if not unique:
        return []
    return unique[: max(1, min(5, limit))]


def _format_school_name(name: str) -> str:
    overrides = {
        "uc berkeley": "UC Berkeley",
        "mit": "MIT",
        "cmu": "CMU",
        "eth zurich": "ETH Zurich",
        "upenn": "UPenn",
    }
    lowered = name.lower()
    return overrides.get(lowered, name.title())


def assess_quality(
    text: str,
    profile: dict[str, Any],
    structure: dict[str, Any],
    work_history: dict[str, Any],
    projects: dict[str, Any],
    public_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute HR-style quality sub-scores, red flags, and strengths."""
    word_count = int(structure.get("word_count", 0))
    bullet_count = int(structure.get("bullet_count", 0))
    sections_found = len(structure.get("found_sections", []))
    sections_total = len(SECTION_ALIASES)
    ft_months = int(work_history.get("total_ft_months", 0))
    intern_months = int(work_history.get("total_intern_months", 0))
    weighted_months = int(work_history.get("weighted_ft_months", 0))
    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    low_rigor_role_count = int(work_history.get("low_rigor_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))
    academic = academic_cv_signals(text)
    publication_count = int(academic["publication_count"])
    award_count = int(academic["award_count"])
    prestigious_education_count = int(academic["prestigious_education_count"])
    is_research_cv = publication_count >= 3 or (
        prestigious_education_count > 0 and int(academic["degree_hits"]) > 0
    )

    # Experience sub-score: dominated by rigor-weighted FT months, with a
    # progression bonus and a prestige bonus.
    if ft_months <= 0 and intern_months <= 0:
        experience_score = 5 if word_count > 0 else 0
    else:
        experience_score = min(100, int(weighted_months * 1.0 + intern_months * 0.2))
        if work_history.get("has_progression"):
            experience_score = min(100, experience_score + 8)
        if prestigious_company_count > 0:
            experience_score = min(100, experience_score + 6)
    if is_research_cv:
        academic_experience_bonus = (
            prestigious_education_count * 8
            + min(publication_count, 12) * 2
            + min(award_count, 4) * 3
            + int(academic["degree_hits"]) * 4
        )
        experience_score = min(
            100, experience_score + min(40, academic_experience_bonus)
        )

    # Impact sub-score: quantified outcomes density.
    impact_total = int(projects.get("impact_total", 0))
    if bullet_count == 0:
        impact_score = 0
    else:
        density = impact_total / max(1, bullet_count)
        impact_score = min(100, int(density * 220))
    if publication_count >= 3:
        research_impact_score = min(
            100,
            45
            + publication_count * 4
            + min(award_count, 4) * 5
            + min(int(academic["referee_count"]), 4) * 3,
        )
        impact_score = max(impact_score, research_impact_score)

    # Specificity sub-score: action verbs vs vague phrases.
    action_total = int(projects.get("action_total", 0))
    vague_total = int(projects.get("vague_total", 0))
    raw_specificity = action_total * 12 - vague_total * 18 + 30
    if publication_count >= 3:
        raw_specificity += 25
    if prestigious_education_count > 0:
        raw_specificity += 8
    specificity_score = max(0, min(100, raw_specificity))

    # Structure sub-score: sections present + bullets + reasonable length.
    structure_score = 0
    if sections_total:
        structure_score += int(60 * sections_found / sections_total)
    if bullet_count >= 6:
        structure_score += 25
    elif bullet_count >= 3:
        structure_score += 15
    if 350 <= word_count <= 1200:
        structure_score += 15
    elif word_count >= 200:
        structure_score += 8
    if is_research_cv and publication_count >= 3:
        structure_score += 12
    structure_score = max(0, min(100, structure_score))

    overall = int(
        experience_score * 0.45
        + impact_score * 0.20
        + specificity_score * 0.20
        + structure_score * 0.15
    )

    # School-project penalty applied to overall.
    n_total = int(projects.get("n_total", 0))
    n_school = int(projects.get("n_school", 0))
    if n_total and n_school / n_total > 0.5:
        overall = max(0, overall - 12)
    if is_research_cv:
        academic_overall_bonus = min(
            12,
            prestigious_education_count * 2
            + min(publication_count, 10) // 2
            + min(award_count, 3),
        )
        overall = min(100, overall + academic_overall_bonus)

    if overall >= 75:
        band_label = "Strong"
    elif overall >= 55:
        band_label = "Mixed"
    elif overall >= 30:
        band_label = "Weak"
    else:
        band_label = "Thin"

    red_flags: list[str] = []
    strengths: list[str] = []
    date_example = work_history_date_example(work_history)
    quantified_example = find_resume_line(
        text,
        lambda line: IMPACT_REGEX.search(line) is not None,
    )
    action_example = find_resume_line(
        text,
        lambda line: any(
            re.search(r"\b" + verb + r"\b", line.lower()) for verb in ACTION_VERBS
        ),
    )
    vague_example = find_resume_line(
        text,
        lambda line: any(phrase in line.lower() for phrase in VAGUE_PHRASES),
    )
    first_bullet = find_first_bullet(text)
    skills_example = find_resume_line(
        text,
        lambda line: (
            "skill" in line.lower()
            or any(
                skill.lower() in line.lower()
                for skill in SAMPLE_FIELD_SKILLS.get(profile["track"], [])[:4]
            )
        ),
    )

    if ft_months == 0 and intern_months == 0:
        red_flags.append(
            "No parseable employment date ranges were detected, so seniority defaults to Entry."
        )
    elif ft_months < 6 and intern_months > 0:
        red_flags.append(
            "Experience reads as internships only; full-time evidence is limited"
            + (f" despite date evidence like {date_example}." if date_example else ".")
        )

    if n_total and n_school / max(1, n_total) > 0.5:
        red_flags.append(
            "Most listed work appears to be coursework or class projects, which weakens job-level evidence."
        )
    if low_rigor_role_count > 0 and rigorous_role_count == 0:
        red_flags.append(
            "Listed roles read as service or retail positions, so the resume needs stronger role-scope evidence."
        )
    if vague_total >= 3 and impact_total == 0:
        red_flags.append(
            "Frequent vague phrasing appears without measurable outcomes"
            + (f", for example {vague_example}." if vague_example else ".")
        )
    if impact_total == 0 and bullet_count >= 3 and not is_research_cv:
        red_flags.append(
            "No quantified outcomes were detected across the bullet list"
            + (f"; add scale to lines like {first_bullet}." if first_bullet else ".")
        )
    if word_count < 150:
        red_flags.append(
            f"Resume is short at {word_count} words, which limits evidence depth."
        )
    missing_sections = structure.get("missing_sections") or []
    if len(missing_sections) >= 3:
        red_flags.append(
            "Several standard sections missing: "
            + ", ".join(missing_sections[:3])
            + "."
        )
    elif missing_sections and structure_score < 70:
        red_flags.append(
            "Resume organization could be clearer; missing "
            + ", ".join(missing_sections[:2])
            + "."
        )
    if (
        impact_score < 45
        and impact_total > 0
        and bullet_count >= 6
        and not is_research_cv
    ):
        red_flags.append(
            "Only a small share of bullets are quantified; add more numbers beyond "
            + quantified_example
            + "."
        )
    if specificity_score < 45 and vague_total > 0:
        red_flags.append(
            "Some bullets still rely on broad activity language instead of concrete ownership."
        )
    if experience_score < 45 and ft_months > 0:
        red_flags.append(
            "Work-history depth is still limited relative to the claimed level."
        )

    if weighted_months >= 24 and date_example:
        strengths.append(
            "Multi-year work history is supported by date evidence"
            + (f" such as {date_example}." if date_example else ".")
        )
    if weighted_months >= 60 and int(work_history.get("ft_role_count", 0)) >= 2:
        strengths.append(
            "Experience spans multiple full-time roles with enough tenure to support level calibration."
        )
    if prestigious_company_count > 0:
        strengths.append("History includes recognized, selective employers or labs.")
    if prestigious_education_count > 0:
        schools = ", ".join(
            _format_school_name(str(school))
            for school in academic["prestigious_education"][:3]
        )
        strengths.append(
            f"Educational background includes highly selective institutions: {schools}."
        )
    if publication_count >= 5:
        strengths.append(
            f"Research output is unusually strong, with {publication_count}+ publications or preprints detected."
        )
    elif publication_count >= 3:
        strengths.append(
            f"Research output is substantial, with {publication_count}+ publications or preprints detected."
        )
    if award_count >= 2:
        strengths.append(
            f"Awards and fellowships add external validation, with {award_count} honor signals detected."
        )
    if rigorous_role_count >= 2:
        strengths.append("Multiple rigorous, high-bar roles in the work history.")
    if work_history.get("has_progression"):
        strengths.append("Title progression visible across roles.")
    if impact_total >= 3:
        strengths.append(
            "Several bullets contain quantified impact"
            + (f", including {quantified_example}." if quantified_example else ".")
        )
    elif impact_total >= 1:
        strengths.append(
            "At least one bullet shows quantified impact"
            + (f": {quantified_example}." if quantified_example else ".")
        )
    if action_total >= 5:
        strengths.append(
            "Action verbs create a strong ownership signal"
            + (f", as in {action_example}." if action_example else ".")
        )
    if specificity_score >= 70 and first_bullet:
        strengths.append(
            f"Bullet phrasing is reasonably concrete about work performed, for example {first_bullet}."
        )
    if sections_found == sections_total:
        strengths.append("All standard resume sections present.")
    elif sections_found >= 3:
        strengths.append(
            "The resume has a usable structure with sections including "
            + ", ".join(structure.get("found_sections", [])[:3])
            + "."
        )
    if 350 <= word_count <= 1200:
        strengths.append(
            f"The resume has enough depth for review at {word_count} words without becoming bloated."
        )

    if public_signals and public_signals.get("ready"):
        public_domain = public_signals.get("domain", {})
        domain_confidence = float(public_domain.get("confidence", 0.0) or 0.0)
        if domain_confidence >= 0.18:
            strengths.append(
                "Public resume-domain model independently recognizes this as a "
                f"{str(public_domain.get('label', 'professional')).title()} profile."
            )
        public_sections = public_signals.get("sections", {}).get("counts", {})
        if public_sections.get("Exp", 0) and public_sections.get("Edu", 0):
            strengths.append(
                "Public section model finds separate experience and education evidence."
            )
        public_entities = public_signals.get("entities", {}).get("counts", {})
        entity_evidence = [
            label
            for label in (
                "Companies worked at",
                "College Name",
                "Degree",
                "Designation",
                "Skills",
            )
            if int(public_entities.get(label, 0)) > 0
        ]
        if len(entity_evidence) >= 2:
            strengths.append(
                "Public entity model detects structured resume evidence: "
                + ", ".join(entity_evidence[:3])
                + "."
            )

    if not strengths:
        if skills_example:
            strengths.append(
                f"The profile has at least one concrete domain signal in {skills_example}."
            )
        elif profile.get("track"):
            strengths.append(
                f"The language is specific enough to infer a {profile['track']} focus."
            )
        else:
            strengths.append("The resume provides an initial base for evaluation.")

    if not red_flags:
        weakest = min(
            (
                ("experience", experience_score),
                ("quantified impact", impact_score),
                ("specificity", specificity_score),
                ("structure", structure_score),
            ),
            key=lambda item: item[1],
        )[0]
        if weakest == "quantified impact" and first_bullet:
            red_flags.append(
                f"The next improvement is sharper quantified impact; add outcome scale to {first_bullet}."
            )
        elif weakest == "structure":
            red_flags.append(
                "Structure is the main remaining constraint; add or clarify standard sections."
            )
        elif weakest == "specificity" and first_bullet:
            red_flags.append(
                f"Some phrasing could still be more specific about ownership and scope, starting with {first_bullet}."
            )
        else:
            red_flags.append(
                "The resume is strong overall, but it can still add more context on scope, collaborators, and business outcome."
            )

    return {
        "overall": overall,
        "band_label": band_label,
        "experience_score": experience_score,
        "impact_score": impact_score,
        "specificity_score": specificity_score,
        "structure_score": structure_score,
        "red_flags": _select_feedback_items(red_flags, overall, positive=False),
        "strengths": _select_feedback_items(strengths, overall, positive=True),
    }


def assess_capability_tier(
    text: str,
    profile: dict[str, Any],
    quality: dict[str, Any],
    work_history: dict[str, Any],
    projects: dict[str, Any],
    public_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate within-level capability strength for salary calibration.

    This is intentionally separate from seniority. Seniority answers "what
    level is supported?" Capability answers "within that level, how strong is
    the evidence?"
    """
    lowered = text.lower()
    track = str(profile.get("track", ""))
    track_skills = SAMPLE_FIELD_SKILLS.get(track, TRACK_SKILLS.get(track, []))
    skill_hits = [
        skill for skill in track_skills if skill.lower().replace("/", " ") in lowered
    ]
    skill_depth = min(100.0, len(skill_hits) / max(4, len(track_skills[:8])) * 100.0)

    academic = academic_cv_signals(text)
    publication_count = int(academic["publication_count"])
    award_count = int(academic["award_count"])
    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))
    prestige_score = min(
        100.0,
        rigorous_role_count * 22
        + prestigious_company_count * 18
        + publication_count * 4
        + award_count * 5,
    )

    public_score = 0.0
    if public_signals and public_signals.get("ready"):
        domain_conf = float(
            public_signals.get("domain", {}).get("confidence", 0.0) or 0.0
        )
        entity_counts = public_signals.get("entities", {}).get("counts", {})
        public_score = min(
            100.0,
            domain_conf * 40.0
            + int(entity_counts.get("Skills", 0)) * 8
            + int(entity_counts.get("Designation", 0)) * 8
            + int(entity_counts.get("Companies worked at", 0)) * 6
            + int(entity_counts.get("Degree", 0)) * 5,
        )

    project_score = max(
        0.0,
        min(
            100.0,
            float(projects.get("mean_score", 0)) + 45.0,
        ),
    )
    score = (
        float(quality.get("impact_score", 0)) * 0.24
        + float(quality.get("specificity_score", 0)) * 0.18
        + float(quality.get("experience_score", 0)) * 0.16
        + skill_depth * 0.16
        + prestige_score * 0.12
        + project_score * 0.08
        + public_score * 0.06
    )
    score = max(0.0, min(100.0, score))

    if score >= 78:
        tier = "Standout"
        salary_multiplier = 1.10
        summary = "top-of-level evidence"
    elif score >= 55:
        tier = "Competitive"
        salary_multiplier = 1.00
        summary = "market-level evidence"
    else:
        tier = "Developing"
        salary_multiplier = 0.93
        summary = "thin within-level evidence"

    notes: list[str] = []
    if skill_hits:
        notes.append(
            "Track-specific skills detected: " + ", ".join(skill_hits[:4]) + "."
        )
    if float(quality.get("impact_score", 0)) >= 70:
        notes.append("Impact evidence is strong for the claimed level.")
    elif float(quality.get("impact_score", 0)) < 40:
        notes.append("Impact evidence is light relative to similar-level candidates.")
    if prestige_score >= 55:
        notes.append(
            "High-rigor employers, titles, publications, or awards lift the tier."
        )
    if public_score >= 35:
        notes.append(
            "Public-data models find corroborating skills, roles, or credentials."
        )
    if not notes:
        notes.append(
            "Capability tier is driven by the resume's specificity and experience evidence."
        )

    return {
        "score": round(score, 1),
        "tier": tier,
        "summary": summary,
        "salary_multiplier": salary_multiplier,
        "salary_effect_pct": round((salary_multiplier - 1.0) * 100.0, 1),
        "skill_hits": skill_hits[:6],
        "notes": notes[:3],
    }
