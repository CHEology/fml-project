from __future__ import annotations

from typing import Any

from ml.resume_assessment.profile import detect_profile
from ml.resume_assessment.projects import score_projects
from ml.resume_assessment.quality import assess_capability_tier, assess_quality
from ml.resume_assessment.salary import (
    add_salary_evidence_note,
    apply_capability_adjustment,
    apply_quality_discount,
    seniority_filtered_salary_matches,
)
from ml.resume_assessment.structure import (
    enhance_structure_with_public_sections,
    resume_structure,
)
from ml.resume_assessment.work_history import extract_work_history


def assess_resume_text(
    text: str,
    public_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structure = resume_structure(text)
    structure = enhance_structure_with_public_sections(structure, public_signals)
    work_history = extract_work_history(text)
    projects = score_projects(text)
    profile = detect_profile(text, work_history, projects, structure)
    quality = assess_quality(
        text,
        profile,
        structure,
        work_history,
        projects,
        public_signals,
    )
    capability = assess_capability_tier(
        text,
        profile,
        quality,
        work_history,
        projects,
        public_signals,
    )
    return {
        "structure": structure,
        "work_history": work_history,
        "projects": projects,
        "profile": profile,
        "quality": quality,
        "capability": capability,
    }


__all__ = [
    "add_salary_evidence_note",
    "apply_capability_adjustment",
    "apply_quality_discount",
    "assess_capability_tier",
    "assess_quality",
    "assess_resume_text",
    "detect_profile",
    "enhance_structure_with_public_sections",
    "extract_work_history",
    "resume_structure",
    "score_projects",
    "seniority_filtered_salary_matches",
]
