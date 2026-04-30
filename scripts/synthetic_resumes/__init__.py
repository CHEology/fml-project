from __future__ import annotations

from ml.taxonomy import (
    MULTI_WORD_SKILLS,
    PROFILE_BY_FAMILY,
    ROLE_PROFILES,
    RoleProfile,
    quality_label_from_score,
)
from scripts.synthetic_resumes.generator import (
    DEGREES_BY_FAMILY,
    PERSONA_SALARY_RANGES,
    generate_paired_synthetic_resumes,
    generate_synthetic_resumes,
    load_jobs,
    write_synthetic_resumes,
)

__all__ = [
    "DEGREES_BY_FAMILY",
    "MULTI_WORD_SKILLS",
    "PERSONA_SALARY_RANGES",
    "PROFILE_BY_FAMILY",
    "ROLE_PROFILES",
    "RoleProfile",
    "generate_paired_synthetic_resumes",
    "generate_synthetic_resumes",
    "load_jobs",
    "quality_label_from_score",
    "write_synthetic_resumes",
]
