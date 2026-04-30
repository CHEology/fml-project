from __future__ import annotations

import re
from typing import Any

from ml.resume_assessment.taxonomy import SECTION_ALIASES

PUBLIC_SECTION_MAP = {
    "Sum": "Summary",
    "Exp": "Experience",
    "Edu": "Education",
    "Skill": "Skills",
}


def resume_structure(text: str) -> dict[str, Any]:
    lowered = text.lower()
    found_sections = [
        label
        for label, aliases in SECTION_ALIASES.items()
        if any(alias in lowered for alias in aliases)
    ]
    missing_sections = [
        label for label in SECTION_ALIASES if label not in found_sections
    ]
    bullet_count = sum(
        1 for line in text.splitlines() if line.strip().startswith(("-", "*"))
    )
    link_count = len(re.findall(r"(https?://|linkedin\.com/|github\.com/)", lowered))
    return {
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "bullet_count": bullet_count,
        "word_count": len(text.split()),
        "link_count": link_count,
    }


def enhance_structure_with_public_sections(
    structure: dict[str, Any],
    public_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    if not public_signals or not public_signals.get("ready"):
        return structure
    counts = public_signals.get("sections", {}).get("counts", {})
    found = list(structure.get("found_sections", []))
    for public_label, app_label in PUBLIC_SECTION_MAP.items():
        if counts.get(public_label, 0) > 0 and app_label not in found:
            found.append(app_label)
    missing = [label for label in SECTION_ALIASES if label not in found]
    updated = dict(structure)
    updated["found_sections"] = found
    updated["missing_sections"] = missing
    return updated
