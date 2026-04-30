from __future__ import annotations

import re
from typing import Any

from ml.resume_assessment.taxonomy import (
    ACTION_VERBS,
    IMPACT_REGEX,
    SCHOOL_PROJECT_MARKERS,
    VAGUE_PHRASES,
)


def _split_project_blocks(text: str) -> list[str]:
    """Heuristically pull project/experience bullet blocks for quality scoring."""
    if not text.strip():
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith(("-", "*", "•")):
            current.append(line.lstrip("-*• ").strip())
        else:
            if current:
                blocks.append(" ".join(current))
                current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def score_projects(text: str) -> dict[str, Any]:
    blocks = _split_project_blocks(text)
    if not blocks:
        return {
            "n_total": 0,
            "n_school": 0,
            "n_vague_dominant": 0,
            "vague_total": 0,
            "action_total": 0,
            "impact_total": 0,
            "mean_score": 0,
        }

    n_school = 0
    n_vague_dominant = 0
    vague_total = 0
    action_total = 0
    impact_total = 0
    block_scores: list[int] = []

    for block in blocks:
        lower = block.lower()
        is_school = any(marker in lower for marker in SCHOOL_PROJECT_MARKERS)
        vague_count = sum(1 for phrase in VAGUE_PHRASES if phrase in lower)
        action_count = sum(
            1 for verb in ACTION_VERBS if re.search(r"\b" + verb + r"\b", lower)
        )
        impact_count = len(IMPACT_REGEX.findall(block))

        score = 0
        if is_school:
            score -= 25
        score -= vague_count * 12
        score += action_count * 10
        score += impact_count * 18
        score = max(-100, min(100, score))

        block_scores.append(score)
        if is_school:
            n_school += 1
        if vague_count >= 2 and impact_count == 0:
            n_vague_dominant += 1
        vague_total += vague_count
        action_total += action_count
        impact_total += impact_count

    mean_score = int(sum(block_scores) / len(block_scores)) if block_scores else 0
    return {
        "n_total": len(blocks),
        "n_school": n_school,
        "n_vague_dominant": n_vague_dominant,
        "vague_total": vague_total,
        "action_total": action_total,
        "impact_total": impact_total,
        "mean_score": mean_score,
    }
