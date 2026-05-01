"""Detect tier-S compensation signals the LinkedIn dataset cannot anchor.

The retrieval-based salary band is anchored on LinkedIn job postings, which
under-represent frontier AI labs (Anthropic, DeepMind, OpenAI, ...) and
elite research roles. A Senior Staff researcher at DeepMind, Tsinghua-PhD
with 14 PRL/JHEP publications, ends up matched to "high school physics
teacher" and "postdoc at SwRI" — anchoring the band at ~$120k when real
total comp is >$500k.

This module produces a numeric `elite_score` (0-100) derived from
co-occurring signals — frontier employer, senior/staff research title,
prolific peer-reviewed record, and an elite PhD. The score is consumed by
`assess_capability_tier` to lift the candidate above the rule scorer's
"Competitive" ceiling and to apply a stronger salary multiplier.

Single-signal hits are intentionally weak: a candidate who merely lists
"Stanford" or one publication should not be tagged Elite. The score is
designed so two or more signals must co-occur to clear the 70 threshold.
"""

from __future__ import annotations

import re
from typing import Any

# Frontier AI / ML labs known to pay well above LinkedIn-typical bands.
# Distinct from PRESTIGIOUS_COMPANY_TOKENS (which lumps top tech, finance,
# healthcare, law together) — the comp profile here is specifically the
# AI research scientist track at frontier labs.
FRONTIER_AI_EMPLOYERS = (
    "anthropic",
    "openai",
    "deepmind",
    "google deepmind",
    "google brain",
    "google research",
    "meta ai",
    "fair",  # Facebook AI Research
    "apple intelligence",
    "apple machine learning research",
    "microsoft research",
    "msr",
    "mistral",
    "xai",
    "x.ai",
    "cohere",
    "adept",
    "scale ai",
    "character.ai",
    "inflection",
    "perplexity",
    "stability ai",
    "huggingface",
    "hugging face",
)

SENIOR_RESEARCH_TITLES = (
    "senior staff",
    "staff research scientist",
    "principal research scientist",
    "principal scientist",
    "distinguished engineer",
    "distinguished scientist",
    "senior research scientist",
    "senior staff research scientist",
    "research scientist",
    "research engineer",
    "applied scientist",
    "member of technical staff",
    "technical staff",
    "fellow",
)

ELITE_PHD_INSTITUTIONS = (
    "stanford university",
    "stanford",
    "massachusetts institute of technology",
    "mit",
    "carnegie mellon",
    "cmu",
    "princeton",
    "harvard",
    "uc berkeley",
    "berkeley",
    "caltech",
    "california institute of technology",
    "eth zurich",
    "oxford",
    "cambridge",
    "tsinghua",
    "peking university",
)


def detect_elite_signals(
    text: str,
    work_history: dict[str, Any],
    academic: dict[str, Any],
) -> dict[str, Any]:
    """Score whether a resume sits in tier-S compensation territory.

    `elite_score` is on a 0-100 scale; capability_tier promotes to
    "Elite" once it crosses 70 and applies an even larger multiplier
    above 85.
    """
    lowered = text.lower()

    frontier_hits = [
        token for token in FRONTIER_AI_EMPLOYERS if _word_token_in(token, lowered)
    ]
    has_frontier_employer = bool(frontier_hits)

    senior_title_hits = [
        token for token in SENIOR_RESEARCH_TITLES if _word_token_in(token, lowered)
    ]
    has_senior_research_title = bool(senior_title_hits)

    publication_count = int(academic.get("publication_count", 0))
    prolific_publications = publication_count >= 5

    prestigious_education = academic.get("prestigious_education") or []
    has_phd = int(academic.get("degree_hits", 0)) > 0 and re.search(
        r"\bph\.?d\b|\bdoctor of philosophy\b", lowered
    )
    elite_phd = bool(has_phd) and any(
        school in ELITE_PHD_INSTITUTIONS for school in prestigious_education
    )

    rigorous_role_count = int(work_history.get("rigorous_role_count", 0))
    prestigious_company_count = int(work_history.get("prestigious_company_count", 0))

    score = 0.0
    if has_frontier_employer:
        score += 45.0
    if has_senior_research_title:
        score += 25.0
    if prolific_publications:
        score += 20.0 + min(15.0, (publication_count - 5) * 1.5)
    if elite_phd:
        score += 15.0
    # Multi-employer tenure at prestigious orgs amplifies the signal — a
    # single hit is already covered above; two-plus says it isn't a fluke.
    if prestigious_company_count >= 2:
        score += 8.0
    if rigorous_role_count >= 3:
        score += 5.0
    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 1),
        "has_frontier_employer": has_frontier_employer,
        "frontier_hits": frontier_hits[:5],
        "has_senior_research_title": has_senior_research_title,
        "senior_title_hits": senior_title_hits[:3],
        "prolific_publications": prolific_publications,
        "publication_count": publication_count,
        "elite_phd": elite_phd,
    }


def _word_token_in(token: str, lowered_text: str) -> bool:
    """Word-boundary substring match — avoids "openai" matching "open ai".

    For multi-word tokens we accept any non-word char between words so
    "Google DeepMind" matches both "google deepmind" and "google-deepmind".
    """
    pattern = (
        r"(?<![a-z0-9])" + re.escape(token).replace(r"\ ", r"[\s\-]+") + r"(?![a-z0-9])"
    )
    return re.search(pattern, lowered_text) is not None
