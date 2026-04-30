from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml.taxonomy import (
    MULTI_WORD_SKILLS,
    PROFILE_BY_FAMILY,
    ROLE_PROFILES,
    RoleProfile,
    quality_label_from_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_JOBS = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "synthetic_resumes.parquet"

DEFAULT_SEED = 42

LEVELS = (
    ("entry", 0, 2),
    ("junior", 1, 3),
    ("mid", 3, 6),
    ("senior", 6, 10),
)

PERSONAS = (
    "direct_match",
    "under_qualified",
    "over_qualified",
    "career_switcher",
)

PERSONA_SALARY_RANGES: dict[str, tuple[float, float]] = {
    "direct_match": (0.95, 1.05),
    "over_qualified": (1.10, 1.25),
    "under_qualified": (0.75, 0.90),
    "career_switcher": (0.80, 1.05),
}

LEVEL_ORDINAL = {"entry": 1, "junior": 2, "mid": 3, "senior": 4}

STYLES = (
    "concise_bullets",
    "first_person_bullets",
    "third_person_summary",
    "abbreviated_keywords",
)

LOCATIONS = (
    "New York, NY",
    "San Francisco, CA",
    "Seattle, WA",
    "Austin, TX",
    "Chicago, IL",
    "Remote, US",
)

DEGREES = (
    "B.S. Computer Science",
    "B.S. Statistics",
    "M.S. Data Science",
    "M.S. Computer Science",
    "B.A. Economics",
    "B.S. Marketing",
)

DEGREES_BY_FAMILY = {
    "data": (
        "B.S. Statistics",
        "M.S. Data Science",
        "M.S. Computer Science",
    ),
    "ml": (
        "B.S. Computer Science",
        "M.S. Computer Science",
        "M.S. Data Science",
    ),
    "analytics": (
        "B.S. Statistics",
        "B.A. Economics",
        "M.S. Data Science",
    ),
    "software": (
        "B.S. Computer Science",
        "M.S. Computer Science",
    ),
    "marketing": (
        "B.S. Marketing",
        "B.A. Economics",
    ),
}

STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "by",
    "as",
    "is",
    "are",
    "be",
    "this",
    "that",
    "you",
    "we",
    "our",
    "job",
    "role",
    "work",
    "team",
    "company",
    "experience",
    "skills",
}


def generate_synthetic_resumes(n: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return profile-based synthetic resumes without source JD pairing.

    Prefer `generate_paired_synthetic_resumes` for retrieval evaluation. This
    fallback remains useful for unit tests and UI demos before job data exists.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        profile = ROLE_PROFILES[i % len(ROLE_PROFILES)]
        level, min_years, max_years = LEVELS[int(rng.integers(0, len(LEVELS)))]
        persona = PERSONAS[i % len(PERSONAS)]
        style = STYLES[i % len(STYLES)]
        row = _make_resume_row(
            i=i,
            profile=profile,
            source_job=None,
            source_skills=list(profile.core_skills + profile.nice_to_have),
            level=level,
            min_years=min_years,
            max_years=max_years,
            persona=persona,
            style=style,
            rng=rng,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def generate_paired_synthetic_resumes(
    jobs: pd.DataFrame,
    n: int = 100,
    seed: int = DEFAULT_SEED,
    n_hard_negatives: int = 1,
) -> pd.DataFrame:
    """Create synthetic resumes paired to real source jobs and hard negatives.

    `n_hard_negatives` >= 1 controls how many same-family / different-level
    negatives we attach. The first negative is exposed in the legacy scalar
    columns (`hard_negative_job_id`, `hard_negative_title`, ...) for
    backwards compatibility; the full ranked list is exposed in
    `hard_negative_job_ids`.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n_hard_negatives < 1:
        raise ValueError(f"n_hard_negatives must be >= 1, got {n_hard_negatives}")
    if jobs.empty and n > 0:
        raise ValueError("jobs must contain at least one row when n > 0")

    rng = np.random.default_rng(seed)
    prepared = _prepare_jobs(jobs)
    source_indices = _stratified_source_indices(prepared, n, rng)
    hard_negative_map = _build_hard_negative_map(
        prepared, source_indices, rng, n_negatives=n_hard_negatives
    )

    rows = []
    for i, source_idx in enumerate(source_indices):
        source = prepared.iloc[int(source_idx)]
        family = str(source["_role_family"])
        profile = PROFILE_BY_FAMILY.get(family, _profile_from_job(source))
        level, min_years, max_years = _level_bounds(str(source["_level"]))
        persona = PERSONAS[i % len(PERSONAS)]
        style = STYLES[(i // len(PERSONAS)) % len(STYLES)]
        source_skills = _extract_job_skills(source, profile)

        row = _make_resume_row(
            i=i,
            profile=profile,
            source_job=source,
            source_skills=source_skills,
            level=level,
            min_years=min_years,
            max_years=max_years,
            persona=persona,
            style=style,
            rng=rng,
        )
        neg_indices = hard_negative_map.get(int(source.name), [])
        if neg_indices:
            primary = prepared.iloc[int(neg_indices[0])]
            row.update(
                {
                    "hard_negative_job_id": _optional_int(primary.get("job_id")),
                    "hard_negative_title": _clean_text(primary.get("title")),
                    "hard_negative_role_family": str(primary["_role_family"]),
                    "hard_negative_reason": _negative_reason(source, primary),
                }
            )
        row["hard_negative_job_ids"] = [
            _optional_int(prepared.iloc[int(idx)].get("job_id")) for idx in neg_indices
        ]
        rows.append(row)

    return pd.DataFrame(rows)


def load_jobs(path: str | Path) -> pd.DataFrame:
    """Load processed parquet or raw CSV job postings."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported jobs extension '{suffix}'. Use .parquet or .csv.")


def write_synthetic_resumes(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write resumes as parquet, csv, or jsonl based on file extension."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(out_path, orient="records", lines=True)
    else:
        raise ValueError(
            f"Unsupported output extension '{suffix}'. Use .parquet, .csv, or .jsonl."
        )
    return out_path


def _prepare_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    jobs = _normalize_columns(jobs)
    required = {"job_id", "title"}
    missing = [column for column in required if column not in jobs.columns]
    if missing:
        raise ValueError(f"jobs is missing required columns: {missing}")

    prepared = jobs.copy().reset_index(drop=True)
    if "text" not in prepared.columns:
        prepared["text"] = [
            _embedding_text(
                row.get("title"), row.get("description"), row.get("skills_desc")
            )
            for _, row in prepared.iterrows()
        ]
    prepared["text"] = prepared["text"].fillna("").astype(str)
    prepared = prepared[prepared["text"].str.len() > 0].reset_index(drop=True)
    if prepared.empty:
        raise ValueError(
            "jobs has no usable rows with non-empty title/description/skills text"
        )

    prepared["_role_family"] = [
        _infer_role_family(title, text)
        for title, text in zip(prepared["title"], prepared["text"], strict=True)
    ]
    prepared["_level"] = [_infer_level(row) for _, row in prepared.iterrows()]
    prepared["_tokens"] = [
        _token_set(f"{row.get('title', '')} {row.get('text', '')}")
        for _, row in prepared.iterrows()
    ]
    return prepared


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    def snake_case(name: Any) -> str:
        text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
        return re.sub(r"_+", "_", text).strip("_").lower()

    return frame.rename(
        columns={column: snake_case(column) for column in frame.columns}
    )


def _embedding_text(title: Any, description: Any, skills: Any) -> str:
    pieces = [_clean_text(title), _strip_html(description), _clean_text(skills)]
    return (
        re.sub(r"\s+", " ", " ".join(piece for piece in pieces if piece))
        .strip()
        .lower()
    )


def _strip_html(value: Any) -> str:
    text = _clean_text(value)
    return re.sub(r"<[^>]+>", " ", text)


def _infer_role_family(title: Any, text: Any = "") -> str:
    haystack = f"{title} {text}".lower()
    rules = (
        (
            "ml",
            (
                "machine learning",
                "ml engineer",
                "pytorch",
                "tensorflow",
                "model serving",
            ),
        ),
        ("data", ("data scientist", "predictive", "statistics", "scientist")),
        (
            "analytics",
            ("data analyst", "business analyst", "tableau", "excel", "dashboard"),
        ),
        (
            "software",
            ("software engineer", "backend", "api", "java", "postgresql", "developer"),
        ),
        ("marketing", ("marketing", "campaign", "brand", "seo", "social media")),
    )
    for family, needles in rules:
        if any(needle in haystack for needle in needles):
            return family
    return "general"


def _infer_level(row: pd.Series) -> str:
    text = f"{row.get('experience_level', '')} {row.get('title', '')}".lower()
    ordinal = row.get("experience_level_ordinal")
    if "intern" in text or "entry" in text:
        return "entry"
    if "senior" in text or "lead" in text or "principal" in text:
        return "senior"
    if "junior" in text or "associate" in text:
        return "junior"
    if pd.notna(ordinal):
        ordinal = float(ordinal)
        if ordinal <= 1:
            return "entry"
        if ordinal <= 2:
            return "junior"
        if ordinal >= 4:
            return "senior"
    return "mid"


def _profile_from_job(source: pd.Series) -> RoleProfile:
    family = str(source.get("_role_family", "general"))
    title = _clean_text(source.get("title")) or "Professional"
    return RoleProfile(
        title=title,
        family=family,
        aliases=(f"{title} Candidate", "Cross-functional Professional"),
        core_skills=(
            "communication",
            "execution",
            "analysis",
            "planning",
            "collaboration",
        ),
        nice_to_have=(
            "reporting",
            "stakeholder management",
            "documentation",
            "process improvement",
        ),
        project_templates=(
            "supported operational projects that improved cycle time by {metric}%",
            "coordinated cross-functional work across {metric} stakeholder groups",
            "maintained reporting workflows for {metric} recurring decisions",
        ),
    )


def _stratified_source_indices(
    jobs: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    if n == 0:
        return []

    groups = []
    for _, group in jobs.groupby(["_role_family", "_level"], sort=True):
        groups.append(group.index.to_numpy())

    if not groups:
        return []

    rng.shuffle(groups)
    selected: list[int] = []
    group_pos = {idx: 0 for idx in range(len(groups))}
    while len(selected) < n:
        for group_idx, group in enumerate(groups):
            if len(selected) >= n:
                break
            if len(group) == 0:
                continue
            pos = group_pos[group_idx]
            pick = int(rng.choice(group)) if pos >= len(group) else int(group[pos])
            group_pos[group_idx] = pos + 1
            selected.append(pick)
    return selected


def _build_hard_negative_map(
    jobs: pd.DataFrame,
    source_indices: list[int],
    rng: np.random.Generator,
    max_candidates_per_source: int = 750,
    n_negatives: int = 1,
) -> dict[int, list[int]]:
    """Pick up to `n_negatives` ranked hard negatives per unique source row.

    Returns a mapping from source row-index to a *list* of candidate row
    indices, ordered by descending score (most-similar-but-wrong first).
    """
    mapping: dict[int, list[int]] = {}
    unique_sources = sorted(set(int(idx) for idx in source_indices))
    by_family = {
        family: group.index.to_numpy()
        for family, group in jobs.groupby("_role_family", sort=False)
    }

    for idx in unique_sources:
        row = jobs.iloc[int(idx)]
        row_tokens = row["_tokens"]
        family_pool = by_family.get(str(row["_role_family"]), np.array([], dtype=int))
        family_pool = family_pool[family_pool != int(idx)]
        if len(family_pool) > max_candidates_per_source:
            candidate_indices = rng.choice(
                family_pool,
                size=max_candidates_per_source,
                replace=False,
            )
        elif len(family_pool) > 0:
            candidate_indices = family_pool
        else:
            all_indices = jobs.index.to_numpy()
            all_indices = all_indices[all_indices != int(idx)]
            sample_size = min(max_candidates_per_source, len(all_indices))
            candidate_indices = rng.choice(all_indices, size=sample_size, replace=False)

        scored: list[tuple[float, int]] = []
        for candidate_idx in candidate_indices:
            candidate = jobs.iloc[int(candidate_idx)]
            if int(candidate_idx) == int(idx):
                continue
            if _optional_int(candidate.get("job_id")) == _optional_int(
                row.get("job_id")
            ):
                continue
            score = _jaccard(row_tokens, candidate["_tokens"])
            same_family = row["_role_family"] == candidate["_role_family"]
            different_level = row["_level"] != candidate["_level"]
            different_title = (
                _clean_text(row.get("title")).lower()
                != _clean_text(candidate.get("title")).lower()
            )
            if same_family:
                score += 0.10
            if different_level:
                score += 0.05
            if not different_title:
                score -= 0.20
            scored.append((score, int(candidate_idx)))

        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            mapping[int(idx)] = [cand_idx for _, cand_idx in scored[:n_negatives]]
    return mapping


def _make_resume_row(
    *,
    i: int,
    profile: RoleProfile,
    source_job: pd.Series | None,
    source_skills: list[str],
    level: str,
    min_years: int,
    max_years: int,
    persona: str,
    style: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    source_skills = _dedupe(source_skills)
    core_pool = _dedupe(list(profile.core_skills) + source_skills[:4])
    nice_pool = _dedupe(list(profile.nice_to_have) + source_skills[4:8])

    years = _years_for_persona(persona, min_years, max_years, rng)
    core_count, nice_count, typo_count, has_metrics = _quality_parameters(
        persona, len(core_pool), len(nice_pool), rng
    )
    core_skills = _sample(tuple(core_pool), core_count, rng)
    nice_skills = _sample(tuple(nice_pool), nice_count, rng)
    skills = _dedupe(core_skills + nice_skills)
    missing_skills = [skill for skill in core_pool[:6] if skill not in core_skills]

    project_count = int(np.clip(rng.poisson(2.0) + 1, 1, 5))
    projects = _make_projects(profile, project_count, has_metrics, rng)
    education = _choose_education(profile.family, persona, rng)
    location = (
        _clean_text(source_job.get("location"))
        if source_job is not None
        else str(rng.choice(LOCATIONS))
    )
    if not location:
        location = str(rng.choice(LOCATIONS))

    score = _quality_score(
        core_count=core_count,
        total_core=len(core_pool),
        nice_count=nice_count,
        years=years,
        min_years=min_years,
        project_count=project_count,
        has_metrics=has_metrics,
        typo_count=typo_count,
        persona=persona,
    )
    label = _quality_label(score)
    resume_text = _resume_text(
        name=f"Candidate {i + 1:04d}",
        profile=profile,
        level=level,
        years=years,
        location=location,
        skills=skills,
        projects=projects,
        education=education,
        typo_count=typo_count,
        persona=persona,
        style=style,
        rng=rng,
    )

    source_job_id = (
        _optional_int(source_job.get("job_id")) if source_job is not None else None
    )
    source_title = (
        _clean_text(source_job.get("title"))
        if source_job is not None
        else profile.title
    )
    source_company = (
        _clean_text(source_job.get("company_name")) if source_job is not None else None
    )

    source_salary_annual = (
        _optional_float(source_job.get("salary_annual"))
        if source_job is not None
        else None
    )
    source_salary_min = (
        _optional_float(source_job.get("min_salary"))
        if source_job is not None
        else None
    )
    source_salary_max = (
        _optional_float(source_job.get("max_salary"))
        if source_job is not None
        else None
    )
    expected_salary_annual = _expected_salary(source_salary_annual, persona, rng)

    if source_job is not None and pd.notna(source_job.get("experience_level_ordinal")):
        experience_level_ordinal = int(float(source_job["experience_level_ordinal"]))
    else:
        experience_level_ordinal = LEVEL_ORDINAL.get(level, 3)

    return {
        "resume_id": f"syn-{i + 1:05d}",
        "source_job_id": source_job_id,
        "source_title": source_title,
        "source_company_name": source_company,
        "target_title": profile.title,
        "role_family": profile.family,
        "level": level,
        "experience_level_ordinal": experience_level_ordinal,
        "persona": persona,
        "writing_style": style,
        "years_experience": years,
        "location": location,
        "skills": "; ".join(skills),
        "missing_core_skills": "; ".join(missing_skills),
        "project_count": project_count,
        "has_metrics": int(has_metrics),
        "typo_count": typo_count,
        "education": education,
        "quality_score": score,
        "quality_label": label,
        "source_salary_annual": source_salary_annual,
        "source_salary_min": source_salary_min,
        "source_salary_max": source_salary_max,
        "expected_salary_annual": expected_salary_annual,
        "hard_negative_job_id": None,
        "hard_negative_title": None,
        "hard_negative_role_family": None,
        "hard_negative_reason": None,
        "hard_negative_job_ids": [],
        "generation_notes": "No JD sentences copied; only limited skill/title signals sampled.",
        "resume_text": resume_text,
    }


def _level_bounds(level: str) -> tuple[str, int, int]:
    for name, min_years, max_years in LEVELS:
        if name == level:
            return name, min_years, max_years
    return "mid", 3, 6


def _years_for_persona(
    persona: str,
    min_years: int,
    max_years: int,
    rng: np.random.Generator,
) -> int:
    if persona == "under_qualified":
        return max(0, min_years - int(rng.integers(0, 2)))
    if persona == "over_qualified":
        return max_years + int(rng.integers(2, 6))
    if persona == "career_switcher":
        return int(rng.integers(max(1, min_years), max_years + 3))
    return int(rng.integers(min_years, max_years + 1))


def _quality_parameters(
    persona: str,
    n_core: int,
    n_nice: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, bool]:
    if persona == "over_qualified":
        return (
            int(rng.integers(max(1, n_core - 2), n_core + 1)),
            int(rng.integers(max(1, n_nice // 2), n_nice + 1)),
            int(rng.integers(0, 2)),
            True,
        )
    if persona == "direct_match":
        return (
            int(rng.integers(max(1, n_core - 3), n_core + 1)),
            int(rng.integers(1, max(2, n_nice))),
            int(rng.integers(0, 3)),
            bool(rng.random() < 0.8),
        )
    if persona == "career_switcher":
        return (
            int(rng.integers(max(1, n_core // 2), max(2, n_core))),
            int(rng.integers(1, max(2, n_nice // 2 + 1))),
            int(rng.integers(0, 4)),
            bool(rng.random() < 0.55),
        )
    return (
        int(rng.integers(1, max(2, n_core // 2 + 1))),
        int(rng.integers(0, max(1, n_nice // 3 + 1))),
        int(rng.integers(2, 6)),
        bool(rng.random() < 0.25),
    )


def _extract_job_skills(source: pd.Series, profile: RoleProfile) -> list[str]:
    """Extract a small skill list from the source job.

    Matches against a curated lexicon of role-profile skills and known
    multi-word phrases (e.g., "machine learning") *before* falling back to
    delimiter splitting, so multi-word skills survive.
    """
    raw = source.get("skills_desc")
    if pd.isna(raw) or not str(raw).strip():
        raw = source.get("skills_desc_clean")

    description = source.get("description") or source.get("text") or ""
    haystack = f"{raw or ''} {description}".lower()

    matches: list[str] = []
    if haystack.strip():
        lexicon = _skill_lexicon()
        for skill in lexicon:
            if skill.lower() in haystack and skill not in matches:
                matches.append(skill)

    # Then add anything from the raw delimiter-split that survives a length
    # filter — picks up domain-specific tokens not in the lexicon.
    if raw and not pd.isna(raw):
        parts = re.split(r"[,;|/\n]+|\s{2,}", str(raw))
        for part in parts:
            cleaned = _clean_skill(part)
            if cleaned and len(cleaned) <= 40 and cleaned not in matches:
                matches.append(cleaned)

    return _dedupe(matches)[:10] or list(profile.core_skills + profile.nice_to_have)


def _skill_lexicon() -> list[str]:
    """Union of every role profile's skill list plus known multi-word phrases.

    Multi-word phrases are checked first via the caller's substring scan,
    so single-token entries don't accidentally consume them.
    """
    lexicon: list[str] = list(MULTI_WORD_SKILLS)
    for profile in ROLE_PROFILES:
        for skill in profile.core_skills + profile.nice_to_have:
            if skill not in lexicon:
                lexicon.append(skill)
    # Sort longest-first so "machine learning" matches before "learning".
    return sorted(lexicon, key=len, reverse=True)


def _clean_skill(value: Any) -> str:
    text = _clean_text(value)
    text = re.sub(r"\s+", " ", text).strip(" .:-")
    return text


def _sample(
    values: tuple[str, ...],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    count = min(max(count, 0), len(values))
    if count == 0:
        return []
    idxs = rng.choice(len(values), size=count, replace=False)
    return [values[int(idx)] for idx in idxs]


def _make_projects(
    profile: RoleProfile,
    count: int,
    has_metrics: bool,
    rng: np.random.Generator,
) -> list[str]:
    templates = _sample(profile.project_templates, count, rng)
    projects = []
    for template in templates:
        metric = int(rng.integers(8, 80))
        text = template.format(metric=metric)
        if not has_metrics:
            text = re.sub(r"\b\d+(K|%| ms)?\b", "several", text)
        projects.append(text)
    return projects


def _choose_education(
    family: str,
    persona: str,
    rng: np.random.Generator,
) -> str:
    if persona == "career_switcher":
        return str(rng.choice(DEGREES))
    degree_pool = DEGREES_BY_FAMILY.get(family, DEGREES)
    return str(rng.choice(degree_pool))


def _quality_score(
    *,
    core_count: int,
    total_core: int,
    nice_count: int,
    years: int,
    min_years: int,
    project_count: int,
    has_metrics: bool,
    typo_count: int,
    persona: str,
) -> int:
    skill_score = 45.0 * (core_count / max(total_core, 1))
    # Tapered nice-skill weights so each marginal skill contributes less.
    nice_weights = (3.5, 2.5, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0)
    nice_score = min(sum(nice_weights[: max(0, nice_count)]), 15.0)
    experience_score = min(max(years - min_years + 1, 0) * 5.0, 15.0)
    project_score = min(project_count * 5.0, 15.0)
    metric_score = 10.0 if has_metrics else 0.0
    persona_adjustment = {
        "under_qualified": -12.0,
        "career_switcher": -4.0,
        "direct_match": 4.0,
        "over_qualified": 2.0,
    }.get(persona, 0.0)
    penalty = typo_count * 2.5
    score = (
        skill_score
        + nice_score
        + experience_score
        + project_score
        + metric_score
        + persona_adjustment
        - penalty
    )
    return int(round(np.clip(score, 0, 100)))


def _quality_label(score: int) -> str:
    return quality_label_from_score(score)


def _resume_text(
    *,
    name: str,
    profile: RoleProfile,
    level: str,
    years: int,
    location: str,
    skills: list[str],
    projects: list[str],
    education: str,
    typo_count: int,
    persona: str,
    style: str,
    rng: np.random.Generator,
) -> str:
    alias = str(rng.choice(profile.aliases))
    skill_text = ", ".join(skills) if skills else "general problem solving"
    project_lines = "\n".join(f"- {project.capitalize()}." for project in projects)
    persona_line = {
        "under_qualified": "Looking for a stretch opportunity with room to grow.",
        "over_qualified": "Experienced contributor comfortable owning ambiguous work.",
        "career_switcher": "Brings adjacent experience and recent hands-on project work.",
        "direct_match": "Focused on practical delivery and measurable outcomes.",
    }.get(persona, "Focused on practical delivery.")

    if style == "first_person_bullets":
        text = (
            f"{name}\n{level.title()} {alias} | {location}\n\n"
            f"Profile\nI have {years} years of relevant experience. {persona_line}\n\n"
            f"Tools\n{skill_text}\n\nRecent Work\n{project_lines}\n\nEducation\n{education}\n"
        )
    elif style == "third_person_summary":
        text = (
            f"{name}\n{alias} | {location}\n\n"
            f"Summary\nCandidate with {years} years of experience. {persona_line} "
            f"Primary strengths include {skill_text}.\n\n"
            f"Selected Projects\n{project_lines}\n\nEducation\n{education}\n"
        )
    elif style == "abbreviated_keywords":
        compact_skills = " / ".join(skills[:8])
        text = (
            f"{name}\n{alias}; {years} yrs; {location}\n\n"
            f"Skills: {compact_skills}\n"
            f"Work: {'; '.join(projects)}.\n"
            f"Edu: {education}\n"
        )
    else:
        text = (
            f"{name}\n{level.title()} {alias} | {location}\n\n"
            f"Summary\n{years} years of relevant experience. {persona_line}\n\n"
            f"Skills\n{skill_text}\n\nExperience\n{project_lines}\n\nEducation\n{education}\n"
        )
    return _add_resume_noise(text, typo_count, rng)


def _add_resume_noise(
    text: str,
    typo_count: int,
    rng: np.random.Generator,
) -> str:
    replacements = {
        "experience": "experiance",
        "analysis": "analysys",
        "model": "modle",
        "database": "databse",
        "pipeline": "pipline",
        "campaign": "campagin",
        "stakeholder": "stakehldr",
    }
    noisy = text
    keys = list(replacements)
    rng.shuffle(keys)
    for key in keys[:typo_count]:
        noisy = noisy.replace(key, replacements[key], 1)
    return noisy


def _negative_reason(source: pd.Series, negative: pd.Series) -> str:
    if (
        source["_role_family"] == negative["_role_family"]
        and source["_level"] != negative["_level"]
    ):
        return "same role family but different seniority"
    if source["_role_family"] == negative["_role_family"]:
        return "same role family with different posting"
    return "lexically similar but different role family"


def _token_set(text: Any) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", str(text).lower())
    return {token for token in tokens if token not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value.strip())
    return out


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _expected_salary(
    source_salary: float | None,
    persona: str,
    rng: np.random.Generator,
) -> float | None:
    """Apply a persona-specific multiplier to the source salary.

    Returns None when the source has no usable salary so downstream
    callers can ignore the row in salary-prediction evaluation.
    """
    if source_salary is None or source_salary <= 0:
        return None
    low, high = PERSONA_SALARY_RANGES.get(persona, (0.95, 1.05))
    multiplier = float(rng.uniform(low, high))
    return float(round(source_salary * multiplier, 2))
