from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "have",
    "into",
    "our",
    "that",
    "the",
    "this",
    "with",
    "work",
    "role",
    "roles",
    "team",
    "teams",
    "job",
    "jobs",
    "will",
    "your",
    "you",
    "experience",
    "required",
    "preferred",
    "candidate",
    "company",
}

EDUCATION_PATTERNS = (
    r"\b(?:bachelor'?s|bachelors|bs|b\.s\.) degree\b",
    r"\b(?:master'?s|masters|ms|m\.s\.) degree\b",
    r"\bmba\b",
    r"\bph\.?d\.?\b",
    r"\bcertification in [a-z][a-z0-9/ +&-]{2,48}",
)
EXPERIENCE_PATTERN = re.compile(
    r"\b\d+\+?(?:\s*-\s*\d+\+?)?\s+years?\s+of\s+"
    r"[a-z0-9/ +&-]{2,80}?experience\b",
    re.IGNORECASE,
)


def cluster_options(
    cluster_labels: dict[str, Any] | None,
    current_cluster_id: int | None,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for cluster_id, info in _iter_cluster_labels(cluster_labels):
        if current_cluster_id is not None and cluster_id == current_cluster_id:
            continue
        options.append(
            {
                "cluster_id": cluster_id,
                "label": str(info.get("label") or f"Cluster {cluster_id}"),
                "top_terms": list(info.get("top_terms") or []),
                "size": info.get("size"),
            }
        )
    return sorted(options, key=lambda option: int(option["cluster_id"]))


def salary_growth_advice(
    jobs: pd.DataFrame,
    assignments: np.ndarray | list[int] | None,
    *,
    cluster_labels: dict[str, Any] | None,
    current_cluster_id: int | None,
    resume_text: str,
    max_items: int = 5,
) -> dict[str, Any]:
    frame = _jobs_for_cluster(jobs, assignments, current_cluster_id)
    if frame.empty:
        return _unavailable("No postings are available for the current cluster.")

    salaries = pd.to_numeric(frame["salary_annual"], errors="coerce")
    valid = frame[np.isfinite(salaries) & (salaries > 0)].copy()
    if valid.empty:
        return _unavailable("Salary evidence is unavailable in the current cluster.")

    threshold = float(valid["salary_annual"].quantile(0.75))
    cohort = valid[valid["salary_annual"] >= threshold].copy()
    cohort = cohort.sort_values("salary_annual", ascending=False)
    if cohort.empty:
        return _unavailable("No high-salary cohort could be built for this cluster.")

    label_info = _cluster_label(cluster_labels, current_cluster_id)
    missing_terms = _missing_terms(
        resume_text,
        _candidate_terms(cohort, label_info.get("top_terms") or []),
        max_items=max_items,
    )
    target_titles = _common_titles(cohort, max_items=max_items)

    return {
        "available": True,
        "kind": "salary",
        "cluster_id": current_cluster_id,
        "cluster_label": str(
            label_info.get("label") or f"Cluster {current_cluster_id}"
        ),
        "salary_threshold": int(round(threshold)),
        "cohort_size": int(len(cohort)),
        "target_titles": target_titles,
        "missing_terms": missing_terms,
        "education_requirements": _requirements(cohort, EDUCATION_PATTERNS, max_items),
        "experience_requirements": _experience_requirements(cohort, max_items),
        "representative_jobs": _representative_jobs(cohort, max_items=max_items),
        "career_actions": _career_actions(target_titles, missing_terms),
    }


def cluster_transition_advice(
    jobs: pd.DataFrame,
    assignments: np.ndarray | list[int] | None,
    *,
    cluster_labels: dict[str, Any] | None,
    current_cluster_id: int | None,
    target_cluster_id: int,
    resume_text: str,
    max_items: int = 5,
) -> dict[str, Any]:
    if current_cluster_id is not None and int(target_cluster_id) == current_cluster_id:
        return _unavailable("Choose a different cluster to see lateral-move advice.")

    frame = _jobs_for_cluster(jobs, assignments, int(target_cluster_id))
    if frame.empty:
        return _unavailable("No postings are available for the selected cluster.")

    frame = frame.sort_values("salary_annual", ascending=False, na_position="last")
    label_info = _cluster_label(cluster_labels, int(target_cluster_id))
    missing_terms = _missing_terms(
        resume_text,
        _candidate_terms(frame, label_info.get("top_terms") or []),
        max_items=max_items,
    )
    target_titles = _common_titles(frame, max_items=max_items)

    return {
        "available": True,
        "kind": "transition",
        "current_cluster_id": current_cluster_id,
        "target_cluster_id": int(target_cluster_id),
        "target_cluster_label": str(
            label_info.get("label") or f"Cluster {target_cluster_id}"
        ),
        "target_titles": target_titles,
        "missing_terms": missing_terms,
        "education_requirements": _requirements(frame, EDUCATION_PATTERNS, max_items),
        "experience_requirements": _experience_requirements(frame, max_items),
        "representative_jobs": _representative_jobs(frame, max_items=max_items),
        "career_actions": _career_actions(target_titles, missing_terms),
    }


def _iter_cluster_labels(
    cluster_labels: dict[str, Any] | None,
) -> list[tuple[int, dict[str, Any]]]:
    if not cluster_labels:
        return []
    normalized = []
    for raw_id, raw_info in cluster_labels.items():
        try:
            cluster_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        info = raw_info if isinstance(raw_info, dict) else {"label": str(raw_info)}
        normalized.append((cluster_id, info))
    return normalized


def _cluster_label(
    cluster_labels: dict[str, Any] | None, cluster_id: int | None
) -> dict[str, Any]:
    if cluster_id is None:
        return {}
    for candidate_id, info in _iter_cluster_labels(cluster_labels):
        if candidate_id == int(cluster_id):
            return info
    return {}


def _jobs_for_cluster(
    jobs: pd.DataFrame,
    assignments: np.ndarray | list[int] | None,
    cluster_id: int | None,
) -> pd.DataFrame:
    if cluster_id is None or assignments is None or jobs.empty:
        return pd.DataFrame()
    assignment_array = np.asarray(assignments)
    if len(assignment_array) != len(jobs):
        return pd.DataFrame()
    frame = jobs.copy().reset_index(drop=True)
    frame["cluster_id"] = assignment_array.astype(int)
    frame = frame[frame["cluster_id"] == int(cluster_id)].copy()
    if "salary_annual" not in frame:
        frame["salary_annual"] = np.nan
    if "text" not in frame:
        frame["text"] = ""
    if "title" not in frame:
        frame["title"] = "Untitled role"
    return frame


def _candidate_terms(frame: pd.DataFrame, cluster_terms: list[Any]) -> list[str]:
    terms = [str(term).strip().lower() for term in cluster_terms if str(term).strip()]
    text = " ".join(frame.get("text", pd.Series(dtype=str)).fillna("").astype(str))
    terms.extend(_top_tokens(text))
    return terms


def _top_tokens(text: str, *, limit: int = 18) -> list[str]:
    counts: Counter[str] = Counter()
    for token in re.findall(r"[a-z][a-z0-9+#.-]{2,}", text.lower()):
        cleaned = token.strip(".-")
        if cleaned and cleaned not in STOPWORDS:
            counts[cleaned] += 1
    return [term for term, _ in counts.most_common(limit)]


def _missing_terms(resume_text: str, terms: list[str], *, max_items: int) -> list[str]:
    resume_lower = resume_text.lower()
    missing: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = re.sub(r"\s+", " ", str(term).lower()).strip()
        if len(cleaned) < 3 or cleaned in seen or cleaned in STOPWORDS:
            continue
        seen.add(cleaned)
        if cleaned not in resume_lower:
            missing.append(cleaned)
        if len(missing) >= max_items:
            break
    return missing


def _common_titles(frame: pd.DataFrame, *, max_items: int) -> list[str]:
    titles = frame.get("title", pd.Series(dtype=str)).fillna("").astype(str)
    titles = titles[titles.str.strip().astype(bool)]
    return titles.value_counts().head(max_items).index.tolist()


def _requirements(
    frame: pd.DataFrame, patterns: tuple[str, ...], max_items: int
) -> list[str]:
    text = " ".join(frame.get("text", pd.Series(dtype=str)).fillna("").astype(str))
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            requirement = _clean_requirement(match.group(0))
            if requirement and requirement not in seen:
                found.append(requirement)
                seen.add(requirement)
            if len(found) >= max_items:
                return found
    return found


def _experience_requirements(frame: pd.DataFrame, max_items: int) -> list[str]:
    text = " ".join(frame.get("text", pd.Series(dtype=str)).fillna("").astype(str))
    text = re.sub(r"(?<=[a-z])(?=\d)", ". ", text.lower())
    found: list[str] = []
    seen: set[str] = set()
    for match in EXPERIENCE_PATTERN.finditer(text):
        requirement = _clean_requirement(match.group(0))
        if requirement and requirement not in seen:
            found.append(requirement)
            seen.add(requirement)
        if len(found) >= max_items:
            break
    return found


def _clean_requirement(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.lower()).strip(" .,:;")
    return cleaned


def _representative_jobs(
    frame: pd.DataFrame, *, max_items: int
) -> list[dict[str, Any]]:
    jobs = []
    for _, row in frame.head(max_items).iterrows():
        salary = row.get("salary_annual")
        jobs.append(
            {
                "title": str(row.get("title") or "Untitled role"),
                "salary_annual": (
                    None
                    if salary is None or pd.isna(salary)
                    else int(round(float(salary)))
                ),
            }
        )
    return jobs


def _career_actions(target_titles: list[str], missing_terms: list[str]) -> list[str]:
    actions: list[str] = []
    if target_titles:
        actions.append(
            "Pursue projects or rotations that resemble "
            + ", ".join(target_titles[:2])
            + " responsibilities."
        )
    if missing_terms:
        actions.append(
            "Build truthful work evidence around "
            + ", ".join(missing_terms[:3])
            + " before adding those signals to the resume."
        )
    actions.append(
        "Document measurable scope, ownership, and outcomes once the experience is real."
    )
    return actions


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}
