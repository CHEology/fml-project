from __future__ import annotations

import re
from datetime import date
from typing import Any

from ml.resume_assessment.taxonomy import (
    _MONTH_INDEX,
    DATE_RANGE_REGEX,
    LOW_RIGOR_TITLE_TOKENS,
    NON_WORK_DATE_TOKENS,
    NON_WORK_SECTION_LABELS,
    PRESTIGIOUS_COMPANY_TOKENS,
    PRESTIGIOUS_EDUCATION_TOKENS,
    RIGOROUS_TITLE_TOKENS,
    WORK_SECTION_LABELS,
)
from ml.resume_assessment.text_utils import quote_resume_line, resume_lines


def _section_label_from_line(line: str) -> str | None:
    label = re.sub(r"[^a-z/& ]+", "", line.lower()).strip(" :")
    label = re.sub(r"\s+", " ", label)
    if not label or len(label.split()) > 4:
        return None
    if label in WORK_SECTION_LABELS or label in NON_WORK_SECTION_LABELS:
        return label
    return None


def _date_context_looks_like_non_work(
    line: str, context: str, section: str | None
) -> bool:
    lowered_line = line.lower()
    lowered_context = context.lower()
    if section in NON_WORK_SECTION_LABELS:
        return True
    if section in WORK_SECTION_LABELS:
        return False
    if any(token in lowered_line for token in NON_WORK_DATE_TOKENS):
        work_tokens = (
            RIGOROUS_TITLE_TOKENS
            + LOW_RIGOR_TITLE_TOKENS
            + (
                "manager",
                "engineer",
                "analyst",
                "designer",
                "nurse",
                "teacher",
                "attorney",
                "coordinator",
                "specialist",
                "associate",
                "consultant",
                "intern",
            )
        )
        return not any(token in lowered_context for token in work_tokens)
    return False


def academic_cv_signals(text: str) -> dict[str, Any]:
    lowered = text.lower()
    school_hits = [
        school
        for school in PRESTIGIOUS_EDUCATION_TOKENS
        if re.search(r"(?<![a-z0-9])" + re.escape(school) + r"(?![a-z0-9])", lowered)
    ]
    prestigious_education = [
        school
        for school in school_hits
        if not any(
            school != other and school in other and other in school_hits
            for other in school_hits
        )
    ]
    degree_hits = sum(
        1
        for token in (
            "ph.d",
            "phd",
            "doctor of philosophy",
            "m.sc",
            "m.s.",
            "b.sc",
            "b.s.",
        )
        if token in lowered
    )
    publication_count = len(
        re.findall(
            r"\b(?:journal articles|preprints|physical review letters|journal of high energy physics|conference|proceedings|preprint)\b",
            lowered,
        )
    )
    numbered_publications = len(re.findall(r"(?m)^\s*\d+\s+[A-Z][^,\n]+,", text))
    publication_count = max(publication_count, numbered_publications)
    award_count = len(
        re.findall(
            r"\b(?:fellowship|award|prize|honor|highest honor|presidential award|dean's list)\b",
            lowered,
        )
    )
    referee_count = len(re.findall(r"\b(?:referee|reviewer|review service)\b", lowered))
    return {
        "prestigious_education": sorted(set(prestigious_education)),
        "prestigious_education_count": len(set(prestigious_education)),
        "degree_hits": degree_hits,
        "publication_count": publication_count,
        "award_count": award_count,
        "referee_count": referee_count,
    }


def _months_between(y1: int, m1: int, y2: int, m2: int) -> int:
    return max(0, (y2 - y1) * 12 + (m2 - m1) + 1)


def _date_match_month(
    match: re.Match[str], named_month: str, numeric_month: str, default: int
) -> int:
    month_name = match.group(named_month)
    if month_name:
        return _MONTH_INDEX.get(month_name.lower(), default)
    month_number = match.group(numeric_month)
    if month_number:
        return max(1, min(12, int(month_number)))
    return default


def extract_work_history(text: str) -> dict[str, Any]:
    """Parse date ranges from resume text and quantify employment.

    Discounts internships and academic roles vs full-time months.
    """
    if not text.strip():
        return {
            "total_ft_months": 0,
            "total_intern_months": 0,
            "weighted_ft_months": 0,
            "role_count": 0,
            "ft_role_count": 0,
            "internship_role_count": 0,
            "rigorous_role_count": 0,
            "low_rigor_role_count": 0,
            "prestigious_company_count": 0,
            "max_seniority_keyword": None,
            "has_progression": False,
            "spans": [],
        }

    today_y, today_m = date.today().year, date.today().month
    spans: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, int, int]] = set()

    lines = resume_lines(text)
    current_section: str | None = None
    for idx, line in enumerate(lines):
        section_label = _section_label_from_line(line)
        if section_label is not None:
            current_section = section_label
            continue

        window = " ".join(lines[max(0, idx - 1) : min(len(lines), idx + 2)])
        if _date_context_looks_like_non_work(line, window, current_section):
            continue

        for match in DATE_RANGE_REGEX.finditer(line):
            y1 = int(match.group("y1"))
            m1 = _date_match_month(match, "m1", "n1", 1)
            if match.group("present"):
                y2, m2 = today_y, today_m
            else:
                y2 = int(match.group("y2"))
                m2 = _date_match_month(match, "m2", "n2", 12)
            if (y2, m2) < (y1, m1):
                continue
            key = (y1, m1, y2, m2)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            months = _months_between(y1, m1, y2, m2)
            if months > 240:
                continue

            context = window.lower()

            is_intern = any(
                token in context
                for token in (
                    "intern",
                    "internship",
                    "co-op",
                    "coop",
                    "practicum",
                    "summer associate",
                )
            )
            is_academic = any(
                token in context
                for token in (
                    "teaching assistant",
                    "research assistant",
                    " ta ",
                    " ra ",
                    "graduate student",
                    "phd student",
                    "undergraduate research",
                )
            )
            seniority_kw = None
            for kw, label in (
                ("chief", "Lead / Executive"),
                ("vice president", "Lead / Executive"),
                ("director", "Lead / Executive"),
                ("principal", "Lead / Executive"),
                ("staff ", "Lead / Executive"),
                ("lead ", "Senior"),
                ("senior", "Senior"),
                ("sr.", "Senior"),
                ("associate", "Associate"),
                ("junior", "Intern / Entry"),
                ("intern", "Intern / Entry"),
            ):
                if kw in context:
                    seniority_kw = label
                    break

            rigorous_title = any(token in context for token in RIGOROUS_TITLE_TOKENS)
            low_rigor_title = any(token in context for token in LOW_RIGOR_TITLE_TOKENS)
            prestigious_company = any(
                token in context for token in PRESTIGIOUS_COMPANY_TOKENS
            )

            # Per-span rigor weight in [0.2, 1.0]: thin "default" job at 0.55,
            # bumped up by rigorous-title or prestigious-company evidence,
            # pulled down by clearly low-rigor titles. Internships are scored
            # separately by the internship multiplier later.
            weight = 0.55
            if rigorous_title:
                weight += 0.25
            if prestigious_company:
                weight += 0.25
            if low_rigor_title:
                weight = min(weight, 0.3)
            weight = max(0.2, min(1.0, weight))

            spans.append(
                {
                    "months": months,
                    "weight": weight,
                    "is_intern": is_intern,
                    "is_academic": is_academic,
                    "rigorous_title": rigorous_title,
                    "low_rigor_title": low_rigor_title,
                    "prestigious_company": prestigious_company,
                    "seniority_kw": seniority_kw,
                    "context": context,
                    "line": line,
                    "section": current_section,
                }
            )

    ft_spans = [s for s in spans if not s["is_intern"] and not s["is_academic"]]
    intern_spans = [s for s in spans if s["is_intern"] or s["is_academic"]]

    ft_months = sum(s["months"] for s in ft_spans)
    intern_months = sum(s["months"] for s in intern_spans)
    ft_role_count = len(ft_spans)
    internship_role_count = len(intern_spans)

    # Weighted FT months: each span's months scaled by its rigor weight.
    # Internships get an additional flat 0.15 multiplier on top of their weight.
    weighted_ft_months = int(round(sum(s["months"] * s["weight"] for s in ft_spans)))
    weighted_intern_contribution = int(
        round(sum(s["months"] * s["weight"] * 0.15 for s in intern_spans))
    )

    rigorous_role_count = sum(
        1 for s in ft_spans if s["rigorous_title"] or s["prestigious_company"]
    )
    low_rigor_role_count = sum(1 for s in ft_spans if s["low_rigor_title"])
    prestigious_company_count = sum(1 for s in spans if s["prestigious_company"])

    progression_order = [
        "Intern / Entry",
        "Associate",
        "Mid",
        "Senior",
        "Lead / Executive",
    ]
    seen_levels = [s["seniority_kw"] for s in spans if s["seniority_kw"]]
    indices = [
        progression_order.index(lv) for lv in seen_levels if lv in progression_order
    ]
    has_progression = len(indices) >= 2 and indices[-1] > indices[0]

    max_seniority_keyword = None
    if indices:
        max_seniority_keyword = progression_order[max(indices)]

    return {
        "total_ft_months": ft_months,
        "total_intern_months": intern_months,
        "weighted_ft_months": weighted_ft_months + weighted_intern_contribution,
        "role_count": len(spans),
        "ft_role_count": ft_role_count,
        "internship_role_count": internship_role_count,
        "rigorous_role_count": rigorous_role_count,
        "low_rigor_role_count": low_rigor_role_count,
        "prestigious_company_count": prestigious_company_count,
        "max_seniority_keyword": max_seniority_keyword,
        "has_progression": has_progression,
        "spans": spans,
    }


def _work_history_date_example(work_history: dict[str, Any]) -> str:
    spans = work_history.get("spans") or []
    for span in spans:
        if span.get("is_intern") or span.get("is_academic"):
            continue
        line = str(span.get("line") or "").strip()
        if line:
            return quote_resume_line(line)
    for span in spans:
        line = str(span.get("line") or "").strip()
        if line:
            return quote_resume_line(line)
    return ""
