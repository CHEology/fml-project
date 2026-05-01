from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from html import escape
from typing import Any

import pandas as pd
import streamlit as st
from ml.resume_assessment import assess_resume_text
from ml.resume_assessment.taxonomy import (
    IMPACT_REGEX,
    SAMPLE_FIELD_SKILLS,
    SECTION_ALIASES,
    TRACK_SKILLS,
)
from ml.resume_assessment.text_utils import find_first_bullet, find_resume_line
from ml.resume_assessment.work_history import (
    _section_label_from_line,
    _work_history_date_example,
)
from streamlit.components.v1 import html as components_html

from app.demo.sample_data import (
    TRACK_INITIATIVES,
    TRACK_METRICS,
    TRACK_PROJECTS,
    TRACK_TITLES,
)
from app.demo.samples import compose_headline
from app.runtime.cache import load_public_assessment_resource, public_resume_signals

QUALITY_DIMENSIONS = (
    ("experience", "Experience", "experience_score"),
    ("quantified_impact", "Quantified impact", "impact_score"),
    ("specificity", "Specificity", "specificity_score"),
    ("structure", "Structure", "structure_score"),
)

REVISION_STATE_DEFAULTS = {
    "revised_resume_text": "",
    "revised_resume_notes": [],
    "revised_resume_source": "",
    "show_resume_revision_diff": True,
}


def ensure_revision_state() -> None:
    for key, default in REVISION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_revision_state() -> None:
    for key, default in REVISION_STATE_DEFAULTS.items():
        st.session_state[key] = default


def weakest_quality_aspects(
    quality: dict[str, Any],
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    scored = [
        {
            "key": key,
            "label": label,
            "score_key": score_key,
            "score": int(quality.get(score_key, 0)),
        }
        for key, label, score_key in QUALITY_DIMENSIONS
    ]
    scored.sort(key=lambda item: (item["score"], item["label"]))
    return scored[:limit]


def build_strengthening_plan(
    quality: dict[str, Any],
    *,
    resume_text: str,
    work_history: dict[str, Any],
) -> list[dict[str, Any]]:
    weakest = weakest_quality_aspects(quality, limit=2)
    first_bullet = find_first_bullet(resume_text)
    date_example = _work_history_date_example(work_history)
    quantified_example = find_resume_line(
        resume_text,
        lambda line: IMPACT_REGEX.search(line) is not None,
    )

    detail_map = {
        "experience": (
            "Make tenure and scope explicit with clean date ranges and clear ownership."
            + (
                f" The parser already sees evidence like {date_example}."
                if date_example
                else ""
            )
        ),
        "quantified_impact": (
            "Attach counts, percentages, dollars, or time saved to the most important bullets."
            + (
                f" Use a line like {quantified_example} as the model."
                if quantified_example
                else ""
            )
        ),
        "specificity": (
            "Replace vague phrasing with named systems, tools, stakeholders, and outcomes."
            + (f" Start by upgrading {first_bullet}." if first_bullet else "")
        ),
        "structure": (
            "Tighten section order and keep the strongest roles at the highest bullet density."
        ),
    }

    return [{**item, "detail": detail_map.get(item["key"], "")} for item in weakest]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").strip().split())
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(cleaned)
    return output


def _canonical_section_from_line(line: str) -> str | None:
    normalized = _section_label_from_line(line)
    if normalized is None:
        return None
    for label, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return label
    return None


def extract_resume_sections(text: str) -> dict[str, list[str]]:
    sections = {label: [] for label in ("Header", *SECTION_ALIASES)}
    current = "Header"
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        canonical = _canonical_section_from_line(line)
        if canonical is not None:
            current = canonical
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _is_bullet_line(line: str) -> bool:
    return str(line).strip().startswith(("-", "*", "•"))


def _clean_resume_bullet_seed(line: str) -> str:
    cleaned = str(line or "").lstrip("-*• ").strip()
    cleaned = re.sub(
        r"^(?:responsible for|helped with|worked on|assisted with|support(?:ed|ing)?|tasked with)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.rstrip(".")
    if cleaned:
        cleaned = cleaned[0].lower() + cleaned[1:]
    return cleaned


def build_revised_skill_list(
    profile: dict[str, Any],
    matches: pd.DataFrame | None,
    *,
    limit: int = 10,
) -> list[str]:
    track = str(profile.get("track", ""))
    seeds: list[str] = []
    seeds.extend(str(skill) for skill in profile.get("skills_present") or [])
    seeds.extend(SAMPLE_FIELD_SKILLS.get(track, TRACK_SKILLS.get(track, [])))
    if matches is not None and not matches.empty and "title" in matches.columns:
        seeds.extend(str(title) for title in matches["title"].head(3).tolist())
    normalized = _dedupe_preserve_order(seeds)
    return normalized[:limit]


def _revision_result_phrase(track: str, index: int) -> str:
    metric_name = TRACK_METRICS.get(track, "delivery quality")
    phrases = [
        f"improved {metric_name} by [{18 + index * 3}%] across [{5 + index}] workflows",
        f"reduced turnaround time by [{14 + index * 2}%] for [{4 + index}] stakeholder groups",
        f"saved [{4 + index}] hours per week while stabilizing {metric_name}",
    ]
    return phrases[index % len(phrases)]


def rewrite_resume_bullet(
    seed_line: str,
    *,
    track: str,
    strongest_skills: list[str],
    weakest_keys: set[str],
    index: int,
) -> str:
    base = _clean_resume_bullet_seed(seed_line)
    initiatives = TRACK_INITIATIVES.get(track, ["cross-functional execution"])
    if not base:
        base = f"delivered the {initiatives[index % len(initiatives)]}"

    verbs = ["Led", "Built", "Automated", "Optimized", "Redesigned", "Standardized"]
    skill = strongest_skills[index % len(strongest_skills)] if strongest_skills else ""
    clause = f"{verbs[index % len(verbs)]} {base}"
    if "specificity" in weakest_keys and skill:
        clause += f" using {skill}"
    if "experience" in weakest_keys:
        clause += f" for [{5 + index}] cross-functional stakeholders"
    clause += f", {_revision_result_phrase(track, index)}"
    return "- " + clause.rstrip(".") + "."


def _revised_role_headers(
    sections: dict[str, list[str]],
    work_history: dict[str, Any],
) -> list[str]:
    headers = [
        line for line in sections.get("Experience", []) if not _is_bullet_line(line)
    ]
    if not headers:
        headers = [
            str(span.get("line", "")).strip()
            for span in work_history.get("spans") or []
            if str(span.get("line", "")).strip()
        ]
    return _dedupe_preserve_order(headers)[:2]


def build_revised_experience_lines(
    *,
    sections: dict[str, list[str]],
    profile: dict[str, Any],
    work_history: dict[str, Any],
    strongest_skills: list[str],
    weakest_keys: set[str],
) -> list[str]:
    track = str(profile.get("track", ""))
    bullet_seeds = [
        line for line in sections.get("Experience", []) if _is_bullet_line(line)
    ]
    if not bullet_seeds:
        initiatives = TRACK_INITIATIVES.get(track, ["cross-functional execution"])
        bullet_seeds = [
            f"- Delivered {initiatives[index % len(initiatives)]}" for index in range(3)
        ]

    rewritten = [
        rewrite_resume_bullet(
            bullet_seeds[index % len(bullet_seeds)],
            track=track,
            strongest_skills=strongest_skills,
            weakest_keys=weakest_keys,
            index=index,
        )
        for index in range(6)
    ]

    headers = _revised_role_headers(sections, work_history)
    if not headers:
        return rewritten

    output: list[str] = []
    slices = [rewritten[:3], rewritten[3:]]
    for index, header in enumerate(headers[:2]):
        output.append(header)
        output.extend(slices[index])
        if index != len(headers[:2]) - 1:
            output.append("")
    if len(headers) == 1:
        output.extend(["", "Earlier experience"])
        output.extend(slices[1])
    return output


def build_revised_project_lines(
    *,
    sections: dict[str, list[str]],
    profile: dict[str, Any],
    strongest_skills: list[str],
) -> list[str]:
    track = str(profile.get("track", ""))
    project_lines = sections.get("Projects", [])
    bullet_seeds = [line for line in project_lines if _is_bullet_line(line)]
    if not bullet_seeds:
        track_projects = TRACK_PROJECTS.get(track, ["operating improvement project"])
        bullet_seeds = [
            f"- Built the {track_projects[index % len(track_projects)]}"
            for index in range(2)
        ]
    return [
        rewrite_resume_bullet(
            bullet_seeds[index % len(bullet_seeds)],
            track=track,
            strongest_skills=strongest_skills,
            weakest_keys={"quantified_impact", "specificity"},
            index=index + 6,
        )
        for index in range(2)
    ]


def build_revised_education_lines(sections: dict[str, list[str]]) -> list[str]:
    education_lines = sections.get("Education", [])
    if education_lines:
        return education_lines[:4]
    return [
        "- Add your degree, institution, graduation year, and honors from the original resume here."
    ]


def revised_resume_identity(
    *,
    sections: dict[str, list[str]],
    profile: dict[str, Any],
    matches: pd.DataFrame | None,
) -> tuple[str, str, str]:
    header_lines = sections.get("Header", [])
    name = header_lines[0] if header_lines else "Candidate Name"
    if _canonical_section_from_line(name) is not None or len(name.split()) > 6:
        name = "Candidate Name"

    top_title = ""
    top_location = ""
    if matches is not None and not matches.empty:
        top_title = str(matches.iloc[0].get("title", "") or "")
        top_location = str(matches.iloc[0].get("location", "") or "")

    headline = next(
        (
            line
            for line in header_lines[1:]
            if "@" not in line and "http" not in line and len(line.split()) <= 10
        ),
        compose_headline(
            str(profile.get("seniority", "Mid")),
            top_title
            or TRACK_TITLES.get(str(profile.get("track", "")), ["Specialist"])[0],
        ),
    )

    contact_lines = [line for line in header_lines[1:] if line != headline]
    contact_line = " | ".join(contact_lines[:2]).strip(" |")
    if not contact_line:
        contact_line = top_location
    return name, headline, contact_line


def build_resume_rewrite_assessment(
    resume_text: str,
    assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(resume_text or "").strip()
    if not text:
        return {}

    if assessment and str(assessment.get("resume_text", "") or "").strip() == text:
        rewrite_assessment = dict(assessment)
    else:
        public_models = load_public_assessment_resource()
        public_signals = (
            public_resume_signals(public_models, text) if public_models else None
        )
        assessed = assess_resume_text(text, public_signals)
        rewrite_assessment = {
            **assessed,
            "resume_text": text,
            "public_signals": public_signals,
            "matches": None,
        }

    rewrite_assessment["resume_text"] = text
    rewrite_assessment["strengthening_plan"] = build_strengthening_plan(
        rewrite_assessment.get("quality") or {},
        resume_text=text,
        work_history=rewrite_assessment.get("work_history") or {},
    )
    return rewrite_assessment


def generate_revised_resume_document(
    assessment: dict[str, Any],
) -> dict[str, Any]:
    resume_text = str(assessment.get("resume_text", "") or "")
    profile = assessment.get("profile") or {}
    work_history = assessment.get("work_history") or {}
    quality = assessment.get("quality") or {}
    matches = assessment.get("matches")
    sections = extract_resume_sections(resume_text)
    strongest_skills = build_revised_skill_list(profile, matches)
    weakest = assessment.get("strengthening_plan") or build_strengthening_plan(
        quality,
        resume_text=resume_text,
        work_history=work_history,
    )
    weakest_keys = {str(item.get("key")) for item in weakest}
    weakest_labels = [str(item.get("label")) for item in weakest]

    name, headline, contact_line = revised_resume_identity(
        sections=sections,
        profile=profile,
        matches=matches if isinstance(matches, pd.DataFrame) else None,
    )

    years = max(1, int((work_history.get("weighted_ft_months", 0) or 0) / 12))
    target_title = headline
    focus_summary = (
        ", ".join(strongest_skills[:4])
        if strongest_skills
        else str(profile.get("track", "cross-functional execution"))
    )
    summary_lines = [
        (
            f"{str(profile.get('seniority', 'Mid'))} {str(profile.get('track', 'professional')).lower()} candidate targeting {target_title.lower()} roles. "
            f"Brings roughly {years}+ years of experience translating ambiguous priorities into structured execution with stronger evidence in {focus_summary}."
        ),
        (
            "This revised version is optimized for section coverage, quantified impact, specificity, and bullet clarity. Replace every bracketed metric with your exact real result before using it externally."
        ),
    ]

    experience_lines = build_revised_experience_lines(
        sections=sections,
        profile=profile,
        work_history=work_history,
        strongest_skills=strongest_skills,
        weakest_keys=weakest_keys,
    )
    project_lines = build_revised_project_lines(
        sections=sections,
        profile=profile,
        strongest_skills=strongest_skills,
    )
    education_lines = build_revised_education_lines(sections)
    skills_lines = [
        "- " + ", ".join(strongest_skills[:10])
        if strongest_skills
        else "- Add your strongest tools, domains, and methods here."
    ]

    parts = [name, headline]
    if contact_line:
        parts.append(contact_line)
    parts.extend(
        [
            "",
            "SUMMARY",
            *summary_lines,
            "",
            "EXPERIENCE",
            *experience_lines,
            "",
            "PROJECTS",
            *project_lines,
            "",
            "EDUCATION",
            *education_lines,
            "",
            "SKILLS",
            *skills_lines,
        ]
    )

    targeted_dimensions = (
        ", ".join(weakest_labels[:2]) or "structure and measurable impact"
    )
    notes = [
        "Replace every bracketed number, count, and percentage with your exact real metric before sending this externally.",
        "The rewrite is optimized around: " + targeted_dimensions + ".",
        "Keep the section order and bullet density even if you further personalize the wording.",
    ]
    return {
        "text": "\n".join(parts).strip() + "\n",
        "notes": notes,
        "weakest_labels": weakest_labels[:2],
    }


def build_highlighted_resume_diff_markup(
    original_text: str,
    revised_text: str,
) -> dict[str, Any]:
    original_lines = [line.rstrip() for line in str(original_text or "").splitlines()]
    revised_lines = [line.rstrip() for line in str(revised_text or "").splitlines()]
    matcher = SequenceMatcher(a=original_lines, b=revised_lines)

    rows: list[str] = []
    added_lines = 0
    deleted_lines = 0

    def add_row(css_class: str, prefix: str, line: str) -> None:
        line_html = escape(line) if line else "&nbsp;"
        rows.append(
            f'<div class="revision-line {css_class}">'
            f'<span class="revision-prefix">{prefix}</span>'
            f"<span>{line_html}</span>"
            "</div>"
        )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in original_lines[i1:i2]:
                add_row("same", "·", line)
            continue
        if tag in {"replace", "delete"}:
            for line in original_lines[i1:i2]:
                if line.strip():
                    deleted_lines += 1
                add_row("deleted", "-", line)
        if tag in {"replace", "insert"}:
            for line in revised_lines[j1:j2]:
                if line.strip():
                    added_lines += 1
                add_row("added", "+", line)

    if not rows:
        add_row("same", "·", "")

    return {
        "html": '<div class="revision-diff-shell">' + "".join(rows) + "</div>",
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
    }


def render_revision_copy_button(text: str, *, key: str) -> None:
    dom_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", key)
    components_html(
        f"""
        <div style="display:flex;align-items:center;gap:0.75rem;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
            <button id="{dom_id}-button" style="border:none;border-radius:999px;background:#166534;color:#ffffff;padding:0.62rem 1rem;font-weight:700;cursor:pointer;">
                Copy revised resume
            </button>
            <span id="{dom_id}-status" style="color:#475467;font-size:0.92rem;">Copies the full revised draft.</span>
        </div>
        <script>
        const button = document.getElementById("{dom_id}-button");
        const status = document.getElementById("{dom_id}-status");
        const payload = {json.dumps(str(text or ""))};
        button.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(payload);
                button.textContent = "Copied";
                status.textContent = "Copied to clipboard.";
                setTimeout(() => {{
                    button.textContent = "Copy revised resume";
                    status.textContent = "Copies the full revised draft.";
                }}, 1800);
            }} catch (error) {{
                status.textContent = "Clipboard access was blocked in this browser.";
            }}
        }});
        </script>
        """,
        height=64,
    )


def generate_revised_resume_into_state(
    current_resume_text: str,
    assessment: dict[str, Any] | None,
) -> None:
    resume_text = str(current_resume_text or "").strip()
    if not resume_text:
        return
    rewrite_assessment = build_resume_rewrite_assessment(resume_text, assessment)
    if not rewrite_assessment:
        return
    revision = generate_revised_resume_document(rewrite_assessment)
    st.session_state.revised_resume_text = revision["text"]
    st.session_state.revised_resume_notes = revision["notes"]
    st.session_state.revised_resume_source = resume_text


def render_resume_revision_panel(
    current_resume_text: str,
    assessment: dict[str, Any] | None,
    *,
    key_prefix: str,
) -> None:
    ensure_revision_state()

    current_text = str(current_resume_text or "").strip()
    revised_resume_text = str(st.session_state.get("revised_resume_text", "") or "")
    revised_resume_notes = list(st.session_state.get("revised_resume_notes") or [])
    revised_resume_source = str(st.session_state.get("revised_resume_source", "") or "")
    using_current_assessment = bool(
        assessment is not None
        and str(assessment.get("resume_text", "") or "").strip() == current_text
    )

    with st.container(border=True):
        action_left, action_right = st.columns([0.34, 0.66], gap="medium")
        with action_left:
            generate_clicked = st.button(
                "Generate revised resume",
                type="primary",
                width="stretch",
                disabled=not bool(current_text),
                key=f"{key_prefix}_generate_revised_resume",
            )
        with action_right:
            if using_current_assessment:
                st.caption(
                    "This rewrite uses the latest analyzed resume signals, including the two weakest quality dimensions."
                )
            else:
                st.caption(
                    "This rewrite can be generated directly from the current resume text before the rest of the recommendation panels are refreshed."
                )

        if generate_clicked:
            generate_revised_resume_into_state(current_text, assessment)
            revised_resume_text = str(
                st.session_state.get("revised_resume_text", "") or ""
            )
            revised_resume_notes = list(
                st.session_state.get("revised_resume_notes") or []
            )
            revised_resume_source = str(
                st.session_state.get("revised_resume_source", "") or ""
            )

        if revised_resume_text:
            if revised_resume_source != current_text:
                st.warning(
                    "The revised draft was generated from an older resume snapshot. Generate it again to sync it with the current text."
                )
            if revised_resume_notes:
                st.markdown(
                    '<div class="chip-cloud">'
                    + "".join(
                        f'<span class="mini-chip">{escape(str(note))}</span>'
                        for note in revised_resume_notes[:3]
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

            show_diff = st.checkbox(
                "Show difference",
                value=bool(st.session_state.get("show_resume_revision_diff", True)),
                key=f"{key_prefix}_show_resume_revision_diff",
            )
            st.session_state.show_resume_revision_diff = show_diff

            render_revision_copy_button(
                revised_resume_text,
                key=f"{key_prefix}_copy_revised_resume",
            )

            if show_diff:
                diff_markup = build_highlighted_resume_diff_markup(
                    current_text,
                    revised_resume_text,
                )
                st.markdown(
                    """
                    <div class="revision-key">
                        <span class="revision-key-item"><span class="revision-swatch added"></span>Added or rewritten lines</span>
                        <span class="revision-key-item"><span class="revision-swatch deleted"></span>Deleted lines</span>
                        <span class="revision-key-item"><span class="revision-swatch same"></span>Unchanged lines</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Diff view. {diff_markup['added_lines']} added lines are highlighted in green and {diff_markup['deleted_lines']} deleted lines are highlighted in red."
                )
                st.markdown(diff_markup["html"], unsafe_allow_html=True)
            else:
                st.caption("Plain revised draft.")
                st.code(revised_resume_text, language="text")
