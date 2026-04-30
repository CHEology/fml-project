from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit.components.v1 as components
from app.demo.components import info_dot
from app.runtime import ml as runtime


def encoded_image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def seniority_ladder_html(current_seniority: str) -> str:
    levels = sorted(
        runtime.SENIORITY_RANKS, key=runtime.SENIORITY_RANKS.get, reverse=True
    )
    current_rank = runtime.SENIORITY_RANKS.get(current_seniority)
    steps = []
    for level in levels:
        rank = runtime.SENIORITY_RANKS[level]
        state = (
            "current"
            if rank == current_rank
            else "complete"
            if rank < current_rank
            else ""
        )
        steps.append(
            f'<div class="seniority-step {state}">'
            '<span class="seniority-dot"></span>'
            f'<span class="seniority-step-label">{escape(level)}</span>'
            "</div>"
        )
    return f'<div class="seniority-ladder">{"".join(steps)}</div>'


def evidence_chip_group_html(
    items: list[tuple[str, str] | tuple[str, str, str]], class_name: str
) -> str:
    chips = ""
    for item in items:
        label, value = item[:2]
        if not value:
            continue
        explanation = item[2] if len(item) > 2 else ""
        label_html = escape(label)
        if explanation:
            label_html += info_dot(explanation, extra_class="inline-info")
        chips += (
            '<div class="evidence-chip">'
            f'<div class="evidence-chip-label">{label_html}</div>'
            f"<strong>{escape(value)}</strong>"
            "</div>"
        )
    return f'<div class="{class_name}">{chips}</div>' if chips else ""


def focus_evidence_html(profile: dict[str, object]) -> str:
    skills = [str(skill) for skill in profile.get("skills_present", [])][:3]
    missing = [str(skill) for skill in profile.get("skills_missing", [])][:2]
    items = [
        ("Matched skills", ", ".join(skills)),
        ("Model confidence", f"{profile.get('confidence', 0)}%"),
    ]
    if missing:
        items.append(("Less visible", ", ".join(missing)))
    return evidence_chip_group_html(items, "focus-evidence-grid")


def seniority_evidence_html(
    current_seniority: str, work_history: dict[str, object]
) -> str:
    weighted_months = int(work_history.get("weighted_ft_months", 0) or 0)
    role_count = int(work_history.get("role_count", 0) or 0)
    title_signal = str(work_history.get("max_seniority_keyword") or current_seniority)
    items = [
        ("Highest title signal", title_signal),
        ("Experience basis", f"{weighted_months} weighted months"),
        ("Roles parsed", str(role_count)),
    ]
    return evidence_chip_group_html(items, "seniority-evidence-grid")


def capability_evidence_html(
    capability: dict[str, object], quality: dict[str, object]
) -> str:
    skill_hits = [str(skill) for skill in capability.get("skill_hits", [])][:3]
    notes = [str(note).rstrip(".") for note in capability.get("notes", [])][:2]
    items = [
        ("Tier readout", str(capability.get("summary", ""))),
        (
            "Impact signal",
            f"{float(quality.get('impact_score', 0) or 0):.0f}/100",
            "Measures evidence of measurable outcomes, business results, quantified improvements, or concrete delivery impact in the resume.",
        ),
        (
            "Specificity",
            f"{float(quality.get('specificity_score', 0) or 0):.0f}/100",
            "Measures how concrete the resume evidence is, including named tools, domains, responsibilities, scale, and clear achievement detail.",
        ),
    ]
    if skill_hits:
        items.append(("Skill evidence", ", ".join(skill_hits)))
    if notes:
        items.append(("Why this tier", notes[0]))
    return evidence_chip_group_html(items, "capability-evidence-grid")


def render_scroll_to_top() -> None:
    components.html(
        """
        <script>
        const scrollToTop = () => {
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;
            const target = parentDocument.getElementById("candidate-snapshot-top");
            const candidates = [
                parentDocument.scrollingElement,
                parentDocument.documentElement,
                parentDocument.body,
                parentDocument.querySelector('[data-testid="stAppViewContainer"]'),
                parentDocument.querySelector('section.main'),
                parentDocument.querySelector('.main')
            ].filter(Boolean);

            candidates.forEach((element) => {
                if (typeof element.scrollTo === "function") {
                    element.scrollTo({ top: 0, left: 0, behavior: "auto" });
                }
                element.scrollTop = 0;
            });
            parentDocument
                .querySelectorAll("*")
                .forEach((element) => {
                    if (element.scrollHeight > element.clientHeight) {
                        element.scrollTop = 0;
                    }
                });
            if (target) {
                target.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
            }
            parentWindow.scrollTo(0, 0);
        };

        let attempts = 0;
        const resetScroll = () => {
            scrollToTop();
            attempts += 1;
            if (attempts < 16) {
                setTimeout(resetScroll, 100);
            }
        };

        requestAnimationFrame(resetScroll);
        </script>
        """,
        height=0,
        width=0,
    )
