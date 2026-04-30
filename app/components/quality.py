from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def render_quality_scorecard(
    quality: dict[str, Any],
    learned_quality: dict[str, Any] | None = None,
) -> None:
    overall = int(quality.get("overall", 0))
    band_label = str(quality.get("band_label", "Mixed"))
    sub_pairs = [
        ("Experience", int(quality.get("experience_score", 0))),
        ("Quantified impact", int(quality.get("impact_score", 0))),
        ("Specificity", int(quality.get("specificity_score", 0))),
        ("Structure", int(quality.get("structure_score", 0))),
    ]
    sub_html = "".join(
        f'<div><div class="quality-sub-label">{escape(label)}</div>'
        f'<div class="quality-bar-track"><div class="quality-bar-fill" '
        f'style="width:{max(0, min(100, value))}%;"></div></div></div>'
        for label, value in sub_pairs
    )
    flags = quality.get("red_flags") or []
    strengths = quality.get("strengths") or []
    strengths_html = (
        '<div class="quality-feedback-panel">'
        '<div class="quality-section-label">What stood out positively</div>'
        '<ul class="quality-list strengths">'
        + "".join(f"<li>{escape(str(s))}</li>" for s in strengths[:5])
        + "</ul></div>"
    )
    flags_html = (
        '<div class="quality-feedback-panel">'
        '<div class="quality-section-label">What needs work</div>'
        '<ul class="quality-list flags">'
        + "".join(f"<li>{escape(str(f))}</li>" for f in flags[:5])
        + "</ul></div>"
    )
    band_label_safe = escape(band_label)
    overall_html = (
        f'<div class="quality-overall">{overall}'
        f'<span style="font-size:0.9rem;color:var(--muted);font-weight:500;"> / 100</span>'
        f"</div>"
    )
    headline = (
        '<div class="quality-headline">'
        f'<div><div class="metric-label">Resume quality</div>{overall_html}</div>'
        f'<span class="quality-band-pill quality-band-{band_label_safe}">{band_label_safe}</span>'
        "</div>"
    )
    learned_html = ""
    if learned_quality:
        learned_score = float(learned_quality.get("score", 0.0) or 0.0)
        learned_label = str(learned_quality.get("label", "unknown")).title()
        delta = learned_score - float(overall)
        if abs(delta) <= 10:
            agreement = "Close to the rule-based score"
        elif delta > 10:
            agreement = "More favorable than the rule-based score"
        else:
            agreement = "More conservative than the rule-based score"
        learned_html = (
            '<div class="quality-learned-check">'
            '<div class="metric-label">Learned MLP cross-check</div>'
            f'<div class="signal-copy"><strong>{learned_score:.0f}/100</strong>'
            f" · {escape(learned_label)} · {escape(agreement)}. "
            "Advisory only; strengths, gaps, and salary adjustments still use the explainable rule score."
            "</div></div>"
        )
    body = (
        '<div class="quality-card">'
        + headline
        + f'<div class="quality-subscores">{sub_html}</div>'
        + learned_html
        + f'<div class="quality-feedback-grid">{strengths_html}{flags_html}</div>'
        + "</div>"
    )
    st.markdown(body, unsafe_allow_html=True)


def render_public_model_card(public_signals: dict[str, Any] | None) -> None:
    if not public_signals or not public_signals.get("ready"):
        return
    domain = public_signals.get("domain", {})
    domain_label = str(domain.get("label", "Unknown")).title()
    domain_confidence = float(domain.get("confidence", 0.0) or 0.0)
    sections = public_signals.get("sections", {}).get("counts", {})
    entities = public_signals.get("entities", {}).get("counts", {})
    chips = [
        f"Domain: {domain_label} ({domain_confidence * 100:.0f}%)",
        "Sections: "
        + ", ".join(
            f"{label}:{count}"
            for label, count in sections.items()
            if label in {"Exp", "Edu", "Skill", "Sum"}
        ),
        "Entities: "
        + ", ".join(
            f"{label}:{count}"
            for label, count in entities.items()
            if label
            in {
                "Companies worked at",
                "College Name",
                "Degree",
                "Designation",
                "Skills",
            }
        ),
    ]
    chips = [chip for chip in chips if not chip.endswith(": ")]
    if not chips:
        return
    st.markdown(
        '<div class="section-label" style="margin-top:0.9rem;">Public-data model checks</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chip-cloud">'
        + "".join(
            f'<span class="mini-chip">{escape(chip)}</span>' for chip in chips[:5]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
