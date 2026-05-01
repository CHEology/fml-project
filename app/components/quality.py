from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.demo.components import info_dot

PROFILE_QUALITY_INFO = (
    "Résumé Quality is the explainable resume-quality readout from "
    "ml.resume_assessment.assess_quality(). It combines experience depth, "
    "quantified impact, specificity, and structure into a 0-100 score. The "
    "positive and work-needed notes are selected from parsed work history, "
    "section coverage, action verbs, vague wording, quantified outcomes, and "
    "optional public-data model signals. The learned MLP cross-check is shown "
    "as an advisory neural comparison only; strengths, gaps, salary discounts, "
    "and capability adjustments continue to use the explainable rule score."
)

LEARNED_CHECK_INFO = (
    "This is an advisory neural cross-check from the public resume-style model. "
    "It is useful for sanity checking the rule-based score, but it does not "
    "drive strengths, gaps, salary adjustments, or the final résumé quality score."
)

FEEDBACK_INFO = (
    "These bullets are selected from the strongest evidence and constraints found "
    "by the quality scorer: parsed dates and roles, section coverage, quantified "
    "impact, action verbs, vague phrasing, recognized employers or schools, and "
    "public model signals when available."
)

PUBLIC_CHECKS_INFO = (
    "These are independent public-data classifiers for resume domain, section "
    "tokens, and named entities. Domain can differ from Detected focus because "
    "Detected focus comes from local track keyword and resume-evidence scoring, "
    "while this domain label comes from a separate public-resume model."
)

EVIDENCE_TAGS_INFO = (
    "These tags come from résumé content detection rather than the quality scorer. "
    "They summarize skills and domain terms found in the résumé text, so they "
    "support the positive readout but are not the same as the written quality notes."
)

MARKET_GAPS_INFO = (
    "These chips come from market matching and retrieval, not the rule-based quality "
    "score. They compare resume wording with matched roles and cluster terms, so "
    "they are shown as related work-on items rather than additional quality red flags."
)


def _chips_html(items: list[str]) -> str:
    if not items:
        return '<span class="mini-chip">None detected</span>'
    return "".join(
        f'<span class="mini-chip">{escape(str(item))}</span>' for item in items
    )


def _public_model_chips(public_signals: dict[str, Any] | None) -> list[str]:
    if not public_signals or not public_signals.get("ready"):
        return []
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
    return [chip for chip in chips if not chip.endswith(": ")]


def render_quality_scorecard(
    quality: dict[str, Any],
    learned_quality: dict[str, Any] | None = None,
    evidence_tags: list[str] | None = None,
    gap_terms: list[str] | None = None,
    detail_html: str = "",
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
    evidence_tags_html = ""
    if evidence_tags:
        evidence_tags_html = (
            '<div class="quality-chip-subsection">'
            '<div class="quality-subsection-label">Evidence tags'
            f"{info_dot(EVIDENCE_TAGS_INFO, extra_class='inline-info')}</div>"
            f'<div class="chip-cloud">{_chips_html(evidence_tags[:8])}</div>'
            "</div>"
        )
    gap_chips = [f"Add stronger evidence for {term}" for term in (gap_terms or [])[:8]]
    gaps_html = ""
    if gap_chips:
        gaps_html = (
            '<div class="quality-chip-subsection">'
            '<div class="quality-subsection-label">Market-language gaps'
            f"{info_dot(MARKET_GAPS_INFO, extra_class='inline-info')}</div>"
            f'<div class="chip-cloud">{_chips_html(gap_chips)}</div>'
            "</div>"
        )
    strengths_html = (
        '<div class="quality-feedback-panel">'
        '<div class="quality-section-label">What stood out positively'
        f"{info_dot(FEEDBACK_INFO, extra_class='inline-info')}</div>"
        '<ul class="quality-list strengths">'
        + "".join(f"<li>{escape(str(s))}</li>" for s in strengths[:5])
        + "</ul>"
        + evidence_tags_html
        + "</div>"
    )
    flags_html = (
        '<div class="quality-feedback-panel">'
        '<div class="quality-section-label">What needs work'
        f"{info_dot(FEEDBACK_INFO, extra_class='inline-info')}</div>"
        '<ul class="quality-list flags">'
        + "".join(f"<li>{escape(str(f))}</li>" for f in flags[:5])
        + "</ul>"
        + gaps_html
        + "</div>"
    )
    band_label_safe = escape(band_label)
    overall_html = (
        f'<div class="quality-overall">{overall}'
        f'<span style="font-size:0.9rem;color:var(--muted);font-weight:500;"> / 100</span>'
        f"</div>"
    )
    learned_summary_html = ""
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
        learned_summary_html = (
            '<div class="quality-peer-metric">'
            '<div class="metric-label">Learned MLP cross-check'
            f"{info_dot(LEARNED_CHECK_INFO, extra_class='inline-info')}</div>"
            f'<div class="quality-overall">{learned_score:.0f}'
            '<span style="font-size:0.9rem;color:var(--muted);font-weight:500;"> / 100</span>'
            "</div></div>"
            f'<div class="quality-agreement-copy">{escape(learned_label)} · '
            f"{escape(agreement)}.</div>"
        )
    headline = (
        '<div class="quality-headline">'
        '<div class="quality-score-row">'
        f'<div class="quality-peer-metric"><div class="metric-label">Resume quality</div>{overall_html}</div>'
        + learned_summary_html
        + "</div>"
        f'<span class="quality-band-pill quality-band-{band_label_safe}">{band_label_safe}</span>'
        "</div>"
    )
    body = (
        '<div class="quality-card">'
        + headline
        + f'<div class="quality-subscores">{sub_html}</div>'
        + detail_html
        + f'<div class="quality-feedback-grid">{strengths_html}{flags_html}</div>'
        + "</div>"
    )
    st.markdown(body, unsafe_allow_html=True)


def render_profile_quality_section(
    *,
    quality: dict[str, Any],
    learned_quality: dict[str, Any] | None,
    public_signals: dict[str, Any] | None,
    resume_stats: dict[str, int],
    strengths: list[str],
    sections: list[str],
    missing_sections: list[str],
    missing_terms: list[str],
    resume_text: str = "",
) -> None:
    word_count = int(resume_stats.get("word_count", 0))
    bullet_count = int(resume_stats.get("bullet_count", 0))
    link_count = int(resume_stats.get("link_count", 0))
    found_sections_count = int(resume_stats.get("found_sections_count", 0))
    total_sections_count = int(resume_stats.get("total_sections_count", 0))
    stats_html = "".join(
        f'<span class="snapshot-stat">{escape(label)}</span>'
        for label in (
            f"{word_count} words",
            f"{bullet_count} bullets",
            f"{link_count} links",
            f"{found_sections_count}/{total_sections_count} sections",
        )
    )
    with st.container(key="profile-quality-section"):
        st.markdown(
            f"""
            <div class="profile-quality-hero">
                <div class="profile-quality-title-row">
                    <h1 class="snapshot-title">Profile Quality</h1>
                    <div class="snapshot-stat-row profile-quality-stats">{stats_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Read more about Profile Quality"):
            st.markdown(PROFILE_QUALITY_INFO)

        public_chips = _public_model_chips(public_signals)
        public_html = ""
        if public_chips:
            public_html = (
                '<div class="quality-subsection">'
                '<div class="snapshot-label">Public-data model checks'
                f"{info_dot(PUBLIC_CHECKS_INFO, extra_class='inline-info')}</div>"
                '<div class="snapshot-copy public-checks-copy">'
                "These chips are a secondary public-resume model readout, separate "
                "from the rule-based quality score. They show the model's inferred "
                "domain, section counts, and entity counts so you can sanity-check "
                "what resume evidence was recognized."
                "</div>"
                f'<div class="chip-cloud">{_chips_html(public_chips[:5])}</div>'
                "</div>"
            )

        missing_sections_html = (
            '<div class="snapshot-copy">Missing sections: '
            + escape(", ".join(missing_sections))
            + "</div>"
            if missing_sections
            else '<div class="snapshot-copy">Core resume sections are represented.</div>'
        )
        body = (
            '<div class="profile-quality-detail-grid">'
            + public_html
            + '<div class="snapshot-card profile-quality-organization-card">'
            '<div class="snapshot-label">Resume organization</div>'
            f'<div class="chip-cloud">{_chips_html(sections)}</div>'
            f"{missing_sections_html}</div>" + "</div>"
        )
        render_quality_scorecard(
            quality,
            learned_quality,
            evidence_tags=strengths,
            gap_terms=missing_terms,
            detail_html=body,
        )


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
