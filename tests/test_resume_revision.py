from __future__ import annotations

import pandas as pd
from app.components.resume_revision import (
    build_highlighted_resume_diff_markup,
    generate_revised_resume_document,
)


def test_build_highlighted_resume_diff_markup_marks_added_and_deleted_lines() -> None:
    original = "Alex Rivera\nSUMMARY\nBuilt reporting workflows.\n"
    revised = "Alex Rivera\nSUMMARY\nBuilt reporting workflows.\n- Increased coverage by [25%].\n"

    diff = build_highlighted_resume_diff_markup(original, revised)

    assert diff["added_lines"] == 1
    assert diff["deleted_lines"] == 0
    assert "revision-line added" in diff["html"]
    assert "Increased coverage" in diff["html"]


def test_generate_revised_resume_document_returns_full_resume_sections() -> None:
    assessment = {
        "resume_text": (
            "Alex Rivera\n"
            "Software Engineer\n"
            "SUMMARY\n"
            "Builder of internal tools.\n"
            "EXPERIENCE\n"
            "Acme | Software Engineer | 2022 - Present\n"
            "- Built internal APIs.\n"
            "EDUCATION\n"
            "- B.S. in Computer Science, Example University\n"
        ),
        "profile": {
            "track": "Software Engineering",
            "seniority": "Mid",
            "skills_present": ["Python", "APIs", "AWS"],
        },
        "work_history": {
            "weighted_ft_months": 48,
            "spans": [{"line": "Acme | Software Engineer | 2022 - Present"}],
        },
        "quality": {
            "experience_score": 48,
            "impact_score": 32,
            "specificity_score": 58,
            "structure_score": 44,
        },
        "matches": pd.DataFrame(
            [{"title": "Backend Software Engineer", "location": "New York, NY"}]
        ),
        "strengthening_plan": [
            {"key": "quantified_impact", "label": "Quantified impact"},
            {"key": "structure", "label": "Structure"},
        ],
    }

    revision = generate_revised_resume_document(assessment)

    assert "SUMMARY" in revision["text"]
    assert "EXPERIENCE" in revision["text"]
    assert "PROJECTS" in revision["text"]
    assert "EDUCATION" in revision["text"]
    assert "SKILLS" in revision["text"]
    assert revision["weakest_labels"] == ["Quantified impact", "Structure"]
