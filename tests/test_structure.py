from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_PYTHON_LINES = 1_000


def test_resume_assessment_public_api_is_available() -> None:
    import ml.resume_assessment as assessment

    required_names = {
        "assess_resume_text",
        "resume_structure",
        "extract_work_history",
        "score_projects",
        "detect_profile",
        "apply_quality_discount",
        "apply_capability_adjustment",
    }

    assert required_names.issubset(set(dir(assessment)))


def test_python_source_files_stay_under_line_budget() -> None:
    ignored_parts = {
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "notebooks",
    }
    over_budget: list[tuple[str, int]] = []

    for path in PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in ignored_parts for part in relative.parts):
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_PYTHON_LINES:
            over_budget.append((str(relative), line_count))

    assert over_budget == []
