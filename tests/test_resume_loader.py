from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.resume_loader import (  # noqa: E402
    MIN_RESUME_CHARS,
    LoadedResume,
    load_resume,
    load_resume_dir,
    load_resume_table,
)

_SAMPLE_RESUME = """\
Jane Doe
Senior Machine Learning Engineer
Contact: jane.doe@example.com | (212) 555-0143 | https://github.com/jane

Summary
- 6 years building production ML systems at retail-scale.
- Shipped a recommender service handling 80K requests per day.

Skills
Python, PyTorch, AWS, Docker, model serving, system design

Experience
- Led migration of training pipelines, reducing retraining time by 35%.
- Built drift monitoring across 12 production models.

Education
M.S. Computer Science, NYU
"""


def test_load_resume_from_txt_redacts_pii(tmp_path: Path) -> None:
    file_ = tmp_path / "jane.txt"
    file_.write_text(_SAMPLE_RESUME, encoding="utf-8")

    loaded = load_resume(file_)

    assert isinstance(loaded, LoadedResume)
    assert loaded.resume_id == "jane"
    assert "jane.doe@example.com" not in loaded.text
    assert "[email]" in loaded.text
    assert "[phone]" in loaded.text
    assert "[url]" in loaded.text
    assert loaded.n_chars == len(loaded.text)
    assert not loaded.truncated


def test_load_resume_preserves_bullet_lines(tmp_path: Path) -> None:
    file_ = tmp_path / "ml.txt"
    file_.write_text(_SAMPLE_RESUME, encoding="utf-8")
    loaded = load_resume(file_, redact_pii=False)
    assert "\n- Led migration" in loaded.text
    assert "\n- Built drift monitoring" in loaded.text


def test_load_resume_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_ = tmp_path / "weird.docx"
    file_.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="unsupported resume extension"):
        load_resume(file_)


def test_load_resume_rejects_empty_extraction(tmp_path: Path) -> None:
    file_ = tmp_path / "blank.txt"
    file_.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match=f"< {MIN_RESUME_CHARS}"):
        load_resume(file_)


def test_load_resume_truncates_oversize_inputs(tmp_path: Path) -> None:
    file_ = tmp_path / "long.txt"
    file_.write_text(_SAMPLE_RESUME + "filler " * 5000, encoding="utf-8")
    loaded = load_resume(file_, max_chars=2_000)
    assert loaded.truncated is True
    assert "[...resume truncated...]" in loaded.text
    assert loaded.n_chars <= 2_000 + 30


def test_load_resume_dir_loads_all_supported(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text(_SAMPLE_RESUME, encoding="utf-8")
    (tmp_path / "b.md").write_text(_SAMPLE_RESUME, encoding="utf-8")
    (tmp_path / "skipme.bin").write_bytes(b"binary blob")

    resumes = load_resume_dir(tmp_path)

    ids = {r.resume_id for r in resumes}
    assert ids == {"a", "b"}


def test_load_resume_table_csv(tmp_path: Path) -> None:
    file_ = tmp_path / "real.csv"
    pd.DataFrame(
        {
            "resume_id": ["r1", "r2"],
            "resume_text": [_SAMPLE_RESUME, _SAMPLE_RESUME.replace("Jane", "Bob")],
        }
    ).to_csv(file_, index=False)

    resumes = load_resume_table(file_)
    assert [r.resume_id for r in resumes] == ["r1", "r2"]
    assert "[email]" in resumes[0].text


def test_load_resume_table_jsonl_with_alternate_column(tmp_path: Path) -> None:
    file_ = tmp_path / "real.jsonl"
    payload = [
        {"id": 1, "Resume": _SAMPLE_RESUME},
        {"id": 2, "Resume": _SAMPLE_RESUME.replace("Jane", "Carol")},
    ]
    with open(file_, "w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(row) + "\n")

    resumes = load_resume_table(file_)
    assert len(resumes) == 2
    assert all("Carol" in r.text or "Jane" in r.text for r in resumes)


def test_load_resume_table_skips_empty_rows(tmp_path: Path) -> None:
    file_ = tmp_path / "mix.csv"
    pd.DataFrame(
        {
            "resume_text": [_SAMPLE_RESUME, "", "   ", _SAMPLE_RESUME],
        }
    ).to_csv(file_, index=False)
    resumes = load_resume_table(file_)
    assert len(resumes) == 2
