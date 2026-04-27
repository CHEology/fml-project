"""
Robust resume loader for real user input.

Reads .pdf / .txt / .md / .csv / .json / .jsonl files (or a directory of
them) and returns a `LoadedResume` with normalised text. Designed to
absorb the messiness of real uploads:

* PDFs go through `pdfplumber` (already a project dependency). Per-page
  text is concatenated; we deliberately keep the structure light because
  downstream rule-based scoring relies on bullet-line markers.
* CSV/JSONL inputs must expose a `resume_text` (or `text`) column; an
  optional `resume_id` is preserved.
* Light PII redaction strips email addresses and obvious phone numbers
  before passing text downstream.
* Whitespace is collapsed but `\\n` line breaks are preserved so the
  rule-based scorer can still detect "- " bullet markers and section
  headings.
* `MAX_RESUME_CHARS` clamps very long texts to keep encoder batches
  bounded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_RESUME_CHARS = 20_000  # ~3-4 dense pages; trims truly absurd uploads
MIN_RESUME_CHARS = 80  # below this we treat as "empty / extraction failed"

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)
_URL_RE = re.compile(r"\bhttps?://\S+\b")


@dataclass(frozen=True)
class LoadedResume:
    """One normalised resume ready for downstream scoring/encoding."""

    resume_id: str
    text: str
    source_path: str
    n_chars: int
    truncated: bool


def load_resume(
    path: str | Path,
    *,
    resume_id: str | None = None,
    max_chars: int = MAX_RESUME_CHARS,
    redact_pii: bool = True,
) -> LoadedResume:
    """Load and normalise a single resume file (PDF/text/markdown).

    Raises:
        FileNotFoundError: when `path` does not exist.
        ValueError: when the file format is unsupported or the extracted
            text is shorter than `MIN_RESUME_CHARS` (likely a scanned
            image PDF or an empty file).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"resume not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix in {".txt", ".md", ""}:
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(
            f"unsupported resume extension '{suffix}'. "
            f"Use .pdf, .txt, .md or pre-load via load_resume_table()."
        )

    return _finalise_resume(
        raw,
        resume_id=resume_id or path.stem,
        source_path=str(path),
        max_chars=max_chars,
        redact_pii=redact_pii,
    )


def load_resume_dir(
    path: str | Path,
    *,
    glob: str = "*",
    max_chars: int = MAX_RESUME_CHARS,
    redact_pii: bool = True,
) -> list[LoadedResume]:
    """Load every supported resume file inside a directory."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    resumes: list[LoadedResume] = []
    for entry in sorted(path.glob(glob)):
        if entry.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        try:
            resumes.append(
                load_resume(
                    entry,
                    max_chars=max_chars,
                    redact_pii=redact_pii,
                )
            )
        except ValueError:
            # Skip empty / unparseable files; the harness logs the gap.
            continue
    return resumes


def load_resume_table(
    path: str | Path,
    *,
    text_column: str = "resume_text",
    id_column: str = "resume_id",
    max_chars: int = MAX_RESUME_CHARS,
    redact_pii: bool = True,
) -> list[LoadedResume]:
    """Load resumes from a tabular file (CSV / JSONL / Parquet).

    `text_column` may also be `"text"`, `"Resume"`, `"resume"`, or
    `"resume_str"` — common column names in public datasets — which we
    auto-detect when the explicit name is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path}")

    rows = _read_table(path)
    column = _resolve_text_column(rows, preferred=text_column)
    out: list[LoadedResume] = []
    for i, row in enumerate(rows):
        text = row.get(column)
        if not isinstance(text, str) or not text.strip():
            continue
        rid = str(row.get(id_column) or row.get("id") or f"{path.stem}-{i:05d}")
        try:
            out.append(
                _finalise_resume(
                    text,
                    resume_id=rid,
                    source_path=f"{path}#{i}",
                    max_chars=max_chars,
                    redact_pii=redact_pii,
                )
            )
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dep is in pyproject
        raise RuntimeError(
            "pdfplumber is required to load PDF resumes. Install via uv sync."
        ) from exc

    pieces: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pieces.append(text)
    return "\n\n".join(pieces)


def _read_table(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    raise ValueError(f"unsupported table extension: {suffix}")


def _resolve_text_column(rows: list[dict], *, preferred: str) -> str:
    if not rows:
        return preferred
    candidates = [preferred, "resume_text", "text", "Resume", "resume", "resume_str"]
    for name in candidates:
        if name in rows[0]:
            return name
    raise ValueError(
        "no usable text column found. Expected one of "
        f"{candidates}; got {sorted(rows[0])}."
    )


def _finalise_resume(
    raw: str,
    *,
    resume_id: str,
    source_path: str,
    max_chars: int,
    redact_pii: bool,
) -> LoadedResume:
    cleaned = _normalise_text(raw)
    if redact_pii:
        cleaned = _redact_pii(cleaned)
    truncated = False
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "\n[...resume truncated...]"
        truncated = True
    if len(cleaned) < MIN_RESUME_CHARS:
        raise ValueError(
            f"resume {source_path!r} extracted to {len(cleaned)} chars "
            f"(< {MIN_RESUME_CHARS}). Likely scanned/empty."
        )
    return LoadedResume(
        resume_id=resume_id,
        text=cleaned,
        source_path=source_path,
        n_chars=len(cleaned),
        truncated=truncated,
    )


def _normalise_text(raw: str) -> str:
    if not raw:
        return ""
    # Normalise newlines, collapse 3+ blank lines to 2, strip trailing space
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _URL_RE.sub("[url]", text)
    return text


__all__ = [
    "LoadedResume",
    "MAX_RESUME_CHARS",
    "MIN_RESUME_CHARS",
    "load_resume",
    "load_resume_dir",
    "load_resume_table",
]
