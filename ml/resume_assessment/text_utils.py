from __future__ import annotations

import re
from collections.abc import Callable


def resume_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()
    ]


def quote_resume_line(line: str, max_chars: int = 120) -> str:
    cleaned = line.lstrip("-*• ").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return f'"{cleaned}"'


def find_resume_line(
    text: str,
    predicate: Callable[[str], bool],
    *,
    default: str = "",
) -> str:
    for line in resume_lines(text):
        if predicate(line):
            return quote_resume_line(line)
    return default


def find_first_bullet(text: str) -> str:
    return find_resume_line(
        text,
        lambda line: line.startswith(("-", "*", "•")),
        default="",
    )
