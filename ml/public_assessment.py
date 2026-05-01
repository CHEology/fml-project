"""Inference helpers for public-data resume assessment models.

These models are deliberately advisory. They are lightweight supervised
baselines trained by `scripts/train_public_assessment_models.py` from public
resume datasets, and they complement the app's deterministic parsing rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
MODEL_FILES = {
    "domain": "public_domain_model.pt",
    "ats_fit": "public_ats_fit_model.pt",
    "entity": "public_entity_model.pt",
    "section": "public_section_model.pt",
}


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PublicAssessmentModels:
    hash_dim: int
    metrics: dict[str, Any]
    domain_labels: list[str]
    entity_labels: list[str]
    section_labels: list[str]
    domain_model: MLPClassifier
    ats_model: MLPRegressor
    entity_model: MLPClassifier
    section_model: MLPClassifier


def public_models_ready(project_root: Path) -> bool:
    models_dir = Path(project_root) / "models"
    return (models_dir / "public_assessment_metrics.json").exists() and all(
        (models_dir / filename).exists() for filename in MODEL_FILES.values()
    )


def load_public_assessment_models(project_root: Path) -> PublicAssessmentModels:
    models_dir = Path(project_root) / "models"
    with (models_dir / "public_assessment_metrics.json").open(encoding="utf-8") as f:
        metrics = json.load(f)

    hash_dim = int(metrics.get("hash_dim", 2048))
    hidden_dim = 128
    datasets = metrics["datasets"]
    domain_labels = list(datasets["domain"]["labels"])
    entity_labels = list(datasets["entity"]["labels"])
    section_labels = list(datasets["section"]["labels"])

    domain_model = MLPClassifier(hash_dim, hidden_dim, len(domain_labels))
    domain_model.load_state_dict(
        torch.load(
            models_dir / MODEL_FILES["domain"], map_location="cpu", weights_only=True
        )
    )
    domain_model.eval()

    ats_model = MLPRegressor(hash_dim + 8, hidden_dim)
    ats_model.load_state_dict(
        torch.load(
            models_dir / MODEL_FILES["ats_fit"], map_location="cpu", weights_only=True
        )
    )
    ats_model.eval()

    entity_model = MLPClassifier(hash_dim, hidden_dim, len(entity_labels))
    entity_model.load_state_dict(
        torch.load(
            models_dir / MODEL_FILES["entity"], map_location="cpu", weights_only=True
        )
    )
    entity_model.eval()

    section_model = MLPClassifier(hash_dim, hidden_dim, len(section_labels))
    section_model.load_state_dict(
        torch.load(
            models_dir / MODEL_FILES["section"], map_location="cpu", weights_only=True
        )
    )
    section_model.eval()

    return PublicAssessmentModels(
        hash_dim=hash_dim,
        metrics=metrics,
        domain_labels=domain_labels,
        entity_labels=entity_labels,
        section_labels=section_labels,
        domain_model=domain_model,
        ats_model=ats_model,
        entity_model=entity_model,
        section_model=section_model,
    )


def resume_public_signals(
    models: PublicAssessmentModels | None,
    resume_text: str,
) -> dict[str, Any]:
    if models is None:
        return {"ready": False}
    domain = predict_domain(models, resume_text)
    sections = predict_sections(models, resume_text)
    entities = predict_entities(models, resume_text)
    return {
        "ready": True,
        "domain": domain,
        "sections": sections,
        "entities": entities,
        "metrics": {
            key: value
            for key, value in models.metrics.get("datasets", {}).items()
            if key in {"domain", "entity", "section", "ats_fit"}
        },
    }


# Patterns that strongly indicate a resume (date ranges, contact info,
# section headers). Each match adds positive evidence.
_DATE_RANGE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)\.?"
    r"\s+\d{4}\s*[-–—to]+\s*"
    r"(?:present|current|now|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)\.?\s+\d{4}|"
    r"\d{4})",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|current|now)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}\b")
_DEGREE_RE = re.compile(
    r"\b(?:b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|ph\.?d\.?|"
    r"bachelor(?:'s)?|master(?:'s)?|doctorate|doctor of philosophy)\b",
    re.IGNORECASE,
)

# Patterns typical of websites / blogs / social bios but NOT resumes.
_WEB_NAV_TOKENS = (
    "subscribe",
    "sign up",
    "log in",
    "sign in",
    "read more",
    "privacy policy",
    "terms of service",
    "cookie",
    "all rights reserved",
    "newsletter",
    "follow me on",
    "follow us on",
    "buy now",
    "add to cart",
    "shopping cart",
    "checkout",
    "© ",
    "powered by",
    "back to top",
    "skip to content",
    "menu",
    "search...",
    "load more",
    "view all posts",
)
_HTML_RESIDUE_RE = re.compile(r"</[a-z]+>|<[a-z]+\s|&nbsp;|&amp;|&lt;|&gt;|&#\d+;", re.IGNORECASE)
_CODE_RESIDUE_RE = re.compile(
    r"^\s*(?:def |class |function |import |const |let |var |#include|public class|"
    r"\{|\}|//|/\*|\*/)",
    re.MULTILINE,
)


def _count_resume_section_keywords(lower_text: str) -> int:
    keywords = (
        "education",
        "experience",
        "skills",
        "projects",
        "employment",
        "work history",
        "summary",
        "objective",
        "publications",
        "awards",
        "certifications",
    )
    return sum(1 for k in keywords if k in lower_text)


def validate_resume_quality(
    models: PublicAssessmentModels | None,
    text: str,
) -> dict[str, Any]:
    """Score whether `text` looks like a resume / CV.

    Returns a tiered judgment:
      - confidence: "high" | "medium" | "low" | "empty"
      - is_resume: True for high/medium, False for low/empty (back-compat)
      - score: 0.0–1.0 blended score
      - reasons: brief negative findings (always populated when non-high)
      - signals: positive evidence found, surfaced for UI explainability

    Tiers are tuned so a clear CV — even a poor one — lands in
    high/medium, while portfolio pages, social bios, blog posts, code
    dumps, and short bios drop to low. Borderline cases (medium) should
    be soft-warned in the UI rather than hard-rejected, with an override.
    """
    text = str(text or "").strip()
    if not text:
        return {
            "is_resume": False,
            "confidence": "empty",
            "score": 0.0,
            "reasons": ["No text provided."],
            "signals": [],
        }

    lower = text.lower()
    char_count = len(text)
    word_count = len(text.split())

    # ---------- Positive signals ----------
    positive: list[str] = []
    pos_score = 0.0

    date_hits = len(_DATE_RANGE_RE.findall(text)) + len(_YEAR_RANGE_RE.findall(text))
    if date_hits >= 2:
        pos_score += 0.30
        positive.append(f"{date_hits} employment / education date range(s)")
    elif date_hits == 1:
        pos_score += 0.15
        positive.append("1 date range")

    section_hits = _count_resume_section_keywords(lower)
    if section_hits >= 3:
        pos_score += 0.25
        positive.append(f"{section_hits} resume section headers")
    elif section_hits == 2:
        pos_score += 0.15
        positive.append("2 resume section headers")
    elif section_hits == 1:
        pos_score += 0.05
        positive.append("1 resume section header")

    if _EMAIL_RE.search(text) or _PHONE_RE.search(text):
        pos_score += 0.10
        positive.append("contact details")

    if _DEGREE_RE.search(text):
        pos_score += 0.10
        positive.append("degree mentioned")

    bullet_lines = sum(1 for ln in text.splitlines() if ln.lstrip().startswith(("-", "*", "•", "·", "▪", "◦")))
    if bullet_lines >= 5:
        pos_score += 0.15
        positive.append(f"{bullet_lines} bullet lines")
    elif bullet_lines >= 2:
        pos_score += 0.08
        positive.append(f"{bullet_lines} bullet lines")

    # ---------- Negative signals ----------
    reasons: list[str] = []
    neg_score = 0.0

    # Length penalties intentionally fire only on really thin text — a
    # one-page resume can be ~120-200 words. We don't want to false-reject
    # entry-level / minimal CVs.
    if char_count < 100:
        reasons.append(f"Very short ({char_count} characters) — most resumes have more.")
        neg_score += 0.35
    if word_count < 25:
        reasons.append(f"Very few words ({word_count}).")
        neg_score += 0.25

    web_nav_hits = sum(1 for token in _WEB_NAV_TOKENS if token in lower)
    if web_nav_hits >= 3:
        reasons.append(
            f"{web_nav_hits} website / nav phrases detected (Subscribe, Read more, "
            "Privacy Policy, ...). This looks like a webpage, not a resume."
        )
        neg_score += 0.45
    elif web_nav_hits == 2:
        reasons.append("Website / nav phrases detected — may be a portfolio page, not a resume.")
        neg_score += 0.20

    html_hits = len(_HTML_RESIDUE_RE.findall(text))
    if html_hits >= 3:
        reasons.append("Raw HTML / markup residue detected. Paste plain text instead.")
        neg_score += 0.25

    if _CODE_RESIDUE_RE.search(text):
        reasons.append("Looks like source code rather than a resume.")
        neg_score += 0.30

    # Article / blog-text smell: long uniform paragraphs without bullets / dates.
    long_paragraphs = sum(
        1
        for paragraph in re.split(r"\n\s*\n", text)
        if len(paragraph.split()) > 80
    )
    if (
        long_paragraphs >= 2
        and bullet_lines == 0
        and date_hits == 0
        and section_hits <= 1
    ):
        reasons.append(
            "Reads like an article / essay (long paragraphs, no bullets or dates)."
        )
        neg_score += 0.30

    # Lexical diversity sanity (catches generated repetition / lorem-ipsum).
    long_words = [w for w in lower.split() if len(w) > 3]
    if len(long_words) > 20:
        unique_ratio = len(set(long_words)) / len(long_words)
        if unique_ratio < 0.30:
            reasons.append("Highly repetitive content.")
            neg_score += 0.20

    # ---------- Optional ML signals (advisory) ----------
    if models is not None:
        sections = predict_sections(models, text)
        entities = predict_entities(models, text)
        section_count = len(sections.get("counts", {}))
        entity_count = sum(entities.get("counts", {}).values())
        if section_count >= 2:
            pos_score += 0.10
            positive.append(f"{section_count} learned section types")
        if entity_count >= 3:
            pos_score += 0.10
            positive.append(f"{entity_count} learned resume entities")

    # ---------- Combine ----------
    pos_score = min(1.0, pos_score)
    neg_score = min(1.0, neg_score)
    score = max(0.0, min(1.0, pos_score - 0.6 * neg_score))

    # Strong-resume-signal floor: don't false-reject minimal real CVs.
    # If two clear resume markers co-occur (degree + section header, or a
    # date range + section header), we hold the score at >= 0.30 ("medium")
    # so the UI soft-warns instead of blocking.
    has_degree = _DEGREE_RE.search(text) is not None
    if (
        score < 0.30
        and (
            (has_degree and section_hits >= 1)
            or (date_hits >= 1 and section_hits >= 1)
            or section_hits >= 3
        )
    ):
        score = 0.30

    if score >= 0.55:
        confidence = "high"
    elif score >= 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "is_resume": confidence in {"high", "medium"},
        "confidence": confidence,
        "score": round(score, 2),
        "reasons": reasons,
        "signals": positive,
        "metadata": {
            "char_count": char_count,
            "word_count": word_count,
            "date_hits": date_hits,
            "section_hits": section_hits,
            "bullet_lines": bullet_lines,
        },
    }


def predict_domain(models: PublicAssessmentModels, text: str) -> dict[str, Any]:
    probs = _classifier_probs(
        models.domain_model, hashed_features([text], models.hash_dim)
    )[0]
    order = np.argsort(probs)[::-1][:3]
    return {
        "label": models.domain_labels[int(order[0])],
        "confidence": float(probs[int(order[0])]),
        "top": [
            {"label": models.domain_labels[int(i)], "confidence": float(probs[int(i)])}
            for i in order
        ],
    }


def predict_sections(models: PublicAssessmentModels, text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"counts": {}, "examples": {}, "line_count": 0}
    X = hashed_features(lines, models.hash_dim)
    probs = _classifier_probs(models.section_model, X)
    preds = np.argmax(probs, axis=1)
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for line, pred_idx, row_probs in zip(lines, preds, probs, strict=True):
        label = models.section_labels[int(pred_idx)]
        confidence = float(row_probs[int(pred_idx)])
        if confidence < 0.34:
            continue
        counts[label] = counts.get(label, 0) + 1
        examples.setdefault(label, line[:140])
    return {"counts": counts, "examples": examples, "line_count": len(lines)}


def predict_entities(models: PublicAssessmentModels, text: str) -> dict[str, Any]:
    candidates = _entity_candidates(text)
    if not candidates:
        return {"counts": {}, "examples": {}}
    samples = [sample for sample, _display in candidates]
    X = hashed_features(samples, models.hash_dim)
    probs = _classifier_probs(models.entity_model, X)
    preds = np.argmax(probs, axis=1)
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for (_sample, display), pred_idx, row_probs in zip(
        candidates, preds, probs, strict=True
    ):
        label = models.entity_labels[int(pred_idx)]
        confidence = float(row_probs[int(pred_idx)])
        if label == "UNKNOWN" or confidence < 0.30:
            continue
        counts[label] = counts.get(label, 0) + 1
        examples.setdefault(label, [])
        if len(examples[label]) < 3 and display not in examples[label]:
            examples[label].append(display)
    return {"counts": counts, "examples": examples}


def score_matches_with_ats_model(
    models: PublicAssessmentModels | None,
    resume_text: str,
    matches: pd.DataFrame,
    *,
    weight: float = 0.18,
) -> pd.DataFrame:
    if models is None or matches.empty or "text" not in matches.columns:
        return matches
    pairs = [
        f"{resume_text} SEP {row.get('title', '')} {row.get('experience_level', '')} {row.get('text', '')}"
        for _, row in matches.iterrows()
    ]
    X = ats_pair_features(pairs, models.hash_dim)
    with torch.no_grad():
        pred = models.ats_model(torch.tensor(X)).numpy()
    ats_scores = np.clip(pred * 100.0, 0.0, 100.0)
    adjusted = matches.copy()
    adjusted["public_ats_score"] = np.round(ats_scores, 2)
    adjusted["raw_match_score"] = pd.to_numeric(
        adjusted.get("raw_match_score", adjusted["match_score"]),
        errors="coerce",
    ).fillna(adjusted["match_score"])
    current = pd.to_numeric(adjusted["match_score"], errors="coerce").fillna(0.0)
    adjusted["match_score"] = np.round(
        current * (1.0 - weight) + ats_scores * weight, 2
    )
    return adjusted.sort_values(
        ["match_score", "similarity"], ascending=[False, False]
    ).reset_index(drop=True)


def hashed_features(texts: list[str], dim: int) -> np.ndarray:
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = TOKEN_RE.findall(str(text).lower())
        if not tokens:
            continue
        for token in tokens[:1600]:
            X[row, _stable_hash(token, dim)] += 1.0
        norm = np.linalg.norm(X[row])
        if norm > 1e-6:
            X[row] /= norm
    return X


def ats_pair_features(texts: list[str], dim: int) -> np.ndarray:
    X = np.zeros((len(texts), dim + 8), dtype=np.float32)
    for row, raw in enumerate(texts):
        resume, job = _ats_split(raw)
        resume_tokens = TOKEN_RE.findall(resume.lower())[:1400]
        job_tokens = TOKEN_RE.findall(job.lower())[:1400]
        resume_set = set(resume_tokens)
        job_set = set(job_tokens)
        overlap = resume_set & job_set

        for token in resume_tokens:
            X[row, _stable_hash("r:" + token, dim)] += 1.0
        for token in job_tokens:
            X[row, _stable_hash("j:" + token, dim)] += 1.0
        for token in overlap:
            X[row, _stable_hash("x:" + token, dim)] += 1.5

        base_norm = np.linalg.norm(X[row, :dim])
        if base_norm > 1e-6:
            X[row, :dim] /= base_norm

        resume_len = max(1, len(resume_tokens))
        job_len = max(1, len(job_tokens))
        union_len = max(1, len(resume_set | job_set))
        X[row, dim:] = np.array(
            [
                len(overlap) / max(1, len(job_set)),
                len(overlap) / max(1, len(resume_set)),
                len(overlap) / union_len,
                np.log1p(resume_len) / 10.0,
                np.log1p(job_len) / 10.0,
                min(resume_len / job_len, 5.0) / 5.0,
                _count_year_tokens(resume_tokens) / 10.0,
                _count_year_tokens(job_tokens) / 10.0,
            ],
            dtype=np.float32,
        )
    return X


def _classifier_probs(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X))
        return torch.softmax(logits, dim=1).numpy()


def _entity_candidates(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        lowered = line.lower()
        context = " ".join(lines[max(0, idx - 1) : min(len(lines), idx + 2)])
        if any(
            token in lowered
            for token in ("university", "college", "school", "ph.d", "b.s", "m.s")
        ):
            candidates.append((f"{line} context {context}", line[:120]))
        if any(
            token in lowered
            for token in (
                "engineer",
                "scientist",
                "analyst",
                "manager",
                "designer",
                "teacher",
                "nurse",
                "attorney",
            )
        ):
            candidates.append((f"{line} context {context}", line[:120]))
        if any(
            token in lowered
            for token in (
                "python",
                "sql",
                "java",
                "machine learning",
                "excel",
                "salesforce",
                "tableau",
            )
        ):
            candidates.append((f"{line} context {context}", line[:120]))
        if re.search(r"\b(?:19|20)\d{2}\b", line):
            candidates.append((f"{line} context {context}", line[:120]))
    return candidates[:120]


def _stable_hash(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def _ats_split(text: str) -> tuple[str, str]:
    parts = re.split(r"\s+SEP\s+", str(text), maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    midpoint = max(1, len(str(text)) // 2)
    return str(text)[:midpoint], str(text)[midpoint:]


def _count_year_tokens(tokens: list[str]) -> float:
    count = 0.0
    for idx, token in enumerate(tokens):
        if (
            token.isdigit()
            and 0 <= idx + 1 < len(tokens)
            and tokens[idx + 1] in {"year", "years", "yr", "yrs"}
        ):
            count += float(token)
    return min(count, 30.0)
