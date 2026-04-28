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
