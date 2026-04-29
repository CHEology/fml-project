"""
Resume-quality predictor.

Maps a resume embedding to a 0–100 quality score using a small
**raw PyTorch** MLP (consistent with the course constraint that bans
sklearn for primary models). The training target is the synthetic
`quality_score` produced by `scripts/generate_synthetic_resumes.py`,
so the resulting model is a *proxy* for the generator's own scoring
rule until real labels exist — predictions should be treated as such.

Public API:
    `ResumeQualityModel` — nn.Module with a single regression head.
    `QualityScaler` — z-score scaler over the 0–100 target.
    `QualityDataset`, `split_data` — training utilities.
    `predict_quality(model, embedding)` — vector inference.
    `predict_quality_from_text(model, encoder, text)` — full pipeline,
    also returns a `weakest_dim` derived from engineered side-features
    (not the learned head).
    `quality_features_from_text` / `weakest_dim_from_features` —
    rule-based component breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scripts.generate_synthetic_resumes import (
    MULTI_WORD_SKILLS,
    ROLE_PROFILES,
    quality_label_from_score,
)
from torch.utils.data import Dataset

DEFAULT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
SEED = 42

QUALITY_DIMENSIONS = ("skills", "experience", "projects", "metrics", "typos")
DEFAULT_ONET_SKILLS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "external" / "onet_skills.parquet"
)


# ---------------------------------------------------------------------------
# Quality Scaler (z-score normalisation, mirrors SalaryScaler)
# ---------------------------------------------------------------------------


@dataclass
class QualityScaler:
    """z-score scaler over the 0–100 quality_score target.

    Keeps gradients well-conditioned; the inverse_transform is applied
    at inference so callers see scores back in the original scale.
    """

    mean: float = 0.0
    std: float = 1.0

    def fit(self, scores: np.ndarray) -> QualityScaler:
        self.mean = float(np.mean(scores))
        self.std = float(np.std(scores))
        if self.std < 1e-8:
            self.std = 1.0
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.mean) / self.std

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict[str, float]) -> QualityScaler:
        return cls(mean=float(d["mean"]), std=float(d["std"]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResumeQualityModel(nn.Module):
    """MLP regressor: embedding_dim → 256 → 128 → 64 → 1.

    Single scalar head trained with MSE (or Huber) on the scaled
    quality score. BatchNorm + Dropout for regularisation, matching
    the architectural style of `ml.salary_model.SalaryQuantileNet`.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class QualityDataset(Dataset):
    """(embedding, score) pairs."""

    def __init__(self, embeddings: np.ndarray, scores: np.ndarray):
        assert len(embeddings) == len(scores), "Length mismatch"
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(scores, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def split_data(
    embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = SEED,
    scale: bool = True,
) -> tuple[QualityDataset, QualityDataset, QualityDataset, QualityScaler]:
    """Split embeddings + scores into train/val/test datasets.

    When `labels` is provided, the split is stratified by label so each
    bucket carries the same weak/medium/strong mix. Stratification is
    purely for split balance — labels never enter the regression target.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)

    if labels is not None and len(labels) == n:
        train_idx, val_idx, test_idx = _stratified_split_indices(
            labels, train_frac, val_frac, rng
        )
    else:
        order = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]

    scaler = QualityScaler()
    if scale:
        scaler.fit(scores[train_idx])
        target = scaler.transform(scores)
    else:
        target = scores.astype(np.float32)

    return (
        QualityDataset(embeddings[train_idx], target[train_idx]),
        QualityDataset(embeddings[val_idx], target[val_idx]),
        QualityDataset(embeddings[test_idx], target[test_idx]),
        scaler,
    )


def _stratified_split_indices(
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        bucket = np.where(labels == label)[0]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(bucket[:n_train])
        val_parts.append(bucket[n_train : n_train + n_val])
        test_parts.append(bucket[n_train + n_val :])
    return (
        np.concatenate(train_parts) if train_parts else np.array([], dtype=int),
        np.concatenate(val_parts) if val_parts else np.array([], dtype=int),
        np.concatenate(test_parts) if test_parts else np.array([], dtype=int),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> ResumeQualityModel:
    """Load a trained ResumeQualityModel from a .pt checkpoint."""
    model = ResumeQualityModel(embedding_dim=embedding_dim)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def predict_quality(
    model: ResumeQualityModel,
    resume_embedding: np.ndarray,
    scaler: QualityScaler | None = None,
) -> dict[str, float | str]:
    """Run vector inference and return `{"score": float, "label": str}`.

    Score is clipped to [0, 100] after inverse-scaling. The label is
    derived via `quality_label_from_score` so callers do not need to
    duplicate the cutoffs.
    """
    x = torch.tensor(resume_embedding, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    device = next(model.parameters()).device
    x = x.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            pred = model(x).cpu().numpy()
    finally:
        if was_training:
            model.train()

    if scaler is not None:
        pred = scaler.inverse_transform(pred)
    score = float(np.clip(pred[0], 0.0, 100.0))
    return {"score": round(score, 2), "label": quality_label_from_score(score)}


def predict_quality_from_text(
    model: ResumeQualityModel,
    encoder: Any,
    resume_text: str,
    scaler: QualityScaler | None = None,
) -> dict[str, float | str]:
    """Encode `resume_text` and run inference plus weakest-dimension attribution.

    `encoder` is duck-typed (matches `ml.retrieval.Retriever`) — any object
    exposing `.encode(texts: list[str]) -> np.ndarray` works.

    Returns `{"score": float, "label": str, "weakest_dim": str}`.
    """
    embedding = np.asarray(encoder.encode([resume_text]), dtype=np.float32)
    if embedding.ndim == 2:
        embedding = embedding[0]
    out = predict_quality(model, embedding, scaler=scaler)
    out["weakest_dim"] = weakest_dim_from_features(
        quality_features_from_text(resume_text)
    )
    return out


# ---------------------------------------------------------------------------
# Side-feature attribution (weakest dimension)
# ---------------------------------------------------------------------------


_TYPO_REPLACEMENTS = (
    "experiance",
    "analysys",
    "modle",
    "databse",
    "pipline",
    "campagin",
    "stakehldr",
)

_CLICHE_PHRASES = (
    "hard worker",
    "team player",
    "self starter",
    "self-starter",
    "go getter",
    "go-getter",
    "fast learner",
    "detail oriented",
    "detail-oriented",
    "excellent communication skills",
    "motivated professional",
    "responsible for",
)

_VAGUE_PHRASES = (
    "worked on",
    "helped with",
    "assisted with",
    "various tasks",
    "improved things",
    "improved processes",
    "made improvements",
    "handled duties",
    "participated in",
    "contributed to",
)

_ACTION_VERBS = (
    "built",
    "shipped",
    "launched",
    "led",
    "owned",
    "designed",
    "implemented",
    "automated",
    "reduced",
    "increased",
    "improved",
    "optimized",
    "migrated",
    "trained",
    "mentored",
)

_METRIC_RE = re.compile(
    r"(?:\$\s*)?\b\d+(?:\.\d+)?\s*"
    r"(?:%|k\b|m\b|ms\b|seconds?\b|minutes?\b|hours?\b|days?\b|"
    r"users?\b|customers?\b|requests?\b|models?\b|teams?\b|projects?\b|"
    r"revenue\b|cost\b|latency\b)"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*")
_DATE_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\b\s*(?:-|to|through|until)\s*"
    r"\b(present|current|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def _legacy_quality_features_from_text(text: str) -> dict[str, float]:
    """Heuristic per-dimension scores derived from raw resume text.

    Each value lives on its own scale; only the *relative* gap to the
    dimension's expected ceiling matters in `weakest_dim_from_features`.
    These mirror the components used inside `_quality_score` in the
    synthetic generator so the attribution lines up with the training
    target's structure.
    """
    lowered = text.lower()
    skill_lexicon = _quality_skill_lexicon()
    skill_hits = sum(1 for skill in skill_lexicon if skill.lower() in lowered)

    years_match = re.search(r"(\d+)\s+(?:\+\s*)?years?", lowered)
    years = float(years_match.group(1)) if years_match else 0.0

    project_count = max(
        text.count("\n- "),
        text.count("•"),
        sum(1 for line in text.splitlines() if line.strip().startswith("-")),
    )
    has_metrics = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:%|k\b|ms\b|hours?\b|requests?\b)", lowered)
    )
    typo_count = sum(lowered.count(token) for token in _TYPO_REPLACEMENTS)

    return {
        "skills": float(min(skill_hits, 12)),
        "experience": float(years),
        "projects": float(project_count),
        "metrics": 1.0 if has_metrics else 0.0,
        "typos": float(typo_count),
    }


def _legacy_weakest_dim_from_features(features: dict[str, float]) -> str:
    """Return the dimension with the largest gap-to-target.

    Targets approximate the saturation points used in the generator's
    quality formula: 8 strong skills, 5 years experience, 3 projects,
    metrics present, zero typos. The returned label is one of
    `QUALITY_DIMENSIONS`.
    """
    targets = _RULE_TARGETS
    gaps: dict[str, float] = {}
    for dim, target in targets.items():
        value = float(features.get(dim, 0.0))
        if dim == "typos":
            # Higher typo count = bigger problem.
            gaps[dim] = max(value - target, 0.0) / 3.0
        else:
            gaps[dim] = max(target - value, 0.0) / max(target, 1.0)
    weakest = max(gaps.items(), key=lambda pair: pair[1])
    return weakest[0]


# ---------------------------------------------------------------------------
# Rule-based quality scoring (real-resume-safe; no learned proxy)
# ---------------------------------------------------------------------------


_RULE_TARGETS: dict[str, float] = {
    "skills": 10.0,
    "experience": 6.0,
    "projects": 4.0,
    "metrics": 3.0,
    "typos": 0.0,
}

_RULE_WEIGHTS: dict[str, float] = {
    "skills": 0.40,
    "experience": 0.20,
    "projects": 0.20,
    "metrics": 0.10,
    "typos": 0.10,
}

_TYPO_STEP = 25.0  # each typo subtracts this from the typo-dimension score


def score_resume_quality(text: str) -> dict[str, object]:
    """Rule-based 0–100 quality score that operates directly on text.

    Real-resume-safe: no trained model, no synthetic-formula assumptions
    beyond the per-dimension targets. Computes each component, weights
    them, and returns an interpretable breakdown the UI / feedback
    engine can consume directly.

    Returns a dict with:
        score (float, 0–100), label (str, weak/medium/strong),
        dimension_scores (dict[str, float]) — per-dim score in [0, 100],
        weakest_dim (str) — the dimension with the largest gap,
        strengths (list[str]) — dims with score >= 80,
        gaps (list[str]) — dims with score < 50.
    """
    features = quality_features_from_text(text)
    dim_scores = _dimension_scores(features)
    score = float(sum(dim_scores[d] * _RULE_WEIGHTS[d] for d in QUALITY_DIMENSIONS))
    score = float(np.clip(score, 0.0, 100.0))
    feedback = _feedback_notes(features, dim_scores)
    priority_gap = _priority_gap(dim_scores)
    return {
        "score": round(score, 2),
        "label": quality_label_from_score(score),
        "dimension_scores": {d: round(dim_scores[d], 1) for d in QUALITY_DIMENSIONS},
        "weakest_dim": weakest_dim_from_features(features),
        "priority_gap": priority_gap,
        "strengths": [d for d in QUALITY_DIMENSIONS if dim_scores[d] >= 80.0],
        "gaps": [d for d in QUALITY_DIMENSIONS if dim_scores[d] < 50.0],
        "feedback": feedback,
        "strength_notes": feedback["strengths"],
        "gap_notes": feedback["gaps"],
        "matched_skills": features.get("matched_skills", []),
        "career_issues": features.get("career_issues", []),
        "vague_phrases": features.get("vague_phrases", []),
        "cliche_phrases": features.get("cliche_phrases", []),
    }


def _legacy_dimension_scores(features: dict[str, float]) -> dict[str, float]:
    targets = _RULE_TARGETS
    return {
        "skills": float(
            min(features.get("skills", 0.0) / targets["skills"] * 100.0, 100.0)
        ),
        "experience": float(
            min(features.get("experience", 0.0) / targets["experience"] * 100.0, 100.0)
        ),
        "projects": float(
            min(features.get("projects", 0.0) / targets["projects"] * 100.0, 100.0)
        ),
        "metrics": 100.0 if features.get("metrics", 0.0) >= 1.0 else 0.0,
        "typos": float(max(100.0 - features.get("typos", 0.0) * _TYPO_STEP, 0.0)),
    }


def quality_features_from_text(text: str) -> dict[str, Any]:
    """Heuristic signals used by the real-resume-safe quality scorer."""
    lowered = text.lower()
    matched_skills = _matched_skills(lowered)
    metric_matches = _unique_hits(_METRIC_RE.findall(lowered))
    typo_hits = _matched_phrases(lowered, _TYPO_REPLACEMENTS)
    cliche_hits = _matched_phrases(lowered, _CLICHE_PHRASES)
    vague_hits = _matched_phrases(lowered, _VAGUE_PHRASES)
    years = _max_years_experience(lowered)
    career = _career_progression(lowered, years)

    return {
        "skills": float(len(matched_skills)),
        "experience": float(years),
        "projects": float(_bullet_count(text)),
        "metrics": float(len(metric_matches)),
        "typos": float(len(typo_hits)),
        "matched_skills": matched_skills[:25],
        "metric_count": float(len(metric_matches)),
        "action_verb_count": float(_action_verb_count(lowered)),
        "word_count": float(len(_WORD_RE.findall(text))),
        "cliche_phrases": cliche_hits,
        "vague_phrases": vague_hits,
        "career_issues": career["issues"],
        "career_gap_years": career["max_gap_years"],
        "seniority_level": career["seniority_level"],
    }


def weakest_dim_from_features(features: dict[str, Any]) -> str:
    """Compatibility helper; use `priority_gap` for true weak areas."""
    dim_scores = _dimension_scores(features)
    return min(dim_scores.items(), key=lambda pair: pair[1])[0]


def _dimension_scores(features: dict[str, Any]) -> dict[str, float]:
    targets = _RULE_TARGETS
    writing_penalty = (
        float(features.get("typos", 0.0)) * _TYPO_STEP
        + len(features.get("vague_phrases", [])) * 10.0
        + len(features.get("cliche_phrases", [])) * 8.0
        + max(float(features.get("word_count", 0.0)) - 850.0, 0.0) / 25.0
    )
    return {
        "skills": _diminishing_score(
            float(features.get("skills", 0.0)), targets["skills"]
        ),
        "experience": _diminishing_score(
            float(features.get("experience", 0.0)), targets["experience"]
        ),
        "projects": _diminishing_score(
            float(features.get("projects", 0.0)), targets["projects"]
        ),
        "metrics": _diminishing_score(
            float(features.get("metrics", 0.0)), targets["metrics"]
        ),
        "typos": float(max(100.0 - writing_penalty, 0.0)),
    }


def _diminishing_score(value: float, target: float) -> float:
    if value <= 0.0:
        return 0.0
    scale = max(target / 1.6, 1.0)
    return float(min(100.0 * (1.0 - np.exp(-value / scale)), 100.0))


def _priority_gap(dim_scores: dict[str, float]) -> str | None:
    weak = {dim: score for dim, score in dim_scores.items() if score < 70.0}
    if not weak:
        return None
    return min(weak.items(), key=lambda pair: pair[1])[0]


def _feedback_notes(
    features: dict[str, Any], dim_scores: dict[str, float]
) -> dict[str, list[str]]:
    matched_skills = list(features.get("matched_skills", []))
    metric_count = int(features.get("metric_count", 0.0))
    project_count = int(features.get("projects", 0.0))
    years = float(features.get("experience", 0.0))
    vague = list(features.get("vague_phrases", []))
    cliches = list(features.get("cliche_phrases", []))
    career_issues = list(features.get("career_issues", []))

    strengths: list[str] = []
    gaps: list[str] = []
    if matched_skills:
        skills_preview = ", ".join(matched_skills[:6])
        suffix = (
            "" if len(matched_skills) <= 6 else f", +{len(matched_skills) - 6} more"
        )
        strengths.append(
            f"Matched {len(matched_skills)} domain skills: {skills_preview}{suffix}."
        )
    if years >= 3:
        strengths.append(f"Shows {years:g} years of experience.")
    if project_count >= 3:
        strengths.append(
            f"Uses {project_count} bullet-level accomplishment statements."
        )
    if metric_count >= 2:
        strengths.append(f"Quantifies impact in {metric_count} places.")
    elif metric_count == 1:
        strengths.append("Includes one quantified impact statement.")

    if dim_scores["skills"] < 50.0:
        gaps.append(
            "Add more role-specific skills so the resume reads as domain-targeted."
        )
    if project_count < 3:
        gaps.append("Add at least 3 accomplishment bullets tied to concrete work.")
    if metric_count < 2:
        gaps.append(
            "Quantify more outcomes with numbers such as %, dollars, users, time, or volume."
        )
    if vague:
        gaps.append(f"Replace vague phrasing: {', '.join(vague[:4])}.")
    if cliches:
        gaps.append(f"Replace generic resume cliches: {', '.join(cliches[:4])}.")
    gaps.extend(career_issues)

    return {"strengths": strengths[:5], "gaps": gaps[:8]}


def _phrase_in_text(phrase: str, lowered_text: str) -> bool:
    if not phrase:
        return False
    if re.search(r"\w", phrase[-1]):
        return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered_text))
    return phrase in lowered_text


def _matched_skills(lowered_text: str) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9+#./-]+", lowered_text))
    matched: list[str] = []
    for skill in _quality_skill_lexicon():
        phrase = skill.lower().strip()
        if not phrase:
            continue
        if re.fullmatch(r"[a-z0-9+#./-]+", phrase):
            if phrase in tokens:
                matched.append(skill)
        elif phrase in lowered_text:
            matched.append(skill)
    return matched


def _matched_phrases(lowered_text: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if _phrase_in_text(phrase, lowered_text)]


def _unique_hits(matches: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in matches:
        key = match.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(match.strip())
    return out


def _max_years_experience(lowered_text: str) -> float:
    """Detect years of experience from explicit mentions and date ranges."""
    # 1. Explicit mentions (e.g. "8 years experience")
    explicit_matches = re.findall(r"\b(\d{1,2})\s*(?:\+\s*)?years?\b", lowered_text)
    explicit_max = (
        float(max(int(m) for m in explicit_matches)) if explicit_matches else 0.0
    )

    # 2. Date range duration (e.g. "2019 - 2024")
    current_year = date.today().year
    date_matches = _DATE_RANGE_RE.findall(lowered_text)
    if date_matches:
        starts = [int(m[0]) for m in date_matches]
        ends = [
            current_year if m[1].lower() in {"present", "current"} else int(m[1])
            for m in date_matches
        ]
        # Total span from first start to last end (rough approximation)
        span = float(max(ends) - min(starts))
        # Or sum of individual durations (clamped to avoid overlaps)
        # For simplicity in heuristics, we'll take the max span or a floor of 1.0 per range
        duration = max(span, float(len(date_matches)))
    else:
        duration = 0.0

    # 3. Educational proxies (PhD usually implies 4-6 years of research)
    education_bonus = 0.0
    if "phd" in lowered_text or "doctorate" in lowered_text:
        education_bonus = 4.0
    elif "m.s." in lowered_text or "master" in lowered_text:
        education_bonus = 1.5

    return float(max(explicit_max, duration, education_bonus))


def _bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(("-", "*")))


def _action_verb_count(lowered_text: str) -> int:
    return sum(1 for verb in _ACTION_VERBS if re.search(rf"\b{verb}\b", lowered_text))


def _career_progression(lowered_text: str, years: float) -> dict[str, Any]:
    seniority = _seniority_level(lowered_text)
    gaps = _career_gaps(lowered_text)
    issues: list[str] = []
    if years >= 7.0 and seniority < 3:
        issues.append("At 7+ years, add a senior/lead scope signal if accurate.")
    if gaps:
        max_gap = max(gaps)
        if max_gap >= 2:
            issues.append(f"Explain the largest career gap of about {max_gap:g} years.")
    return {
        "issues": issues,
        "max_gap_years": float(max(gaps) if gaps else 0.0),
        "seniority_level": float(seniority),
    }


def _seniority_level(lowered_text: str) -> int:
    if re.search(r"\b(chief|vp|vice president|director|head of)\b", lowered_text):
        return 5
    if re.search(r"\b(manager|principal|staff|architect)\b", lowered_text):
        return 4
    if re.search(r"\b(senior|sr\.?|lead)\b", lowered_text):
        return 3
    if re.search(
        r"\b(engineer|analyst|specialist|consultant|nurse|accountant)\b", lowered_text
    ):
        return 2
    if re.search(r"\b(junior|jr\.?|intern|associate)\b", lowered_text):
        return 1
    return 0


def _career_gaps(lowered_text: str) -> list[float]:
    ranges: list[tuple[int, int]] = []
    current_year = date.today().year
    for start_raw, end_raw in _DATE_RANGE_RE.findall(lowered_text):
        start = int(start_raw)
        end = (
            current_year if end_raw.lower() in {"present", "current"} else int(end_raw)
        )
        if start <= end:
            ranges.append((start, end))
    ranges.sort()
    gaps: list[float] = []
    last_end: int | None = None
    for start, end in ranges:
        if last_end is not None and start > last_end + 1:
            gaps.append(float(start - last_end))
        last_end = max(last_end or end, end)
    return gaps


def _quality_skill_lexicon(external_path: str | Path | None = None) -> list[str]:
    return list(
        _quality_skill_lexicon_cached(str(external_path or DEFAULT_ONET_SKILLS_PATH))
    )


@lru_cache(maxsize=4)
def _quality_skill_lexicon_cached(path: str) -> tuple[str, ...]:
    skills: list[str] = list(MULTI_WORD_SKILLS)
    for profile in ROLE_PROFILES:
        for skill in profile.core_skills + profile.nice_to_have:
            if skill not in skills:
                skills.append(skill)
    for skill in _load_external_skill_terms(path):
        if skill not in skills:
            skills.append(skill)
    return tuple(skills)


@lru_cache(maxsize=4)
def _load_external_skill_terms(path: str) -> tuple[str, ...]:
    """Load optional O*NET-derived terms, falling back silently if absent."""
    skill_path = Path(path)
    if not skill_path.exists():
        return ()
    try:
        import pandas as pd

        df = pd.read_parquet(skill_path, columns=["skill"])
    except Exception:
        return ()

    seen: set[str] = set()
    terms: list[str] = []
    for raw in df["skill"].dropna().astype(str):
        term = raw.strip()
        key = term.lower()
        if 2 <= len(term) <= 80 and key not in seen:
            seen.add(key)
            terms.append(term)
    return tuple(terms)
