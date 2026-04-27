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


def quality_features_from_text(text: str) -> dict[str, float]:
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


def weakest_dim_from_features(features: dict[str, float]) -> str:
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
    "skills": 8.0,
    "experience": 5.0,
    "projects": 3.0,
    "metrics": 1.0,
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
    return {
        "score": round(score, 2),
        "label": quality_label_from_score(score),
        "dimension_scores": {d: round(dim_scores[d], 1) for d in QUALITY_DIMENSIONS},
        "weakest_dim": weakest_dim_from_features(features),
        "strengths": [d for d in QUALITY_DIMENSIONS if dim_scores[d] >= 80.0],
        "gaps": [d for d in QUALITY_DIMENSIONS if dim_scores[d] < 50.0],
    }


def _dimension_scores(features: dict[str, float]) -> dict[str, float]:
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


def _quality_skill_lexicon() -> list[str]:
    skills: list[str] = list(MULTI_WORD_SKILLS)
    for profile in ROLE_PROFILES:
        for skill in profile.core_skills + profile.nice_to_have:
            if skill not in skills:
                skills.append(skill)
    return skills
