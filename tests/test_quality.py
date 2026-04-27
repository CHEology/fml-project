from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.quality import (  # noqa: E402
    QUALITY_DIMENSIONS,
    QualityDataset,
    QualityScaler,
    ResumeQualityModel,
    _load_external_skill_terms,
    predict_quality,
    predict_quality_from_text,
    quality_features_from_text,
    score_resume_quality,
    split_data,
    weakest_dim_from_features,
)
from scripts.generate_synthetic_resumes import (  # noqa: E402
    generate_paired_synthetic_resumes,
    quality_label_from_score,
)


def _toy_jobs():
    import pandas as pd

    return pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4, 5, 6],
            "title": [
                "Data Scientist",
                "Senior Data Scientist",
                "Machine Learning Engineer",
                "Marketing Coordinator",
                "Backend Software Engineer",
                "Data Analyst",
            ],
            "company_name": list("ABCDEF"),
            "location": ["NY"] * 6,
            "experience_level": ["Mid-Senior level"] * 6,
            "experience_level_ordinal": [3.0] * 6,
            "skills_desc": [
                "Python, SQL, statistics, machine learning",
                "Python, SQL, forecasting",
                "Python, PyTorch, Docker, model serving",
                "campaign planning, CRM",
                "Python, REST APIs, PostgreSQL, system design",
                "SQL, Tableau, Excel",
            ],
            "salary_annual": [
                110_000.0,
                165_000.0,
                145_000.0,
                70_000.0,
                130_000.0,
                90_000.0,
            ],
            "min_salary": [
                95_000.0,
                150_000.0,
                130_000.0,
                60_000.0,
                115_000.0,
                80_000.0,
            ],
            "max_salary": [
                125_000.0,
                185_000.0,
                160_000.0,
                80_000.0,
                145_000.0,
                100_000.0,
            ],
            "text": [
                "data scientist python sql statistics machine learning",
                "senior data scientist python sql forecasting",
                "machine learning engineer pytorch docker model serving",
                "marketing coordinator campaign crm",
                "backend software engineer python rest apis postgresql system design",
                "data analyst sql tableau excel",
            ],
        }
    )


@pytest.fixture
def toy_synthetic_resumes():
    return generate_paired_synthetic_resumes(_toy_jobs(), n=40, seed=2026)


def test_resume_quality_model_forward_shape() -> None:
    torch.manual_seed(0)
    model = ResumeQualityModel(embedding_dim=16)
    model.eval()
    out = model(torch.randn(4, 16))
    assert out.shape == (4,)


def test_quality_scaler_round_trip() -> None:
    scaler = QualityScaler().fit(np.array([10.0, 50.0, 90.0]))
    scaled = scaler.transform(np.array([10.0, 50.0, 90.0]))
    assert pytest.approx(scaler.inverse_transform(scaled)) == [10.0, 50.0, 90.0]


def test_split_data_uses_label_stratification(toy_synthetic_resumes) -> None:
    rng = np.random.default_rng(0)
    n = len(toy_synthetic_resumes)
    embeddings = rng.standard_normal((n, 16)).astype(np.float32)
    scores = toy_synthetic_resumes["quality_score"].to_numpy(dtype=np.float32)
    labels = toy_synthetic_resumes["quality_label"].to_numpy()

    train_ds, val_ds, test_ds, scaler = split_data(
        embeddings, scores, labels=labels, seed=7
    )

    assert len(train_ds) + len(val_ds) + len(test_ds) == n
    assert isinstance(train_ds, QualityDataset)
    assert scaler.std > 0.0


def test_predict_quality_returns_label_within_range() -> None:
    torch.manual_seed(0)
    model = ResumeQualityModel(embedding_dim=8)
    model.eval()
    scaler = QualityScaler(mean=60.0, std=20.0)
    embedding = np.zeros(8, dtype=np.float32)
    out = predict_quality(model, embedding, scaler=scaler)
    assert 0.0 <= out["score"] <= 100.0
    assert out["label"] in {"weak", "medium", "strong"}
    assert out["label"] == quality_label_from_score(out["score"])


def test_predict_quality_is_deterministic_with_seed() -> None:
    torch.manual_seed(42)
    model = ResumeQualityModel(embedding_dim=8)
    model.eval()
    scaler = QualityScaler(mean=50.0, std=10.0)
    rng = np.random.default_rng(0)
    embedding = rng.standard_normal(8).astype(np.float32)

    a = predict_quality(model, embedding, scaler=scaler)
    b = predict_quality(model, embedding, scaler=scaler)
    assert a == b


class _FakeEncoder:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        return rng.standard_normal((len(texts), self.dim)).astype(np.float32)


def test_predict_quality_from_text_returns_weakest_dim() -> None:
    torch.manual_seed(1)
    model = ResumeQualityModel(embedding_dim=8)
    model.eval()
    scaler = QualityScaler(mean=50.0, std=10.0)
    encoder = _FakeEncoder(dim=8)

    text = (
        "Candidate 0001\nMid Predictive Modeling Analyst | Remote, US\n\n"
        "Skills\nPython, SQL, machine learning, statistics\n\n"
        "Experience\n- Built retention models that improved precision by 22%.\n"
        "Education\nM.S. Data Science\n"
    )

    out = predict_quality_from_text(model, encoder, text, scaler=scaler)
    assert out["weakest_dim"] in QUALITY_DIMENSIONS
    assert 0.0 <= out["score"] <= 100.0
    assert out["label"] in {"weak", "medium", "strong"}


def test_weakest_dim_picks_largest_gap() -> None:
    # Lots of skills, no projects, no metrics, lots of typos → weakest should
    # be one of the deficient dims, not "skills".
    weak = weakest_dim_from_features(
        {
            "skills": 8.0,
            "experience": 5.0,
            "projects": 0.0,
            "metrics": 0.0,
            "typos": 5.0,
        }
    )
    assert weak in {"projects", "metrics", "typos"}

    # All saturated → still returns one of the documented categories.
    full = weakest_dim_from_features(
        {
            "skills": 8.0,
            "experience": 5.0,
            "projects": 3.0,
            "metrics": 1.0,
            "typos": 0.0,
        }
    )
    assert full in QUALITY_DIMENSIONS


def test_quality_features_from_text_detects_components() -> None:
    text = (
        "Candidate\nSkills\nPython, SQL, machine learning\n\n"
        "Experience\n5 years working on data pipelines.\n"
        "- Built models that improved retention by 23%.\n"
        "- Ran experiance analysys reviews.\n"
    )
    features = quality_features_from_text(text)
    assert features["skills"] >= 3
    assert features["experience"] == 5.0
    assert features["projects"] >= 1
    assert features["metrics"] == 1.0
    assert features["typos"] >= 2


def test_predict_quality_score_increases_after_targeted_finetune() -> None:
    """Sanity check: training on (embedding, score) pairs lets the model
    discriminate strong vs weak resumes well above chance."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 240
    embedding_dim = 16

    # Build a synthetic linear relationship: half the dataset has a
    # systematic +signal, the other half has -signal.
    direction = rng.standard_normal(embedding_dim).astype(np.float32)
    direction /= np.linalg.norm(direction)
    high = (
        rng.standard_normal((n // 2, embedding_dim)).astype(np.float32)
        + 2.0 * direction
    )
    low = (
        rng.standard_normal((n // 2, embedding_dim)).astype(np.float32)
        - 2.0 * direction
    )
    embeddings = np.concatenate([high, low])
    scores = np.concatenate([np.full(n // 2, 80.0), np.full(n // 2, 20.0)]).astype(
        np.float32
    )
    perm = rng.permutation(n)
    embeddings = embeddings[perm]
    scores = scores[perm]

    train_ds, val_ds, _test_ds, scaler = split_data(embeddings, scores, seed=0)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    model = ResumeQualityModel(embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = torch.nn.MSELoss()
    for _ in range(20):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    pred_high = predict_quality(model, high.mean(axis=0), scaler=scaler)["score"]
    pred_low = predict_quality(model, low.mean(axis=0), scaler=scaler)["score"]
    assert pred_high > pred_low + 10.0


# ---------------------------------------------------------------------------
# Rule-based scorer (real-resume-safe)
# ---------------------------------------------------------------------------


_STRONG_RESUME = """\
Jane Doe
Senior Machine Learning Engineer | Remote, US

Summary
- 6 years building production ML systems and recommender services.

Skills
Python, PyTorch, AWS, Docker, model serving, system design, CI/CD,
monitoring, machine learning

Experience
- Shipped a recommender service handling 80K requests per day.
- Reduced retraining time by 35% across 12 production models.
- Built drift monitoring pipelines used by 4 product teams.

Education
M.S. Computer Science
"""

_WEAK_RESUME = """\
Bob Smith
Looking for a job

I have done some work. I want to learn new things and grow in my career.
Please consider me. I am very motivated.
"""

_TYPO_RESUME = """\
Carol Lee
Junior Data Scientist

I have 2 years of experiance in analysys.
- Built a modle for our databse.
- Updated the campagin pipline weekly.

Skills: Python, SQL
"""


def test_score_resume_quality_strong_resume_scores_higher() -> None:
    strong = score_resume_quality(_STRONG_RESUME)
    weak = score_resume_quality(_WEAK_RESUME)

    assert strong["score"] > weak["score"]
    assert strong["score"] >= 60.0
    assert weak["score"] <= 30.0
    assert strong["label"] in {"medium", "strong"}
    assert weak["label"] == "weak"


def test_score_resume_quality_returns_full_breakdown() -> None:
    out = score_resume_quality(_STRONG_RESUME)
    assert set(out["dimension_scores"].keys()) == set(QUALITY_DIMENSIONS)
    for value in out["dimension_scores"].values():
        assert 0.0 <= value <= 100.0
    assert out["weakest_dim"] in QUALITY_DIMENSIONS
    assert isinstance(out["strengths"], list)
    assert isinstance(out["gaps"], list)
    assert "skills" in out["strengths"] or out["dimension_scores"]["skills"] >= 70.0


def test_score_resume_quality_flags_typos() -> None:
    out = score_resume_quality(_TYPO_RESUME)
    assert out["dimension_scores"]["typos"] < 100.0
    # Either typos themselves or low experience/projects should appear in gaps.
    assert out["gaps"], "expected at least one gap dimension for the noisy resume"


def test_score_resume_quality_is_deterministic() -> None:
    a = score_resume_quality(_STRONG_RESUME)
    b = score_resume_quality(_STRONG_RESUME)
    assert a == b


def test_quality_features_can_use_external_onet_lexicon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = Path(".tmp_smoke") / f"quality-onet-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=True)
    skills_path = scratch / "onet_skills.parquet"
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"skill": ["neonatal care"]}).to_parquet(skills_path, index=False)
    monkeypatch.setattr("ml.quality.DEFAULT_ONET_SKILLS_PATH", skills_path)
    _load_external_skill_terms.cache_clear()

    features = quality_features_from_text(
        "Registered nurse with neonatal care experience."
    )

    assert features["skills"] >= 1.0
