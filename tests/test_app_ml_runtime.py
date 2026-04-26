from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from app.ml_runtime import (
    artifact_status,
    cluster_position,
    enrich_retrieval_matches,
    salary_band_from_model,
)
from ml.retrieval import JobMatch
from ml.salary_model import SalaryScaler


def test_artifact_status_tracks_salary_and_cluster_outputs(tmp_path: Path) -> None:
    status_by_path = {item["path"]: item for item in artifact_status(tmp_path)}

    assert "models/salary_model.pt" in status_by_path
    assert "models/salary_model.scaler.json" in status_by_path
    assert "models/kmeans_k8.pkl" in status_by_path
    assert "models/cluster_assignments.npy" in status_by_path
    assert "models/cluster_labels.json" in status_by_path


def test_enrich_retrieval_matches_preserves_faiss_order_and_similarity() -> None:
    jobs = pd.DataFrame(
        {
            "job_id": [10, 20, 30],
            "title": ["Keyword heavy role", "Middle role", "FAISS winner"],
            "company_name": ["A", "B", "C"],
            "salary_annual": [120_000.0, 130_000.0, 140_000.0],
            "location": ["New York, NY", "Austin, TX", "Seattle, WA"],
            "state": ["NY", "TX", "WA"],
            "experience_level": ["Associate", "Mid-Senior level", "Director"],
            "work_type": ["Remote", "Hybrid", "On-site"],
            "text": [
                "python python python sql sql",
                "ordinary posting",
                "semantic nearest neighbor",
            ],
        }
    )
    matches = [
        JobMatch(
            2, 30, "FAISS winner", "C", 140_000.0, "Seattle, WA", "Director", 0.83
        ),
        JobMatch(
            0,
            10,
            "Keyword heavy role",
            "A",
            120_000.0,
            "New York, NY",
            "Associate",
            0.21,
        ),
    ]

    enriched = enrich_retrieval_matches(matches, jobs, top_k=2)

    assert enriched["job_id"].tolist() == [30, 10]
    assert enriched["match_score"].tolist() == [83.0, 21.0]
    assert enriched["similarity"].tolist() == [0.83, 0.21]
    assert enriched.loc[0, "text"] == "semantic nearest neighbor"


def test_enrich_retrieval_matches_applies_location_and_remote_filters() -> None:
    jobs = pd.DataFrame(
        {
            "job_id": [1, 2, 3],
            "title": ["Remote NY", "Remote CA", "Onsite NY"],
            "company_name": ["A", "B", "C"],
            "salary_annual": [1.0, 2.0, 3.0],
            "location": ["New York, NY", "San Francisco, CA", "New York, NY"],
            "state": ["NY", "CA", "NY"],
            "experience_level": ["mid", "mid", "mid"],
            "work_type": ["Remote", "Remote", "On-site"],
            "text": ["a", "b", "c"],
        }
    )
    matches = [
        JobMatch(i, job_id, title, company, salary, location, "mid", similarity)
        for i, (job_id, title, company, salary, location, similarity) in enumerate(
            [
                (1, "Remote NY", "A", 1.0, "New York, NY", 0.9),
                (2, "Remote CA", "B", 2.0, "San Francisco, CA", 0.8),
                (3, "Onsite NY", "C", 3.0, "New York, NY", 0.7),
            ]
        )
    ]

    enriched = enrich_retrieval_matches(
        matches,
        jobs,
        preferred_location="NY",
        remote_only=True,
        top_k=3,
    )

    assert enriched["job_id"].tolist() == [1]


class ConstantSalaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[3.0, 1.0, 5.0, 2.0, 4.0]], device=x.device)


def test_salary_band_from_model_uses_quantile_model_and_scaler() -> None:
    band = salary_band_from_model(
        ConstantSalaryModel(),
        np.ones(4, dtype=np.float32),
        SalaryScaler(mean=100_000.0, std=10_000.0),
    )

    assert band == {
        "q10": 110_000,
        "q25": 120_000,
        "q50": 130_000,
        "q75": 140_000,
        "q90": 150_000,
    }


class FakeKMeans:
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([1])


def test_cluster_position_uses_saved_model_prediction_and_labels() -> None:
    position = cluster_position(
        FakeKMeans(),
        {
            "0": {"label": "General roles", "top_terms": ["ops"]},
            "1": {"label": "Data science roles", "top_terms": ["python", "sql"]},
        },
        np.array([0.75, 0.75], dtype=np.float32),
    )

    assert position["cluster_id"] == 1
    assert position["label"] == "Data science roles"
    assert position["top_terms"] == ["python", "sql"]
    assert position["distance"] < position["next_best_distance"]
