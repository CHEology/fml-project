from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    path: str
    required_for: str


ARTIFACT_SPECS = (
    ArtifactSpec("LinkedIn postings", "data/raw/postings.csv", "raw_data"),
    ArtifactSpec("LinkedIn companies", "data/raw/companies/companies.csv", "raw_data"),
    ArtifactSpec("Public resumes", "data/raw/public_hf/Resume.csv", "raw_public"),
    ArtifactSpec("Public ATS train", "data/raw/public_hf/train.csv", "raw_public"),
    ArtifactSpec(
        "Public ATS validation", "data/raw/public_hf/validation.csv", "raw_public"
    ),
    ArtifactSpec(
        "Public resume sections", "data/raw/public_hf/resume.txt", "raw_public"
    ),
    ArtifactSpec(
        "Public NER train",
        "data/raw/public_dataturks/traindata.json",
        "raw_public",
    ),
    ArtifactSpec(
        "Public NER test", "data/raw/public_dataturks/testdata.json", "raw_public"
    ),
    ArtifactSpec("Processed jobs", "data/processed/jobs.parquet", "data"),
    ArtifactSpec("Salary targets", "data/processed/salaries.npy", "training"),
    ArtifactSpec("Job embeddings", "models/job_embeddings.npy", "retrieval"),
    ArtifactSpec("FAISS index", "models/jobs.index", "retrieval"),
    ArtifactSpec("Retrieval metadata", "models/jobs_meta.parquet", "retrieval"),
    ArtifactSpec(
        "Resume salary model",
        "models/resume_salary_model.pt",
        "salary_optional",
    ),
    ArtifactSpec(
        "Resume salary scaler",
        "models/resume_salary_model.scaler.json",
        "salary_optional",
    ),
    ArtifactSpec(
        "Resume salary features",
        "models/resume_salary_model.features.json",
        "salary_optional",
    ),
    ArtifactSpec("Quality model", "models/quality_model.pt", "quality"),
    ArtifactSpec("Quality scaler", "models/quality_model.scaler.json", "quality"),
    ArtifactSpec("Salary model", "models/salary_model.pt", "salary"),
    ArtifactSpec("Salary scaler", "models/salary_model.scaler.json", "salary"),
    ArtifactSpec("Salary features", "models/salary_model.features.json", "salary"),
    ArtifactSpec("O*NET skills", "data/external/onet_skills.parquet", "occupation"),
    ArtifactSpec("BLS wage bands", "data/external/bls_wages.parquet", "wage"),
    ArtifactSpec("KMeans model", "models/kmeans_k8.pkl", "clustering"),
    ArtifactSpec("Cluster centroids", "models/cluster_centroids.npy", "clustering"),
    ArtifactSpec("Cluster assignments", "models/cluster_assignments.npy", "clustering"),
    ArtifactSpec("Cluster labels", "models/cluster_labels.json", "clustering"),
    ArtifactSpec(
        "Public assessment metrics",
        "models/public_assessment_metrics.json",
        "public_assessment",
    ),
    ArtifactSpec(
        "Public domain model", "models/public_domain_model.pt", "public_assessment"
    ),
    ArtifactSpec(
        "Public ATS fit model", "models/public_ats_fit_model.pt", "public_assessment"
    ),
    ArtifactSpec(
        "Public entity model", "models/public_entity_model.pt", "public_assessment"
    ),
    ArtifactSpec(
        "Public section model", "models/public_section_model.pt", "public_assessment"
    ),
)


PIPELINE_SETUP_COMMANDS = {
    "raw_data": "Follow data/README.md to download Kaggle LinkedIn Job Postings 2023-2024 into data/raw/.",
    "raw_public": "Follow data/README.md to download the public resume assessment files into data/raw/public_hf/ and data/raw/public_dataturks/.",
    "data": "uv run python scripts/preprocess_data.py",
    "training": "uv run python scripts/preprocess_data.py",
    "retrieval": "uv run python scripts/build_index.py",
    "salary_optional": "uv run python scripts/train_resume_salary_model.py",
    "quality": "uv run python scripts/train_quality_model.py",
    "salary": "uv run python scripts/train_salary_model.py --embeddings models/job_embeddings.npy --salaries data/processed/salaries.npy --jobs-parquet data/processed/jobs.parquet",
    "occupation": "uv run python scripts/load_onet_skills.py --download",
    "wage": "uv run python scripts/load_bls_oews.py --download",
    "clustering": "uv run python scripts/build_clusters.py",
    "public_assessment": "uv run python scripts/train_public_assessment_models.py",
}


IMPORTANT_ARTIFACT_PATHS = {
    "data/processed/jobs.parquet",
    "models/jobs.index",
    "models/resume_salary_model.pt",
    "models/quality_model.pt",
    "models/salary_model.pt",
    "models/kmeans_k8.pkl",
    "models/public_assessment_metrics.json",
    "data/external/onet_skills.parquet",
    "data/external/bls_wages.parquet",
}


def artifact_status(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    root = Path(project_root)
    rows: list[dict[str, Any]] = []
    for spec in ARTIFACT_SPECS:
        path = root / spec.path
        ready = path.exists()
        stat = path.stat() if ready else None
        modified_at = (
            datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            if stat is not None
            else None
        )
        modified_label = (
            f"Created {datetime.fromtimestamp(stat.st_mtime).strftime('%b %d, %Y %I:%M %p')}"
            if stat is not None
            else "Not created"
        )
        rows.append(
            {
                "label": spec.label,
                "path": spec.path,
                "ready": ready,
                "required_for": spec.required_for,
                "size_bytes": stat.st_size if stat is not None else None,
                "modified_at": modified_at,
                "modified_label": modified_label,
                "setup_command": PIPELINE_SETUP_COMMANDS.get(spec.required_for, ""),
                "important": spec.path in IMPORTANT_ARTIFACT_PATHS,
            }
        )
    return rows


def pipeline_readiness(status: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in status:
        groups.setdefault(str(item.get("required_for", "other")), []).append(item)

    group_rows: list[dict[str, Any]] = []
    setup_commands: list[str] = []
    for group, items in sorted(groups.items()):
        ready_count = sum(1 for item in items if item.get("ready"))
        ready = ready_count == len(items)
        command = PIPELINE_SETUP_COMMANDS.get(group, "")
        if not ready and command and command not in setup_commands:
            setup_commands.append(command)
        group_rows.append(
            {
                "group": group,
                "label": group.replace("_", " ").title(),
                "ready": ready,
                "ready_count": ready_count,
                "total_count": len(items),
                "missing": [item for item in items if not item.get("ready")],
                "setup_command": command,
            }
        )

    important_artifacts = [
        item for item in status if item.get("important") and item.get("ready")
    ]
    important_artifacts.sort(
        key=lambda item: str(item.get("modified_at") or ""), reverse=True
    )
    ready_groups = sum(1 for item in group_rows if item["ready"])
    return {
        "fully_established": bool(group_rows) and ready_groups == len(group_rows),
        "ready_groups": ready_groups,
        "total_groups": len(group_rows),
        "groups": group_rows,
        "setup_commands": setup_commands,
        "important_artifacts": important_artifacts,
    }


def artifacts_ready(
    status: list[dict[str, Any]], required_for: str | None = None
) -> bool:
    relevant = [
        item
        for item in status
        if required_for is None or item["required_for"] == required_for
    ]
    return bool(relevant) and all(item["ready"] for item in relevant)
