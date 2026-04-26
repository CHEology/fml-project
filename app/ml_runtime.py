from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml.clustering import KMeans
from ml.embeddings import Encoder
from ml.retrieval import JobMatch, Retriever
from ml.salary_model import SalaryScaler, load_model, predict_salary

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    path: str
    required_for: str


ARTIFACT_SPECS = (
    ArtifactSpec("Processed jobs", "data/processed/jobs.parquet", "data"),
    ArtifactSpec("Salary targets", "data/processed/salaries.npy", "training"),
    ArtifactSpec("Job embeddings", "models/job_embeddings.npy", "retrieval"),
    ArtifactSpec("FAISS index", "models/jobs.index", "retrieval"),
    ArtifactSpec("Retrieval metadata", "models/jobs_meta.parquet", "retrieval"),
    ArtifactSpec("Salary model", "models/salary_model.pt", "salary"),
    ArtifactSpec("Salary scaler", "models/salary_model.scaler.json", "salary"),
    ArtifactSpec("KMeans model", "models/kmeans_k8.pkl", "clustering"),
    ArtifactSpec("Cluster centroids", "models/cluster_centroids.npy", "clustering"),
    ArtifactSpec("Cluster assignments", "models/cluster_assignments.npy", "clustering"),
    ArtifactSpec("Cluster labels", "models/cluster_labels.json", "clustering"),
)


def artifact_status(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    root = Path(project_root)
    return [
        {
            "label": spec.label,
            "path": spec.path,
            "ready": (root / spec.path).exists(),
            "required_for": spec.required_for,
        }
        for spec in ARTIFACT_SPECS
    ]


def artifacts_ready(
    status: list[dict[str, Any]], required_for: str | None = None
) -> bool:
    relevant = [
        item
        for item in status
        if required_for is None or item["required_for"] == required_for
    ]
    return bool(relevant) and all(item["ready"] for item in relevant)


def load_jobs(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    path = Path(project_root) / "data" / "processed" / "jobs.parquet"
    frame = pd.read_parquet(path)
    return _ensure_app_columns(frame)


def load_retriever(project_root: Path = PROJECT_ROOT, encoder: Any | None = None):
    root = Path(project_root)
    index = _read_faiss_index(root / "models" / "jobs.index")
    metadata = pd.read_parquet(root / "models" / "jobs_meta.parquet")
    encoder = encoder if encoder is not None else Encoder()
    return Retriever(encoder, index, metadata), encoder


def load_salary_artifacts(project_root: Path = PROJECT_ROOT):
    root = Path(project_root)
    embeddings = np.load(root / "models" / "job_embeddings.npy", mmap_mode="r")
    model = load_model(
        str(root / "models" / "salary_model.pt"),
        embedding_dim=int(embeddings.shape[1]),
    )
    scaler = load_salary_scaler(root / "models" / "salary_model.scaler.json")
    return model, scaler


def load_salary_scaler(path: Path) -> SalaryScaler:
    with Path(path).open() as f:
        return SalaryScaler.from_state_dict(json.load(f))


def load_cluster_artifacts(project_root: Path = PROJECT_ROOT):
    root = Path(project_root)
    model = KMeans.load(root / "models" / "kmeans_k8.pkl")
    assignments = np.load(root / "models" / "cluster_assignments.npy")
    with (root / "models" / "cluster_labels.json").open() as f:
        labels = json.load(f)
    return model, assignments, labels


def encode_resume(encoder: Any, resume_text: str) -> np.ndarray:
    vector = encoder.encode([resume_text])
    return np.asarray(vector, dtype=np.float32).reshape(1, -1)


def retrieve_matches(
    retriever: Retriever,
    jobs: pd.DataFrame,
    resume_embedding: np.ndarray,
    *,
    preferred_location: str = "Anywhere",
    remote_only: bool = False,
    top_k: int = 6,
    candidate_k: int = 120,
) -> pd.DataFrame:
    matches = retriever.search_by_vector(resume_embedding, k=max(top_k, candidate_k))
    return enrich_retrieval_matches(
        matches,
        jobs,
        preferred_location=preferred_location,
        remote_only=remote_only,
        top_k=top_k,
    )


def enrich_retrieval_matches(
    matches: list[JobMatch],
    jobs: pd.DataFrame,
    *,
    preferred_location: str = "Anywhere",
    remote_only: bool = False,
    top_k: int = 6,
) -> pd.DataFrame:
    jobs = _ensure_app_columns(jobs)
    rows: list[dict[str, Any]] = []

    for match in matches:
        row = _job_row_for_match(match, jobs)
        if row is None:
            row = pd.Series(dtype=object)

        record = row.to_dict()
        record.update(match.to_dict())
        record["similarity"] = float(match.similarity)
        record["match_score"] = round(float(match.similarity) * 100.0, 2)
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=list(jobs.columns) + ["similarity", "match_score"])

    frame = _ensure_app_columns(pd.DataFrame(rows))
    frame = _filter_matches(
        frame,
        preferred_location=preferred_location,
        remote_only=remote_only,
    )
    return frame.head(top_k).reset_index(drop=True)


def salary_band_from_model(
    model: Any,
    resume_embedding: np.ndarray,
    scaler: SalaryScaler | None,
) -> dict[str, int]:
    predictions = predict_salary(
        model, np.asarray(resume_embedding).reshape(-1), scaler=scaler
    )
    return {key: int(round(float(value), -3)) for key, value in predictions.items()}


def cluster_position(
    kmeans_model: Any,
    labels: dict[str, Any],
    resume_embedding: np.ndarray,
) -> dict[str, Any]:
    vector = np.asarray(resume_embedding, dtype=np.float32).reshape(1, -1)
    predicted = int(kmeans_model.predict(vector)[0])

    centroids = np.asarray(kmeans_model.centroids, dtype=np.float32)
    distances = np.linalg.norm(centroids - vector, axis=1)
    ordered = np.argsort(distances)
    next_best = next((int(idx) for idx in ordered if int(idx) != predicted), predicted)

    label_info = labels.get(str(predicted), labels.get(predicted, {}))
    if isinstance(label_info, str):
        label_info = {"label": label_info, "top_terms": []}

    return {
        "cluster_id": predicted,
        "label": label_info.get("label", f"Cluster {predicted}"),
        "top_terms": list(label_info.get("top_terms", [])),
        "size": label_info.get("size"),
        "distance": float(distances[predicted]),
        "next_best_cluster_id": int(next_best),
        "next_best_distance": float(distances[next_best]),
    }


def feedback_terms(
    resume_text: str,
    matches: pd.DataFrame,
    cluster: dict[str, Any] | None,
    *,
    max_terms: int = 6,
) -> list[str]:
    resume_lower = resume_text.lower()
    candidates: list[str] = []

    if cluster:
        candidates.extend(str(term) for term in cluster.get("top_terms", []))

    if "text" in matches.columns and not matches.empty:
        candidates.extend(_top_terms_from_text(matches["text"].head(5).astype(str)))

    seen: set[str] = set()
    missing: list[str] = []
    for term in candidates:
        cleaned = str(term).strip().lower()
        if len(cleaned) < 3 or cleaned in seen:
            continue
        seen.add(cleaned)
        if cleaned not in resume_lower:
            missing.append(cleaned)
        if len(missing) >= max_terms:
            break
    return missing


def _read_faiss_index(path: Path):
    import faiss

    buffer = Path(path).read_bytes()
    return faiss.deserialize_index(np.frombuffer(buffer, dtype=np.uint8))


def _ensure_app_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    defaults: dict[str, Any] = {
        "job_id": np.arange(len(frame), dtype=np.int64),
        "title": "",
        "company_name": "",
        "salary_annual": np.nan,
        "location": "",
        "state": "",
        "experience_level": "",
        "work_type": "",
        "text": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    if frame["state"].astype(str).str.len().eq(0).all():
        frame["state"] = (
            frame["location"].astype(str).str.split(",").str[-1].str.strip()
        )

    if frame["work_type"].astype(str).str.len().eq(0).all():
        if "formatted_work_type" in frame.columns:
            frame["work_type"] = frame["formatted_work_type"].fillna("")
        elif "work_type_remote" in frame.columns:
            frame["work_type"] = np.where(
                frame["work_type_remote"].astype(bool), "Remote", ""
            )

    return frame


def _job_row_for_match(match: JobMatch, jobs: pd.DataFrame) -> pd.Series | None:
    if 0 <= match.row_id < len(jobs):
        row = jobs.iloc[match.row_id]
        if pd.isna(match.job_id) or row.get("job_id") == match.job_id:
            return row

    if match.job_id is not None and "job_id" in jobs.columns:
        matches = jobs.index[jobs["job_id"] == match.job_id].tolist()
        if matches:
            return jobs.iloc[matches[0]]
    return None


def _filter_matches(
    frame: pd.DataFrame,
    *,
    preferred_location: str,
    remote_only: bool,
) -> pd.DataFrame:
    filtered = frame
    if preferred_location and preferred_location != "Anywhere":
        location = (
            filtered["location"]
            .astype(str)
            .str.contains(
                preferred_location,
                case=False,
                na=False,
                regex=False,
            )
        )
        state = (
            filtered["state"]
            .astype(str)
            .str.fullmatch(
                preferred_location,
                case=False,
                na=False,
            )
        )
        filtered = filtered[location | state]

    if remote_only:
        work_type = (
            filtered["work_type"]
            .astype(str)
            .str.contains(
                "remote",
                case=False,
                na=False,
            )
        )
        if "work_type_remote" in filtered.columns:
            work_type = work_type | filtered["work_type_remote"].fillna(0).astype(bool)
        filtered = filtered[work_type]

    return filtered


def _top_terms_from_text(texts: pd.Series, max_terms: int = 12) -> list[str]:
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "have",
        "our",
        "that",
        "the",
        "this",
        "with",
        "you",
        "your",
        "will",
        "work",
        "team",
        "role",
        "experience",
        "skills",
        "job",
    }
    counts: dict[str, int] = {}
    for text in texts:
        for token in str(text).lower().replace("/", " ").replace("-", " ").split():
            cleaned = "".join(ch for ch in token if ch.isalnum())
            if len(cleaned) < 3 or cleaned in stopwords:
                continue
            counts[cleaned] = counts.get(cleaned, 0) + 1
    return [
        term
        for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :max_terms
        ]
    ]
