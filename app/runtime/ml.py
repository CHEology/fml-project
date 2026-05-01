from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml.clustering import KMeans
from ml.embeddings import Encoder
from ml.feedback import compute_gap_analysis
from ml.retrieval import JobMatch, Retriever

from app.runtime import artifacts as artifact_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
artifact_status = artifact_runtime.artifact_status
artifacts_ready = artifact_runtime.artifacts_ready
pipeline_readiness = artifact_runtime.pipeline_readiness


def load_model(path: str, *, embedding_dim: int, n_extra_features: int = 0) -> Any:
    from ml.salary_model import load_model as _load_model

    return _load_model(
        path, embedding_dim=embedding_dim, n_extra_features=n_extra_features
    )


def predict_salary(
    model: Any,
    embedding: np.ndarray,
    *,
    extra_features: np.ndarray | None = None,
    scaler: Any | None,
) -> Any:
    from ml.salary_model import predict_salary as _predict_salary

    return _predict_salary(model, embedding, extra_features, scaler=scaler)


QUANTILE_KEYS = ("q10", "q25", "q50", "q75", "q90")
MIN_RETRIEVED_SALARIES = 3
HIGH_CONFIDENCE_SIMILARITY = 0.45
DISAGREEMENT_RATIO = 0.25


def load_jobs(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    path = Path(project_root) / "data" / "processed" / "jobs.parquet"
    frame = pd.read_parquet(path)
    return _ensure_app_columns(frame)


def load_job_embeddings(project_root: Path = PROJECT_ROOT) -> np.ndarray:
    path = Path(project_root) / "models" / "job_embeddings.npy"
    return np.load(path).astype(np.float32, copy=False)


def load_retriever(project_root: Path = PROJECT_ROOT, encoder: Any | None = None):
    root = Path(project_root)
    index = _read_faiss_index(root / "models" / "jobs.index")
    metadata = pd.read_parquet(root / "models" / "jobs_meta.parquet")
    encoder = encoder if encoder is not None else Encoder()
    return Retriever(encoder, index, metadata), encoder


def load_salary_artifacts(project_root: Path = PROJECT_ROOT):
    from ml.salary_features import load_salary_feature_metadata
    from ml.salary_model import SalaryScaler

    root = Path(project_root)
    model_path, scaler_path = _preferred_salary_paths(root)
    features_path = _preferred_salary_feature_path(root, model_path)
    scaler_state = _read_scaler_state(scaler_path)
    embedding_dim = scaler_state.get("embedding_dim")
    if embedding_dim is None:
        embeddings = np.load(root / "models" / "job_embeddings.npy", mmap_mode="r")
        embedding_dim = int(embeddings.shape[1])
    feature_metadata = None
    n_extra_features = 0
    if features_path is not None and features_path.exists():
        feature_metadata = load_salary_feature_metadata(features_path)
        n_extra_features = int(feature_metadata.get("n_features", 0))
    model = load_model(
        str(model_path),
        embedding_dim=int(embedding_dim),
        n_extra_features=n_extra_features,
    )
    scaler = SalaryScaler.from_state_dict(scaler_state)
    return model, scaler, feature_metadata


def salary_artifacts_ready(project_root: Path = PROJECT_ROOT) -> bool:
    root = Path(project_root)
    try:
        _, scaler_path = _preferred_salary_paths(root)
    except FileNotFoundError:
        return False
    if (root / "models" / "job_embeddings.npy").exists():
        return True
    return "embedding_dim" in _read_scaler_state(scaler_path)


def load_salary_scaler(path: Path) -> Any:
    from ml.salary_model import SalaryScaler

    with Path(path).open() as f:
        return SalaryScaler.from_state_dict(json.load(f))


def load_quality_artifacts(project_root: Path = PROJECT_ROOT):
    from ml.quality import QualityScaler
    from ml.quality import load_model as load_quality_model

    root = Path(project_root)
    scaler_state = _read_scaler_state(root / "models" / "quality_model.scaler.json")
    embedding_dim = int(scaler_state.get("embedding_dim", 384))
    model = load_quality_model(
        str(root / "models" / "quality_model.pt"),
        embedding_dim=embedding_dim,
    )
    scaler = QualityScaler.from_state_dict(scaler_state)
    return model, scaler


def learned_quality_signal(
    model: Any,
    resume_embedding: np.ndarray,
    scaler: Any | None,
) -> dict[str, Any]:
    from ml.quality import predict_quality

    signal = predict_quality(model, np.asarray(resume_embedding).reshape(-1), scaler)
    return {**signal, "source": "quality_model"}


def load_occupation_router(
    project_root: Path = PROJECT_ROOT,
    encoder: Any | None = None,
):
    root = Path(project_root)
    skills_path = root / "data" / "external" / "onet_skills.parquet"
    if not skills_path.exists():
        return None
    from ml.occupation_router import OccupationRouter

    encoder = encoder if encoder is not None else Encoder()
    return OccupationRouter.from_onet_skills(skills_path, encoder)


def load_wage_table(project_root: Path = PROJECT_ROOT):
    root = Path(project_root)
    wages_path = root / "data" / "external" / "bls_wages.parquet"
    if not wages_path.exists():
        return None
    from ml.wage_bands import WageBandTable

    return WageBandTable.from_parquet(wages_path)


def load_cluster_artifacts(project_root: Path = PROJECT_ROOT):
    root = Path(project_root)
    model = KMeans.load(root / "models" / "kmeans_k8.pkl")
    assignments = np.load(root / "models" / "cluster_assignments.npy")
    with (root / "models" / "cluster_labels.json").open() as f:
        labels = json.load(f)
    return model, assignments, labels


def load_public_assessment_artifacts(project_root: Path = PROJECT_ROOT):
    from ml.public_assessment import load_public_assessment_models

    return load_public_assessment_models(Path(project_root))


def public_resume_signals(
    public_models: Any | None, resume_text: str
) -> dict[str, Any]:
    from ml.public_assessment import resume_public_signals

    return resume_public_signals(public_models, resume_text)


def apply_public_ats_fit(
    public_models: Any | None,
    resume_text: str,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    from ml.public_assessment import score_matches_with_ats_model

    return score_matches_with_ats_model(public_models, resume_text, matches)


def validate_resume(
    public_models: Any | None,
    resume_text: str,
) -> dict[str, Any]:
    from ml.public_assessment import validate_resume_quality

    return validate_resume_quality(public_models, resume_text)


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
    target_seniority: str | None = None,
    top_k: int = 6,
    candidate_k: int = 120,
) -> pd.DataFrame:
    matches = retriever.search_by_vector(resume_embedding, k=max(top_k, candidate_k))
    return enrich_retrieval_matches(
        matches,
        jobs,
        preferred_location=preferred_location,
        remote_only=remote_only,
        target_seniority=target_seniority,
        top_k=top_k,
    )


def enrich_retrieval_matches(
    matches: list[JobMatch],
    jobs: pd.DataFrame,
    *,
    preferred_location: str = "Anywhere",
    remote_only: bool = False,
    target_seniority: str | None = None,
    top_k: int = 6,
) -> pd.DataFrame:
    jobs = _ensure_app_columns(jobs)
    rows: list[dict[str, Any]] = []

    for match in matches:
        row = _job_row_for_match(match, jobs)
        if row is None:
            row = pd.Series(dtype=object)

        record = row.to_dict()
        match_record = match.to_dict()
        if not match_record.get("job_posting_url"):
            match_record.pop("job_posting_url", None)
        record.update(match_record)
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
    frame = _apply_seniority_fit(frame, target_seniority)
    return frame.head(top_k).reset_index(drop=True)


def salary_band_from_model(
    model: Any,
    resume_embedding: np.ndarray,
    scaler: Any | None,
    feature_metadata: dict[str, Any] | None = None,
    resume_features: dict[str, Any] | None = None,
) -> dict[str, int]:
    extra_features = None
    if feature_metadata is not None:
        feature_frame = _salary_feature_frame(resume_features or {})
        from ml.salary_features import build_resume_salary_features

        extra_features = build_resume_salary_features(feature_frame, feature_metadata)[
            0
        ]
    predictions = predict_salary(
        model,
        np.asarray(resume_embedding).reshape(-1),
        extra_features=extra_features,
        scaler=scaler,
    )
    return {key: int(round(float(value), -3)) for key, value in predictions.items()}


def hybrid_salary_band(
    matches: pd.DataFrame,
    *,
    neural_band: dict[str, float | int] | None = None,
    bls_band: Any | None = None,
    occupation_match: Any | None = None,
) -> dict[str, Any] | None:
    """Combine retrieved, BLS, and neural salary signals into one evidence band."""
    retrieved = _retrieved_salary_signal(matches)
    bls = _band_from_bls(bls_band)
    neural = _normalized_band(neural_band)

    if retrieved is not None and retrieved["salary_count"] >= MIN_RETRIEVED_SALARIES:
        primary_source = "retrieved_jobs"
        band = retrieved["band"]
    elif bls is not None:
        primary_source = "bls"
        band = bls
    elif neural is not None:
        primary_source = "neural_model"
        band = neural
    else:
        return None

    disagreement = _has_supporting_disagreement(band, neural=neural, bls=bls)
    salary_count = 0 if retrieved is None else retrieved["salary_count"]
    median_similarity = None if retrieved is None else retrieved["median_similarity"]
    evidence: dict[str, Any] = {
        "salary_count": salary_count,
        "median_similarity": median_similarity,
        "model_bls_disagreement": disagreement,
    }
    if occupation_match is not None:
        evidence["soc_code"] = getattr(occupation_match, "soc_code", None)
        evidence["occupation_title"] = getattr(
            occupation_match, "occupation_title", None
        )
        evidence["soc_similarity"] = getattr(occupation_match, "similarity", None)
    if bls is not None:
        evidence["bls_band"] = bls
    if neural is not None:
        evidence["neural_band"] = neural

    return {
        **band,
        "primary_source": primary_source,
        "confidence": _salary_confidence(
            primary_source=primary_source,
            salary_count=salary_count,
            median_similarity=median_similarity,
            disagreement=disagreement,
        ),
        "evidence": evidence,
    }


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

    # Task 5.1: Compute direction vector to centroid
    gap = compute_gap_analysis(vector, centroids[predicted])

    return {
        "cluster_id": predicted,
        "label": label_info.get("label", f"Cluster {predicted}"),
        "top_terms": list(label_info.get("top_terms", [])),
        "size": label_info.get("size"),
        "distance": float(distances[predicted]),
        "direction_vector": gap["direction_vector"],
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


def gap_advice(
    encoder: Encoder,
    direction_vector: np.ndarray,
    candidate_terms: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Translates a direction vector into specific skill/term suggestions.

    Args:
        encoder: The embedding model to use for candidate terms
        direction_vector: The (target - resume) vector
        candidate_terms: List of strings to check for alignment
        top_n: Number of suggestions

    Returns:
        List of (term, score) sorted by alignment
    """
    from ml.feedback import project_vector_to_terms

    # Embed candidate terms to find which ones align with the gap
    term_embeddings = np.asarray(encoder.encode(candidate_terms), dtype=np.float32)
    return project_vector_to_terms(
        direction_vector, term_embeddings, candidate_terms, top_n
    )


def cluster_migration_advice(
    kmeans_model: Any,
    encoder: Encoder,
    resume_embedding: np.ndarray,
    target_cluster_id: int,
    candidate_terms: list[str],
) -> dict[str, Any]:
    """
    Provides advice on how to move from current position to a target cluster.

    Args:
        kmeans_model: The fitted KMeans model
        encoder: The embedding model
        resume_embedding: User's resume embedding
        target_cluster_id: The ID of the cluster to move towards
        candidate_terms: List of terms to suggest

    Returns:
        Dictionary with gap magnitude and suggested terms
    """
    from ml.feedback import compute_gap_analysis

    centroids = np.asarray(kmeans_model.centroids, dtype=np.float32)
    target_centroid = centroids[target_cluster_id]

    gap = compute_gap_analysis(resume_embedding, target_centroid)
    advice = gap_advice(encoder, gap["direction_vector"], candidate_terms)

    return {
        "gap_magnitude": gap["gap_magnitude"],
        "suggestions": advice,
    }


def _preferred_salary_paths(root: Path) -> tuple[Path, Path]:
    resume_model = root / "models" / "resume_salary_model.pt"
    resume_scaler = root / "models" / "resume_salary_model.scaler.json"
    if resume_model.exists() and resume_scaler.exists():
        return resume_model, resume_scaler

    legacy_model = root / "models" / "salary_model.pt"
    legacy_scaler = root / "models" / "salary_model.scaler.json"
    if legacy_model.exists() and legacy_scaler.exists():
        return legacy_model, legacy_scaler

    raise FileNotFoundError("No usable salary model/scaler pair found.")


def _preferred_salary_feature_path(root: Path, model_path: Path) -> Path | None:
    if model_path.name == "resume_salary_model.pt":
        resume_features = root / "models" / "resume_salary_model.features.json"
        if resume_features.exists():
            return resume_features
    legacy_features = root / "models" / "salary_model.features.json"
    if legacy_features.exists():
        return legacy_features
    return None


def _read_scaler_state(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _salary_feature_frame(values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experience_level_ordinal": values.get("experience_level_ordinal", 0.0),
                "work_type_remote": values.get("work_type_remote", 0.0),
                "work_type_hybrid": values.get("work_type_hybrid", 0.0),
                "work_type_onsite": values.get("work_type_onsite", 0.0),
                "state": values.get("state", ""),
            }
        ]
    )


def _retrieved_salary_signal(matches: pd.DataFrame) -> dict[str, Any] | None:
    if matches.empty or "salary_annual" not in matches.columns:
        return None
    salaries = pd.to_numeric(matches["salary_annual"], errors="coerce")
    valid = salaries[np.isfinite(salaries) & (salaries > 0)]
    if valid.empty:
        return None
    quantiles = np.percentile(valid.to_numpy(dtype=np.float64), [10, 25, 50, 75, 90])
    if "similarity" in matches.columns:
        similarities = pd.to_numeric(matches["similarity"], errors="coerce")
    else:
        similarities = pd.Series(dtype=float)
    sim_valid = similarities[np.isfinite(similarities)]
    return {
        "band": _quantile_array_to_band(quantiles),
        "salary_count": int(len(valid)),
        "median_similarity": float(sim_valid.median()) if len(sim_valid) else None,
    }


def _band_from_bls(bls_band: Any | None) -> dict[str, int] | None:
    if bls_band is None:
        return None
    values = {
        "q10": getattr(bls_band, "p10", None),
        "q25": getattr(bls_band, "p25", None),
        "q50": getattr(bls_band, "p50", None),
        "q75": getattr(bls_band, "p75", None),
        "q90": getattr(bls_band, "p90", None),
    }
    return _normalized_band(values)


def _normalized_band(band: dict[str, float | int] | None) -> dict[str, int] | None:
    if not band:
        return None
    try:
        values = [float(band[key]) for key in QUANTILE_KEYS]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(values)):
        return None
    return _quantile_array_to_band(np.sort(np.asarray(values, dtype=np.float64)))


def _quantile_array_to_band(values: np.ndarray) -> dict[str, int]:
    return {
        key: int(round(float(value), -3))
        for key, value in zip(QUANTILE_KEYS, values, strict=True)
    }


def _has_supporting_disagreement(
    primary: dict[str, int],
    *,
    neural: dict[str, int] | None,
    bls: dict[str, int] | None,
) -> bool:
    primary_q50 = max(float(primary["q50"]), 1.0)
    for support in (neural, bls):
        if support is None or support == primary:
            continue
        if abs(float(support["q50"]) - primary_q50) / primary_q50 >= DISAGREEMENT_RATIO:
            return True
    return False


def _salary_confidence(
    *,
    primary_source: str,
    salary_count: int,
    median_similarity: float | None,
    disagreement: bool,
) -> str:
    if primary_source == "retrieved_jobs":
        if (
            salary_count >= 5
            and median_similarity is not None
            and median_similarity >= HIGH_CONFIDENCE_SIMILARITY
            and not disagreement
        ):
            return "high"
        return "medium"
    if primary_source == "bls":
        return "medium" if not disagreement else "low"
    return "low"


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
        "job_posting_url": "",
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


SENIORITY_RANKS = {
    "Intern / Entry": 0,
    "Associate": 1,
    "Mid": 2,
    "Senior": 3,
    "Lead / Executive": 4,
}


def _normalise_seniority(label: str | None) -> int | None:
    if not label:
        return None
    lowered = str(label).lower()
    if any(
        token in lowered
        for token in ("internship", "intern ", "entry", "junior", "jr.")
    ):
        return 0
    if "associate" in lowered:
        return 1
    if any(
        token in lowered for token in ("director", "executive", "vp", "vice president")
    ):
        return 4
    if any(token in lowered for token in ("principal", "staff", "head of", "chief")):
        return 4
    if any(token in lowered for token in ("mid-senior", "senior", "sr.", "sr ")):
        return 3
    if any(token in lowered for token in ("lead", "technical leadership")):
        return 3
    if "mid" in lowered:
        return 2
    return SENIORITY_RANKS.get(str(label))


def _infer_job_seniority(row: pd.Series) -> int | None:
    experience_level = _normalise_seniority(str(row.get("experience_level", "")))
    if experience_level is not None:
        return experience_level

    title_rank = _normalise_seniority(str(row.get("title", "")))
    if title_rank is not None:
        return title_rank

    text = str(row.get("text", ""))[:600]
    return _normalise_seniority(text)


def _seniority_penalty(target_rank: int | None, job_rank: int | None) -> float:
    if target_rank is None or job_rank is None or pd.isna(job_rank):
        return 0.0
    gap = int(job_rank) - target_rank
    if gap == 0:
        return 0.0
    if gap < 0:
        if gap == -1:
            return 6.0 if target_rank <= 1 else 14.0
        if gap == -2:
            return 18.0 if target_rank <= 2 else 28.0
        return 40.0
    if gap == 1:
        return 7.0
    if gap == 2:
        return 18.0
    if gap == 3:
        return 32.0
    return 45.0


def _apply_seniority_fit(
    frame: pd.DataFrame,
    target_seniority: str | None,
) -> pd.DataFrame:
    if frame.empty or not target_seniority:
        return frame

    target_rank = _normalise_seniority(target_seniority)
    if target_rank is None:
        return frame

    adjusted = frame.copy()
    job_ranks = adjusted.apply(_infer_job_seniority, axis=1)
    penalties = job_ranks.map(lambda rank: _seniority_penalty(target_rank, rank))
    adjusted["job_seniority_rank"] = job_ranks
    adjusted["seniority_gap"] = job_ranks.map(
        lambda rank: (
            np.nan if rank is None or pd.isna(rank) else int(rank) - target_rank
        )
    )
    adjusted["seniority_fit"] = adjusted["seniority_gap"].map(_seniority_fit_label)
    adjusted["salary_eligible"] = adjusted["seniority_gap"].map(
        lambda gap: _salary_eligible_for_gap(target_rank, gap)
    )
    adjusted["salary_eligibility_note"] = adjusted["seniority_gap"].map(
        lambda gap: _salary_eligibility_note(target_rank, gap)
    )
    adjusted["seniority_penalty"] = penalties.astype(float)
    adjusted["raw_match_score"] = pd.to_numeric(
        adjusted["match_score"], errors="coerce"
    ).fillna(0.0)
    adjusted["match_score"] = (
        adjusted["raw_match_score"] - adjusted["seniority_penalty"]
    ).clip(lower=0.0)
    return adjusted.sort_values(["match_score", "similarity"], ascending=[False, False])


def _seniority_fit_label(gap: float | int | None) -> str:
    if gap is None or pd.isna(gap):
        return "unknown"
    gap_int = int(gap)
    if gap_int == 0:
        return "level-aligned"
    if gap_int < 0:
        return "below-candidate-level"
    if gap_int == 1:
        return "stretch"
    return "above-candidate-level"


def _salary_eligible_for_gap(target_rank: int, gap: float | int | None) -> bool:
    if gap is None or pd.isna(gap):
        return True
    gap_int = int(gap)
    if gap_int < 0:
        return False
    if gap_int == 0:
        return True
    # Early-career candidates often accept one-level stretch roles. For senior
    # and executive candidates, one-level-up roles overstate market salary.
    return gap_int == 1 and target_rank <= 2


def _salary_eligibility_note(target_rank: int, gap: float | int | None) -> str:
    if gap is None or pd.isna(gap):
        return "included: seniority unknown"
    gap_int = int(gap)
    if gap_int < 0:
        return "excluded: role is below candidate level"
    if gap_int == 0:
        return "included: same seniority"
    if gap_int == 1:
        return (
            "included: one-level stretch"
            if target_rank <= 2
            else "excluded: role is above candidate level"
        )
    return "excluded: role is above candidate level"


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
