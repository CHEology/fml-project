from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_VERSION = 1
TOP_STATES = 15
BASE_FEATURES = (
    "experience_level_ordinal",
    "work_type_remote",
    "work_type_hybrid",
    "work_type_onsite",
)


def build_job_salary_features(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    top_states = _top_states(df.get("state", pd.Series(index=df.index, dtype=object)))
    metadata = _metadata_for_states(top_states)
    return _build_features(df, metadata), metadata


def build_resume_salary_features(
    df: pd.DataFrame, metadata: dict[str, Any]
) -> np.ndarray:
    return _build_features(df, metadata)


def save_salary_feature_metadata(path: Path, metadata: dict[str, Any]) -> None:
    payload = {
        "version": int(metadata["version"]),
        "feature_names": [str(name) for name in metadata["feature_names"]],
        "top_states": [str(state) for state in metadata["top_states"]],
        "n_features": int(metadata["n_features"]),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_salary_feature_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["feature_names"] = [str(name) for name in payload.get("feature_names", [])]
    payload["top_states"] = [str(state) for state in payload.get("top_states", [])]
    payload["n_features"] = int(payload.get("n_features", 0))
    payload["version"] = int(payload.get("version", FEATURE_VERSION))
    return payload


def _metadata_for_states(top_states: list[str]) -> dict[str, Any]:
    feature_names = list(BASE_FEATURES)
    feature_names.extend(f"state_{state}" for state in top_states)
    feature_names.append("state_other")
    return {
        "version": FEATURE_VERSION,
        "feature_names": feature_names,
        "top_states": top_states,
        "n_features": len(feature_names),
    }


def _build_features(df: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    top_states = [str(state) for state in metadata.get("top_states", [])]
    feature_names = [str(name) for name in metadata.get("feature_names", [])]
    n_rows = len(df)
    features = np.zeros((n_rows, len(feature_names)), dtype=np.float32)
    column_index = {name: idx for idx, name in enumerate(feature_names)}

    for name in BASE_FEATURES:
        if name not in column_index:
            continue
        features[:, column_index[name]] = _numeric_column(df, name)

    if "state_other" in column_index and "state" in df.columns:
        states = df["state"].fillna("").astype(str).str.strip().str.upper()
        other_idx = column_index["state_other"]

        # Vectorized one-hot for top states
        for state in top_states:
            key = f"state_{state}"
            if key in column_index:
                features[:, column_index[key]] = (states == state).astype(np.float32)

        # Vectorized "state_other" (includes non-top states AND empty/missing states)
        is_top = states.isin(top_states)
        features[:, other_idx] = (~is_top).astype(np.float32)

    return features.astype(np.float32, copy=False)


def _numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return values.to_numpy(dtype=np.float32, copy=False)


def _normalized_state_series(series: pd.Series) -> list[str]:
    values = series.fillna("").astype(str).str.strip().str.upper()
    return [value if value else "" for value in values.tolist()]


def _top_states(series: pd.Series) -> list[str]:
    states = _normalized_state_series(series)
    if not states:
        return []
    counts = pd.Series(states)
    counts = counts[counts != ""].value_counts()
    return counts.head(TOP_STATES).index.astype(str).tolist()
