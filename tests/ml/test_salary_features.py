from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ml.salary_features import (
    build_job_salary_features,
    build_resume_salary_features,
    load_salary_feature_metadata,
    save_salary_feature_metadata,
)


def test_build_job_salary_features_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "experience_level_ordinal": [1, 3],
            "work_type_remote": [1, 0],
            "work_type_hybrid": [0, 1],
            "work_type_onsite": [0, 0],
            "state": ["NY", "CA"],
        }
    )

    features_a, metadata_a = build_job_salary_features(df)
    features_b, metadata_b = build_job_salary_features(df)

    assert np.allclose(features_a, features_b)
    assert metadata_a == metadata_b


def test_unknown_state_maps_to_state_other() -> None:
    train_df = pd.DataFrame(
        {
            "experience_level_ordinal": [1],
            "work_type_remote": [1],
            "work_type_hybrid": [0],
            "work_type_onsite": [0],
            "state": ["NY"],
        }
    )
    _, metadata = build_job_salary_features(train_df)
    resume_df = pd.DataFrame({"experience_level_ordinal": [2], "state": ["ZZ"]})

    features = build_resume_salary_features(resume_df, metadata)
    state_other_idx = metadata["feature_names"].index("state_other")

    assert features.shape[1] == metadata["n_features"]
    assert features[0, state_other_idx] == 1.0


def test_missing_values_become_zero() -> None:
    df = pd.DataFrame({"experience_level_ordinal": [np.nan], "state": [None]})
    features, metadata = build_job_salary_features(df)
    state_other_idx = metadata["feature_names"].index("state_other")

    assert features.shape == (1, metadata["n_features"])
    assert np.isfinite(features).all()
    assert features[0, 0] == 0.0
    assert features[0, state_other_idx] == 1.0


def test_metadata_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"experience_level_ordinal": [1], "state": ["NY"]})
    _, metadata = build_job_salary_features(df)
    path = tmp_path / "features.json"

    save_salary_feature_metadata(path, metadata)
    restored = load_salary_feature_metadata(path)

    assert restored == metadata
