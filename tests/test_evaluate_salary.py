from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_salary import (  # noqa: E402
    QUANTILE_KEYS,
    evaluate_salary,
)


def _eval_df(targets: list[float], personas: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resume_id": [f"r{i}" for i in range(len(targets))],
            "source_salary_annual": targets,
            "persona": personas,
        }
    )


def _make_predictor(quants_per_row: list[dict[str, float]]):
    counter = {"i": 0}

    def predictor(_embedding: np.ndarray) -> dict[str, float]:
        out = quants_per_row[counter["i"]]
        counter["i"] += 1
        return dict(out)

    return predictor


def test_evaluate_salary_perfect_predictor_yields_zero_mae() -> None:
    targets = [80_000.0, 120_000.0, 160_000.0]
    personas = ["direct_match", "over_qualified", "under_qualified"]
    df = _eval_df(targets, personas)
    embeddings = np.zeros((3, 4), dtype=np.float32)

    predictor = _make_predictor(
        [
            {"q10": t * 0.8, "q25": t * 0.9, "q50": t, "q75": t * 1.1, "q90": t * 1.2}
            for t in targets
        ]
    )

    metrics, per_row = evaluate_salary(df, embeddings, predictor)

    assert metrics["n"] == 3
    assert metrics["median_mae"] == pytest.approx(0.0)
    assert metrics["coverage_80"] == pytest.approx(1.0)
    assert metrics["coverage_50"] == pytest.approx(1.0)
    assert set(QUANTILE_KEYS).issubset(per_row.columns)
    assert set(metrics["per_persona"].keys()) == set(personas)


def test_evaluate_salary_skips_missing_targets() -> None:
    df = pd.DataFrame(
        {
            "resume_id": ["a", "b", "c"],
            "source_salary_annual": [100_000.0, np.nan, 0.0],
            "persona": ["direct_match", "direct_match", "direct_match"],
        }
    )
    embeddings = np.zeros((3, 4), dtype=np.float32)

    predictor = _make_predictor(
        [
            {
                "q10": 80_000.0,
                "q25": 90_000.0,
                "q50": 100_000.0,
                "q75": 110_000.0,
                "q90": 120_000.0,
            },
        ]
    )

    metrics, per_row = evaluate_salary(df, embeddings, predictor)

    assert metrics["n"] == 1
    assert len(per_row) == 1
    assert per_row.iloc[0]["resume_id"] == "a"


def test_evaluate_salary_coverage_below_target() -> None:
    # All true salaries fall outside the predicted [q10, q90] band.
    df = _eval_df([200_000.0, 220_000.0], ["direct_match", "direct_match"])
    embeddings = np.zeros((2, 4), dtype=np.float32)
    predictor = _make_predictor(
        [
            {
                "q10": 50_000.0,
                "q25": 60_000.0,
                "q50": 70_000.0,
                "q75": 80_000.0,
                "q90": 90_000.0,
            },
            {
                "q10": 50_000.0,
                "q25": 60_000.0,
                "q50": 70_000.0,
                "q75": 80_000.0,
                "q90": 90_000.0,
            },
        ]
    )

    metrics, _per_row = evaluate_salary(df, embeddings, predictor)
    assert metrics["coverage_80"] == pytest.approx(0.0)
    assert metrics["coverage_50"] == pytest.approx(0.0)
    assert metrics["median_mae"] > 0


def test_evaluate_salary_requires_target_column() -> None:
    df = pd.DataFrame({"resume_id": ["a"], "persona": ["direct_match"]})
    embeddings = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="source_salary_annual"):
        evaluate_salary(df, embeddings, lambda _e: dict.fromkeys(QUANTILE_KEYS, 0.0))


def test_evaluate_salary_rejects_mismatched_embeddings() -> None:
    df = _eval_df([100_000.0], ["direct_match"])
    with pytest.raises(ValueError, match="must match embeddings"):
        evaluate_salary(
            df,
            np.zeros((2, 4), dtype=np.float32),
            lambda _e: dict.fromkeys(QUANTILE_KEYS, 0.0),
        )
