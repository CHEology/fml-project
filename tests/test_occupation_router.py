from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.occupation_router import OccupationRouter  # noqa: E402


def _scratch_dir() -> Path:
    path = Path(".tmp_smoke") / f"occupation-router-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _ToyEncoder:
    dim = 3

    def encode(self, texts):
        vectors = {
            "Software Developers": [1.0, 0.0, 0.0],
            "Registered Nurses": [0.0, 1.0, 0.0],
            "Accountants and Auditors": [0.0, 0.0, 1.0],
            "resume software": [0.9, 0.1, 0.0],
            "resume nurse": [0.0, 1.0, 0.1],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_occupation_router_routes_embedding_to_nearest_title() -> None:
    router = OccupationRouter.from_titles(
        ["15-1252.00", "29-1141.00"],
        ["Software Developers", "Registered Nurses"],
        encoder=_ToyEncoder(),
    )

    matches = router.route(np.array([0.8, 0.2, 0.0], dtype=np.float32), k=2)

    assert [m.soc_code for m in matches] == ["15-1252.00", "29-1141.00"]
    assert matches[0].similarity > matches[1].similarity


def test_occupation_router_routes_text_when_encoder_is_available() -> None:
    router = OccupationRouter.from_titles(
        ["15-1252.00", "29-1141.00"],
        ["Software Developers", "Registered Nurses"],
        encoder=_ToyEncoder(),
    )

    matches = router.route("resume nurse", k=1)

    assert matches[0].soc_code == "29-1141.00"


def test_occupation_router_from_onet_skills_deduplicates_soc() -> None:
    skills_path = _scratch_dir() / "onet_skills.parquet"
    pd.DataFrame(
        {
            "soc_code": ["15-1252.00", "15-1252.00", "29-1141.00"],
            "occupation_title": [
                "Software Developers",
                "Software Developers",
                "Registered Nurses",
            ],
            "skill": ["Python", "SQL", "patient care"],
        }
    ).to_parquet(skills_path, index=False)

    router = OccupationRouter.from_onet_skills(skills_path, _ToyEncoder())

    assert len(router) == 2


def test_occupation_router_rejects_text_without_encoder() -> None:
    router = OccupationRouter(
        ["15-1252.00"],
        ["Software Developers"],
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="requires an encoder"):
        router.route("resume software")
