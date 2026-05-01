from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from app.runtime.live_jobs import (
    build_live_job_query,
    exp_level_for_seniority,
    fetch_live_jobs,
    fetch_serpdog_linkedin_jobs,
    parse_arbeitnow_jobs,
    parse_himalayas_jobs,
    parse_remoteok_jobs,
    parse_remotive_jobs,
    parse_serpdog_jobs,
    rank_live_jobs,
)


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "machine learning" in lowered:
                vectors.append([1.0, 0.0])
            elif "python" in lowered:
                vectors.append([0.8, 0.2])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_build_live_job_query_uses_model_outputs_not_resume_text() -> None:
    profile = {
        "track": "Machine Learning",
        "seniority": "Senior",
        "skills_present": ["Python", "PyTorch", "Retrieval", "Ignored Skill"],
    }
    cluster = {
        "label": "Software / Engineering",
        "top_terms": ["vector search", "ranking", "platform", "ignored"],
    }
    matches = pd.DataFrame(
        [
            {
                "title": "Applied Machine Learning Engineer",
                "text": "secret raw resume phrase should not leak",
            }
        ]
    )

    query = build_live_job_query(profile, cluster, matches)

    assert "Machine Learning" in query
    assert "Applied Machine Learning Engineer" in query
    assert "Ignored Skill" not in query
    assert "secret raw resume phrase" not in query
    assert len(query) <= 120


def test_exp_level_for_seniority_maps_app_labels_to_serpdog_values() -> None:
    assert exp_level_for_seniority("Intern / Entry") == "entry_level"
    assert exp_level_for_seniority("Associate") == "associate"
    assert exp_level_for_seniority("Mid") == "mid_senior_level"
    assert exp_level_for_seniority("Senior") == "mid_senior_level"
    assert exp_level_for_seniority("Lead / Executive") == "director"


def test_parse_serpdog_jobs_accepts_expected_shape_and_ignores_bad_rows() -> None:
    frame = parse_serpdog_jobs(
        {
            "job_results": [
                {
                    "job_position": "Machine Learning Engineer",
                    "job_link": "https://www.linkedin.com/jobs/view/ml-engineer-1",
                    "job_id": "1",
                    "company_name": "Example AI",
                    "job_location": "New York, NY",
                    "job_posting_date": "2026-05-01",
                },
                {
                    "job_position": "",
                    "job_link": "https://www.linkedin.com/jobs/view/missing-title",
                },
                {
                    "job_position": "Bad Link",
                    "job_link": "https://example.com/jobs/view/2",
                },
            ]
        }
    )

    assert frame["title"].tolist() == ["Machine Learning Engineer"]
    assert frame.loc[0, "job_link"].startswith("https://www.linkedin.com/jobs/view")
    assert frame.loc[0, "provider_rank"] == 1


def test_fetch_serpdog_jobs_handles_missing_key_and_transport_errors() -> None:
    missing_key = fetch_serpdog_linkedin_jobs("python", api_key="")
    assert missing_key.empty
    assert "SERPDOG_API_KEY" in missing_key.attrs["reason"]

    def failing_open(url: str, timeout: float) -> bytes:
        raise TimeoutError

    failed = fetch_serpdog_linkedin_jobs(
        "python",
        api_key="key",
        opener=failing_open,
    )
    assert failed.empty
    assert "TimeoutError" in failed.attrs["reason"]


def test_fetch_serpdog_jobs_parses_opener_response() -> None:
    def opener(url: str, timeout: float) -> bytes:
        assert "q=python" in url
        assert "geoId=103644278" in url
        return (
            b'{"job_results":[{"job_position":"Python Engineer",'
            b'"job_link":"https://www.linkedin.com/jobs/view/python-engineer-1",'
            b'"job_id":"1","company_name":"Example","job_location":"Remote",'
            b'"job_posting_date":"2026-05-01"}]}'
        )

    frame = fetch_serpdog_linkedin_jobs("python", api_key="key", opener=opener)

    assert frame["title"].tolist() == ["Python Engineer"]


def test_keyless_provider_parsers_normalize_expected_shapes() -> None:
    himalayas = parse_himalayas_jobs(
        {
            "jobs": [
                {
                    "title": "ML Platform Engineer",
                    "companyName": "Himalayas AI",
                    "locationRestrictions": ["United States"],
                    "pubDate": "2026-05-01",
                    "applicationLink": "https://himalayas.app/jobs/ml-platform",
                    "guid": "h1",
                    "description": "<p>Python and retrieval systems</p>",
                }
            ]
        }
    )
    remotive = parse_remotive_jobs(
        {
            "jobs": [
                {
                    "id": 2,
                    "url": "https://remotive.com/remote-jobs/software-dev/ml",
                    "title": "Remote Machine Learning Engineer",
                    "company_name": "Remote AI",
                    "publication_date": "2026-05-01T10:00:00",
                    "candidate_required_location": "Worldwide",
                    "description": "<strong>PyTorch</strong>",
                }
            ]
        }
    )
    arbeitnow = parse_arbeitnow_jobs(
        {
            "data": [
                {
                    "slug": "a3",
                    "url": "https://www.arbeitnow.com/jobs/company-ml-engineer-a3",
                    "title": "Data Engineer",
                    "company_name": "Berlin Data",
                    "location": "Berlin",
                    "created_at": "2026-04-30",
                    "description": "<p>Python pipelines</p>",
                }
            ]
        }
    )
    remoteok = parse_remoteok_jobs(
        [
            {"legal": "notice"},
            {
                "id": 4,
                "url": "https://remoteok.com/remote-jobs/ml-engineer",
                "position": "AI Engineer",
                "company": "RemoteOK AI",
                "location": "USA",
                "date": "2026-04-29",
                "tags": ["python", "ml"],
            },
        ]
    )

    combined = pd.concat([himalayas, remotive, arbeitnow, remoteok])
    assert combined["source"].tolist() == [
        "Himalayas",
        "Remotive",
        "Arbeitnow",
        "RemoteOK",
    ]
    assert combined["job_link"].str.startswith("https://").all()
    assert "PyTorch" in remotive.iloc[0]["snippet"]


def test_fetch_live_jobs_uses_keyless_providers_without_serpdog_key() -> None:
    def opener(url: str, timeout: float) -> bytes:
        if "himalayas.app" in url:
            return (
                b'{"jobs":[{"title":"Machine Learning Engineer",'
                b'"companyName":"Himalayas AI","locationRestrictions":["US"],'
                b'"pubDate":"2026-05-01",'
                b'"applicationLink":"https://himalayas.app/jobs/ml",'
                b'"guid":"h1","description":"Python ranking"}]}'
            )
        if "remotive.com" in url:
            return b'{"jobs":[]}'
        if "arbeitnow.com" in url:
            return (
                b'{"data":[{"title":"Python Backend Engineer",'
                b'"company_name":"Work Co","location":"Remote",'
                b'"created_at":"2026-05-01",'
                b'"url":"https://www.arbeitnow.com/jobs/python-backend",'
                b'"slug":"a1","description":"Python APIs"}]}'
            )
        if "remoteok.com" in url:
            return (
                b'[{"legal":"notice"},{"position":"Ranking Engineer",'
                b'"company":"Remote Co","location":"Remote",'
                b'"date":"2026-05-01",'
                b'"url":"https://remoteok.com/remote-jobs/ranking",'
                b'"id":"r1","tags":["ranking","python"]}]'
            )
        raise AssertionError(url)

    frame = fetch_live_jobs(
        "machine learning python ranking",
        serpdog_key="",
        opener=opener,
    )

    assert frame["source"].tolist() == ["Himalayas", "Arbeitnow", "RemoteOK"]
    assert frame["title"].tolist() == [
        "Machine Learning Engineer",
        "Python Backend Engineer",
        "Ranking Engineer",
    ]


def test_rank_live_jobs_uses_embeddings_overlap_and_freshness() -> None:
    live_jobs = pd.DataFrame(
        [
            {
                "title": "Marketing Manager",
                "company_name": "Brand Co",
                "location": "Remote",
                "posting_date": "2026-04-30",
                "job_link": "https://www.linkedin.com/jobs/view/marketing-1",
                "job_id": "1",
                "provider_rank": 1,
            },
            {
                "title": "Machine Learning Engineer",
                "company_name": "Example AI",
                "location": "New York, NY",
                "posting_date": "2026-04-29",
                "job_link": "https://www.linkedin.com/jobs/view/ml-2",
                "job_id": "2",
                "provider_rank": 2,
            },
        ]
    )

    ranked = rank_live_jobs(
        live_jobs,
        FakeEncoder(),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        "machine learning python",
        today=date(2026, 5, 1),
    )

    assert ranked.iloc[0]["title"] == "Machine Learning Engineer"
    assert ranked.iloc[0]["live_match_score"] > ranked.iloc[1]["live_match_score"]
