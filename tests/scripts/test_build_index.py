from __future__ import annotations

import pandas as pd
from scripts.build_index import write_meta


def test_write_meta_preserves_optional_linkedin_posting_url(tmp_path) -> None:
    out = tmp_path / "jobs_meta.parquet"
    write_meta(
        pd.DataFrame(
            {
                "job_id": [1],
                "title": ["Machine Learning Engineer"],
                "company_name": ["Example AI"],
                "salary_annual": [150000.0],
                "location": ["New York, NY"],
                "experience_level": ["Mid-Senior level"],
                "job_posting_url": ["https://www.linkedin.com/jobs/view/1"],
            }
        ),
        out,
    )

    meta = pd.read_parquet(out)

    assert meta["job_posting_url"].tolist() == ["https://www.linkedin.com/jobs/view/1"]


def test_write_meta_keeps_optional_url_absent_when_input_has_none(tmp_path) -> None:
    out = tmp_path / "jobs_meta.parquet"
    write_meta(
        pd.DataFrame(
            {
                "job_id": [1],
                "title": ["Machine Learning Engineer"],
                "company_name": ["Example AI"],
                "salary_annual": [150000.0],
                "location": ["New York, NY"],
                "experience_level": ["Mid-Senior level"],
            }
        ),
        out,
    )

    meta = pd.read_parquet(out)

    assert "job_posting_url" not in meta.columns
