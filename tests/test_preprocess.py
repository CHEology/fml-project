"""
Tests for scripts/preprocess_data.py

Run:
    pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.preprocess_data import preprocess_jobs, write_outputs


def test_preprocess_jobs_builds_expected_processed_frame(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    jobs_dir = raw_dir / "jobs"
    company_dir = raw_dir / "company_details"
    jobs_dir.mkdir(parents=True)
    company_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4],
            "company_id": [10, 20, 30, 10],
            "title": ["Data Scientist", "ML Engineer", "Analyst", "Missing Salary"],
            "description": [
                "<p>Build models</p>",
                "Train <b>systems</b>",
                "Uses spreadsheets",
                "No salary information",
            ],
            "med_salary": [50.0, np.nan, 8000.0, np.nan],
            "min_salary": [np.nan, 100000.0, np.nan, np.nan],
            "max_salary": [np.nan, 140000.0, np.nan, np.nan],
            "pay_period": ["hourly", "YEARLY", "monthly", np.nan],
            "location": [
                "New York, NY",
                "San Francisco, CA",
                "Austin, TX",
                "New York, NY",
            ],
            "formatted_experience_level": [
                "Entry level",
                "Mid-Senior level",
                "Associate",
                "Entry level",
            ],
            "skills_desc": ["Python, SQL, Python", "PyTorch; NLP", "Excel", ""],
            "work_type": ["Remote", "Hybrid", "On-site", "Remote"],
            "formatted_work_type": ["Full-time", "Full-time", "Contract", "Full-time"],
            "remote_allowed": [True, False, False, True],
        }
    ).to_csv(jobs_dir / "postings.csv", index=False)

    pd.DataFrame(
        {
            "company_id": [10, 20],
            "name": ["Acme", "Globex"],
            "description": ["<div>AI company</div>", "Robotics"],
            "company_size": [3, 5],
            "state": ["NY", "CA"],
            "country": ["US", "US"],
        }
    ).to_csv(company_dir / "companies.csv", index=False)

    pd.DataFrame(
        {
            "job_id": [1, 1, 2],
            "type": ["Medical", "Dental", "401k"],
        }
    ).to_csv(jobs_dir / "benefits.csv", index=False)

    pd.DataFrame(
        {
            "company_id": [10, 10, 20],
            "employee_count": [100, 120, 500],
            "follower_count": [1000, 1100, 9000],
            "time_recorded": [100, 200, 150],
        }
    ).to_csv(company_dir / "employee_counts.csv", index=False)

    frame = preprocess_jobs(raw_dir)

    assert len(frame) == 2
    assert list(frame["job_id"]) == [1, 2]
    assert frame["salary_annual"].tolist() == [104000.0, 120000.0]
    assert frame["company_name"].tolist() == ["Acme", "Globex"]
    assert frame["state"].tolist() == ["NY", "CA"]
    assert frame["experience_level_ordinal"].tolist() == [1.0, 3.0]
    assert frame["work_type_remote"].tolist() == [1, 0]
    assert frame["work_type_hybrid"].tolist() == [0, 1]
    assert frame["work_type_onsite"].tolist() == [0, 0]
    assert frame["benefit_count"].tolist() == [2, 1]
    assert frame["benefits_text"].tolist() == ["medical; dental", "401k"]
    assert frame.loc[0, "text"] == "data scientist build models python sql"
    assert frame.loc[1, "text"] == "ml engineer train systems pytorch nlp"

    jobs_out = tmp_path / "processed" / "jobs.parquet"
    salaries_out = tmp_path / "processed" / "salaries.npy"
    write_outputs(frame, jobs_out, salaries_out)

    written = pd.read_parquet(jobs_out)
    salaries = np.load(salaries_out)

    assert written["job_id"].tolist() == [1, 2]
    assert salaries.tolist() == [104000.0, 120000.0]
