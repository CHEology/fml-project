# Vibecode Context

## Project
ResuMatch — user uploads a resume PDF or pastes resume text (or both), app returns projected salary range (with uncertainty), matching job openings, and job market positioning. Built for NYU CSCI-UA 473 (Fundamentals of Machine Learning) final project.

## Dataset
LinkedIn Job Postings (2023-2024) from Kaggle — `arshkon/linkedin-job-postings`, ~124K postings, ~159 MB zipped / ~531 MB unzipped. Raw files go in `data/raw/` (gitignored), joined and cleaned parquet in `data/processed/` (gitignored).

Actual layout after `kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw --unzip`:

```
data/raw/
├── postings.csv                          # 123,849 rows — main job postings
├── companies/
│   ├── companies.csv                     # company name, size, location
│   ├── company_industries.csv
│   ├── company_specialities.csv
│   └── employee_counts.csv
├── jobs/
│   ├── benefits.csv
│   ├── job_industries.csv
│   ├── job_skills.csv
│   └── salaries.csv                      # extra/alt salary records per job_id
└── mappings/
    ├── industries.csv                    # industry_id → name
    └── skills.csv                        # skill_abr → name
```

Key columns in `postings.csv` (relevant subset): `job_id`, `company_id`, `company_name`, `title`, `description`, `skills_desc`, `min_salary` / `med_salary` / `max_salary`, `pay_period`, `currency`, `normalized_salary`, `compensation_type`, `location`, `formatted_experience_level`, `formatted_work_type`, `work_type`, `remote_allowed`.

## Stack
- **App**: Streamlit (multi-page via `app/pages/`)
- **ML**: PyTorch, sentence-transformers, FAISS
- **Data**: pandas, pyarrow
- **Viz**: Plotly
- **Tests**: pytest

## ML Methods
1. **Embeddings + Retrieval** — encode resume and job descriptions with sentence-transformers, cosine similarity search via FAISS (`ml/embeddings.py`, `ml/retrieval.py`)
2. **Quantile Regression** — predict salary with confidence intervals. MUST be implemented in raw PyTorch (not sklearn). This is a grading requirement. (`ml/salary_model.py`)
3. **K-means Clustering** — cluster job embeddings to show market segments. Implement with numpy/torch, not sklearn. (`ml/clustering.py`)

## Structure
- `ml/` — all ML logic, no Streamlit imports. Must be importable by both `app/` and `scripts/`.
- `app/` — Streamlit UI only. Calls into `ml/` for computation.
- `scripts/` — offline data processing and model training. Run once, not at app startup.
- `tests/` — pytest tests for `ml/` and `scripts/`.
- `models/` — saved `.pt` weights and `.index` files. Gitignored.

## Run
```
uv sync
uv run streamlit run app/app.py
```
