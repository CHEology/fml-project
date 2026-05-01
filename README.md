# ResuMatch

ResuMatch is a Streamlit machine-learning app for resume market assessment. It
parses a resume, retrieves similar job postings, estimates a salary band,
places the candidate in market clusters, and returns evidence-based feedback.

The app can start with a small sample catalog, but the full demo requires local
data and model artifacts in `data/` and `models/`.

## Quick Start

```bash
uv sync
uv run streamlit run app/app.py
```

Live job links use public no-key job feeds by default. Optional LinkedIn
enrichment uses Serpdog:

```bash
export SERPDOG_API_KEY=your_key
export LINKEDIN_GEO_ID=103644278
uv run streamlit run app/app.py
```

## Data And Models

Place Kaggle's LinkedIn Job Postings 2023-2024 dataset under `data/raw/`.
See `data/README.md` for the expected file layout.

Build the main artifacts in this order:

```bash
uv run python scripts/preprocess_data.py
uv run python scripts/build_index.py
uv run python scripts/build_clusters.py
uv run python scripts/train_resume_salary_model.py
uv run python scripts/train_quality_model.py
uv run python scripts/train_public_assessment_models.py
```

Optional public data loaders:

```bash
uv run python scripts/load_onet_skills.py --download
uv run python scripts/load_bls_oews.py --download
```

Generated datasets, embeddings, FAISS indexes, model checkpoints, and external
downloads are gitignored.

## Evaluation

Generate synthetic resume/job pairs:

```bash
uv run python scripts/generate_synthetic_resumes.py \
    --jobs data/processed/jobs.parquet \
    --n 100 \
    --out data/eval/synthetic_resumes.parquet
```

Evaluate retrieval:

```bash
uv run python scripts/evaluate_retrieval.py
```

Validate on a real-resume corpus:

```bash
uv run python scripts/load_real_resumes.py \
    --input data/raw/resumes/UpdatedResumeDataSet.csv \
    --out data/eval/real_resumes.parquet

uv run python scripts/validate_on_real_resumes.py \
    --resumes data/eval/real_resumes.parquet
```

The validation script skips optional checks when matching artifacts are absent.

## Tests

```bash
uv run ruff format --check app ml scripts tests
uv run ruff check app ml scripts tests
uv run pytest
```

`tests/test_structure.py` also enforces the project rule that Python files stay
under 1,000 lines.

## Project Layout

```text
app/        Streamlit entrypoint, pages, components, styles, and runtime loaders
ml/         ML modules, retrieval, salary models, and resume assessment logic
scripts/    Data processing, artifact generation, training, and evaluation CLIs
tests/      Pytest coverage for app, ML, and script behavior
data/       Local raw, processed, eval, and external data
models/     Local trained models, embeddings, and indexes
notebooks/  Exploration notebooks
```

`app/app.py` is intentionally thin. UI rendering lives in `app/pages/`,
`app/components/`, `app/demo/`, and `app/styles/`; runtime loading belongs in
`app/runtime/`; ML logic belongs in `ml/`.

## Team

- [@ohortig](https://github.com/ohortig)
- [@trp8625](https://github.com/trp8625)
- [@CHEology](https://github.com/CHEology)
- [@ZQH003](https://github.com/ZQH003)
- [@Eliguli712](https://github.com/Eliguli712)
