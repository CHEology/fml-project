# ResuMatch

Upload a resume (PDF or text) and get projected salary ranges, matching job openings, and job market positioning — powered by ML techniques from NYU CSCI-UA 473.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for package management and the virtual environment. Install `uv` before syncing dependencies.

```bash
uv sync
```

## Run

```bash
uv run streamlit run app/app.py
```

The app can start without real artifacts by using its synthetic fallback data,
but the full demo path needs the generated files described below.

## Test

```bash
uv run pytest
```

## Code Quality

Install the pre-commit hooks once after setup:

```bash
uv run pre-commit install
```

Run the same checks locally before opening a PR:

```bash
uv run ruff format --check app ml scripts tests
uv run ruff check app ml scripts tests
uv run pytest
```

CI runs formatting, linting, and tests against code directories only. Notebooks are excluded from the strict lint/format gate.

## Data And Artifacts

Download the dataset from [Kaggle: LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) and place the unzipped contents in `data/raw/`.

If you use the Kaggle CLI, authenticate it first with `~/.kaggle/kaggle.json`,
then run:

```bash
uv run kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw --unzip
```

Build the processed parquet used by embeddings, retrieval, clustering, and
salary training:

```bash
uv run python scripts/preprocess_data.py
```

Expected processed outputs:

```text
data/processed/jobs.parquet
data/processed/salaries.npy
```

Build the embedding matrix, FAISS index, and retrieval metadata:

```bash
uv run python scripts/build_index.py
```

Expected model/index outputs:

```text
models/job_embeddings.npy
models/jobs.index
models/jobs_meta.parquet
```

These files are intentionally gitignored. Do not commit raw Kaggle data,
processed parquet files, embeddings, FAISS indexes, model checkpoints, or eval
artifacts to GitHub. If a teammate needs prebuilt artifacts, share them through
Drive or regenerate them locally with the commands above.

To sanity-check the index build without Kaggle data, run:

```bash
uv run python scripts/build_index.py --smoke
```

See `data/README.md` for the expected raw layout and additional details.

## Retrieval Evaluation

After building the real index, generate synthetic resume/job pairs and evaluate
retrieval:

```bash
uv run python scripts/generate_synthetic_resumes.py --jobs data/processed/jobs.parquet --n 100 --out data/eval/synthetic_resumes.parquet
uv run python scripts/evaluate_retrieval.py
```

Expected eval outputs:

```text
data/eval/synthetic_resumes.parquet
data/eval/retrieval_metrics.json
data/eval/retrieval_errors.csv
```

Eval artifacts are also gitignored. Current synthetic retrieval metrics should
be treated as diagnostics for improving retrieval quality, not as final model
performance claims.

## Repo Structure

```
app/           Streamlit UI (entry point: app.py)
ml/            ML modules (embeddings, retrieval, salary model, clustering)
scripts/       Data preprocessing and model training
notebooks/     Exploration and experiments
tests/         pytest tests
data/          Raw and processed data (gitignored)
models/        Saved weights and indexes (gitignored)
```

## Team

- [@ohortig](https://github.com/ohortig)
- [@trp8625](https://github.com/trp8625)
- [@CHEology](https://github.com/CHEology)
- [@alanhe1219-web](https://github.com/alanhe1219-web)
- [@Eliguli712](https://github.com/Eliguli712)
