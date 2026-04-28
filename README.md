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

### Optional federal occupation data

For broader non-tech coverage, build the optional O*NET skill lexicon and BLS
OEWS wage table. Both sources are public-domain federal data and are written
under `data/external/`, which is gitignored:

```bash
uv run python scripts/load_onet_skills.py --download
uv run python scripts/load_bls_oews.py --download
```

The first command creates `data/external/onet_skills.parquet`, which
`ml.quality.score_resume_quality` uses automatically when present. The second
creates `data/external/bls_wages.parquet`, which the real-resume validation
harness can use with O*NET routing to report SOC-level wage bands.

If BLS blocks scripted downloads with HTTP 403, download `oesm24nat.zip` from
the BLS OEWS tables page in a browser, unzip the national XLSX, then run:

```bash
uv run python scripts/load_bls_oews.py --input data/external/bls/national_M2024_dl.xlsx
```

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

### Real resumes (optional, for validation)

For end-to-end validation against real candidate resumes, download a public corpus into `data/raw/resumes/`. Recommended:
[Kaggle "Updated Resume Dataset"](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (~960 resumes, category labels, no salary).

Then normalise it into the canonical schema:

```bash
uv run python scripts/load_real_resumes.py \
    --input data/raw/resumes/UpdatedResumeDataSet.csv \
    --out   data/eval/real_resumes.parquet
```

The loader auto-detects CSV / parquet / JSONL files, directories of PDFs / TXTs (via `pdfplumber`), redacts emails / phones / URLs, and clamps very long uploads. See `scripts/load_real_resumes.py` for options.

### Real-resume validation harness

Once Task 2.1 (`ml.embeddings.Encoder`) is available, the validation harness exercises the full pipeline on real input:

```bash
uv run python scripts/validate_on_real_resumes.py \
    --resumes data/eval/real_resumes.parquet \
    --index   models/jobs.index \
    --meta    models/jobs_meta.parquet \
    --onet-skills data/external/onet_skills.parquet \
    --bls-wages   data/external/bls_wages.parquet \
    --salary-model  models/resume_salary_model.pt \
    --salary-scaler models/resume_salary_model.scaler.json \
    --quality-model models/quality_model.pt
```

The harness reports rule-based quality (real-resume-safe; uses `ml.quality.score_resume_quality`), human-readable strength/gap notes, the learned MLP score plus its rank correlation with the rule, and a **self-consistency** salary metric: predicted q50 vs. the median salary of the top-k retrieved jobs. Each section degrades gracefully when artifacts are missing; pass `--smoke` to run with deterministic random embeddings.
When O*NET/BLS artifacts are present, it also reports the nearest SOC
occupation, federal p10-p90 wage band, per-category quality distribution, and
retrieval role-family mismatch rate.

> **Calibration caveat.** The learned quality MLP was trained on the synthetic generator's `quality_score` formula and the JD-side salary model has a domain shift on resume embeddings. Treat numbers from these as proxies until real labels exist. The rule-based scorer + self-consistency salary check are what we trust on real input.

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
