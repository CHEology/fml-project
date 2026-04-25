# Data Setup

This directory is intentionally mostly gitignored. Raw Kaggle CSVs live under
`data/raw/`, and generated training/indexing artifacts live under
`data/processed/`.

## Source Dataset

Use Kaggle's LinkedIn Job Postings 2023-2024 dataset:

https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

The project proposal, design document, and preprocessing code all assume this
dataset.

## Expected Raw Layout

After downloading and unzipping, `scripts/preprocess_data.py` searches
recursively under `data/raw/` for the required files. The preferred layout is:

```text
data/raw/
├── postings.csv
├── companies/
│   ├── companies.csv
│   ├── company_industries.csv
│   ├── company_specialities.csv
│   └── employee_counts.csv
├── jobs/
│   ├── benefits.csv
│   ├── job_industries.csv
│   ├── job_skills.csv
│   └── salaries.csv
└── mappings/
    ├── industries.csv
    └── skills.csv
```

Required files:

- `postings.csv` or `job_postings.csv`
- `companies.csv`

Optional files used when present:

- `benefits.csv`
- `employee_counts.csv`

## Download

Manual download from Kaggle is fine. Put the unzipped contents in `data/raw/`.

If the Kaggle CLI is installed and authenticated with `~/.kaggle/kaggle.json`,
this command downloads the dataset directly:

```bash
kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw --unzip
```

The Kaggle CLI is not a project dependency, so install/configure it separately
if you want to use this command.

## Build Processed Data

Once the raw files exist:

```bash
uv run python scripts/preprocess_data.py
```

Expected outputs:

```text
data/processed/jobs.parquet
data/processed/salaries.npy
```

Those processed files are then used by:

```bash
uv run python scripts/build_index.py
```

Expected model/index outputs:

```text
models/job_embeddings.npy
models/jobs.index
models/jobs_meta.parquet
```
