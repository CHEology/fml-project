# Data Setup

This directory is intentionally mostly gitignored. Raw Kaggle CSVs live under
`data/raw/`, and generated training/indexing artifacts live under
`data/processed/`.

## LinkedIn Job Postings

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

## Public Resume Assessment Data

`scripts/train_public_assessment_models.py` expects additional public resume
datasets in fixed local paths. These files are gitignored and must be
downloaded before running that trainer.

Expected layout:

```text
data/raw/
├── public_hf/
│   ├── Resume.csv
│   ├── train.csv
│   ├── validation.csv
│   └── resume.txt
└── public_dataturks/
    ├── traindata.json
    └── testdata.json
```

Create the directories:

```bash
mkdir -p data/raw/public_hf data/raw/public_dataturks
```

Download the Hugging Face-hosted files. `hf` is available through the project
environment after `uv sync` because `sentence-transformers` depends on
`huggingface-hub`.

```bash
uv run hf download Divyaamith/Kaggle-Resume Resume.csv \
    --repo-type dataset \
    --local-dir data/raw/public_hf

uv run hf download 0xnbk/resume-ats-score-v1-en train.csv \
    --repo-type dataset \
    --local-dir data/raw/public_hf

uv run hf download 0xnbk/resume-ats-score-v1-en validation.csv \
    --repo-type dataset \
    --local-dir data/raw/public_hf

uv run hf download ganchengguang/resume_seven_class resume.txt \
    --repo-type dataset \
    --local-dir data/raw/public_hf
```

Download the DataTurks resume NER files from GitHub:

```bash
curl -L \
    https://raw.githubusercontent.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy/master/traindata.json \
    -o data/raw/public_dataturks/traindata.json

curl -L \
    https://raw.githubusercontent.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy/master/testdata.json \
    -o data/raw/public_dataturks/testdata.json
```

Verify the expected files exist:

```bash
ls data/raw/public_hf/Resume.csv \
   data/raw/public_hf/train.csv \
   data/raw/public_hf/validation.csv \
   data/raw/public_hf/resume.txt \
   data/raw/public_dataturks/traindata.json \
   data/raw/public_dataturks/testdata.json
```

Then train the public assessment checkpoints:

```bash
uv run python scripts/train_public_assessment_models.py
```

Expected outputs:

```text
models/public_domain_model.pt
models/public_ats_fit_model.pt
models/public_entity_model.pt
models/public_section_model.pt
models/public_assessment_metrics.json
```

Dataset sources:

- `Divyaamith/Kaggle-Resume` provides `Resume.csv`.
- `0xnbk/resume-ats-score-v1-en` provides `train.csv` and `validation.csv`.
- `ganchengguang/resume_seven_class` provides `resume.txt`.
- `DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy` provides
  `traindata.json` and `testdata.json`.

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

## Optional External Data

These files are also gitignored and can be regenerated locally:

```bash
uv run python scripts/load_onet_skills.py --download
uv run python scripts/load_bls_oews.py --download
```

Expected outputs:

```text
data/external/onet_skills.parquet
data/external/bls_wages.parquet
```

O*NET expands the rule-based quality scorer's skill lexicon beyond the bundled
tech/business terms. BLS OEWS adds SOC-level federal wage bands for real-resume
validation. Both loaders also accept `--input` if the source files were
downloaded manually.

As of April 28, 2026, the loader defaults are O*NET 30.2 and BLS May 2024
OEWS. The May 2025 OEWS release is scheduled for May 15, 2026. If the BLS
website rejects command-line downloads with HTTP 403, download the national zip
from the OEWS tables page in a browser and pass the extracted XLSX via
`scripts/load_bls_oews.py --input`.
