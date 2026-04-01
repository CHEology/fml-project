# ResuMatch

Upload a resume (PDF or text) and get projected salary ranges, matching job openings, and job market positioning — powered by ML techniques from NYU CSCI-UA 473.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/app.py
```

## Data

Download the dataset from [Kaggle: LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) and place the contents in `data/raw/`.

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
