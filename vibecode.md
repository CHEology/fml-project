# Vibecode Context

## Project
ResuMatch — user uploads a resume PDF or pastes resume text (or both), app returns projected salary range (with uncertainty), matching job openings, and job market positioning. Built for NYU CSCI-UA 473 (Fundamentals of Machine Learning) final project.

## Dataset
LinkedIn Jobs & Skills 2024 from Kaggle (~1.3M rows). Fields include job title, description, skills, salary, location. Raw CSVs go in `data/raw/`, cleaned parquet in `data/processed/`.

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
pip install -r requirements.txt
streamlit run app/app.py
```
