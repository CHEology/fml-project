# ResuMatch — Progress Tracker

> Last updated: **2026-04-25**

---

## Status Legend

| Icon | Meaning |
|------|---------|
| ⬜ | Not started |
| 🟡 | In progress |
| ✅ | Complete |
| 🔴 | Blocked |

---

## Phase 1 — Data Ingestion & Preprocessing

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 1.1 | Download raw Kaggle data to `data/raw/` | — | 🔴 | Missing locally. Download `arshkon/linkedin-job-postings` from Kaggle; see `data/README.md`. |
| 1.2 | Join CSVs (`scripts/preprocess_data.py`) | — | ✅ | Implemented and tested with synthetic CSV layout. |
| 1.3 | Normalize salary to annual | — | ✅ | Implemented for hourly/daily/weekly/biweekly/monthly/yearly variants. |
| 1.4 | Clean text fields (strip HTML, combine columns) | — | ✅ | Produces embedding-ready `text` column. |
| 1.5 | Feature engineering (ordinals, one-hot, location) | — | ✅ | Experience ordinal, work-type flags, and state extraction implemented. |
| 1.6 | Write `data/processed/jobs.parquet` | — | 🔴 | Code path implemented; blocked until raw Kaggle files are present. |
| 1.7 | EDA notebook (`01_data_exploration.ipynb`) | — | ⬜ | |

**Phase status:** 🟡 Implementation ready; real processed data is blocked on Kaggle download.

---

## Phase 2 — Embedding & Retrieval Pipeline

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 2.1 | Embedding module (`ml/embeddings.py`) | @ohortig | ✅ | `Encoder` wraps sentence-transformers, returns float32 L2-normalized vectors, and has mocked unit tests. |
| 2.2 | Batch embed all jobs (`scripts/build_index.py`) | Ryan | ✅ | Complete. `--smoke` flag generates synthetic data without Phase 1; real path lazy-imports `Encoder` from Task 2.1. |
| 2.3 | Build FAISS index → `models/jobs.index` | Ryan | ✅ | Complete. `IndexFlatIP` over L2-normalized vectors. |
| 2.4 | Retrieval module (`ml/retrieval.py`) | Ryan | ✅ | Complete. `Retriever` class + `JobMatch` dataclass with DI; module-level default `search()` still depends on generated model artifacts. |
| 2.5 | Embedding experiments notebook | — | ⬜ | Owned by @trp8625 per design doc. |

**Phase status:** 🟡 In progress (embedding/retrieval code complete; 2.5 pending) · **Blocked by:** Phase 1 for real-data embedding/index validation.

---

## Phase 3 — K-Means Clustering

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 3.1 | K-Means from scratch (`ml/clustering.py`) | — | ✅ | NumPy implementation with unit tests. **No sklearn KMeans.** |
| 3.2 | Choose K (elbow + silhouette) | — | ⬜ | |
| 3.3 | Auto-label clusters (TF-IDF top terms) | — | ⬜ | |
| 3.4 | Assign user embedding to cluster | — | ⬜ | |

**Phase status:** 🟡 Core algorithm implemented; K selection/labels need real embeddings.

---

## Phase 4 — Quantile Regression (Salary Prediction)

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 4.1 | `SalaryQuantileNet` architecture (`ml/salary_model.py`) | Alan | ✅ | Complete. Added `SalaryScaler` for target normalization. **Raw PyTorch.** |
| 4.2 | `SalaryDataset` + DataLoader | Alan | ✅ | Complete |
| 4.3 | Training loop (`scripts/train_salary_model.py`) | Alan | ✅ | Complete. Features early stopping & scaler saving. |
| 4.4 | Inference API (`predict_salary()`) | Alan | ✅ | Complete. Returns monotonic quantiles. |
| 4.5 | Evaluation notebook (`03_salary_regression.ipynb`) | Alan | ✅ | Complete (evaluated with synthetic data) |

**Phase status:** ✅ Complete (Alan) · **Blocked by:** Waiting on Phase 2 (embeddings needed for running with real data)

---

## Phase 5 — Resume Feedback Engine

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 5.1 | Gap analysis (direction vector) | — | ⬜ | |
| 5.2 | Keyword extraction (missing skills) | — | ⬜ | |
| 5.3 | Phrase-level highlighting (strengths vs gaps) | — | ⬜ | |
| 5.4 | Cluster migration advice | — | ⬜ | |

**Phase status:** ⬜ Not started · **Blocked by:** Phases 3 & 4

---

## Phase 6 — Streamlit UI & Integration

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 6.1 | App shell & navigation (`app/app.py`) | — | ✅ | Local Streamlit shell with synthetic fallback data. |
| 6.2 | Resume upload page (`app/pages/01_upload.py`) | — | ⬜ | |
| 6.3 | Job matching page (`app/pages/02_matches.py`) | — | ⬜ | |
| 6.4 | Salary prediction page (`app/pages/03_salary.py`) | — | ⬜ | |
| 6.5 | Market position page (`app/pages/04_market.py`) | — | ⬜ | |
| 6.6 | Resume feedback page (`app/pages/05_feedback.py`) | — | ⬜ | |
| 6.7 | Reusable components (`app/components/`) | — | ⬜ | |

**Phase status:** ⬜ Not started · **Blocked by:** Phases 2–5 (shell can start early)

---

## Cross-Cutting Tasks

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| `pyproject.toml` / `uv.lock` — manage dependencies and virtual environment with uv | — | ✅ | Switched from `requirements.txt` on 2026-04-25 |
| Ruff / pre-commit / GitHub Actions CI | — | ✅ | Formatting, linting, and tests enforced for code directories only |
| `tests/` — pytest coverage for `ml/` | Alan, Ryan, Omer | ✅ | Salary, retrieval, clustering, preprocessing, and embeddings covered. |
| Set random seeds in all scripts | — | ⬜ | |
| `.gitignore` — verify `data/raw/`, `models/` excluded | — | ⬜ | |
| Final report / presentation | — | ⬜ | |

---

## Overall Progress

```
Phase 1  [███████░░░]  70%  ← implementation ready; raw Kaggle data missing
Phase 2  [████████░░]  80%  ← embeddings + retrieval ready; 2.5 pending
Phase 3  [███░░░░░░░]  30%  ← KMeans implemented; K selection/labels pending
Phase 4  [██████████] 100%  ← Alan
Phase 5  [░░░░░░░░░░]   0%
Phase 6  [░░░░░░░░░░]   0%
─────────────────────────
Total    [█████░░░░░]  48%
```

**Current state:** Preprocessing, embeddings, retrieval, clustering core, salary modeling, and a local Streamlit shell are implemented with synthetic/unit-test validation. Real end-to-end runs still need the Kaggle raw files in `data/raw/`, followed by `scripts/preprocess_data.py` and `scripts/build_index.py`.

---

## Changelog

| Date | Update |
|------|--------|
| 2026-04-24 | Initial plan and progress documents created. Repo skeleton reviewed — all source files are empty stubs. |
| 2026-04-24 | Alan starting Phase 4: `ml/salary_model.py`, `scripts/train_salary_model.py`, `notebooks/03_salary_regression.ipynb`, `tests/test_salary_model.py`. |
| 2026-04-24 | Alan completed Phase 4 code, tests, and notebook evaluation (using synthetic data). |
| 2026-04-25 | Switched dependency management from `requirements.txt` to `uv` with `pyproject.toml` and `uv.lock`. |
| 2026-04-25 | Added Ruff formatting/linting, local pre-commit hooks, and GitHub Actions CI for code directories (`app`, `ml`, `scripts`, `tests`). |
| 2026-04-25 | Ryan completed Phase 2 retrieval slice (Tasks 2.2/2.3/2.4): `ml/retrieval.py` with DI-friendly `Retriever`, `scripts/build_index.py` with `--smoke` flag, `tests/test_retrieval.py` (25 tests, 25/25 passing). Synthetic fixture committed under `tests/fixtures/`. |
| 2026-04-25 | Added Kaggle data setup docs in `data/README.md` and expanded README data instructions. Local raw data remains missing. |
| 2026-04-25 | Omer completed Task 2.1: `ml/embeddings.Encoder` with L2-normalized float32 outputs and mocked unit tests in `tests/test_embeddings.py`. |
