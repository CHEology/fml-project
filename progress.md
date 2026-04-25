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
| 1.1 | Download raw Kaggle data to `data/raw/` | — | ⬜ | Need Kaggle account; ~1 GB download |
| 1.2 | Join CSVs (`scripts/preprocess_data.py`) | — | ⬜ | |
| 1.3 | Normalize salary to annual | — | ⬜ | |
| 1.4 | Clean text fields (strip HTML, combine columns) | — | ⬜ | |
| 1.5 | Feature engineering (ordinals, one-hot, location) | — | ⬜ | |
| 1.6 | Write `data/processed/jobs.parquet` | — | ⬜ | |
| 1.7 | EDA notebook (`01_data_exploration.ipynb`) | — | ⬜ | |

**Phase status:** ⬜ Not started

---

## Phase 2 — Embedding & Retrieval Pipeline

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 2.1 | Embedding module (`ml/embeddings.py`) | — | ⬜ | File exists (empty). Owned by @ohortig per design doc. |
| 2.2 | Batch embed all jobs (`scripts/build_index.py`) | Ryan | ✅ | Complete. `--smoke` flag generates synthetic data without Phase 1; real path lazy-imports `Encoder` from Task 2.1. |
| 2.3 | Build FAISS index → `models/jobs.index` | Ryan | ✅ | Complete. `IndexFlatIP` over L2-normalized vectors. |
| 2.4 | Retrieval module (`ml/retrieval.py`) | Ryan | ✅ | Complete. `Retriever` class + `JobMatch` dataclass with DI; module-level `search()` deferred until 2.1 lands. |
| 2.5 | Embedding experiments notebook | — | ⬜ | Owned by @trp8625 per design doc. |

**Phase status:** 🟡 In progress (Ryan complete; awaiting 2.1 from @ohortig and 2.5 from @trp8625) · **Blocked by:** Phase 1 for end-to-end real-data validation

---

## Phase 3 — K-Means Clustering

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 3.1 | K-Means from scratch (`ml/clustering.py`) | — | ⬜ | File exists (empty). **Must use NumPy/PyTorch, not sklearn.** |
| 3.2 | Choose K (elbow + silhouette) | — | ⬜ | |
| 3.3 | Auto-label clusters (TF-IDF top terms) | — | ⬜ | |
| 3.4 | Assign user embedding to cluster | — | ⬜ | |

**Phase status:** ⬜ Not started · **Blocked by:** Phase 2

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
| 6.1 | App shell & navigation (`app/app.py`) | — | ⬜ | File exists (empty) |
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
| `requirements.txt` — add `sentence-transformers`, `faiss-cpu`, `pdfplumber`, `pytest` | — | ✅ | Added 2026-04-25 |
| `tests/` — pytest coverage for `ml/` | Alan, Ryan | ✅ | `test_salary_model.py` (25/25) + `test_retrieval.py` (25/25) — full suite 50/50 |
| Set random seeds in all scripts | — | ⬜ | |
| `.gitignore` — verify `data/raw/`, `models/` excluded | — | ⬜ | |
| Final report / presentation | — | ⬜ | |

---

## Overall Progress

```
Phase 1  [░░░░░░░░░░]   0%
Phase 2  [██████░░░░]  60%  ← Ryan (2.2/2.3/2.4); 2.1 @ohortig, 2.5 @trp8625
Phase 3  [░░░░░░░░░░]   0%
Phase 4  [██████████] 100%  ← Alan
Phase 5  [░░░░░░░░░░]   0%
Phase 6  [░░░░░░░░░░]   0%
─────────────────────────
Total    [███░░░░░░░]  27%
```

**Current state:** Phase 4 (Alan) and the Ryan-owned slice of Phase 2 (retrieval module, build-index script, tests) are complete with synthetic-data validation. Real end-to-end runs of `scripts/build_index.py` need Phase 1 (`data/processed/jobs.parquet`) and Task 2.1 (`ml/embeddings.Encoder` from @ohortig). Phases 1, 3, 5, 6 still need owners.

---

## Changelog

| Date | Update |
|------|--------|
| 2026-04-24 | Initial plan and progress documents created. Repo skeleton reviewed — all source files are empty stubs. |
| 2026-04-24 | Alan starting Phase 4: `ml/salary_model.py`, `scripts/train_salary_model.py`, `notebooks/03_salary_regression.ipynb`, `tests/test_salary_model.py`. |
| 2026-04-24 | Alan completed Phase 4 code, tests, and notebook evaluation (using synthetic data). |
| 2026-04-25 | Added missing deps (`sentence-transformers`, `faiss-cpu`, `pdfplumber`, `pytest`) to `requirements.txt`. |
| 2026-04-25 | Ryan completed Phase 2 retrieval slice (Tasks 2.2/2.3/2.4): `ml/retrieval.py` with DI-friendly `Retriever`, `scripts/build_index.py` with `--smoke` flag, `tests/test_retrieval.py` (25 tests, 25/25 passing). Synthetic fixture committed under `tests/fixtures/`. Module-level `search()` convenience deferred until @ohortig lands `ml/embeddings.Encoder` (Task 2.1). |
