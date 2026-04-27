# ResuMatch — Progress Tracker

> Last updated: **2026-04-26**

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
| 1.7 | EDA notebook (`01_data_exploration.ipynb`) | @ohortig | ✅ | Built raw-data availability checks, schema/missingness, salary, categorical, text-field, and processed-data exploration. Runs safely before Kaggle data is present. |

**Phase status:** 🟡 Implementation and EDA foundation ready; real processed data is blocked on Kaggle download.

---

## Phase 2 — Embedding & Retrieval Pipeline

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 2.1 | Embedding module (`ml/embeddings.py`) | @ohortig | ✅ | `Encoder` wraps sentence-transformers, returns float32 L2-normalized vectors, and has mocked unit tests. |
| 2.2 | Batch embed all jobs (`scripts/build_index.py`) | Ryan | ✅ | Complete. `--smoke` flag generates synthetic data without Phase 1; real path lazy-imports `Encoder` from Task 2.1. |
| 2.3 | Build FAISS index → `models/jobs.index` | Ryan | ✅ | Complete. `IndexFlatIP` over L2-normalized vectors. |
| 2.4 | Retrieval module (`ml/retrieval.py`) | Ryan | ✅ | Complete. `Retriever` class + `JobMatch` dataclass with DI; module-level default `search()` still depends on generated model artifacts. |
| 2.5 | Embedding experiments notebook (`02_embedding_experiments.ipynb`) | @ohortig | ✅ | Implemented artifact checks, embedding quality checks, PCA/t-SNE projections, FAISS row-alignment sanity checks, clustering handoff notes, and opt-in model/retrieval comparisons. |

**Phase status:** ✅ Code and experiment notebook complete. Remaining work is quality tuning and Streamlit integration, not Phase 2 foundation.

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
| 4.5 | Evaluation notebook (`03_salary_regression.ipynb`) | Alan | ✅ | Complete and rerun with real Phase 2 embeddings/salary targets. Reports calibration, interval coverage, residuals, and median MAE in USD. |

**Phase status:** ✅ Complete (Alan) · Real-data notebook run is unblocked by Phase 2 embeddings.

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
| `tests/` — pytest coverage for `ml/` | Alan, Ryan, Omer | ✅ | Salary, retrieval, clustering, preprocessing, embeddings, plus `test_synthetic_resumes.py` (14) + `test_evaluate_retrieval.py` (6) + `test_quality.py` (13) + `test_evaluate_salary.py` (5) + `test_resume_loader.py` (9) + `test_load_real_resumes.py` (5) + `test_validate_on_real_resumes.py` (5) — 107 Ryan/Alan tests on the new surface alone, all passing. |
| Resume-quality predictor (`ml/quality.py`) | Ryan | ✅ | Single-head PyTorch MLP + rule-based `score_resume_quality(text)` for real-resume input; both reported, agreement quantified. Trained on synthetic `quality_score` via `scripts/train_quality_model.py` (`--smoke` for CI). |
| Salary-prediction evaluation (`scripts/evaluate_salary.py`) | Ryan | ✅ | Median MAE, pinball loss, [q10,q90]/[q25,q75] coverage, per-persona breakdown. Now able to run end-to-end since Omer landed Task 2.1. |
| Resume-side salary model (`scripts/train_resume_salary_model.py`) | Ryan | ✅ | Retrains Alan's `SalaryQuantileNet` on `(resume_embedding, source_salary_annual)` pairs to remove the JD↔resume domain shift. Saves to `models/resume_salary_model.pt`. |
| Real-resume ingest (`ml/resume_loader.py`, `scripts/load_real_resumes.py`) | Ryan | ✅ | PDF/.txt/.md/.csv/JSONL ingest with PII redaction, length cap, sample fixture under `tests/fixtures/sample_real_resumes.csv`. |
| Real-resume validation (`scripts/validate_on_real_resumes.py`) | Ryan | ✅ | Rule-based quality + learned-MLP score + Spearman agreement + retrieval stats + self-consistency salary metric (predicted q50 vs. retrieved-median salary). Each section degrades gracefully when artifacts are missing; `--smoke` mode for CI. |
| External occupation data (`scripts/load_onet_skills.py`, `scripts/load_bls_oews.py`, `ml/occupation_router.py`, `ml/wage_bands.py`) | Ryan/Codex | Complete | Optional O*NET lexicon and BLS OEWS wage bands load from `data/external/`, stay gitignored, and enrich quality/real-resume validation when present. |
| Set random seeds in all scripts | — | ⬜ | |
| `.gitignore` — verify `data/raw/`, `models/` excluded | — | Complete | Also excludes `data/external/` while preserving `.gitkeep`. |
| Final report / presentation | — | ⬜ | |

---

## Overall Progress

```
Phase 1  [███████░░░]  70%  ← implementation ready; raw Kaggle data missing
Phase 2  [██████████] 100%  ← embeddings, retrieval, and experiment notebook ready
Phase 3  [███░░░░░░░]  30%  ← KMeans implemented; K selection/labels pending
Phase 4  [██████████] 100%  ← Alan
Phase 5  [░░░░░░░░░░]   0%
Phase 6  [░░░░░░░░░░]   0%
─────────────────────────
Total    [█████░░░░░]  48%
```

**Current state:** Preprocessing, embeddings, retrieval, clustering core, salary modeling, and a local Streamlit shell are implemented with synthetic/unit-test validation. Ryan shipped the resume-quality predictor (`ml/quality.py`), the salary-prediction evaluator (`scripts/evaluate_salary.py`), the multi-hard-negative + persona-sliced retrieval evaluator, and a salary-aware synthetic resume generator — closing the integration gaps between Phase 2, Phase 4, and the upcoming Phase 5 feedback engine. As of 2026-04-27, Ryan also made the pipeline ready for **real resume input**: PDF/text loader with PII redaction, real-corpus normaliser, rule-based quality scorer (model-free, real-resume-safe), resume-side salary model that fixes the JD↔resume domain shift, and a validation harness with self-consistency salary metric. Real end-to-end runs still need the Kaggle raw files in `data/raw/`, followed by `scripts/preprocess_data.py` and `scripts/build_index.py`; Task 2.1 (`ml/embeddings.Encoder`) has now landed via Omer's PR #10.

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
| 2026-04-25 | Omer completed Task 1.7: `notebooks/01_data_exploration.ipynb` with no-data guardrails and exploration cells aligned to `scripts/preprocess_data.py`. |
| 2026-04-26 | Omer completed Task 2.5: `notebooks/02_embedding_experiments.ipynb` with artifact validation, PCA/t-SNE visualization, FAISS alignment checks, clustering handoff notes, and optional model/retrieval comparison cells. |
| 2026-04-26 | Reran `notebooks/02_embedding_experiments.ipynb` with real artifacts and completed `notebooks/03_salary_regression.ipynb` on real embeddings/salaries. Salary q50 MAE: `$22.6K`; median-baseline improvement: `$20.6K`; q90 calibration remains slightly outside the ±5 pp target. |
| 2026-04-26 | Enabled and ran `02_embedding_experiments.ipynb` hand-crafted retrieval checks and MiniLM-vs-mpnet comparison. Both models were available; retrieval returned plausible ML engineer, data analyst, and product manager roles. |
| 2026-04-26 | Resolved `main`/PR #10 conflict in `02_embedding_experiments.ipynb` by preserving Omer's embedding/retrieval/model-comparison work and integrating Tanvi's PCA/elbow/KMeans clustering handoff section. |
| 2026-04-26 | Ryan added the synthetic-data ↔ salary ↔ quality bridge: synthetic resumes now carry `source_salary_annual` / `expected_salary_annual` (persona-multiplier) / `experience_level_ordinal` / `hard_negative_job_ids` (multi-negative ranked list). Skill extraction merges multi-word phrases ("machine learning", "system design") before falling back to delimiter splits. New `ml/quality.py` (PyTorch single-head regressor on `quality_score`), `scripts/train_quality_model.py` (`--smoke`), `scripts/evaluate_salary.py` (median MAE, pinball, calibration coverage, per-persona). `scripts/evaluate_retrieval.py` now accepts `--k-sweep`, generalizes NDCG / discriminative_accuracy to multi-hard-negatives, and emits per-persona / per-quality_label slices. Test suite grew to 84 Ryan/Alan tests; ruff format + lint clean. |
| 2026-04-27 | Ryan made the pipeline real-resume-ready. Added `ml/resume_loader.py` (PDF/.txt/.md/.csv/JSONL ingest with PII redaction + length cap), `scripts/load_real_resumes.py` (corpus normaliser; auto-detects directory vs. tabular input), and `scripts/validate_on_real_resumes.py` (full validation harness reporting rule-based quality, learned-MLP quality + Spearman agreement, retrieval stats, and self-consistency salary metric). Added a rule-based `score_resume_quality(text)` to `ml/quality.py` so real resumes have a model-free path. New `scripts/train_resume_salary_model.py` retrains Alan's `SalaryQuantileNet` on `(resume_embedding, source_salary_annual)` pairs to remove the JD↔resume domain shift (saves to `models/resume_salary_model.pt`). Sample fixture `tests/fixtures/sample_real_resumes.csv` (5 hand-written real-style resumes) lets the new tests run without an external corpus. Test suite grew from 84 to 107 (added test_resume_loader.py, test_load_real_resumes.py, test_validate_on_real_resumes.py, plus new rule-based scorer cases in test_quality.py). |
| 2026-04-27 | Continued the external-data slice without downloading large files. Finished O*NET/BLS plumbing: optional O*NET skill parquet now augments `ml.quality` when present; `ml/occupation_router.py` routes resumes to SOC titles; `ml/wage_bands.py` does exact/family/major-group BLS wage lookup; `scripts/validate_on_real_resumes.py` now reports SOC matches, BLS p10-p90 bands, and per-category quality distributions. Added focused tests and README/data docs. |
