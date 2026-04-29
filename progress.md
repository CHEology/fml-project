# ResuMatch — Progress Tracker

> Last updated: **2026-04-29**

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
| 1.1 | Download raw Kaggle data to `data/raw/` | — | ✅ | `postings.csv` (517 MB) + company/job/mapping subdirectories present locally. |
| 1.2 | Join CSVs (`scripts/preprocess_data.py`) | — | ✅ | Implemented and tested with synthetic CSV layout. |
| 1.3 | Normalize salary to annual | — | ✅ | Implemented for hourly/daily/weekly/biweekly/monthly/yearly variants. |
| 1.4 | Clean text fields (strip HTML, combine columns) | — | ✅ | Produces embedding-ready `text` column. |
| 1.5 | Feature engineering (ordinals, one-hot, location) | — | ✅ | Experience ordinal, work-type flags, and state extraction implemented. |
| 1.6 | Write `data/processed/jobs.parquet` | — | ✅ | `data/processed/jobs.parquet` exists (231 MB). |
| 1.7 | EDA notebook (`01_data_exploration.ipynb`) | @ohortig | ✅ | Built raw-data availability checks, schema/missingness, salary, categorical, text-field, and processed-data exploration. Runs safely before Kaggle data is present. |

**Phase status:** ✅ Complete — raw Kaggle data downloaded, preprocessing run, and `jobs.parquet` generated.

---

## Phase 2 — Embedding & Retrieval Pipeline

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 2.1 | Embedding module (`ml/embeddings.py`) | @ohortig | ✅ | `Encoder` wraps sentence-transformers, returns float32 L2-normalized vectors, and has mocked unit tests. |
| 2.2 | Batch embed all jobs (`scripts/build_index.py`) | Ryan | ✅ | Complete. `--smoke` flag generates synthetic data without Phase 1; real path lazy-imports `Encoder` from Task 2.1. `models/job_embeddings.npy` exists (51 MB). |
| 2.3 | Build FAISS index → `models/jobs.index` | Ryan | ✅ | Complete. `IndexFlatIP` over L2-normalized vectors. `models/jobs.index` exists (51 MB). |
| 2.4 | Retrieval module (`ml/retrieval.py`) | Ryan | ✅ | Complete. `Retriever` class + `JobMatch` dataclass with DI; `models/jobs_meta.parquet` exists. |
| 2.5 | Embedding experiments notebook (`02_embedding_experiments.ipynb`) | @ohortig | ✅ | Implemented artifact checks, embedding quality checks, PCA/t-SNE projections, FAISS row-alignment sanity checks, clustering handoff notes, and opt-in model/retrieval comparisons. |

**Phase status:** ✅ Complete — all artifacts generated and verified.

---

## Phase 3 — K-Means Clustering

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 3.1 | K-Means from scratch (`ml/clustering.py`) | — | ✅ | NumPy implementation with unit tests. **No sklearn KMeans.** |
| 3.2 | Choose K (elbow + silhouette) | — | ✅ | Elbow method run in `02_embedding_experiments.ipynb` on real embeddings; chose k=8. Silhouette not explicitly computed (see remaining work). |
| 3.3 | Auto-label clusters (TF-IDF top terms) | — | ✅ | `scripts/build_clusters.py` uses TF-IDF + heuristic pattern matching to label clusters. `models/cluster_labels.json` exists. |
| 3.4 | Assign user embedding to cluster | — | ✅ | `cluster_position()` in `app/ml_runtime.py` calls `kmeans.predict()` and returns cluster ID, label, top terms, distances. |

**Phase status:** ✅ Core complete — K selected, clusters labeled, user assignment working. Silhouette score metric not yet computed (see remaining work).

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
| 5.1 | Gap analysis (direction vector) | — | ✅ | `compute_gap_analysis()` implemented in `ml/feedback.py` and exposed via `cluster_position()`. |
| 5.2 | Keyword extraction (missing skills) | — | ✅ | `feedback_terms()` in `app/ml_runtime.py` identifies cluster/match terms missing from the resume. |
| 5.3 | Cluster migration advice | — | ✅ | `cluster_migration_advice()` implemented in `app/ml_runtime.py`. |

**Phase status:** ✅ Complete — all feedback and gap analysis logic implemented.

---

## Phase 6 — Streamlit UI & Integration

| # | Task | Owner | Status | Notes |
|---|------|-------|--------|-------|
| 6.1 | App shell & navigation (`app/app.py`) | Ryan | ✅ | Single-page Streamlit app (~5,000 lines) with dark theme, synthetic fallback, and full ML pipeline integration. |
| 6.2 | Resume upload page | Ryan | ✅ | Text paste + PDF upload + public URL fetch via `app/components/resume_upload.py`. |
| 6.3 | Job matching page | Ryan | ✅ | Job results rendered via `app/components/job_results.py` with similarity scores, seniority fit, salary eligibility. |
| 6.4 | Salary prediction page | Ryan | ✅ | Salary band rendered via `app/components/salary_chart.py`; hybrid salary (retrieved + BLS + neural) with confidence. |
| 6.5 | Market position page | Ryan | ✅ | Cluster position shown via `app/components/cluster_view.py`; cluster browser included. |
| 6.6 | Resume feedback page | Ryan | 🟡 | Missing keywords shown; full phrase-level highlighting and cluster migration advice not yet implemented (depends on Phase 5). |
| 6.7 | Reusable components (`app/components/`) | Ryan | ✅ | `cluster_view.py`, `job_results.py`, `resume_upload.py`, `salary_chart.py` all implemented. |

**Phase status:** 🟡 App is functional end-to-end with most features working. Feedback page lacks depth (Phase 5 dependency).

---

## Cross-Cutting Tasks

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| `pyproject.toml` / `uv.lock` — manage dependencies and virtual environment with uv | — | ✅ | Switched from `requirements.txt` on 2026-04-25 |
| Ruff / pre-commit / GitHub Actions CI | — | ✅ | Formatting, linting, and tests enforced for code directories only |
| `tests/` — pytest coverage for `ml/` | Alan, Ryan, Omer | ✅ | 20 test files covering salary, retrieval, clustering, preprocessing, embeddings, quality, resume loader, and more. |
| Resume-quality predictor (`ml/quality.py`) | Ryan | ✅ | Single-head PyTorch MLP + rule-based `score_resume_quality(text)` for real-resume input. |
| Salary-prediction evaluation (`scripts/evaluate_salary.py`) | Ryan | ✅ | Median MAE, pinball loss, coverage, per-persona breakdown. |
| Resume-side salary model (`scripts/train_resume_salary_model.py`) | Ryan | ✅ | Retrains `SalaryQuantileNet` on `(resume_embedding, source_salary_annual)` pairs. |
| Real-resume ingest (`ml/resume_loader.py`, `scripts/load_real_resumes.py`) | Ryan | ✅ | PDF/.txt/.md/.csv/JSONL ingest with PII redaction, length cap, sample fixture. |
| Real-resume validation (`scripts/validate_on_real_resumes.py`) | Ryan | ✅ | Rule-based quality + learned-MLP score + retrieval stats + self-consistency salary metric. |
| External occupation data (O*NET / BLS) | Ryan | 🟡 | O*NET/BLS plumbing complete (`ml/occupation_router.py`, `ml/wage_bands.py`), but external data files not present in `data/external/`. |
| Set random seeds in all scripts | — | 🟡 | Seeds set in `train_salary_model.py`, `train_resume_salary_model.py`, `train_quality_model.py`, `train_public_assessment_models.py`, `build_index.py`, `build_clusters.py`. **Missing from:** `preprocess_data.py`, `generate_synthetic_resumes.py`. |
| `.gitignore` — verify `data/raw/`, `models/` excluded | — | ✅ | Also excludes `data/external/` while preserving `.gitkeep`. |
| Final report / presentation | — | ⬜ | |

---

## ⚠️ Areas Still Needing Work

### High Priority (core requirements)
### High Priority (core requirements)

1. **Silhouette score (Task 3.2):** K was chosen via the elbow method, but the plan also requires silhouette score ≥ 0.15. This metric has not been computed or reported.

2. **Final report / presentation:** Not started.

### Medium Priority (quality & completeness)

3. **Random seeds in all scripts:** `preprocess_data.py` and `generate_synthetic_resumes.py` do not set `np.random.seed()` / `torch.manual_seed()`. Required by grading constraints for reproducibility.

4. **External data files (O*NET / BLS):** Code exists in `ml/occupation_router.py` and `ml/wage_bands.py`, but `data/external/onet_skills.parquet` and `data/external/bls_wages.parquet` are not present. These are optional enrichments but enhance salary confidence and occupation routing.

5. **Salary model calibration:** q90 calibration is reported as slightly outside the ±5 pp target. May need further tuning.

9. **Stale files in repo root:** `replace_main.py` (27 KB) and `fix_app.py` (2 KB) appear to be one-off scripts that should be cleaned up or moved.

### Low Priority (nice-to-have)

10. **Multi-page Streamlit structure:** The app is currently a single 5,000-line `app.py` file instead of the planned `app/pages/` layout. Functionally complete but harder to maintain.

11. **t-SNE/UMAP market position visualization:** The plan calls for a 2D scatter plot showing job clusters with the user's resume as a highlighted point. The cluster browser exists but the full interactive Plotly scatter is not confirmed.

---

## Overall Progress

```
Phase 1  [██████████] 100%  ← raw data downloaded, preprocessing complete
Phase 2  [██████████] 100%  ← embeddings, retrieval, and experiment notebook ready
Phase 3  [█████████░]  90%  ← KMeans done, clusters labeled; silhouette score missing
Phase 4  [██████████] 100%  ← quantile regression model, training, inference, and evaluation complete
Phase 5  [██████████] 100%  ← gap analysis, keyword extraction, and migration advice complete
Phase 6  [█████████░]  90%  ← app functional; Phase 5 logic integrated into runtime
─────────────────────────
Total    [█████████░]  92%
```

**Current state:** The core ML pipeline is fully operational end-to-end: raw Kaggle data → preprocessing → embeddings → FAISS retrieval → K-Means clustering → salary quantile regression → Streamlit app. Users can upload a resume and get job matches, salary predictions, cluster assignment, and detailed gap analysis with migration advice. The main remaining gaps are **silhouette score reporting**, **random seed consistency**, and the **final report**.

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
| 2026-04-28 | Improved real-resume assessment quality. O*NET default updated to release 30.2 and local download succeeded (`105,526` rows / `26,902` unique skills / `923` SOCs, gitignored). Rule scoring now avoids trivial 100s through diminishing returns, surfaces human-readable positives/gaps, flags vague/cliche wording and career-progression issues, and validation adds role-family mismatch checks. BLS command-line download is still blocked by HTTP 403 in this environment; docs now include the manual `--input` path. |
| 2026-04-29 | Audited full codebase against progress. Updated Phase 1 to ✅ (raw data present), Phase 3.2/3.3/3.4 to ✅, Phase 5/6 to 🟡. Added "Areas Still Needing Work" section. Identified missing silhouette score, random seeds in 2 scripts, stale root files, and Phase 5 gaps as remaining work. |
