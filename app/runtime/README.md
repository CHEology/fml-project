# App Runtime

Runtime modules bridge the Streamlit UI to local artifacts and ML modules.

- `ml.py` loads model artifacts, FAISS indexes, wage tables, public assessment
  models, salary signals, and clustering helpers.
- `artifacts.py` defines the expected raw data and generated artifact files,
  reports readiness, and returns setup commands plus creation timestamps for
  user-facing pipeline status.
- `live_jobs.py` builds compact live-job queries, fetches public no-key job
  feeds plus optional Serpdog LinkedIn results, and reranks them with the
  existing embedding model.
- `cache.py` wraps runtime calls in Streamlit caching decorators and provides
  fallback sample data when local artifacts are missing.

Salary loading prefers `models/resume_salary_model.pt` with
`models/resume_salary_model.scaler.json` because the demo predicts from resume
embeddings. If that pair is absent, runtime falls back to
`models/salary_model.pt` with `models/salary_model.scaler.json`, the older
job-embedding salary baseline.

`app/ml_runtime.py` is a compatibility alias for older imports. New code should
import from `app.runtime` directly.
