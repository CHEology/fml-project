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

`app/ml_runtime.py` is a compatibility alias for older imports. New code should
import from `app.runtime` directly.
