# Tests

Pytest tests mirror the production source areas:

- `app/`: Streamlit page/runtime/component behavior.
- `ml/`: ML modules, shared resume assessment, retrieval, salary, and quality.
- `scripts/`: CLI and data-pipeline behavior.
- `fixtures/`: small committed fixtures used by tests.

Avoid tests that depend on large literal source strings or a specific monolith
layout. Prefer import, behavior, and output-shape tests. `test_structure.py`
enforces the repository file-size budget and public assessment API.
