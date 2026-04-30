# Agent Maintenance Guide

This repo has a strict organization rule: keep Python source files under 1,000
lines. Prefer focused files under 800 lines when adding new code.

## Required Boundaries

- `app/app.py` must stay a thin Streamlit entrypoint. It may register pages,
  initialize state/styles, render the sidebar, and call `page.run()`.
- Streamlit rendering belongs in `app/pages/`, `app/components/`, `app/demo/`,
  or `app/styles/`.
- Runtime artifact loading, Streamlit cache wrappers, retrieval orchestration,
  and model-loading glue belong in `app/runtime/`.
- ML logic belongs in `ml/`. Do not import Streamlit from `ml/`.
- Resume parsing, profile detection, quality scoring, capability scoring, and
  salary adjustments belong in `ml/resume_assessment/`.
- Synthetic resume generation implementation belongs in
  `scripts/synthetic_resumes/`; keep `scripts/generate_synthetic_resumes.py` as
  a compatibility wrapper.

## Duplication Rules

- Do not copy role taxonomies, skill lexicons, seniority thresholds, or quality
  label thresholds into page files.
- Reuse `ml.resume_assessment` for resume evidence and scoring.
- Reuse `ml.taxonomy` for shared synthetic role profiles and quality labels.
- Reuse `app.runtime` for artifact status, retrieval, clustering, salary, wage,
  and public-assessment model access.

## Testing Rules

- Add or update tests in the matching folder under `tests/app/`, `tests/ml/`, or
  `tests/scripts/`.
- Avoid brittle tests that inspect large literal source blocks or assume a file
  remains monolithic.
- Keep `tests/test_structure.py` passing; it enforces the file-size budget and
  public resume-assessment API.
- Run before handing off:

```bash
uv run ruff format --check app ml scripts tests
uv run ruff check app ml scripts tests
uv run pytest
```

## Documentation Rules

- Update the relevant package README when moving files or changing ownership.
- Keep root `README.md` aligned with the actual run/test/data commands.
- Historical plan/progress documents live in `docs/history/`; do not treat them
  as active implementation instructions.
