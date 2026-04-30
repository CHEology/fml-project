# Synthetic Resume Generation

This package implements the synthetic resume/job-pair generator used by
retrieval, salary, and quality-model experiments.

- `generator.py` contains deterministic generation, hard-negative selection,
  salary proxy, resume text rendering, and output writing.
- `cli.py` owns argument parsing for the command-line interface.

The legacy command still works:

```bash
uv run python scripts/generate_synthetic_resumes.py
```

Shared role profiles, multi-word skills, and quality-label thresholds live in
`ml/taxonomy.py` so ML modules do not import from CLI scripts.
