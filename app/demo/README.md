# Demo Workflow

This package owns the interactive resume demo flow.

- `sample_data.py` contains synthetic sample profile fixtures and demo copy.
- `samples.py` renders random sample resumes from the local job catalog context.
- `state.py` initializes Streamlit session state for the demo.
- `components.py` contains small demo-only UI helpers.

Keep this package focused on demo input and presentation workflow. Reusable UI
belongs in `app/components/`; resume scoring and seniority logic belongs in
`ml/resume_assessment/`.
