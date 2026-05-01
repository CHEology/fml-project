# App Package

`app/app.py` is the Streamlit entrypoint and should remain a thin bootstrap for
page registration, global style injection, and sidebar rendering.

Current layout:

- `pages/`: top-level Streamlit page renderers.
- `demo/`: demo workflow state, input flow helpers, and sample resume data.
- `runtime/`: artifact loading and ML orchestration used by the UI.
- `components/`: reusable Streamlit/HTML components.
- `styles/`: CSS and theme injection.
- `assets/`: static app assets.

The Methodology page is rendered from `components/methodology.py`, with static
Plotly figure builders in `components/methodology_figures.py`. Keep it as an
explanatory UI layer: model logic, scoring, training, and artifact loading stay
in `ml/`, `scripts/`, and `app/runtime/`.

The Market Overview page is the model-context layer for the demo. It can
explain catalog shape, salary distributions, geography/work-mode mix, and
cluster labels, but runtime artifact access and model-loading glue still belong
in `app/runtime/`.

Do not put ML scoring, parsing, or model logic in page files. Shared resume
assessment belongs in `ml/resume_assessment/`; runtime artifact and model
loading belongs in `app/runtime/`.
