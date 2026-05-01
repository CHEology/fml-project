from __future__ import annotations

from types import SimpleNamespace

import plotly.graph_objects as go


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_methodology_page_renders_academic_project_explanation(monkeypatch) -> None:
    import app.components.methodology as methodology

    rendered: list[str] = []
    equations: list[str] = []
    charts: list[go.Figure] = []

    fake_streamlit = SimpleNamespace(
        markdown=lambda body, *args, **kwargs: rendered.append(str(body)),
        latex=lambda body, *args, **kwargs: equations.append(str(body)),
        plotly_chart=lambda fig, *args, **kwargs: charts.append(fig),
        columns=lambda *args, **kwargs: [_Context(), _Context()],
        container=lambda *args, **kwargs: _Context(),
        expander=lambda *args, **kwargs: _Context(),
        code=lambda body, *args, **kwargs: rendered.append(str(body)),
        caption=lambda body, *args, **kwargs: rendered.append(str(body)),
    )
    monkeypatch.setattr(methodology, "st", fake_streamlit)

    methodology.render_methodology_page()

    page_text = "\n".join(rendered)
    equation_text = "\n".join(equations)

    for expected in (
        "Lucky Hamsters",
        "Omer Hortig",
        "Abstract",
        "LinkedIn Job Postings 2023-2024",
        "all-MiniLM-L6-v2",
        "FAISS",
        "K-Means",
        "pinball loss",
        "synthetic resumes",
        "public assessment models",
        "uv run python scripts/preprocess_data.py",
        "uv run streamlit run app/app.py",
        "Limitations",
    ):
        assert expected in page_text

    assert r"\operatorname{cos}" in equation_text
    assert r"\sum_{i=1}^{n}" in equation_text
    assert r"\rho_{\tau}" in equation_text
    assert len(charts) >= 3


def test_methodology_figure_builders_return_plotly_figures() -> None:
    from app.components.methodology_figures import (
        build_cluster_snapshot_figure,
        build_experiment_snapshot_figure,
        build_pipeline_figure,
        build_salary_snapshot_figure,
    )

    pipeline = build_pipeline_figure()
    salary = build_salary_snapshot_figure()
    clusters = build_cluster_snapshot_figure()
    experiments = build_experiment_snapshot_figure()

    assert isinstance(pipeline, go.Figure)
    assert isinstance(salary, go.Figure)
    assert isinstance(clusters, go.Figure)
    assert isinstance(experiments, go.Figure)
    assert len(pipeline.data) == 1
    assert len(salary.data) == 1
    assert len(clusters.data) == 1
    assert len(experiments.data) >= 2
    assert "Pipeline" in str(pipeline.layout.title.text)
    assert "Salary" in str(salary.layout.title.text)
    assert "Cluster" in str(clusters.layout.title.text)
    assert "Experiment" in str(experiments.layout.title.text)
