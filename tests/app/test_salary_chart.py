from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from app.components import salary_chart


class StreamlitCapture:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.markdowns: list[str] = []
        self.figures: list[object] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def markdown(self, body: str, **_kwargs: object) -> None:
        self.markdowns.append(body)

    def plotly_chart(self, fig: object, **_kwargs: object) -> None:
        self.figures.append(fig)


def test_cluster_salary_distribution_renders_dataset_and_user_marker(
    monkeypatch,
) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    jobs = pd.DataFrame(
        {
            "title": ["A", "B", "C", "D"],
            "company_name": ["Acme", "Beta", "Core", "Delta"],
            "location": ["NY", "SF", "Remote", "Austin"],
            "experience_level": ["Entry", "Mid", "Senior", "Lead"],
            "work_type": ["Hybrid", "Remote", "Remote", "On-site"],
            "salary_annual": [100_000, 120_000, 150_000, 180_000],
        }
    )
    cluster_labels = {
        "0": {
            "label": "Software / Engineering",
            "top_terms": ["python", "backend"],
            "common_titles": ["Software Engineer"],
        },
        "1": {
            "label": "Data / Analytics",
            "top_terms": ["analytics", "sql"],
            "common_titles": ["Data Analyst"],
        },
    }

    salary_chart.render_cluster_salary_distribution(
        jobs,
        assignments=[0, 0, 1, 1],
        cluster_labels=cluster_labels,
        cluster={"cluster_id": 1, "label": "Data / Analytics"},
        band={"q50": 160_000},
        job_embeddings=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [4.0, 4.0, 0.0],
                [4.2, 4.1, 0.0],
            ],
            dtype=np.float32,
        ),
        resume_embedding=np.array([4.1, 4.0, 0.0], dtype=np.float32),
    )

    assert not capture.infos
    assert len(capture.figures) == 1
    fig = capture.figures[0]
    assert len(fig.data) == 3
    assert fig.data[0].name == "Software / Engineering"
    assert fig.data[1].name == "Data / Analytics"
    assert fig.data[-1].name == "Your predicted position"
    assert isinstance(list(fig.data[-1].x)[0], float)
    assert isinstance(list(fig.data[-1].y)[0], float)
    assert "1 - Data / Analytics" not in {trace.name for trace in fig.data}
    assert fig.layout.showlegend is True
    assert fig.layout.xaxis.title.text == "Embedding component 1"
    assert fig.layout.yaxis.title.text == "Embedding component 2"


def test_cluster_salary_distribution_hover_has_descriptive_context(monkeypatch) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    salary_chart.render_cluster_salary_distribution(
        pd.DataFrame(
            {
                "title": ["Machine Learning Engineer"],
                "company_name": ["North Harbor AI"],
                "location": ["New York, NY"],
                "experience_level": ["Senior"],
                "work_type": ["Remote"],
                "salary_annual": [180_000],
            }
        ),
        assignments=[0],
        cluster_labels={
            "0": {
                "label": "Machine Learning / AI",
                "top_terms": ["machine learning", "pytorch", "nlp"],
                "common_titles": ["Machine Learning Engineer"],
            }
        },
        cluster={"cluster_id": 0, "label": "Machine Learning / AI"},
        band={"q50": 175_000},
        job_embeddings=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        resume_embedding=np.array([0.1, 1.1, 0.0], dtype=np.float32),
    )

    fig = capture.figures[0]
    job_trace = fig.data[0]
    user_trace = fig.data[-1]
    assert "Title: %{customdata[0]}" in job_trace.hovertemplate
    assert "Company: %{customdata[1]}" in job_trace.hovertemplate
    assert "Cluster evidence: %{customdata[5]}" in job_trace.hovertemplate
    assert "Salary: %{customdata[7]}" in job_trace.hovertemplate
    assert job_trace.customdata[0][5] == "machine learning, pytorch, nlp"
    assert "Cluster: Machine Learning / AI" in user_trace.hovertemplate
    assert "Predicted salary: $175,000" in user_trace.hovertemplate


def test_cluster_salary_distribution_samples_points_for_responsiveness(
    monkeypatch,
) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    jobs = pd.DataFrame(
        {
            "title": [f"Role {idx}" for idx in range(18)],
            "salary_annual": [100_000 + idx for idx in range(18)],
        }
    )
    embeddings = np.column_stack(
        [
            np.arange(18, dtype=np.float32),
            np.arange(18, dtype=np.float32) * 0.5,
            np.ones(18, dtype=np.float32),
        ]
    )

    salary_chart.render_cluster_salary_distribution(
        jobs,
        assignments=[idx % 3 for idx in range(18)],
        cluster_labels={
            "0": {"label": "Cluster A"},
            "1": {"label": "Cluster B"},
            "2": {"label": "Cluster C"},
        },
        cluster={"cluster_id": 1},
        band={"q50": 120_000},
        job_embeddings=embeddings,
        resume_embedding=np.array([8.0, 4.0, 1.0], dtype=np.float32),
        sample_size=6,
    )

    fig = capture.figures[0]
    plotted_jobs = sum(len(trace.x) for trace in fig.data[:-1])
    assert plotted_jobs == 6
    assert "random sample of 6 jobs" in capture.markdowns[0]


def test_cluster_salary_distribution_falls_back_without_assignments(
    monkeypatch,
) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    salary_chart.render_cluster_salary_distribution(
        pd.DataFrame({"salary_annual": [100_000]}),
        assignments=None,
        cluster_labels={},
        cluster={"cluster_id": 0},
        band={"q50": 100_000},
        job_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        resume_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert capture.figures == []
    assert capture.infos == [
        "Cluster salary distribution is unavailable until clustering artifacts are built."
    ]


def test_cluster_salary_distribution_renders_without_salary_data(monkeypatch) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    salary_chart.render_cluster_salary_distribution(
        pd.DataFrame({"salary_annual": [None, None]}),
        assignments=[0, 1],
        cluster_labels={},
        cluster={"cluster_id": 0},
        band={"q50": 100_000},
        job_embeddings=np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        resume_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert capture.infos == []
    assert len(capture.figures) == 1
    assert capture.figures[0].data[0].customdata[0][7] == "N/A"


def test_cluster_salary_distribution_falls_back_when_assignments_mismatch(
    monkeypatch,
) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    salary_chart.render_cluster_salary_distribution(
        pd.DataFrame({"salary_annual": [100_000, 120_000]}),
        assignments=[0],
        cluster_labels={},
        cluster={"cluster_id": 0},
        band={"q50": 100_000},
        job_embeddings=np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        resume_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert capture.figures == []
    assert capture.infos == [
        "Cluster salary distribution is unavailable because clustering assignments do not match the job catalog."
    ]


def test_cluster_salary_distribution_falls_back_without_user_prediction(
    monkeypatch,
) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    salary_chart.render_cluster_salary_distribution(
        pd.DataFrame({"salary_annual": [100_000]}),
        assignments=[0],
        cluster_labels={},
        cluster=SimpleNamespace(),
        band=None,
        job_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        resume_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert capture.figures == []
    assert capture.infos == [
        "User salary position is unavailable until both salary and cluster predictions are available."
    ]


def test_cluster_salary_distribution_falls_back_without_embeddings(monkeypatch) -> None:
    capture = StreamlitCapture()
    monkeypatch.setattr(salary_chart, "st", capture)

    salary_chart.render_cluster_salary_distribution(
        pd.DataFrame({"salary_annual": [100_000]}),
        assignments=[0],
        cluster_labels={},
        cluster={"cluster_id": 0},
        band={"q50": 100_000},
    )

    assert capture.figures == []
    assert capture.infos == [
        "2D cluster map is unavailable until job embeddings are built."
    ]
