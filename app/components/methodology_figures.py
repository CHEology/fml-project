from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

PIPELINE_STAGES = (
    "Raw datasets",
    "Preprocess",
    "Embed + index",
    "Train models",
    "Demo analysis",
    "Feedback",
)

SALARY_QUANTILES = {
    "q10": 39_520,
    "q25": 52_020,
    "q50": 82_500,
    "q75": 125_000,
    "q90": 168_750,
}

CLUSTER_SNAPSHOT = (
    ("Software / Engineering", 6430),
    ("Administrative / HR", 5197),
    ("Operations / Logistics", 5230),
    ("Sales / Customer Growth", 5167),
    ("Business / Data Analysis", 4401),
    ("Healthcare / Clinical", 3546),
    ("Healthcare / Clinical", 3037),
    ("Finance / Accounting", 2110),
)

SALARY_CALIBRATION = {
    "q10": 0.110,
    "q25": 0.254,
    "q50": 0.526,
    "q75": 0.772,
    "q90": 0.920,
}

PUBLIC_ASSESSMENT_METRICS = {
    "Domain": 0.4698,
    "Entity": 0.4143,
    "Section": 0.7883,
}


def build_pipeline_figure() -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(PIPELINE_STAGES))),
            y=[1] * len(PIPELINE_STAGES),
            mode="lines+markers+text",
            text=PIPELINE_STAGES,
            textposition="top center",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=18, color="#ffffff", line=dict(color="#2563eb", width=3)),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Pipeline From Dataset To Feedback",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=250,
        margin=dict(l=20, r=20, t=60, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_salary_snapshot_figure() -> go.Figure:
    labels = list(SALARY_QUANTILES)
    values = list(SALARY_QUANTILES.values())
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#93c5fd", "#60a5fa", "#2563eb", "#1d4ed8", "#1e3a8a"],
            text=[f"${value:,.0f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Salary Quantile Snapshot From Processed LinkedIn Jobs",
        xaxis_title="Empirical quantile",
        yaxis_title="Annual salary",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        height=360,
        margin=dict(l=20, r=20, t=70, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_cluster_snapshot_figure() -> go.Figure:
    labels = [label for label, _ in CLUSTER_SNAPSHOT]
    counts = [count for _, count in CLUSTER_SNAPSHOT]
    fig = go.Figure(
        go.Bar(
            x=counts[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color="#0f766e",
            text=[f"{count:,}" for count in counts[::-1]],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:,} postings<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cluster Size Snapshot From K-Means Artifacts",
        xaxis_title="Processed postings",
        yaxis_title="",
        height=390,
        margin=dict(l=20, r=20, t=70, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_experiment_snapshot_figure() -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Salary calibration", "Public assessment baselines"),
    )
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    actual = list(SALARY_CALIBRATION.values())
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=actual,
            mode="lines+markers",
            name="Observed coverage",
            line=dict(color="#2563eb", width=3),
            hovertemplate="Nominal %{x:.2f}<br>Observed %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=quantiles,
            mode="lines",
            name="Ideal",
            line=dict(color="#64748b", width=2, dash="dash"),
            hovertemplate="Ideal %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(PUBLIC_ASSESSMENT_METRICS),
            y=list(PUBLIC_ASSESSMENT_METRICS.values()),
            name="Validation score",
            marker_color="#7c3aed",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Experiment Snapshot From Local Evaluation Artifacts",
        height=390,
        margin=dict(l=20, r=20, t=80, b=55),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="Nominal quantile", row=1, col=1)
    fig.update_yaxes(title_text="Observed fraction <= prediction", row=1, col=1)
    fig.update_yaxes(title_text="Validation metric", row=1, col=2)
    return fig
