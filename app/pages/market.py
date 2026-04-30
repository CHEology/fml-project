from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.job_results import (
    fmt_money,
    render_metric_card,
    render_panel_banner,
)
from app.runtime.cache import artifacts_ready, load_cluster_resource


def render_market_overview_page(
    jobs: pd.DataFrame,
    data_source: str,
    has_real_data: bool,
    status: list[dict[str, str | bool]],
) -> None:
    metric_cols = st.columns(2)
    with metric_cols[0]:
        render_metric_card("Jobs loaded", f"{len(jobs):,}", "local catalog size")
    with metric_cols[1]:
        median_salary = pd.to_numeric(
            jobs.get("salary_annual"), errors="coerce"
        ).dropna()
        render_metric_card(
            "Median salary",
            fmt_money(median_salary.median() if len(median_salary) else None),
            "from current dataset",
        )

    st.write("")
    render_panel_banner(
        "Market Radar",
        "Where the current feed is concentrated",
        "A quick view of geography, seniority, and salary distribution in the available job catalog.",
    )
    display_jobs = jobs.copy()
    display_jobs["salary_annual"] = pd.to_numeric(
        display_jobs.get("salary_annual"), errors="coerce"
    )

    left, right = st.columns([0.52, 0.48], gap="large")
    with left, st.container(border=True):
        st.markdown("**Top locations**")
        location_counts = (
            display_jobs["location"].fillna("Unknown").value_counts().head(8)
        )
        st.bar_chart(location_counts)

        st.markdown("**Experience mix**")
        exp_counts = (
            display_jobs["experience_level"].fillna("Unknown").value_counts().head(8)
        )
        st.bar_chart(exp_counts)

        if artifacts_ready(status, "clustering"):
            _, assignments, cluster_labels = load_cluster_resource()
            cluster_names = [
                cluster_labels.get(str(cluster_id), {}).get(
                    "label", f"Cluster {cluster_id}"
                )
                for cluster_id in assignments
            ]
            st.markdown("**Market segments**")
            st.bar_chart(pd.Series(cluster_names).value_counts())

    with right, st.container(border=True):
        st.markdown("**Salary sample**")
        salary_view = display_jobs[
            ["title", "company_name", "location", "salary_annual"]
        ].copy()
        salary_view = salary_view.sort_values("salary_annual", ascending=False).head(12)
        st.dataframe(salary_view, width="stretch", hide_index=True)

        st.markdown("**Dataset notes**")
        if has_real_data:
            st.success(f"Loaded real project data from `{data_source}`.")
        else:
            st.info("Using sample roles until the local job catalog is prepared.")
