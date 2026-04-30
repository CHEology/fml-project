from __future__ import annotations

import streamlit as st

from app.components.job_results import render_panel_banner
from app.components.team import TEAM_MEMBERS, TEAM_NAME


def render_methodology_page() -> None:
    render_panel_banner(
        "Methodology",
        "How ResuMatch works",
        "A walkthrough of the ML pipeline powering resume analysis, job matching, salary prediction, and market positioning.",
    )

    st.markdown("### Dataset")
    st.markdown(
        """
        ResuMatch is trained and evaluated on the
        **LinkedIn Job Postings (2023–2024)** dataset from Kaggle.
        We process approximately **35,000 postings** filtered from the full dataset
        to those with valid salary data and English descriptions, with fields
        including job descriptions, required skills, experience level, company,
        location, and normalized annual salary.
        """
    )

    st.markdown("### ML Pipeline")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("#### 1. Text Embedding")
            st.markdown(
                """
                Resume and job description text is encoded into dense
                **384-dimensional vectors** using a sentence-transformer model
                (`all-MiniLM-L6-v2`). Vectors are L2-normalized so cosine
                similarity equals dot product — enabling fast FAISS retrieval.

                Unlike TF-IDF, transformer embeddings capture semantic meaning
                rather than exact keyword overlap. A resume mentioning "built APIs"
                matches a job requiring "REST services" even without shared tokens.
                """
            )

        with st.container(border=True):
            st.markdown("#### 2. Job Retrieval")
            st.markdown(
                """
                All 35,000+ job embeddings are indexed in a **FAISS flat
                inner-product index**. At query time, the resume vector is
                searched against the index to return the top-k most similar
                roles in milliseconds via cosine similarity in the normalized
                embedding space.
                """
            )

        with st.container(border=True):
            st.markdown("#### 3. K-Means Clustering *(from scratch)*")
            st.markdown(
                """
                Job postings are grouped into **8 role-family clusters** using
                K-means implemented from scratch in NumPy — no sklearn.
                K was chosen via the **elbow method** on inertia across k=2–12
                on PCA-reduced (50-dim) embeddings.

                - Centroids initialized by random sampling without replacement
                - Convergence when max centroid shift < 1e-4
                - Empty clusters reinitialized to a random sample

                Cluster labels are derived from TF-IDF top terms per cluster.
                """
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### 4. Salary Prediction")
            st.markdown(
                """
                A **quantile regression neural network** (raw PyTorch) predicts
                five salary quantiles (p10–p90) simultaneously using pinball loss,
                giving a calibrated salary band rather than a point estimate.
                A resume-side model removes the domain shift from applying a
                job-description-trained model to resume inputs.
                """
            )

        with st.container(border=True):
            st.markdown("#### 5. Resume Quality Scoring")
            st.markdown(
                """
                **Rule-based scorer** evaluates quantified impact, action verb
                strength, skill density, section completeness, and vague phrasing.
                Produces human-readable strength and gap notes.

                **Learned MLP** (PyTorch) is trained on synthetic resumes with
                known quality scores and used as a cross-check. Spearman
                agreement between the two is reported for validation.
                """
            )

        with st.container(border=True):
            st.markdown("#### 6. Skill Gap Analysis")
            st.markdown(
                """
                Missing skills are surfaced by comparing TF-IDF term weights
                between the resume and top matched postings. Terms with high
                weight in the job corpus but low weight in the resume are ranked
                as highest-priority gaps to address.
                """
            )

    with st.container(border=True):
        st.markdown("#### 7. Domain & Seniority Detection")
        st.markdown(
            """
            Two lightweight **MLP classifiers** (raw PyTorch, hash-based input features)
            are trained on public resume datasets to detect:

            - **Domain** — e.g. Software Engineering, Data & Analytics, Healthcare
            - **Seniority** — e.g. Intern/Entry, Mid, Senior, Executive

            These are used as advisory signals in the candidate snapshot alongside
            the rule-based quality scorer and salary band. Predictions degrade
            gracefully when model artifacts are missing.
            """
        )

    st.markdown("### Course Algorithms Used")
    st.markdown(
        """
        | Algorithm | Where used | Implementation |
        |---|---|---|
        | **K-Means Clustering** | Job market segmentation | From scratch, NumPy only |
        | **Quantile Regression** | Salary band prediction | Raw PyTorch, pinball loss |
        | **MLP Classifier** | Domain & seniority detection | Raw PyTorch |
        | **MLP Regressor** | Resume quality scoring | Raw PyTorch |
        | **TF-IDF** | Skill gap analysis, cluster labeling | scikit-learn |
        | **PCA** | Dimensionality reduction before clustering | scikit-learn |
        | **Cosine Similarity** | Resume-to-job matching | NumPy / FAISS |
        """
    )

    st.markdown(f"### Team: {TEAM_NAME}")
    team_rows = "\n".join(
        f"| {member['name']} | {member['github']} |" for member in TEAM_MEMBERS
    )
    st.markdown(
        f"""
        | Contributor | GitHub |
        |---|---|
        {team_rows}
        """
    )
