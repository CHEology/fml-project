from __future__ import annotations

from html import escape

import streamlit as st

from app.components.methodology_figures import (
    build_cluster_snapshot_figure,
    build_experiment_snapshot_figure,
    build_pipeline_figure,
    build_salary_snapshot_figure,
)
from app.components.team import TEAM_MEMBERS, TEAM_NAME


def render_methodology_page() -> None:
    _render_title_block()
    _render_abstract()
    _render_dataset_section()
    _render_model_section()
    _render_demo_section()
    _render_experiments_section()
    _render_reproducibility_section()
    _render_limitations_section()


def _render_title_block() -> None:
    authors = ", ".join(member["name"] for member in TEAM_MEMBERS)
    st.markdown(
        f"""
        <article class="method-paper">
            <div class="method-kicker">NYU CSCI-UA 473 Fundamentals of Machine Learning</div>
            <h1>ResuMatch: Machine Learning for Resume Market Assessment</h1>
            <div class="method-authors">{escape(TEAM_NAME)} - {escape(authors)}</div>
            <div class="method-meta">
                Streamlit demo, retrieval system, salary quantile model, K-Means
                market segmentation, and explainable resume assessment.
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_pipeline_figure(), use_container_width=True)


def _render_abstract() -> None:
    _paper_heading("Abstract")
    st.markdown(
        """
        ResuMatch was designed from the perspective of college students entering
        a crowded internship and full-time job market. The user-facing question
        is practical: given a resume, what career options does it currently
        resemble, how strong is the evidence, what salary range is supported by
        comparable roles, and what concrete changes could move the candidate
        toward a desired path?

        We answer this with a multi-stage ML system. Job descriptions and
        resumes are embedded with `all-MiniLM-L6-v2`; a FAISS inner-product
        index retrieves semantically similar roles; a NumPy K-Means model
        turns postings into role-family clusters; PyTorch quantile regression
        predicts salary bands; and resume-assessment logic scores experience,
        quantified impact, specificity, structure, capability, public
        assessment models, and skill gaps. The demo is therefore not a hiring
        oracle. It is a data-grounded market explanation tool for students and
        early-career users deciding how to position their experience.
        """
    )


def _render_dataset_section() -> None:
    _paper_heading("1. Dataset")
    st.markdown(
        """
        The primary dataset is Kaggle's **LinkedIn Job Postings 2023-2024**
        catalog. We chose it because it contains real job descriptions,
        titles, companies, locations, experience labels, work-type fields,
        skills, and salary fields in one coherent posting-level source. That
        makes it relevant for our goal: modeling the relationship between job
        language, market segment, seniority, and compensation.

        Our reproduced local run contains **35,118 processed postings** after
        joining postings to company metadata, stripping HTML, normalizing salary
        fields to annual values, and filtering unusable text/salary rows. The
        final job embedding matrix has shape **35,118 x 384**.
        """
    )
    _metric_grid(
        (
            ("Processed jobs", "35,118", "salary-bearing LinkedIn rows"),
            ("Embedding dim", "384", "all-MiniLM-L6-v2 vectors"),
            ("Clusters", "8", "NumPy K-Means segments"),
            ("Median salary", "$82,500", "processed catalog q50"),
        )
    )
    st.plotly_chart(build_salary_snapshot_figure(), use_container_width=True)
    st.markdown(
        """
        We also use public resume datasets for auxiliary supervised baselines:
        `Divyaamith/Kaggle-Resume` for domain classification,
        `0xnbk/resume-ats-score-v1-en` for resume/job ATS fit regression,
        `DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy` for resume entity
        recognition, and `ganchengguang/resume_seven_class` for section labels.
        These public assessment models are intentionally lightweight raw
        PyTorch MLPs over deterministic hashed text features, so a fresh clone
        can train them without a large model-serving stack.

        Dataset pitfalls are important. LinkedIn/Kaggle postings are a sample
        of the labor market, not the labor market itself. Salary is missing or
        noisy in many real postings, geography is skewed toward high-volume
        U.S. metros, job descriptions often contain boilerplate, and public
        resume labels are imperfect. We therefore use the data as contextual
        evidence for market positioning rather than as ground truth about a
        person's worth or guaranteed hiring outcome.
        """
    )


def _render_model_section() -> None:
    _paper_heading("2. Creating The ML Models")
    st.markdown(
        """
        The model pipeline starts with representation learning. We encode each
        cleaned posting text and each resume into a dense vector, then normalize
        rows to unit length. This lets FAISS `IndexFlatIP` use an inner product
        that is equivalent to cosine similarity.
        """
    )
    st.latex(
        r"""
        \hat{x}=\frac{x}{\lVert x\rVert_2},
        \qquad
        \operatorname{cos}(\hat{r},\hat{j})=
        \frac{\hat{r}^{\top}\hat{j}}
        {\lVert \hat{r}\rVert_2\lVert \hat{j}\rVert_2}
        =\hat{r}^{\top}\hat{j}
        """
    )
    st.markdown(
        """
        The retrieval model is deliberately simple and inspectable: encode the
        resume, search the local FAISS index, join the row IDs back to
        `models/jobs_meta.parquet`, then apply app-level seniority/location
        filters. This works because transformer embeddings capture semantic
        proximity beyond exact keywords; for example, "built APIs" can land
        near postings asking for "REST services."

        We used this normalization equation because raw transformer-vector
        magnitude is not the signal we want to rank by. A longer or more
        repetitive posting can produce a vector with a different norm, but the
        demo needs to answer a directional question: does the resume point
        toward the same semantic region as a job? Dividing by the L2 norm
        projects every resume and posting onto the unit hypersphere. Once both
        vectors have norm 1, cosine similarity collapses to a dot product, so
        FAISS can use fast inner-product search without changing the ranking.
        This is also why the match score should be read as semantic proximity,
        not as a probability of being hired.
        """
    )
    st.markdown("#### K-Means Market Segmentation")
    st.markdown(
        """
        We implement K-Means from scratch in `ml/clustering.py` using NumPy.
        It groups job embeddings into market neighborhoods that the demo can
        label with TF-IDF top terms and common titles.
        """
    )
    st.latex(
        r"""
        \min_{\{z_i\}_{i=1}^{n},\{\mu_k\}_{k=1}^{K}}
        \sum_{i=1}^{n}\left\lVert x_i-\mu_{z_i}\right\rVert_2^2
        =
        \min_{\{C_k\}_{k=1}^{K}}\sum_{k=1}^{K}\sum_{i \in C_k}
        \left\lVert x_i-\mu_k \right\rVert_2^2,
        \qquad
        \mu_k=\frac{1}{|C_k|}\sum_{i \in C_k}x_i
        """
    )
    st.plotly_chart(build_cluster_snapshot_figure(), use_container_width=True)
    st.markdown(
        """
        The fitted `k=8` clusters include software/engineering, business/data
        analysis, operations/logistics, healthcare, finance/accounting,
        sales/customer growth, and administrative/HR segments. The labels are
        explanatory summaries, not supervised classes. They come from the text
        distribution inside each centroid neighborhood.

        This objective matches the product goal because we do not initially
        have hand-labeled career families for every LinkedIn posting. K-Means
        gives us an unsupervised way to ask whether the embedding space already
        contains compact neighborhoods of similar roles. The assignment
        variable \(z_i\) chooses the nearest centroid for posting \(x_i\), and
        the update equation replaces each centroid with the mean of the vectors
        assigned to it. Repeating assignment and update steps reduces within
        cluster squared distance, which is exactly the notion of a "market
        neighborhood" we need for the demo's segment card and cluster map.
        The limitation is geometric: K-Means prefers roughly spherical,
        similarly sized clusters, so labels should be treated as summaries of
        nearby evidence rather than hard occupational categories.
        """
    )
    st.latex(
        r"""
        \operatorname{tfidf}(t,d)=
        \operatorname{tf}(t,d)\log\frac{N}{1+\operatorname{df}(t)}
        """
    )
    st.markdown(
        """
        TF-IDF is used only for interpretation, not for the core vector search.
        The term-frequency part rewards words that are common inside a cluster,
        while the inverse-document-frequency part discounts words that appear
        everywhere. That is useful for labeling centroids because generic terms
        such as "experience" or "team" are less explanatory than terms that are
        concentrated in a specific market neighborhood, such as "nursing,"
        "tax," "warehouse," or "backend." This gives the user human-readable
        reasons for a cluster assignment while preserving the transformer
        embedding as the actual representation used for retrieval.
        """
    )
    st.markdown("#### Salary Quantile Regression")
    st.markdown(
        """
        Salary is a distribution, so we predict q10/q25/q50/q75/q90 rather
        than a single scalar. `ml/salary_model.py` defines a raw PyTorch MLP
        with BatchNorm, ReLU, Dropout, and one output head per quantile. The
        loss is pinball loss:
        """
    )
    st.latex(
        r"""
        \rho_{\tau}(y-\hat{y})=
        \max\left(\tau(y-\hat{y}),(\tau-1)(y-\hat{y})\right)
        """
    )
    st.markdown(
        """
        We train two salary variants. The job-side model maps job embeddings
        and structured features to posting salaries. The active demo model
        retrains the same architecture on synthetic resume text paired with
        source-job salaries, reducing resume/job language domain shift.

        We used pinball loss because salary uncertainty is asymmetric and
        heteroskedastic. Being $20k below the true value should not be punished
        the same way for every requested percentile: a q10 prediction is
        supposed to sit below most targets, while a q90 prediction is supposed
        to sit above most targets. The coefficient \(\tau\) controls that
        asymmetry. Underestimating a high quantile receives a larger penalty
        than overestimating it, while the reverse is true for a low quantile.
        Training five heads with this loss lets the demo show a calibrated band
        instead of a brittle point estimate, and our evaluation checks whether
        observed salaries fall below each predicted quantile at approximately
        the advertised rate.
        """
    )
    st.markdown("#### Resume Quality And Capability")
    st.markdown(
        """
        Resume scoring lives in `ml/resume_assessment/`, not in Streamlit
        pages. The quality model combines experience depth, quantified impact,
        action-oriented specificity, structure, public-model signals, and
        evidence notes. Capability is separate from seniority: seniority asks
        which level is supported; capability asks how strong the resume is
        within that level.
        """
    )
    st.latex(
        r"""
        Q=0.45E+0.20I+0.20S+0.15R,
        \qquad
        \tilde{B}_q=B_q \cdot m_{\mathrm{capability}}
        """
    )
    st.markdown(
        """
        The quality equation is deliberately linear because the inputs are
        evidence categories we can explain to a user: experience depth $E$,
        quantified impact $I$, specificity $S$, and structure $R$. We
        gave experience the largest weight because career level and salary
        calibration are most sensitive to verified work history, then used
        impact and specificity to distinguish resumes with similar tenure but
        different evidence quality. Structure receives a smaller weight because
        formatting matters, but it should not dominate substantive experience.

        The capability multiplier is separated from the base salary band for
        the same reason. $B_q$ is the market evidence at quantile $q$, built
        from retrieved jobs, BLS/O*NET, and neural estimates. The multiplier
        $m_{\mathrm{capability}}$ adjusts that market anchor only after the
        resume's within-level evidence is assessed. This keeps the demo from
        mixing two different questions: what similar roles pay, and how strong
        this candidate's evidence looks within the inferred level.
        """
    )


def _render_demo_section() -> None:
    _paper_heading("3. How The Demo Uses The Models")
    st.markdown(
        """
        The Streamlit demo composes model artifacts rather than embedding ML
        logic in the UI. `app/runtime/` loads artifacts, wraps cache behavior,
        retrieves jobs, computes salary evidence, positions the resume in a
        cluster, and degrades gracefully when optional artifacts are absent.

        Synthetic sample resumes are created from source jobs by
        `scripts/synthetic_resumes/`. The generator infers role family and
        seniority, samples core and nice-to-have skills, assigns personas
        (`direct_match`, `under_qualified`, `over_qualified`,
        `career_switcher`), injects writing-style variation and limited noise,
        records quality labels, and attaches hard-negative jobs for retrieval
        evaluation. No full job descriptions are copied into the generated
        resumes.
        """
    )
    _metric_grid(
        (
            ("Profile", "track + seniority", "resume evidence parser"),
            ("Quality", "0-100", "experience, impact, specificity, structure"),
            ("Salary", "q10-q90", "retrieved, BLS, neural evidence"),
            ("Segment", "nearest centroid", "K-Means cluster position"),
        )
    )
    st.markdown(
        """
        A demo run proceeds as follows:

        1. Parse the resume text and extract structure, bullets, work-history
           dates, project specificity, track, seniority, and capability.
        2. Encode the resume and retrieve similar jobs from the FAISS index.
        3. Optionally score retrieved jobs with the public ATS-fit model.
        4. Build a salary band from retrieved-role salary quantiles, optional
           neural salary predictions, optional O*NET/BLS occupation wages, and
           quality/capability adjustments.
        5. Assign the resume to the nearest K-Means centroid and explain that
           segment through top TF-IDF terms.
        6. Show missing terms by comparing resume wording with top matched jobs
           and cluster terms, then build action advice for salary growth or
           cluster transition.
        """
    )


def _render_experiments_section() -> None:
    _paper_heading("4. Experiments And Results")
    st.markdown(
        """
        The local evaluation artifacts report two different kinds of evidence.
        The salary model is evaluated on 500 synthetic resume/source-job pairs.
        Calibration is close to nominal: q10/q25/q50/q75/q90 observed fractions
        are **0.110/0.254/0.526/0.772/0.920**, 80% interval coverage is
        **0.810**, 50% interval coverage is **0.518**, and median MAE is about
        **$27,025**.

        Retrieval exact-source recovery is intentionally interpreted more
        conservatively. On the same synthetic paired set, exact source-job
        recall@20 is **0.072** and MRR is **0.052**. That is too low to claim
        the system reliably recovers the precise original posting. It is still
        useful as a semantic-neighborhood retriever for surfacing comparable
        roles, salary evidence, and gap terms.
        """
    )
    st.plotly_chart(build_experiment_snapshot_figure(), use_container_width=True)
    st.markdown(
        """
        Public assessment baselines are also modest but useful as secondary
        signals: domain validation accuracy is **0.4698** across 24 labels,
        entity validation accuracy is **0.4143** across 11 labels, section
        validation accuracy is **0.7883**, and ATS-fit validation MAE is
        **18.78** points. These models corroborate resume evidence; they do not
        override the explainable scorer.
        """
    )


def _render_reproducibility_section() -> None:
    _paper_heading("5. Recreating The Local Models")
    st.markdown(
        """
        A new clone can reproduce the demo with the project commands below.
        Generated data, embeddings, FAISS indexes, and checkpoints are
        gitignored, so they must be built locally.
        """
    )
    st.code(
        """uv sync

# Download Kaggle LinkedIn Job Postings 2023-2024 into data/raw/.
# Then follow data/README.md for public_hf/ and public_dataturks/ resume files.

uv run python scripts/preprocess_data.py
uv run python scripts/build_index.py
uv run python scripts/build_clusters.py

uv run python scripts/generate_synthetic_resumes.py \\
    --jobs data/processed/jobs.parquet \\
    --n 500 \\
    --out data/eval/synthetic_resumes.parquet

uv run python scripts/train_resume_salary_model.py
uv run python scripts/train_salary_model.py \\
    --embeddings models/job_embeddings.npy \\
    --salaries data/processed/salaries.npy \\
    --jobs-parquet data/processed/jobs.parquet
uv run python scripts/train_quality_model.py
uv run python scripts/train_public_assessment_models.py

uv run python scripts/evaluate_retrieval.py
uv run python scripts/evaluate_salary.py

uv run streamlit run app/app.py""",
        language="bash",
    )


def _render_limitations_section() -> None:
    _paper_heading("6. Limitations And Future Work")
    st.markdown(
        """
        The strongest part of the current system is interpretability: every
        major prediction is tied to retrieved jobs, salary quantiles, cluster
        terms, resume evidence, or public baseline signals. The salary
        calibration experiment is encouraging for a course project, especially
        because quantile bands communicate uncertainty.

        The main weaknesses are data and validation. Exact retrieval of a
        synthetic source posting is weak, so results should be framed as
        comparable-market evidence rather than exact matching. Salary-bearing
        LinkedIn postings are biased toward roles that disclose pay, public
        resume datasets have noisy labels, and synthetic resumes cannot replace
        human-labeled career outcomes. Future work should add larger real
        resume/job outcome datasets, stronger negative mining, occupation-aware
        calibration, human evaluation of advice quality, and fairness checks
        across geography, school prestige, career-switching background, and
        seniority.
        """
    )
    st.markdown("</article>", unsafe_allow_html=True)


def _paper_heading(title: str) -> None:
    st.markdown(
        f"""
        <div class="method-section-heading">
            <h2>{escape(title)}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_grid(items: tuple[tuple[str, str, str], ...]) -> None:
    html = "".join(
        '<div class="method-metric">'
        f'<div class="method-metric-label">{escape(label)}</div>'
        f'<div class="method-metric-value">{escape(value)}</div>'
        f'<div class="method-metric-copy">{escape(copy)}</div>'
        "</div>"
        for label, value, copy in items
    )
    st.markdown(f'<div class="method-metric-grid">{html}</div>', unsafe_allow_html=True)
