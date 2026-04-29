"""
tests/test_real_resume_assessment.py

Tests for resume quality assessment across a wider range of real resume types.
Verifies that score_resume_quality() produces sensible, ordered results for
diverse candidate profiles: entry-level, senior, career-changer, non-technical,
healthcare, typo-heavy, and minimal resumes.

"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.quality import score_resume_quality  # noqa: E402

# ---------------------------------------------------------------------------
# Resume fixtures — diverse candidate profiles
# ---------------------------------------------------------------------------

SENIOR_ENGINEER = """
Jane Smith — San Francisco, CA
Senior Software Engineer | 8 years experience

Skills: Python, Go, Kubernetes, AWS, PostgreSQL, Redis, gRPC, system design, CI/CD

Experience:
- Architected a distributed event pipeline processing 120K events/sec with p99 < 20ms.
- Led migration of monolith to microservices, reducing deploy time from 4 hours to 12 minutes.
- Mentored 6 engineers; 3 promoted to senior during her tenure.
- Reduced infrastructure costs by $340K/year through spot-instance scheduling.

Education: B.S. Computer Science, Stanford University

Publications: Co-authored paper on distributed tracing at SREcon 2023.
"""

JUNIOR_ML = """
Kevin Park — New York, NY
Junior ML Engineer | 1 year experience

Skills: Python, PyTorch, scikit-learn, pandas, NumPy, machine learning

Experience:
- Built a text classification pipeline achieving 91% F1 on internal benchmark.
- Deployed a recommendation model serving 10K daily active users.
- Wrote unit tests raising coverage from 40% to 78%.

Education: B.S. Computer Science, NYU, GPA 3.8
"""

DATA_ANALYST = """
Maria Lopez — Chicago, IL
Data Analyst | 4 years experience

Skills: SQL, Python, Tableau, Looker, dbt, Excel, A/B testing, business metrics

Experience:
- Built dashboards used by 80 weekly stakeholders across product and finance.
- Designed and analyzed 12 A/B experiments, two of which shipped to 2M users.
- Automated reporting workflows saving 30 analyst-hours per month.
- Identified 11% conversion lift in checkout funnel analysis.

Education: B.A. Statistics, University of Chicago
"""

CAREER_CHANGER = """
Tom Richards — Austin, TX
Former teacher transitioning to data analytics.

I spent 5 years teaching high school math and recently completed a data analytics
bootcamp. I learned SQL, Tableau, and some Python. I am excited to break into tech
and apply my analytical thinking from teaching to data problems.

Education: B.A. Education, Texas State University
Data Analytics Certificate, General Assembly 2024
"""

HEALTHCARE = """
Emily Chen, RN — Boston, MA
Registered Nurse | 6 years experience

Clinical Skills: patient assessment, IV placement, triage, EMR documentation,
medication administration, wound care, ACLS certified

Experience:
- Managed care for 8-10 ICU patients per shift in a 32-bed unit.
- Reduced medication errors by 22% through new double-check protocol.
- Trained 15 new nursing graduates over two years.
- Served as charge nurse covering unit staffing decisions.

Education: B.S. Nursing, Boston University
"""

MINIMAL_WEAK = """
John Doe

Looking for a job. I am a hard worker and fast learner.
I have done some projects and worked at a few places.
"""

TYPO_HEAVY = """
Daniel Prk — Auston, TX
Mid-level data anylst with 5 years experiance in retial analytics.

Skils: SQL, Excell, Tableau, Looker
- Bilt dashbords for stakehldrs.
- Wrked with the prodcut team on campains.
- Identifed convertion oppertunities.

Educaton: B.A. Economcis, UT Austni
"""

STRONG_RESEARCH = """
Aisha Johnson — Cambridge, MA
PhD Candidate, Machine Learning | Expected 2025

Research: transformer architectures, efficient attention, multi-modal learning

Publications:
- Johnson et al. (2024). Efficient Sparse Attention for Long Documents. NeurIPS 2024.
- Johnson & Lee (2023). Cross-modal Alignment without Paired Data. ICML 2023.

Experience:
- Research intern at Google Brain (2023): shipped sparse attention kernel reducing
  inference cost by 40% on 16K-token sequences.
- Teaching assistant for graduate ML course, 3 semesters.

Skills: Python, JAX, PyTorch, CUDA, distributed training, research writing
"""

PRODUCT_MANAGER = """
Sarah Kim — Seattle, WA
Senior Product Manager | 6 years experience

Skills: product strategy, roadmap planning, stakeholder management, A/B testing,
SQL, user research, agile, Jira, Figma

Experience:
- Led end-to-end launch of mobile payments feature used by 4M customers.
- Increased checkout conversion by 18% through iterative A/B testing over 6 months.
- Managed cross-functional team of 12 across engineering, design, and marketing.
- Defined and owned OKRs across 3 product areas for 2 fiscal years.

Education: MBA, Kellogg School of Management; B.S. Industrial Engineering, Purdue
"""


# ---------------------------------------------------------------------------
# Basic contract tests — output shape and type
# ---------------------------------------------------------------------------


def test_score_returns_dict_with_required_keys():
    """score_resume_quality() must return a dict with score and label at minimum."""
    result = score_resume_quality(JUNIOR_ML)
    assert isinstance(result, dict)
    assert "score" in result
    assert "label" in result


def test_score_is_numeric_in_range():
    """Score must be a number between 0 and 100."""
    result = score_resume_quality(SENIOR_ENGINEER)
    score = result["score"]
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100


def test_label_is_valid_category():
    """Label must be one of the expected quality tiers."""
    valid_labels = {"weak", "medium", "strong"}
    for resume in [SENIOR_ENGINEER, JUNIOR_ML, MINIMAL_WEAK]:
        result = score_resume_quality(resume)
        assert result["label"] in valid_labels, (
            f"Unexpected label '{result['label']}' for resume starting: {resume[:40]}"
        )


# ---------------------------------------------------------------------------
# Ordering tests — stronger resumes should score higher
# ---------------------------------------------------------------------------


def test_senior_scores_higher_than_junior():
    """A senior engineer resume should score higher than a junior one."""
    senior_score = score_resume_quality(SENIOR_ENGINEER)["score"]
    junior_score = score_resume_quality(JUNIOR_ML)["score"]
    assert senior_score >= junior_score, (
        f"Senior ({senior_score:.1f}) should be >= Junior ({junior_score:.1f})"
    )


def test_strong_resume_scores_higher_than_minimal():
    """A detailed resume should score significantly higher than a minimal one."""
    strong_score = score_resume_quality(SENIOR_ENGINEER)["score"]
    weak_score = score_resume_quality(MINIMAL_WEAK)["score"]
    assert strong_score > weak_score + 10, (
        f"Strong ({strong_score:.1f}) should beat Weak ({weak_score:.1f}) by >10pts"
    )


def test_research_resume_scores_well():
    """A PhD candidate with publications should score in medium or strong tier."""
    result = score_resume_quality(STRONG_RESEARCH)
    assert result["label"] in {"medium", "strong"}, (
        f"Research resume scored '{result['label']}' ({result['score']:.1f})"
    )


def test_minimal_resume_scores_weak():
    """A one-paragraph vague resume should be classified as weak."""
    result = score_resume_quality(MINIMAL_WEAK)
    assert result["label"] == "weak", (
        f"Minimal resume should be 'weak', got '{result['label']}' ({result['score']:.1f})"
    )


# ---------------------------------------------------------------------------
# Domain diversity tests — non-tech resumes should still be handled
# ---------------------------------------------------------------------------


def test_healthcare_resume_does_not_crash():
    """A nursing resume should return a valid score without errors."""
    result = score_resume_quality(HEALTHCARE)
    assert "score" in result
    assert 0 <= result["score"] <= 100


def test_product_manager_resume_does_not_crash():
    """A PM resume should return a valid score without errors."""
    result = score_resume_quality(PRODUCT_MANAGER)
    assert "score" in result
    assert result["label"] in {"weak", "medium", "strong"}


def test_career_changer_resume_does_not_crash():
    """A career changer resume should return a valid score without errors."""
    result = score_resume_quality(CAREER_CHANGER)
    assert "score" in result
    assert 0 <= result["score"] <= 100


def test_data_analyst_scores_medium_or_above():
    """A solid data analyst resume should not be classified as weak."""
    result = score_resume_quality(DATA_ANALYST)
    assert result["label"] in {"medium", "strong"}, (
        f"Data analyst resume should not be 'weak', got '{result['label']}'"
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_typo_heavy_resume_scores_lower_than_clean_equivalent():
    """A typo-heavy resume should score lower than an otherwise similar clean one."""
    typo_score = score_resume_quality(TYPO_HEAVY)["score"]
    clean_score = score_resume_quality(DATA_ANALYST)["score"]
    assert clean_score >= typo_score, (
        f"Clean ({clean_score:.1f}) should be >= Typo-heavy ({typo_score:.1f})"
    )


def test_empty_string_does_not_crash():
    """An empty resume input should return a weak score without raising."""
    result = score_resume_quality("")
    assert result["label"] == "weak"


def test_whitespace_only_does_not_crash():
    """A whitespace-only input should not raise an exception."""
    result = score_resume_quality("   \n\n\t  ")
    assert "score" in result
    assert 0 <= result["score"] <= 100


def test_very_long_resume_does_not_crash():
    """A very long resume (10x normal length) should not raise or time out."""
    long_resume = SENIOR_ENGINEER * 10
    result = score_resume_quality(long_resume)
    assert "score" in result


def test_all_fixture_resumes_return_consistent_types():
    """All sample fixtures should return the same dict shape."""
    fixtures = [
        SENIOR_ENGINEER,
        JUNIOR_ML,
        DATA_ANALYST,
        CAREER_CHANGER,
        HEALTHCARE,
        MINIMAL_WEAK,
        TYPO_HEAVY,
        STRONG_RESEARCH,
        PRODUCT_MANAGER,
    ]
    for resume in fixtures:
        result = score_resume_quality(resume)
        assert isinstance(result.get("score"), (int, float))
        assert isinstance(result.get("label"), str)
