"""Calibration tests for the resume / non-resume validator.

The validator is the gate that scopes the app to actual CVs. Two
failure modes matter and we test both: (a) falsely rejecting real
resumes (especially poor / minimal ones), and (b) falsely accepting
portfolio pages, blog posts, social bios, and code dumps.
"""

from __future__ import annotations

from ml.public_assessment import validate_resume_quality

YAO_CV = """Shunyu Yao, Ph.D. shunyu.yao.physics@gmail.com
Employment History
Sept 2025 - present  Senior Staff research scientist at Google DeepMind, core-Gemini team.
March 2025 - Sept 2025  Member of technical staff, research scientist at Anthropic PBC.
Oct 2024 - Feb 2025  Member of technical staff, resident at Anthropic PBC.
Sept 2024 - Sept 2024  Postdoctoral Researcher, UC Berkeley CTP.
Education
2019 - 2024  Ph.D., Stanford University Theoretical and mathematical physics.
2015 - 2019  B.Sc., Tsinghua University Physics.
Skills
- Reinforcement learning, prompt engineering, Python, Julia, Mathematica
Awards: Clark fellowship Stanford 2022, Tsinghua Presidential award 2018."""

POOR_CV = """John Smith
john@example.com
Work
2022 - 2024 worked at small startup as developer
2021 - 2022 intern at company
Education
2018 - 2021 BS Computer Science State University
Skills
python, javascript"""

MINIMAL_CV = """Jane Doe
Education: BS State University 2024
Skills: Python, SQL"""

PORTFOLIO_PAGE = """Welcome to my site!
Home About Blog Contact
Subscribe to newsletter. Read more about my work.
Privacy Policy | Terms of Service. Cookie consent required.
© 2025 Jane Doe. All rights reserved.
Follow me on Twitter and Instagram. Powered by Wordpress."""

BLOG_POST = (
    "Machine learning is transforming industries across the globe. "
    "From healthcare to finance, the applications are vast and growing every day. "
    "In this post we explore how recent advances in deep learning, reinforcement "
    "learning, and large language models are reshaping how organizations make "
    "decisions and interact with customers. The field has seen remarkable progress "
    "in the past decade. Researchers have published thousands of papers describing "
    "new architectures, training techniques, and applications. Companies large "
    "and small are racing to integrate these capabilities into their products. "
    "Yet many fundamental questions remain unanswered, and the path forward is "
    "anything but clear."
)

CODE_DUMP = """def hello_world():
    print("hello")

class MyClass:
    def __init__(self):
        self.value = 42

import numpy as np
import pandas as pd
"""

SOCIAL_BIO = (
    "Aspiring engineer. Coffee lover. Building things at the intersection of "
    "design and code. Tweets are my own. He/him."
)


def test_real_cv_lands_in_high_confidence() -> None:
    r = validate_resume_quality(None, YAO_CV)
    assert r["confidence"] == "high"
    assert r["is_resume"] is True
    assert r["score"] >= 0.55


def test_poor_real_cv_is_not_falsely_rejected() -> None:
    # A thin but real CV must clear at least "medium" so the UI soft-warns
    # rather than blocking analysis.
    r = validate_resume_quality(None, POOR_CV)
    assert r["confidence"] in {"high", "medium"}
    assert r["is_resume"] is True


def test_minimal_real_cv_lands_at_medium_floor() -> None:
    # Section header + degree -> "medium" floor; we never want a real CV
    # silently blocked just because it is short.
    r = validate_resume_quality(None, MINIMAL_CV)
    assert r["confidence"] in {"high", "medium"}
    assert r["is_resume"] is True


def test_portfolio_page_is_rejected() -> None:
    r = validate_resume_quality(None, PORTFOLIO_PAGE)
    assert r["confidence"] == "low"
    assert r["is_resume"] is False
    # The reasons surface so the UI can explain *why* the input was blocked.
    assert any("nav" in reason.lower() or "page" in reason.lower() for reason in r["reasons"])


def test_blog_post_is_rejected() -> None:
    r = validate_resume_quality(None, BLOG_POST)
    assert r["confidence"] == "low"
    assert r["is_resume"] is False


def test_code_dump_is_rejected() -> None:
    r = validate_resume_quality(None, CODE_DUMP)
    assert r["confidence"] == "low"
    assert r["is_resume"] is False


def test_social_bio_is_rejected() -> None:
    r = validate_resume_quality(None, SOCIAL_BIO)
    assert r["confidence"] == "low"
    assert r["is_resume"] is False


def test_empty_input_is_empty_confidence() -> None:
    r = validate_resume_quality(None, "")
    assert r["confidence"] == "empty"
    assert r["is_resume"] is False
