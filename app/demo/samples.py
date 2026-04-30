from __future__ import annotations

import re

import numpy as np
import pandas as pd
from app.demo.sample_data import (
    FAKE_COMPANIES,
    FAKE_SCHOOLS,
    SAMPLE_RESUME_SPECS,
    TRACK_METRICS,
)
from ml.resume_assessment.taxonomy import (
    FALLBACK_TRACK_SKILLS,
    SAMPLE_FIELD_SKILLS,
    TRACK_KEYWORDS,
    TRACK_SKILLS,
)


def track_job_subset(jobs: pd.DataFrame, track: str) -> pd.DataFrame:
    if jobs.empty:
        return jobs

    searchable = (
        jobs["title"].astype(str)
        + " "
        + jobs["company_name"].astype(str)
        + " "
        + jobs["text"].astype(str)
    ).str.lower()

    mask = pd.Series(False, index=jobs.index)
    for keyword in TRACK_KEYWORDS[track]:
        mask = mask | searchable.str.contains(keyword, regex=False, na=False)

    subset = jobs.loc[mask].copy()
    return subset if not subset.empty else jobs.copy()


def market_skill_stack(jobs: pd.DataFrame, track: str, limit: int = 8) -> list[str]:
    subset = track_job_subset(jobs, track)
    searchable = (
        subset["title"].astype(str) + " " + subset["text"].astype(str)
    ).str.lower()

    track_skills = TRACK_SKILLS.get(track, FALLBACK_TRACK_SKILLS)
    ranked_skills: list[tuple[int, str]] = []
    for skill in track_skills:
        count = int(searchable.str.count(skill.lower()).sum())
        ranked_skills.append((count, skill))

    ranked_skills.sort(key=lambda item: (-item[0], item[1]))
    ordered = [skill for count, skill in ranked_skills if count > 0]
    for skill in track_skills:
        if skill not in ordered:
            ordered.append(skill)
    return ordered[:limit]


def choose_market_examples(
    jobs: pd.DataFrame,
    track: str,
    preferred_location: str,
) -> pd.DataFrame:
    subset = track_job_subset(jobs, track)
    if preferred_location != "Anywhere":
        location_mask = subset["location"].astype(str).str.contains(
            preferred_location,
            case=False,
            na=False,
        ) | subset["state"].astype(str).str.fullmatch(
            preferred_location,
            case=False,
            na=False,
        )
        if location_mask.any():
            subset = subset.loc[location_mask].copy()

    columns = ["title", "company_name", "location", "text"]
    for column in columns:
        if column not in subset.columns:
            subset[column] = ""
    return subset[columns].drop_duplicates().head(6)


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".")
    return slug or "candidate"


def compose_headline(seniority: str, title: str) -> str:
    title_clean = re.sub(r"\s+", " ", title).strip()
    seniority_clean = seniority.strip()
    if title_clean.lower().startswith(seniority_clean.lower()):
        return title_clean
    return f"{seniority_clean} {title_clean}"


def generate_premade_sample_resume(
    spec: tuple[str, str, str, str, str, str, str],
    jobs: pd.DataFrame,
) -> str:
    """Render one curated sample profile from the 50-profile sample library."""
    name, track, title, seniority, location, domain, quality = spec
    seed = sum((idx + 1) * ord(char) for idx, char in enumerate("|".join(spec)))
    rng = np.random.default_rng(seed)
    slug = slugify_name(name)
    school = str(rng.choice(FAKE_SCHOOLS))
    skills = SAMPLE_FIELD_SKILLS.get(
        track, TRACK_SKILLS.get(track, FALLBACK_TRACK_SKILLS)
    )
    primary, secondary, tertiary = skills[:3]
    metric_name = TRACK_METRICS.get(track, "operating quality")
    market_examples = choose_market_examples(jobs, track, location)
    companies = (
        market_examples["company_name"]
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(companies) < 2:
        companies.extend(
            company for company in FAKE_COMPANIES if company not in companies
        )
    company_a, company_b = companies[:2]

    if quality == "excellent":
        impact_one = int(rng.integers(24, 48))
        impact_two = int(rng.integers(16, 34))
        impact_three = int(rng.integers(8, 21))
        scope_one = int(rng.integers(6, 18))
        scope_two = int(rng.integers(120, 900))
        current_start = "2021"
        prior_range = "2017 - 2021"
        earlier_range = "2014 - 2017"
        project_count = int(rng.integers(3, 8))
        return f"""{name}
{title}
{location} | {slug}@example.com | linkedin.com/in/{slug} | github.com/{slug}

PROFESSIONAL SUMMARY
{seniority} {track.lower()} professional focused on {domain}. Known for combining {primary}, {secondary}, and {tertiary} with crisp operating judgment, measurable delivery, and cross-functional leadership across technical and business teams.

CORE SKILLS
{" | ".join(skills)}

EXPERIENCE
{company_a} | {title} | {current_start} - Present
- Led the {domain} roadmap across {scope_one} teams, using {primary} and {secondary} to improve {metric_name} by {impact_one}% while reducing review cycles by {impact_three}%.
- Built a weekly executive scorecard covering {scope_two:,}+ records, surfacing risk, ownership, and next actions for product, finance, and operations leaders.
- Mentored 4 team members on evidence-backed problem framing, raising peer review quality and shortening stakeholder turnaround by {impact_two}%.
- Partnered with legal, finance, and customer-facing teams to convert ambiguous requests into scoped delivery plans with clear metrics, owners, and launch criteria.

{company_b} | {compose_headline("Mid", title.replace("Senior ", "").replace("Lead ", ""))} | {prior_range}
- Owned a high-priority {domain} initiative from discovery through rollout, increasing adoption by {impact_two}% across {scope_one - 1 if scope_one > 7 else scope_one + 2} business units.
- Rebuilt the reporting and documentation workflow in {tertiary}, cutting manual reconciliation from 9 hours to 3 hours per week.
- Presented monthly findings to VP-level stakeholders and turned feedback into a sequenced roadmap with measurable acceptance criteria.

{str(rng.choice(FAKE_COMPANIES))} | Associate {track.split("/")[0].strip()} Specialist | {earlier_range}
- Supported operating reviews, QA checks, and customer research for a {domain} portfolio serving {int(scope_two / 2):,}+ users or records.
- Created reusable templates that improved handoffs between analysts, operators, and managers.

SELECTED PROJECTS
- {domain.title()} Control Room: Built a reusable dashboard and decision log connecting leading indicators, owner actions, and post-launch impact.
- Evidence Quality Rubric: Created a {project_count}-part review framework that helped teams distinguish activity updates from measurable outcomes.
- Stakeholder Briefing Pack: Standardized monthly leadership narratives with metric definitions, risks, and recommended next steps.

EDUCATION
- B.S. in {track.split("/")[0].strip()} / Business Analytics, {school}

CERTIFICATIONS
- Advanced {primary} for Practitioners
- Cross-Functional Leadership Workshop
"""

    if quality == "solid":
        impact_one = int(rng.integers(10, 28))
        impact_two = int(rng.integers(7, 19))
        return f"""{name}
{title}
{location} | {slug}@example.com | linkedin.com/in/{slug}

SUMMARY
{track} professional with practical experience in {domain}, {primary}, and {secondary}. Comfortable working with managers and cross-functional partners to improve recurring workflows.

SKILLS
{" | ".join(skills[:6])}

EXPERIENCE
{company_a} | {title} | 2020 - Present
- Managed day-to-day {domain} work using {primary} and {secondary}, improving {metric_name} by {impact_one}% over two planning cycles.
- Created status reports and process documentation for 5 stakeholder groups.
- Coordinated issue triage, follow-up actions, and handoffs between internal teams.

{company_b} | Associate {track.split("/")[0].strip()} Specialist | 2018 - 2020
- Supported reporting, quality checks, and stakeholder requests for a busy {domain} team.
- Reduced recurring manual work by {impact_two}% by standardizing templates and review steps.

PROJECTS
- {domain.title()} Tracker: Built a shared tracker for open issues, owners, deadlines, and basic trend reporting.

EDUCATION
- B.A. in Business Administration, {school}
"""

    return f"""{name}
{title}
{location} | {slug}@example.com

Summary
Motivated {track.lower()} candidate interested in {domain}. I am organized, dependable, and eager to grow.

Experience
{company_a} | {title} | 2023 - Present
- Responsible for helping with {domain} tasks.
- Worked on reports and team requests.

{company_b} | Intern | Summer 2022
- Helped the team with research and coordination.

Education
- Coursework in business and technology, {school}

Skills
- {", ".join(skills[:4])}
"""


def random_premade_sample_resume(
    jobs: pd.DataFrame,
    previous_index: int | None = None,
) -> tuple[str, str, int]:
    rng = np.random.default_rng()
    choices = list(range(len(SAMPLE_RESUME_SPECS)))
    if previous_index in choices and len(choices) > 1:
        choices.remove(previous_index)
    index = int(rng.choice(choices))
    spec = SAMPLE_RESUME_SPECS[index]
    text = generate_premade_sample_resume(spec, jobs)
    source = f"Random sample resume: {spec[0]}, {spec[2]}"
    return text, source, index


def linkedin_dataset_note(has_real_data: bool) -> str:
    if has_real_data:
        return "Using the local LinkedIn job catalog."
    return "Using sample roles because the local job catalog is not available yet."
