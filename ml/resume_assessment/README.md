# Resume Assessment

This package is the shared source of truth for resume parsing and explainable
assessment.

- `structure.py`: section, bullet, link, and word-count signals.
- `work_history.py`: employment date parsing, seniority evidence, academic CV
  signals, and non-work date filtering.
- `projects.py`: bullet/project specificity and impact scoring.
- `profile.py`: track and seniority detection.
- `quality.py`: explainable quality and capability-tier scoring.
- `salary.py`: quality/capability salary adjustments and seniority salary
  filtering helpers.
- `career_actions.py`: deterministic salary-growth and cluster-transition
  advice derived from local job, salary, and cluster evidence.
- `taxonomy.py`: local role, skill, title, and evidence lexicons.

Streamlit pages and scripts should call this package instead of duplicating
resume assessment heuristics.
