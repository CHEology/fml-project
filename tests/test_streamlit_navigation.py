from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _function_source(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    next_def = source.index("\ndef ", start + 1)
    return source[start:next_def]


def test_methodology_is_extracted_to_shared_renderer() -> None:
    methodology_component = PROJECT_ROOT / "app" / "components" / "methodology.py"

    assert methodology_component.exists()
    assert "render_methodology_page" in methodology_component.read_text(
        encoding="utf-8"
    )


def test_sidebar_navigation_exposes_named_pages() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")

    assert "st.navigation" in app_source
    assert 'position="hidden"' in app_source
    assert "st.page_link" in app_source
    assert 'title="Home"' in app_source
    assert 'title="Demo"' in app_source
    assert 'title="Market Overview"' in app_source
    assert 'title="Methodology"' in app_source
    assert "st.tabs(" not in app_source


def test_home_page_has_team_and_mascot_asset() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    team_source = (PROJECT_ROOT / "app" / "components" / "team.py").read_text(
        encoding="utf-8"
    )
    mascot = PROJECT_ROOT / "app" / "assets" / "resumatch-mascot.png"

    assert "render_home_page" in app_source
    assert "Lucky Hamsters" in team_source
    for contributor in (
        "Omer Hortig",
        "Tanvi Patel",
        "Ryan Lu",
        "Alan He",
        "Eliguli Han",
    ):
        assert contributor in team_source
    assert mascot.exists()


def test_demo_page_uses_single_stage_wizard() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")

    assert 'st.session_state.demo_stage == "input"' in demo_source
    assert 'st.session_state.demo_stage == "snapshot"' in demo_source
    assert 'st.session_state.demo_stage == "market"' in demo_source
    assert 'st.session_state.demo_stage == "gaps"' in demo_source
    assert 'st.session_state.demo_stage = "snapshot"' in demo_source
    assert 'st.session_state.demo_stage = "market"' in demo_source
    assert 'st.session_state.demo_stage = "gaps"' in demo_source
    assert "Start over with new resume/profile" in demo_source
    assert "← Previous" in demo_source


def test_demo_removes_intro_hero_and_static_metrics() -> None:
    app_source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    demo_source = _function_source(app_source, "render_demo_page")
    market_source = _function_source(app_source, "render_market_overview_page")

    assert "Understand role fit, salary range, and market position." not in demo_source
    assert "Resume market intelligence" not in demo_source
    assert "Jobs loaded" not in demo_source
    assert "Median salary" not in demo_source
    assert "Jobs loaded" in market_source
    assert "Median salary" in market_source
