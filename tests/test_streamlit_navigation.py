from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    for contributor in ("Omer Hortig", "Tanvi Patel", "Ryan Lu", "Alan He", "Eliguli Han"):
        assert contributor in team_source
    assert mascot.exists()
