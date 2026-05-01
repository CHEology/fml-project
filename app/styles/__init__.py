from __future__ import annotations

from pathlib import Path

import streamlit as st

THEMES = {
    "Dark": {
        "bg_start": "#0a0a0a",
        "bg_end": "#0a0a0a",
        "flare_a": "transparent",
        "flare_b": "transparent",
        "panel": "#111111",
        "ink": "#f0ede8",
        "muted": "#6b6560",
        "line": "rgba(255,255,255,0.06)",
        "pill_bg": "rgba(232,160,69,0.15)",
        "pill_ink": "#ffd6a0",
        "hero_a": "#1a0e06",
        "hero_b": "#e8a045",
        "shadow": "rgba(0, 0, 0, 0.50)",
        "score_bg": "rgba(232,160,69,0.18)",
        "score_ink": "#ffd6a0",
    },
    "Lavender": {
        "bg_start": "#FFFFFF",
        "bg_end": "#FAF9FF",
        "flare_a": "transparent",
        "flare_b": "transparent",
        "panel": "#FFFFFF",
        "ink": "#111827",
        "muted": "#6B7280",
        "line": "#F3F4F6",
        "pill_bg": "#F5F3FF",
        "pill_ink": "#7C3AED",
        "hero_a": "#F5F3FF",
        "hero_b": "#DDD6FE",
        "shadow": "rgba(124, 58, 237, 0.1)",
        "score_bg": "#EDE9FE",
        "score_ink": "#7C3AED",
    },
}

STYLE_FILES = (
    "foundation.css",
    "snapshot.css",
    "snapshot-evidence.css",
    "demo.css",
    "market.css",
    "home-sidebar.css",
    "quality-responsive.css",
    "buttons-typography.css",
)


def inject_styles(theme_name: str = "Lavender") -> None:
    theme = THEMES.get(theme_name, THEMES["Lavender"])
    css = "\n\n".join(
        (Path(__file__).with_name(filename)).read_text(encoding="utf-8")
        for filename in STYLE_FILES
    )
    for placeholder, value in {
        "__BG_START__": theme["bg_start"],
        "__BG_END__": theme["bg_end"],
        "__PANEL__": theme["panel"],
        "__INK__": theme["ink"],
        "__MUTED__": theme["muted"],
        "__LINE__": theme["line"],
        "__PILL_BG__": theme["pill_bg"],
        "__PILL_INK__": theme["pill_ink"],
        "__HERO_A__": theme["hero_a"],
        "__HERO_B__": theme["hero_b"],
        "__SHADOW__": theme["shadow"],
        "__BUTTON__": theme.get("button", "#7C3AED"),
        "__BUTTON_HOVER__": theme.get("button_hover", "#6D28D9"),
        "__BUTTON_SHADOW__": theme.get("button_shadow", "rgba(124, 58, 237, 0.2)"),
        "__SCORE_BG__": theme["score_bg"],
        "__SCORE_INK__": theme["score_ink"],
    }.items():
        css = css.replace(placeholder, value)
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
