import re

with open("app/app.py", "r") as f:
    content = f.read()

# 1. Update THEMES
content = re.sub(
    r'"Light": \{.*?"bg_start": "#f7f8fa".*?"bg_end": "#eef1f4".*?"ink": "#17202a".*?"line": "#d9dee7".*?"shadow": "rgba\(16, 24, 40, 0.08\)".*?\},',
    r'''"Light": {
        "bg_start": "#f6f8fb",
        "bg_end": "#f6f8fb",
        "flare_a": "transparent",
        "flare_b": "transparent",
        "panel": "#ffffff",
        "ink": "#111827",
        "muted": "#667085",
        "line": "#e5e7eb",
        "pill_bg": "#edf4ff",
        "pill_ink": "#175cd3",
        "hero_a": "#ffffff",
        "hero_b": "#ffffff",
        "shadow": "rgba(15, 23, 42, 0.04)",
        "score_bg": "#ecfdf3",
        "score_ink": "#027a48",
    },''',
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'"Dark": \{.*?"bg_start": "#101418".*?"bg_end": "#151a21".*?"panel": "#1b222b".*?"line": "#2d3642".*?\},',
    r'''"Dark": {
        "bg_start": "#0f1115",
        "bg_end": "#0f1115",
        "flare_a": "transparent",
        "flare_b": "transparent",
        "panel": "#1b1d22",
        "ink": "#f2f4f7",
        "muted": "#a5adba",
        "line": "#374151",
        "pill_bg": "#182b45",
        "pill_ink": "#84caff",
        "hero_a": "#1b222b",
        "hero_b": "#1b222b",
        "shadow": "rgba(0, 0, 0, 0.24)",
        "score_bg": "#12372a",
        "score_ink": "#75e0a7",
    },''',
    content,
    flags=re.DOTALL
)

# 2. Update CSS in inject_styles()
# border-radius on cards
content = re.sub(
    r'\.metric-card, \.info-card, \.job-card \{(.*?border-radius: )8px(.*?)\}',
    r'.metric-card, .info-card, .job-card {\g<1>18px\g<2>}',
    content,
    flags=re.DOTALL
)

# add css to prevent wrapping
css_addition = """
        .signal-value, .metric-value { word-break: normal; overflow-wrap: break-word; hyphens: none; }
"""
content = content.replace('.info-title {', css_addition + '\n        .info-title {')

with open("app/app.py", "w") as f:
    f.write(content)
