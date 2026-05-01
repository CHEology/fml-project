from __future__ import annotations

from base64 import b64encode
from functools import lru_cache


@lru_cache(maxsize=1)
def get_plot_graph_background_uris() -> tuple[str, str]:
    try:
        from scripts.plot_graph import plot_background_png_bytes

        market_png, salary_png = plot_background_png_bytes()
    except Exception:
        return "", ""

    market_uri = "data:image/png;base64," + b64encode(market_png).decode("ascii")
    salary_uri = "data:image/png;base64," + b64encode(salary_png).decode("ascii")
    return market_uri, salary_uri
