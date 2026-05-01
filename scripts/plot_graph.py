from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PRIMARY_FILENAME = "plot_graph_market.png"
SECONDARY_FILENAME = "plot_graph_salary.png"


def _quadratic_curve(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    *,
    steps: int = 140,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    omt = 1.0 - t
    return (
        (omt**2)[:, None] * start
        + (2.0 * omt * t)[:, None] * control
        + (t**2)[:, None] * end
    )


def make_market_background_figure(*, seed: int = 7) -> plt.Figure:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="#061019")
    ax.set_facecolor("#061019")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            1.0,
            facecolor="none",
            edgecolor=(0.96, 0.97, 0.99, 0.22),
            linewidth=1.8,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            0.42,
            facecolor=(0.34, 0.77, 0.94, 0.05),
            edgecolor="none",
        )
    )
    ax.plot([-1, 1], [0, 0], color=(1, 1, 1, 0.08), linewidth=0.8)
    ax.plot([0, 0], [-1, 1], color=(1, 1, 1, 0.08), linewidth=0.8)

    hues = np.linspace(0.56, 0.92, 168)
    for idx, hue in enumerate(hues):
        theta_0 = rng.uniform(0.0, 2.0 * np.pi)
        theta_1 = theta_0 + rng.uniform(0.42, 2.45)
        start = np.array([np.cos(theta_0), np.sin(theta_0)], dtype=np.float32)
        end = np.array([np.cos(theta_1), np.sin(theta_1)], dtype=np.float32)
        midpoint = (start + end) / 2.0
        radial_pull = rng.uniform(0.08, 0.78)
        twist = rng.uniform(-0.18, 0.18)
        control = midpoint * radial_pull + np.array(
            [-midpoint[1] * twist, midpoint[0] * twist],
            dtype=np.float32,
        )
        curve = _quadratic_curve(start, control, end)
        radius = np.linalg.norm(curve, axis=1)
        curve = curve[radius <= 1.02]
        if len(curve) < 24:
            continue

        color = matplotlib.colors.hsv_to_rgb((hue % 1.0, 0.68, 0.88))
        alpha = 0.26 + 0.48 * (1.0 - idx / len(hues))
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color=(*color, alpha),
            linewidth=0.75,
        )

    fig.tight_layout(pad=0)
    return fig


def make_salary_background_figure(*, seed: int = 13) -> plt.Figure:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor="#07121c")
    ax.set_facecolor("#07121c")
    ax.axis("off")
    ax.set_xlim(-8.6, 8.6)
    ax.set_ylim(-0.2, 8.6)

    ax.plot([-9, 9], [0, 0], color=(1, 1, 1, 0.14), linewidth=1.1)
    ax.plot([0, 0], [0, 8.2], color=(1, 1, 1, 0.08), linewidth=0.7)
    ax.add_patch(
        Circle(
            (0.0, 5.4),
            2.1,
            facecolor=(0.35, 0.74, 0.98, 0.06),
            edgecolor="none",
        )
    )

    for idx in range(136):
        center = rng.normal(0.0, 0.42)
        span = rng.uniform(1.15, 7.1)
        height = rng.uniform(0.35, 7.2)
        x = np.linspace(-1.0, 1.0, 220, dtype=np.float32)
        y = np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0)) * height
        x_scaled = center + x * span

        hue = 0.54 + 0.28 * (idx / 136.0)
        color = matplotlib.colors.hsv_to_rgb((hue, 0.64, 0.9))
        alpha = 0.24 + 0.5 * (1.0 - min(height / 7.2, 1.0))
        ax.plot(x_scaled, y, color=(*color, alpha), linewidth=0.72)

    fig.tight_layout(pad=0)
    return fig


def _figure_to_png_bytes(fig: plt.Figure, *, dpi: int = 130) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    return buffer.getvalue()


def plot_background_png_bytes() -> tuple[bytes, bytes]:
    market_png = _figure_to_png_bytes(make_market_background_figure())
    salary_png = _figure_to_png_bytes(make_salary_background_figure())
    return market_png, salary_png


def save_plot_background_assets(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    market_path = output_dir / PRIMARY_FILENAME
    salary_path = output_dir / SECONDARY_FILENAME
    market_png, salary_png = plot_background_png_bytes()
    market_path.write_bytes(market_png)
    salary_path.write_bytes(salary_png)
    return market_path, salary_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    market_path, salary_path = save_plot_background_assets(root / "app" / "assets")
    print(f"Saved {market_path}")
    print(f"Saved {salary_path}")
