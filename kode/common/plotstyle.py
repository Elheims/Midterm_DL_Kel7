"""Shared matplotlib style for report figures (static PDF, light background)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated palette from the dataviz reference).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#a8a7a0"          # de-emphasised marks (e.g. dropped sensors)
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.size": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save(fig, path_without_ext) -> None:
    """Save as PDF (for LaTeX) and PNG (for quick viewing)."""
    fig.savefig(f"{path_without_ext}.pdf")
    fig.savefig(f"{path_without_ext}.png")
    plt.close(fig)
