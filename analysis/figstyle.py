"""Shared figure style for the working paper.

One palette, one type scale, one axis-title convention across every figure
in results/jr16, results/appendix and results/incidence. Presentation only:
nothing here touches data, weights or CSVs.

Palette follows a colourblind-validated categorical order (blue, aqua,
yellow, green, violet, red) with a blue sequential ramp and a blue-grey-red
diverging map for the scenario heatmaps (two hues + neutral midpoint,
CVD-safe; replaces matplotlib RdBu_r for palette coherence).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots (fixed order — never cycled)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
SERIES = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED]

# Ink / chrome
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"  # diverging midpoint
LIGHT_BLUE = "#9ec5f4"  # sequential step 200 (secondary bars)

# Diverging map: blue (negative) — neutral grey — red (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "paper_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL, "#f0a5a4", "#e34948", "#7f1d1d"],
)

DECILE_AXIS = "Income decile (equivalised household disposable income, HBAI)"

# Canonical figure sizes (inches)
SINGLE = (8.0, 4.5)
HEATMAP = (10.0, 5.0)
FACETS = (16.0, 6.5)
TWOPANEL = (11.0, 4.5)
DPI = 200


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def decile_ax(ax, ylabel, xlabel=DECILE_AXIS):
    """Common decile-chart chrome: x ticks 1-10, y grid only."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
