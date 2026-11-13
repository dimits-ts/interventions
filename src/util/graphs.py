# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


# Okabe–Ito palette
COLORBLIND_PALETTE = [
    # --- Core ---
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    # --- Neutral / light extensions ---
    "#999999",
    "#8DD3C7",
    "#FDB462",
    "#B3DE69",
    "#80B1D3",
    "#FB8072",
    "#CAB2D6",
    "#BC80BD",
    # --- Additional medium/dark colors ---
    "#4D4D4D",
    "#1B9E77",
    "#7570B3",
    "#A6761D",
    "#66A61E",
    "#E7298A",
    "#A6CEE3",
    "#FFB000",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
HATCHES = ["//", "\\\\", "++", "oo", "..", "xx", "**", "--"]

# Visually distinct line styles, as (offset, (on, off, ...)) dash tuples so
# every pattern is explicit. Patterns scale with linewidth (matplotlib's
# lines.scale_dashes), so they stay distinct at any line weight.
# Index meaning is fixed; callers pick by index or cycle in order:
#   0 solid, 1 dashed, 2 dotted, 3 dash-dot, 4 dash-dot-dot,
#   5 long dash, 6 loosely dotted
LINESTYLES = [
    "solid",
    (0, (5, 2)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
    (0, (5, 2, 1, 2, 1, 2)),
    (0, (10, 3)),
    (0, (1, 3)),
]

DEFAULT_FIGSIZE = (14, 6)


def graph_setup() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        },
    )

    plt.rcParams.update(
        {
            "text.usetex": True,
            # Figure
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Fonts
            # Same fonts as the ACM acmart class: Linux Libertine for text,
            # newtxmath (Libertine variant) for math. With usetex the font
            # comes from the LaTeX preamble below; font.serif only names a
            # font matplotlib knows for usetex, so it does not log "No
            # LaTeX-compatible font found". The preamble loads after it.
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": (
                r"\usepackage[tt=false,type1=true]{libertine}"
                r"\usepackage[libertine]{newtxmath}"
            ),
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 22,
            "legend.title_fontsize": 22,
            "figure.titlesize": 26,
            "figure.labelsize": 22,
            # Subplots
            "figure.subplot.wspace": 0.3,
            "figure.subplot.hspace": 0.7,
            "figure.subplot.bottom": 0.2,
            # Axes
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Legend
            "legend.frameon": False,
            "legend.loc": "best",
            # Math text
            "mathtext.fontset": "cm",
            # PDF/PS output (important for LaTeX + journals)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_palette(COLORBLIND_PALETTE)


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path: The full path (including filename)
        where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {path.resolve()}")
    plt.close()
