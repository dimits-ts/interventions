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
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util.graphs


def main(input_dir: Path, graph_dir: Path, tables_dir: Path):
    util.graphs.seaborn_setup()
    graph_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    for entry in input_dir.iterdir():
        if entry.is_dir():
            curves_df = pd.read_csv(entry / "pr_curves.csv", index_col=0)
            plot_metrics(curves_df)
            util.graphs.save_plot(graph_dir / f"pr_curves_{entry.name}.png")


def plot_metrics(df):
    # Convert to long format
    df_long = df.melt(
        id_vars="Threshold",
        value_vars=["Precision", "Recall", "F1"],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots()

    # Colorblind-friendly palette
    palette = sns.color_palette("colorblind", 3)

    # Plot lines
    sns.lineplot(
        data=df_long,
        x="Threshold",
        y="value",
        hue="metric",
        style="metric",
        dashes=True,
        palette=palette,
        markers=True,
        ax=ax,
        legend=True,
    )

    # Add shading manually per metric
    for metric, color in zip(["Precision", "Recall", "F1"], palette):
        ax.fill_between(
            df["Threshold"],
            df[metric],
            alpha=0.15,
            color=color,
        )

    # Labels and limits
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(-0.01, 1.1)
    ax.set_ylim(0, 1.1)

    # Clean legend
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(None)  # type: ignore

    sns.despine()
    plt.tight_layout()

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate discussion statistics and moderation plots."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the classifier's output files.",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        required=True,
        help="Directory where the graphs will be exported to",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        required=True,
        help="Directory where the LaTex tables will be exported to",
    )
    args = parser.parse_args()
    main(
        input_dir=Path(args.input_dir),
        graph_dir=Path(args.graph_dir),
        tables_dir=Path(args.tables_dir),
    )
