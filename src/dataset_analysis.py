# The PEFK (Prosocial and Effective Facilitation in Konversations) Dataset
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
import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import util.io
import util.graphs

MAX_COMMENTS_PER_DISCUSSION = 500
MAX_CHARACTERS_PER_COMMENT = 500


def main(args):
    util.graphs.seaborn_setup()

    csv_path = Path(args.dataset_path)
    graph_dir = Path(args.graph_dir)

    print("Loading dataset to extract statistics...")
    df = util.io.progress_load_csv(csv_path)
    df = df[df.dataset != "vmd"]
    df.dataset = df.dataset.replace(
        {
            "ceri": "CeRI",
            "wikidisputes": "WikiDisputes",
            "wikiconv": "WikiConv",
            "fora": "Fora",
            "umod": "UMOD",
            "wikitactics": "WikiTactics",
            "whow": "WHoW",
            "iq2": "IQ2",
            "cmv_awry": "CMV-Awry",
        }
    )

    words_per_comment_plot(df, graph_dir)
    comments_per_discussion_plot(df, graph_dir)
    moderation_plot(
        df[(df.dataset != "UMOD") & (df.dataset != "WikiDisputes")],
        graph_dir=graph_dir,
    )


def moderation_plot(df: pd.DataFrame, graph_dir: Path) -> float:
    moderator_percent = (
        df.groupby("dataset")["is_moderator"]
        .mean()
        .reset_index(name="moderator_percent")
    )
    # exclude datasets where moderation is not supported
    moderator_percent = moderator_percent[
        moderator_percent.moderator_percent != 0
    ]
    moderator_percent["moderator_percent"] *= 100
    order = moderator_percent.sort_values(
        "moderator_percent", ascending=False
    )["dataset"].tolist()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=moderator_percent,
        y="dataset",
        x="moderator_percent",
        color="black",
        order=order,
    )
    plt.ylabel("Dataset")
    plt.xlabel("Overall ratio (%) of facilitator comments")
    plt.tight_layout()

    util.graphs.save_plot(graph_dir / "analysis_moderation_perc.png")


def comments_per_discussion_plot(df: pd.DataFrame, graph_dir: Path) -> None:
    disc_sizes = (
        df.groupby(["dataset", "conv_id"])
        .size()
        .reset_index(name="comments_per_disc")
    )

    # Cap at 2MAX_COMMENT_LEN_CHARS because there is a
    # tail going up to 1200 comments
    disc_sizes["comments_per_disc"] = disc_sizes["comments_per_disc"].clip(
        upper=MAX_COMMENTS_PER_DISCUSSION
    )

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=disc_sizes,
        x="comments_per_disc",
        hue="dataset",
        stat="density",
        common_norm=False,
        bins=40,  # do NOT let this go to auto
    )
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels(["0", "100", "200", "300", "400", "500+"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(None)
    plt.xlabel("#Comments/Discussion ")
    plt.ylabel("Density")
    plt.tight_layout()

    util.graphs.save_plot(graph_dir / "analysis_comments_per_discussion.png")


def words_per_comment_plot(df: pd.DataFrame, graph_dir: Path) -> None:
    df = df.copy()

    df["words_per_comment"] = (
        df.text.astype(str).apply(lambda x: len(x.split())).astype(int)
    )

    # Cap long tail (optional but very useful for readability)
    df["words_per_comment"] = df["words_per_comment"].clip(
        upper=MAX_CHARACTERS_PER_COMMENT
    )

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=df,
        x="words_per_comment",
        hue="dataset",
        bins=50,  # do NOT let this go to auto
        stat="density",
        common_norm=False,
    )
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels(["0", "100", "200", "300", "400", "500+"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(None)
    plt.xlabel("#Words per Comment")
    plt.ylabel("Density")
    plt.tight_layout()

    util.graphs.save_plot(graph_dir / "analysis_words_per_comment.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate discussion statistics and moderation plots."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        required=True,
        help="Directory where the graphs will be exported to",
    )
    args = parser.parse_args()
    main(args)
