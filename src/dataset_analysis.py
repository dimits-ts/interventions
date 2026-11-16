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
import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util.io
import util.graphs
from util.graphs import COLORBLIND_PALETTE, HATCHES, MARKERS

MAX_COMMENTS_PER_DISCUSSION = 500
MAX_WORDS_PER_COMMENT = 500


WRITTEN_DATASETS = [
    "CeRI",
    "WikiDisputes",
    "WikiConv",
    "UMOD",
    "WikiTactics",
    "CMV-Awry",
]
SPOKEN_DATASETS = ["Fora", "IQ2", "WHoW"]
ALL_DATASETS = WRITTEN_DATASETS + SPOKEN_DATASETS

DATASET_RENAMES = {
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
EXCLUDED_DATASETS = {"vmd"}
MODERATION_EXCLUDED_DATASETS = {"UMOD", "WikiDisputes"}


DATASET_COLOR = {
    ds: COLORBLIND_PALETTE[i] for i, ds in enumerate(ALL_DATASETS)
}
DATASET_HATCH = {ds: HATCHES[i] for i, ds in enumerate(ALL_DATASETS)}
DATASET_MARKER = {ds: MARKERS[i] for i, ds in enumerate(ALL_DATASETS)}


# ---------------------------------------------------------------------------
# Calculations (each one is cached to its own file)
# ---------------------------------------------------------------------------
def compute_words_per_comment(df: pd.DataFrame) -> pd.DataFrame:
    """One row per comment: dataset, words_per_comment."""
    return df[["dataset", "words_per_comment"]].reset_index(drop=True)


def compute_comments_per_discussion(df: pd.DataFrame) -> pd.DataFrame:
    """One row per discussion: dataset, conv_id, comments_per_disc."""
    disc_sizes = (
        df.groupby(["dataset", "conv_id"])
        .size()
        .reset_index(name="comments_per_disc")
    )
    disc_sizes["comments_per_disc"] = disc_sizes["comments_per_disc"].clip(
        upper=MAX_COMMENTS_PER_DISCUSSION
    )
    return disc_sizes


def compute_moderation(df: pd.DataFrame) -> pd.DataFrame:
    """One row per discussion: dataset, conv_id, moderator_percent."""
    disc_mod = (
        df.groupby(["dataset", "conv_id"])["is_moderator"]
        .mean()
        .reset_index(name="moderator_percent")
    )
    disc_mod["moderator_percent"] *= 100
    return disc_mod


# name -> (function, does it need the raw text column?)
CALCULATIONS = {
    "words_per_comment": (compute_words_per_comment, True),
    "comments_per_discussion": (compute_comments_per_discussion, False),
    "moderation": (compute_moderation, False),
}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filter, rename, and reduce the raw text to a word count per chunk."""
    chunk = chunk[~chunk["dataset"].isin(EXCLUDED_DATASETS)].copy()
    chunk["dataset"] = chunk["dataset"].replace(DATASET_RENAMES)
    if "text" in chunk.columns:
        # Same result as len(x.split()), but vectorized
        chunk["words_per_comment"] = (
            chunk["text"]
            .astype(str)
            .str.count(r"\S+")
            .clip(upper=MAX_WORDS_PER_COMMENT)
            .astype("int16")
        )
        chunk = chunk.drop(columns="text")
    return chunk


def load_statistics(
    csv_path: Path, cache_dir: Path
) -> dict[str, pd.DataFrame]:
    """
    Load every calculation from the cache where possible. The dataset is
    only read if at least one cache file is missing, and the text column
    is only parsed if the word counts need recomputing. Caches are never
    invalidated automatically; delete a file to force a recompute.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    missing = []

    for name in CALCULATIONS:
        cache_file = _cache_path(cache_dir, name)
        if cache_file.is_file():
            print(f"Using cached '{name}' from {cache_file}")
            stats[name] = pd.read_parquet(cache_file)
        else:
            missing.append(name)

    if not missing:
        return stats

    need_text = any(CALCULATIONS[name][1] for name in missing)
    usecols = ["dataset", "conv_id", "is_moderator"]
    if need_text:
        usecols.append("text")

    print(f"Loading dataset to compute: {', '.join(missing)}")
    df = util.io.progress_load_csv(
        csv_path, usecols=usecols, chunk_fn=_prepare_chunk
    )

    for name in missing:
        compute_fn, _ = CALCULATIONS[name]
        stats[name] = compute_fn(df)
        cache_file = _cache_path(cache_dir, name)
        util.io.write_cache(stats[name], cache_file)
        print(f"Cached '{name}' to {cache_file}")

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main(args):
    util.graphs.seaborn_setup()

    csv_path = Path(args.dataset_path)
    graph_dir = Path(args.graph_dir)
    cache_dir = Path(args.cache_dir)

    stats = load_statistics(csv_path, cache_dir)

    words_per_comment_plot(stats["words_per_comment"], graph_dir)
    comments_per_discussion_plot(stats["comments_per_discussion"], graph_dir)
    moderation_plot(stats["moderation"], graph_dir)


def grouped_legend(ax, datasets_in_plot, **legend_kwargs):
    """
    Build a Written / Spoken grouped legend from scratch using Patch handles
    so that colours *and* hatches are both visible in the legend.
    Only datasets actually present in the current plot are included.
    """
    written_in_plot = [d for d in WRITTEN_DATASETS if d in datasets_in_plot]
    spoken_in_plot = [d for d in SPOKEN_DATASETS if d in datasets_in_plot]

    handles = []

    def _add_section(header, members):
        if not members:
            return
        handles.append(mpatches.Patch(color="none", label=header))
        for d in members:
            handles.append(
                mpatches.Patch(
                    facecolor=DATASET_COLOR[d],
                    hatch=DATASET_HATCH[d],
                    edgecolor="black",
                    label=f"  {d}",
                )
            )

    _add_section(r"\textbf{Written}:", written_in_plot)
    _add_section(r"\textbf{Spoken}:", spoken_in_plot)

    defaults = dict(
        handlelength=1.5,
        handleheight=1.2,
        borderpad=0.6,
        labelspacing=0.3,
        frameon=False,
    )
    legend = ax.legend(handles=handles, **{**defaults, **legend_kwargs})

    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        if text.get_text() in (r"\textbf{Written}:", r"\textbf{Spoken}:"):
            handle.set_visible(False)


def apply_hatches_to_histplot(ax, hue_order):
    """
    seaborn lays histplot patches in contiguous per-dataset blocks:
      [all bins for hue_order[0]] [all bins for hue_order[1]] …
    Walk those blocks and stamp the canonical hatch for each dataset.
    """
    n_datasets = len(hue_order)
    n_patches = len(ax.patches)
    per_ds = n_patches // n_datasets

    for ds_idx, ds_name in enumerate(hue_order):
        hatch = DATASET_HATCH.get(ds_name, "")
        for p in ax.patches[ds_idx * per_ds: (ds_idx + 1) * per_ds]:
            p.set_hatch(hatch)
            p.set_edgecolor("black")


def moderation_plot(disc_mod: pd.DataFrame, graph_dir: Path) -> None:
    disc_mod = disc_mod[
        ~disc_mod.dataset.isin(MODERATION_EXCLUDED_DATASETS)
    ]

    # Drop datasets with zero moderation
    nonzero = disc_mod.groupby("dataset")["moderator_percent"].mean()
    disc_mod = disc_mod[disc_mod.dataset.isin(nonzero[nonzero > 0].index)]

    order = (
        disc_mod.groupby("dataset")["moderator_percent"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure()
    ax = sns.barplot(
        data=disc_mod,
        y="dataset",
        x="moderator_percent",
        order=order,
        palette={d: DATASET_COLOR.get(d, "#888888") for d in order},
        estimator="mean",
        errorbar=("ci", 95),
        err_kws={
            "linewidth": 3.5,
            "color": "#EE4B2B",
        },
    )
    apply_hatches_to_histplot(ax, order)  # reuse the same patch-walker

    ax.set_ylabel("Dataset")
    ax.set_xlabel(r"Overall ratio (\%) of facilitator comments")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    grouped_legend(
        ax,
        set(order),
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    util.graphs.save_plot(graph_dir / "analysis_moderation_perc.png")


def comments_per_discussion_plot(
    disc_sizes: pd.DataFrame, graph_dir: Path
) -> None:
    hue_order = sorted(disc_sizes["dataset"].unique())
    palette = {d: DATASET_COLOR.get(d, "#888888") for d in hue_order}

    plt.figure()
    ax = sns.histplot(
        data=disc_sizes,
        x="comments_per_disc",
        hue="dataset",
        stat="density",
        common_norm=False,
        bins=40,
        hue_order=hue_order,
        palette=palette,
        multiple="stack",
    )
    apply_hatches_to_histplot(ax, hue_order)

    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels(["0", "100", "200", "300", "400", "500+"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(r"\#Comments per discussion")
    plt.ylabel("Density (stacked)")

    grouped_legend(ax, set(hue_order))
    plt.tight_layout()
    util.graphs.save_plot(graph_dir / "analysis_comments_per_discussion.png")


def words_per_comment_plot(
    words_df: pd.DataFrame, graph_dir: Path
) -> None:
    hue_order = sorted(words_df["dataset"].unique())
    palette = {d: DATASET_COLOR.get(d, "#888888") for d in hue_order}

    plt.figure()
    ax = sns.histplot(
        data=words_df,
        x="words_per_comment",
        hue="dataset",
        bins=50,
        stat="density",
        common_norm=False,
        hue_order=hue_order,
        palette=palette,
        multiple="stack",
    )
    apply_hatches_to_histplot(ax, hue_order)

    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels(["0", "100", "200", "300", "400", "500+"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(r"\#Words per comment")
    plt.ylabel("Density (stacked)")

    grouped_legend(ax, set(hue_order))
    plt.tight_layout()
    util.graphs.save_plot(graph_dir / "analysis_words_per_comment.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate discussion statistics and moderation plots."
    )
    parser.add_argument(
        "--dataset-path", required=True, help="Path to the dataset CSV file."
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory where the graphs will be exported to.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help=(
            "Directory for cached calculations (one file per calculation). "
            "Existing cache files are always used; delete them to recompute."
        ),
    )
    args = parser.parse_args()
    main(args)