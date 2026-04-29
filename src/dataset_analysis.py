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
MAX_CHARACTERS_PER_COMMENT = 500


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


DATASET_COLOR = {
    ds: COLORBLIND_PALETTE[i] for i, ds in enumerate(ALL_DATASETS)
}
DATASET_HATCH = {ds: HATCHES[i] for i, ds in enumerate(ALL_DATASETS)}
DATASET_MARKER = {ds: MARKERS[i] for i, ds in enumerate(ALL_DATASETS)}


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

    _add_section("Written:", written_in_plot)
    _add_section("Spoken:", spoken_in_plot)

    defaults = dict(
        handlelength=1.5,
        handleheight=1.2,
        borderpad=0.6,
        labelspacing=0.3,
        frameon=False,
    )
    legend = ax.legend(handles=handles, **{**defaults, **legend_kwargs})

    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        if text.get_text() in ("Written:", "Spoken:"):
            text.set_fontweight("bold")
            text.set_color("#333333")
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


def moderation_plot(df: pd.DataFrame, graph_dir: Path) -> None:
    # One observation per discussion
    disc_mod = (
        df.groupby(["dataset", "conv_id"])["is_moderator"]
        .mean()
        .reset_index(name="moderator_percent")
    )
    disc_mod["moderator_percent"] *= 100

    # Drop datasets with zero moderation
    nonzero = disc_mod.groupby("dataset")["moderator_percent"].mean()
    disc_mod = disc_mod[disc_mod.dataset.isin(nonzero[nonzero > 0].index)]

    order = (
        disc_mod.groupby("dataset")["moderator_percent"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure(figsize=(8, 5))
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
    ax.set_xlabel("Overall ratio (%) of facilitator comments")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    grouped_legend(
        ax,
        set(order),
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    util.graphs.save_plot(graph_dir / "analysis_moderation_perc.png")


def comments_per_discussion_plot(df: pd.DataFrame, graph_dir: Path) -> None:
    disc_sizes = (
        df.groupby(["dataset", "conv_id"])
        .size()
        .reset_index(name="comments_per_disc")
    )
    disc_sizes["comments_per_disc"] = disc_sizes["comments_per_disc"].clip(
        upper=MAX_COMMENTS_PER_DISCUSSION
    )

    hue_order = sorted(df["dataset"].unique())
    palette = {d: DATASET_COLOR.get(d, "#888888") for d in hue_order}

    plt.figure(figsize=(8, 5))
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
    plt.xlabel("#Comments per discussion")
    plt.ylabel("Density (stacked)")

    grouped_legend(ax, set(hue_order))
    plt.tight_layout()
    util.graphs.save_plot(graph_dir / "analysis_comments_per_discussion.png")


def words_per_comment_plot(df: pd.DataFrame, graph_dir: Path) -> None:
    df = df.copy()
    df["words_per_comment"] = (
        df.text.astype(str).apply(lambda x: len(x.split())).astype(int)
    )
    df["words_per_comment"] = df["words_per_comment"].clip(
        upper=MAX_CHARACTERS_PER_COMMENT
    )

    hue_order = sorted(df["dataset"].unique())
    palette = {d: DATASET_COLOR.get(d, "#888888") for d in hue_order}

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=df,
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
    plt.xlabel("#Words per comment")
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
    args = parser.parse_args()
    main(args)
