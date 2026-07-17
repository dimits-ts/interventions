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

"""
judgement_analysis.py

Two graphs, built on top of the same `llm_intervention_<model>_timing_<task>.csv`
files already produced by the inference pipeline and consumed by llm_test.py:

  1. Judgement frequency by model and dataset
     A count plot of the LLM's binary judgements (No / Yes), grouped by
     dataset and coloured by model, so you can see how often each model
     says "yes" vs "no" on each dataset.

  2. LLM vs human judgements by dataset
     A grouped bar chart comparing the "positive judgement rate" (fraction
     of Yes / 1 judgements) of the human ground truth (the task's truth
     column, e.g. should_intervene / is_moderator) against every model's
     predictions, per dataset.

Both graphs are produced once per task ("prediction" and "detection"), the
same two tasks llm_test.py already handles, using the same file-discovery
and response-normalization logic so results stay consistent with the rest
of the analysis pipeline.

Usage
-----
python judgement_analysis.py \\
    --annotation-dir data/llm_output \\
    --graph-dir graphs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import util.graphs
from util.graphs import COLORBLIND_PALETTE, HATCHES
from llm_test import normalize_binary_response

# (task_name, truth_column, datasets_to_exclude)
# Mirrors llm_test.py: prediction drops "umod", detection keeps everything.
TASKS = [
    ("prediction", "should_intervene", {"umod"}),
    ("detection", "is_moderator", set()),
]

JUDGEMENT_LABELS = {0: "No (0)", 1: "Yes (1)"}
HUMAN_LABEL = "Human (ground truth)"


def discover_files(annotations_dir: Path, task_name: str) -> list[Path]:
    pattern = f"llm_intervention_*_timing_{task_name}.csv"
    return sorted(annotations_dir.glob(pattern))


def model_name_from_path(path: Path) -> str:
    return path.stem.split("_")[2]


def load_task_frames(
    files: list[Path], truth_column: str, exclude_datasets: set[str]
) -> dict[str, pd.DataFrame]:
    """
    Loads and lightly normalizes every model's annotation file for a task.

    Returns {model_name: dataframe[dataset, response_binary, truth_binary]}.
    Keeps 'conv_id' too, when present, so human ground truth can be
    deduplicated across model files later on.
    """
    frames = {}
    for file_path in files:
        model = model_name_from_path(file_path)
        df = pd.read_csv(file_path)

        if exclude_datasets:
            df = df[~df["dataset"].isin(exclude_datasets)]

        keep_cols = ["dataset", "response", truth_column]
        if "conv_id" in df.columns:
            keep_cols.append("conv_id")
        df = df[keep_cols].copy()

        df["response_binary"] = df["response"].apply(normalize_binary_response)
        df["truth_binary"] = df[truth_column].apply(normalize_binary_response)

        frames[model] = df

    return frames


# ---------------------------------------------------------------------------
# Graph 1: LLM judgement frequency by model and dataset
# ---------------------------------------------------------------------------
def plot_judgement_frequency(
    frames: dict[str, pd.DataFrame], task_name: str, graph_dir: Path
) -> None:
    rows = []
    for model, df in frames.items():
        valid = df.dropna(subset=["response_binary"])
        for dataset, judgement in zip(
            valid["dataset"], valid["response_binary"]
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "Judgement": JUDGEMENT_LABELS[int(judgement)],
                }
            )

    if not rows:
        print(
            f"[info] No valid LLM responses for '{task_name}' frequency graph."
        )
        return

    plot_df = pd.DataFrame(rows)

    models = sorted(plot_df["model"].unique())
    palette = {
        m: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        for i, m in enumerate(models)
    }
    hatches = {m: HATCHES[i % len(HATCHES)] for i, m in enumerate(models)}
    dataset_order = sorted(plot_df["dataset"].unique())

    g = sns.catplot(
        data=plot_df,
        kind="count",
        x="dataset",
        hue="model",
        col="Judgement",
        order=dataset_order,
        hue_order=models,
        palette=palette,
        col_order=["No (0)", "Yes (1)"],
        height=5,
        aspect=1.1,
        legend_out=True,
    )

    for ax in g.axes.flat:
        # seaborn gives one BarContainer per hue level (model), in hue_order;
        # stamp every bar in that container with its model's hatch.
        for model, container in zip(models, ax.containers):
            for patch in container:
                patch.set_hatch(hatches[model])
                patch.set_edgecolor("black")

        ax.set_xlabel("Dataset")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    g.set_titles("LLM judgement: {col_name}")
    g.fig.suptitle(
        f"LLM Judgement Frequency by Model and Dataset – {task_name.capitalize()}",
        y=1.05,
    )

    util.graphs.save_plot(
        graph_dir / f"judgement_frequency_by_model_dataset_{task_name}.png"
    )


# ---------------------------------------------------------------------------
# Graph 2: LLM vs human judgements by dataset
# ---------------------------------------------------------------------------
def compute_human_rates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pools the ground-truth column across every model's file (since they all
    annotate the same underlying human-labelled support set) and dedupes on
    conv_id when available, so a discussion isn't counted once per model.
    """
    combined = pd.concat(frames.values(), ignore_index=True)

    if "conv_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["dataset", "conv_id"])
    else:
        print(
            "[warn] No 'conv_id' column found; falling back to the first "
            "model's file only for human ground-truth rates, to avoid "
            "double-counting the same discussions across models."
        )
        combined = next(iter(frames.values()))

    combined = combined.dropna(subset=["truth_binary"])
    rates = (
        combined.groupby("dataset")["truth_binary"]
        .mean()
        .reset_index()
        .rename(columns={"truth_binary": "positive_rate"})
    )
    rates["judge"] = HUMAN_LABEL
    return rates


def compute_model_rates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in frames.items():
        valid = df.dropna(subset=["response_binary"])
        rates = (
            valid.groupby("dataset")["response_binary"].mean().reset_index()
        )
        rates = rates.rename(columns={"response_binary": "positive_rate"})
        rates["judge"] = model
        rows.append(rates)
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["dataset", "positive_rate", "judge"])
    )


def plot_llm_vs_human(
    frames: dict[str, pd.DataFrame], task_name: str, graph_dir: Path
) -> None:
    if not frames:
        print(f"[info] No files for '{task_name}' human-vs-LLM graph.")
        return

    human_rates = compute_human_rates(frames)
    model_rates = compute_model_rates(frames)
    plot_df = pd.concat([human_rates, model_rates], ignore_index=True)

    if plot_df.empty:
        print(f"[info] No valid data for '{task_name}' human-vs-LLM graph.")
        return

    models = sorted(model_rates["judge"].unique())
    judge_order = [HUMAN_LABEL] + models
    dataset_order = sorted(plot_df["dataset"].unique())

    palette = {HUMAN_LABEL: "#000000"}
    hatches = {HUMAN_LABEL: ""}
    for i, m in enumerate(models):
        palette[m] = COLORBLIND_PALETTE[(i + 1) % len(COLORBLIND_PALETTE)]
        hatches[m] = HATCHES[(i + 1) % len(HATCHES)]

    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(
        data=plot_df,
        x="dataset",
        y="positive_rate",
        hue="judge",
        order=dataset_order,
        hue_order=judge_order,
        palette=palette,
    )

    for judge, container in zip(judge_order, ax.containers):
        for patch in container:
            patch.set_hatch(hatches.get(judge, ""))
            patch.set_edgecolor("black")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Positive judgement rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"LLM vs Human Judgements by Dataset – {task_name.capitalize()}"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    util.graphs.save_plot(
        graph_dir / f"llm_human_judgement_by_dataset_{task_name}.png"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(annotations_dir: Path, graph_dir: Path) -> None:
    util.graphs.seaborn_setup()
    graph_dir.mkdir(parents=True, exist_ok=True)

    for task_name, truth_column, exclude_datasets in TASKS:
        files = discover_files(annotations_dir, task_name)
        if not files:
            print(f"[info] No files found for task '{task_name}', skipping.")
            continue

        print(f"\n=== {task_name} ({len(files)} model file(s)) ===")
        frames = load_task_frames(files, truth_column, exclude_datasets)

        plot_judgement_frequency(frames, task_name, graph_dir)
        plot_llm_vs_human(frames, task_name, graph_dir)

    print(f"\nAll graphs written to {graph_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate (1) LLM judgement frequency by model and dataset, and "
            "(2) LLM vs human judgement rate by dataset, for the prediction "
            "and detection tasks."
        )
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        required=True,
        help="Directory containing llm_intervention_<model>_timing_<task>.csv files.",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        required=True,
        help="Directory where graphs will be saved.",
    )

    args = parser.parse_args()

    main(
        annotations_dir=Path(args.annotation_dir),
        graph_dir=Path(args.graph_dir),
    )
