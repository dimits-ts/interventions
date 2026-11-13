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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import util.io
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
HUMAN_LABEL = "Human"
HUMAN_COLOR = "#000000"
HUMAN_HATCH = ""


def build_model_styles(models: list[str]) -> tuple[dict, dict]:
    """
    One color + one hatch per model, assigned once from a sorted, global
    list of model names. Every plot in this script must build its
    color/hatch maps by slicing *this* dict (never by re-deriving indices
    locally), so a given model looks identical across every graph and task.

    Index 0 of the palette/hatch lists is reserved for HUMAN_COLOR /
    HUMAN_HATCH (see plot_llm_vs_human) and is skipped here so no model can
    ever be assigned the same black / unhatched style as the human bars.
    """
    color = {
        m: COLORBLIND_PALETTE[(i + 1) % len(COLORBLIND_PALETTE)]
        for i, m in enumerate(models)
    }
    hatch = {m: HATCHES[(i + 1) % len(HATCHES)] for i, m in enumerate(models)}
    return color, hatch


def restyle_legend(ax, order: list[str], hatch_map: dict) -> None:
    """
    seaborn/matplotlib legend swatches are separate proxy artists that do
    NOT inherit hatches set on the actual bar patches after the fact, so
    the legend has to be re-stamped explicitly to stay visually consistent
    with the bars it's labelling.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    for label, handle in zip(order, legend.legend_handles):
        handle.set_hatch(hatch_map.get(label, ""))
        handle.set_edgecolor("black")


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
    wanted = {"dataset", "response", truth_column, "conv_id"}
    frames = {}
    for file_path in files:
        model = model_name_from_path(file_path)
        # Only parse the columns we need, never the discussion text
        df = pd.read_csv(file_path, usecols=lambda c: c in wanted)

        if exclude_datasets:
            df = df[~df["dataset"].isin(exclude_datasets)]

        df = df.copy()
        df["response_binary"] = df["response"].apply(normalize_binary_response)
        df["truth_binary"] = df[truth_column].apply(normalize_binary_response)
        df = df.drop(columns=["response", truth_column])

        frames[model] = df

    return frames


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
        rates = valid.groupby("dataset")["response_binary"].mean().reset_index()
        rates = rates.rename(columns={"response_binary": "positive_rate"})
        rates["judge"] = model
        rows.append(rates)
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["dataset", "positive_rate", "judge"])
    )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def load_task_rates(
    files: list[Path],
    truth_column: str,
    exclude_datasets: set[str],
    task_name: str,
    annotations_dir: Path,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (human_rates, model_rates) for a task, reading each from its own
    cache file where possible:

      human_rates_<task>.parquet         one per task
      model_rates_<task>_<model>.parquet one per (task, model)

    A model's cache is recomputed when its annotation file changes. The
    human cache depends on every model file of the task (it pools them),
    so it is also recomputed when a model file is added or removed.
    Annotation files are only read if some cache needs recomputing.
    """
    model_files = {model_name_from_path(f): f for f in files}

    human_cache = cache_dir / f"human_rates_{task_name}.parquet"
    model_caches = {
        m: cache_dir / f"model_rates_{task_name}_{m}.parquet" for m in model_files
    }

    human_fresh = not refresh and util.io.is_cache_fresh(
        human_cache, annotations_dir, *files
    )
    stale_models = [
        m
        for m, f in model_files.items()
        if refresh or not util.io.is_cache_fresh(model_caches[m], f)
    ]

    # The human rates pool every file, so a stale human cache needs them all
    to_load = stale_models if human_fresh else list(model_files)
    frames = load_task_frames(
        [model_files[m] for m in to_load], truth_column, exclude_datasets
    )

    if human_fresh:
        print(f"Using cached human rates from {human_cache}")
        human_rates = pd.read_parquet(human_cache)
    else:
        human_rates = compute_human_rates(frames)
        util.io.write_cache(human_rates, human_cache)
        print(f"Cached human rates to {human_cache}")

    model_rates = []
    for model in sorted(model_files):
        cache_file = model_caches[model]
        if model in stale_models:
            rates = compute_model_rates({model: frames[model]})
            util.io.write_cache(rates, cache_file)
            print(f"Cached '{model}' rates to {cache_file}")
        else:
            print(f"Using cached '{model}' rates from {cache_file}")
            rates = pd.read_parquet(cache_file)
        model_rates.append(rates)

    return human_rates, pd.concat(model_rates, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_llm_vs_human(
    human_rates: pd.DataFrame,
    model_rates: pd.DataFrame,
    task_name: str,
    graph_dir: Path,
    model_color: dict,
    model_hatch: dict,
    title: str,
) -> None:
    plot_df = pd.concat([human_rates, model_rates], ignore_index=True)

    if plot_df.empty:
        print(f"[info] No valid data for '{task_name}' human-vs-LLM graph.")
        return

    models = sorted(model_rates["judge"].unique())
    judge_order = [HUMAN_LABEL] + models
    dataset_order = sorted(plot_df["dataset"].unique())
    plot_df.positive_rate = plot_df.positive_rate * 100

    # Human always gets its own fixed black/unhatched style; every model
    # reuses the exact color+hatch assigned to it in build_model_styles(),
    # so it looks identical here and in the frequency graph above.
    palette = {HUMAN_LABEL: HUMAN_COLOR, **{m: model_color[m] for m in models}}
    hatches = {HUMAN_LABEL: HUMAN_HATCH, **{m: model_hatch[m] for m in models}}

    plt.figure()
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
    ax.set_ylabel(r"\% Positive interventions")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    restyle_legend(ax, judge_order, hatches)

    plt.tight_layout()
    util.graphs.save_plot(graph_dir / f"llm_human_judgement_by_dataset_{task_name}.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(
    annotations_dir: Path, graph_dir: Path, cache_dir: Path, refresh: bool
) -> None:
    util.graphs.seaborn_setup()
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Discover every model across every task first, so a model gets the
    # exact same color+hatch everywhere, regardless of which task or which
    # graph it happens to show up in.
    task_files = {
        task_name: discover_files(annotations_dir, task_name)
        for task_name, _, _ in TASKS
    }
    all_models = sorted(
        {model_name_from_path(f) for files in task_files.values() for f in files}
    )
    model_color, model_hatch = build_model_styles(all_models)

    for task_name, truth_column, exclude_datasets in TASKS:
        files = task_files[task_name]
        if not files:
            print(f"[info] No files found for task '{task_name}', skipping.")
            continue

        print(f"\n=== {task_name} ({len(files)} model file(s)) ===")
        human_rates, model_rates = load_task_rates(
            files,
            truth_column,
            exclude_datasets,
            task_name,
            annotations_dir,
            cache_dir,
            refresh,
        )

        plot_llm_vs_human(
            human_rates,
            model_rates,
            task_name,
            graph_dir,
            model_color,
            model_hatch,
            title=(
                "Unlike humans, LLMs are influenced by discussion domain "
                f"({task_name.capitalize()})."
            ),
        )

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
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help=(
            "Directory for cached calculations (one file per calculation). "
            "Cache files older than their source files are recomputed."
        ),
    )

    args = parser.parse_args()

    main(
        annotations_dir=Path(args.annotation_dir),
        graph_dir=Path(args.graph_dir),
        cache_dir=Path(args.cache_dir),
        refresh=args.refresh_cache,
    )
