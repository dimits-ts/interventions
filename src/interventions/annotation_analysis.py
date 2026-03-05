import argparse
import re
import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from ..util import graphs


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------


@dataclass
class AnnotationSchema:
    """Describes a reinforcement annotation schema."""

    name: str  # short identifier used in output filenames / print headers
    columns: list[str]  # reinforcement column names present in the raw data


SCHEMA_THREE_WAY = AnnotationSchema(
    name="three_way",
    columns=[
        "positive_reinforcement",
        "negative_reinforcement",
        "no_reinforcement",
    ],
)

SCHEMA_BINARY = AnnotationSchema(
    name="binary",
    columns=["reinforcement"],
)

ALL_SCHEMAS = [SCHEMA_THREE_WAY, SCHEMA_BINARY]

THRESHOLD = 4

HUMAN_COLOR = "#7B2D8B"  # purple
LLM_COLOR = "#FF6600"  # deep orange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def annotator_color(name: str) -> str:
    """Return the label color for a given annotator name."""
    return HUMAN_COLOR if re.match(r"A\d+$", name) else LLM_COLOR


def apply_annotator_label_colors(ax: plt.Axes, axis: str = "y") -> None:
    """Color tick labels purple for human annotators and deep orange for LLMs."""
    tick_labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for label in tick_labels:
        label.set_color(annotator_color(label.get_text()))


def sort_annotators(names: list[str]) -> list[str]:
    def key(name: str):
        if re.match(r"A\d+", name):
            return (0, int(name[1:]))  # humans first, numeric order
        return (1, name.lower())  # LLMs next, alphabetical

    return sorted(names, key=key)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_annotation_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    reinforce_cols = SCHEMA_THREE_WAY.columns
    df = df[["conv_id", "data_malformation"] + reinforce_cols].copy()
    df["conv_id"] = df["conv_id"].astype(str)
    df = df.sort_values("conv_id").reset_index(drop=True)
    df = df.fillna(0)

    for col in reinforce_cols:
        df[col] = df[col].apply(
            lambda x: 0 if x == 0 else int(str(x).split(" – ")[0])
        )

    return df


def load_human_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    dfs = {p.stem: read_annotation_file(p) for p in paths}

    ref_name, ref_df = next(iter(dfs.items()))
    ref_ids = ref_df["conv_id"]
    for name, df in dfs.items():
        if not df["conv_id"].equals(ref_ids):
            raise ValueError(f"conv_id mismatch between {ref_name} and {name}")

    return dfs


def parse_llm_response(text: str) -> dict[str, int]:
    """
    Extract reinforcement counts from the LLM output string.
    Example format:

    Positive: 4
    Negative: 2
    No Reinforcement: 1
    """
    patterns = {
        "positive_reinforcement": r"Positive:\s*(\d+)",
        "negative_reinforcement": r"Negative:\s*(\d+)",
        "no_reinforcement": r"No Reinforcement:\s*(\d+)",
    }
    return {
        col: (
            int(m.group(1))
            if (m := re.search(pat, text, re.IGNORECASE))
            else 0
        )
        for col, pat in patterns.items()
    }


def read_llm_annotation_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    parsed = df["response"].apply(parse_llm_response).apply(pd.Series)
    out = pd.concat([df[["conv_id"]], parsed], axis=1)
    out["data_malformation"] = "no"
    out = out[["conv_id", "data_malformation"] + SCHEMA_THREE_WAY.columns]
    out["conv_id"] = out["conv_id"].astype(str)
    out = out.sort_values("conv_id").reset_index(drop=True)

    return out


def load_llm_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for p in paths:
        name = re.sub(r"^llm_intervention_", "", p.stem).capitalize()
        dfs[name] = read_llm_annotation_file(p)
    return dfs


# ---------------------------------------------------------------------------
# Schema projection
# ---------------------------------------------------------------------------


def project_to_schema(
    dfs: dict[str, pd.DataFrame],
    schema: AnnotationSchema,
) -> dict[str, pd.DataFrame]:
    """
    Return a copy of *dfs* that contains only the columns required by *schema*.

    For SCHEMA_BINARY the three reinforcement columns are collapsed: a row is
    considered reinforced (1) when the sum of positive and negative counts
    meets the threshold, ignoring no_reinforcement.
    """
    if schema is SCHEMA_THREE_WAY:
        return dfs  # already in the right shape

    # Binary: positive + negative → single "reinforcement" column
    out = {}
    for name, df in dfs.items():
        tmp = df[["conv_id", "data_malformation"]].copy()
        tmp["reinforcement"] = (
            df["positive_reinforcement"] + df["negative_reinforcement"]
        )
        out[name] = tmp
    return out


# ---------------------------------------------------------------------------
# Cleaning / alignment
# ---------------------------------------------------------------------------


def align_by_conv_id(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common_ids = set.intersection(*[set(df["conv_id"]) for df in dfs.values()])
    return {
        name: (
            df[df["conv_id"].isin(common_ids)]
            .sort_values("conv_id")
            .reset_index(drop=True)
        )
        for name, df in dfs.items()
    }


def get_malformed_ids(dfs: dict[str, pd.DataFrame]) -> set[str]:
    malformed_sets = [
        set(
            df.loc[
                df["data_malformation"].astype(str).str.strip().eq("yes"),
                "conv_id",
            ]
        )
        for df in dfs.values()
    ]
    return set.union(*malformed_sets) if malformed_sets else set()


def remove_malformed_rows(
    dfs: dict[str, pd.DataFrame],
    malformed_ids: set[str],
) -> dict[str, pd.DataFrame]:
    return {
        name: df[~df["conv_id"].isin(malformed_ids)].reset_index(drop=True)
        for name, df in dfs.items()
    }


# ---------------------------------------------------------------------------
# Binarisation
# ---------------------------------------------------------------------------


def to_binary(
    dfs: dict[str, pd.DataFrame],
    schema: AnnotationSchema,
) -> dict[str, pd.DataFrame]:
    out = {}
    for name, df in dfs.items():
        df_bin = df.copy()
        for col in schema.columns:
            df_bin[col] = (df_bin[col] >= THRESHOLD).astype(int)
        out[name] = df_bin
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def average_kappa(
    dfs_binary: dict[str, pd.DataFrame],
    schema: AnnotationSchema,
) -> dict[str, float]:
    annotators = list(dfs_binary.keys())
    result = {}
    for col in schema.columns:
        kappas = [
            cohen_kappa_score(dfs_binary[a][col], dfs_binary[b][col])
            for a, b in itertools.combinations(annotators, 2)
        ]
        result[col] = sum(kappas) / len(kappas) if kappas else 0.0
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_malformation_agreement_histogram(
    dfs: dict[str, pd.DataFrame],
    graph_output_dir: Path,
) -> None:
    records = []
    for annotator, df in dfs.items():
        tmp = df[["conv_id", "data_malformation"]].copy()
        tmp["annotator"] = annotator
        tmp["malformed"] = (
            tmp["data_malformation"].astype(str).str.strip().eq("yes")
        )
        records.append(tmp[["conv_id", "annotator", "malformed"]])

    long_df = pd.concat(records, ignore_index=True)

    any_malformed = (
        long_df.groupby("conv_id")["malformed"].any().rename("any_malformed")
    )
    long_df = long_df.merge(any_malformed, on="conv_id")
    long_df = long_df[long_df["any_malformed"]]

    def agreement_rate(group: pd.DataFrame) -> float:
        majority = group["malformed"].mode().iloc[0]
        return (group["malformed"] == majority).mean()

    agreement = (
        long_df.groupby("conv_id")
        .apply(agreement_rate)
        .rename("agreement")
        .reset_index()
    )

    plt.hist(agreement["agreement"] * 100, bins=10)
    plt.title("Inter-annotator agreement on malformed discussions")
    plt.ylabel("#Conversations")
    plt.xlabel("Agreement (%)")
    graphs.save_plot(graph_output_dir / "malformation_agreement_histogram.png")
    plt.close()


def plot_annotation_frequency(
    dfs_binary: dict[str, pd.DataFrame],
    schema: AnnotationSchema,
    graph_output_dir: Path,
) -> None:
    counts = [
        {
            "annotator": name,
            "category": col,
            "count": int(df[col].sum()),
        }
        for name, df in dfs_binary.items()
        for col in schema.columns
    ]
    counts_df = pd.DataFrame(counts)

    order = sort_annotators(counts_df["annotator"].unique().tolist())
    ax = sns.barplot(
        data=counts_df, y="annotator", x="count", hue="category", order=order
    )
    apply_annotator_label_colors(ax, axis="y")
    plt.ylabel("")
    plt.xlabel("#Annotations")
    graphs.save_plot(
        graph_output_dir / f"annotation_frequency_{schema.name}.png"
    )
    plt.close()


def plot_kappa_heatmap(
    dfs_binary_cleaned: dict[str, pd.DataFrame],
    schema: AnnotationSchema,
    graph_output_dir: Path,
) -> None:
    plt.figure(figsize=(12, 12))
    annotators = sort_annotators(list(dfs_binary_cleaned.keys()))

    matrix = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

    for i, a in enumerate(annotators):
        for j, b in enumerate(annotators):
            if i > j:
                kappas = [
                    cohen_kappa_score(
                        dfs_binary_cleaned[a][col],
                        dfs_binary_cleaned[b][col],
                    )
                    for col in schema.columns
                ]
                matrix.iloc[i, j] = sum(kappas) / len(kappas)
            else:
                matrix.iloc[i, j] = float("nan")

    ax = sns.heatmap(
        matrix,
        annot=True,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        square=True,
        annot_kws={"size": 10},
        fmt=".2f",
        cbar_kws={"label": "Cohen's κ"},
        mask=matrix.isna(),
    )
    apply_annotator_label_colors(ax, axis="x")
    apply_annotator_label_colors(ax, axis="y")
    plt.title("Cohen's κ per annotator pair")
    graphs.save_plot(graph_output_dir / f"kappa_heatmap_{schema.name}.png")
    plt.close()


# ---------------------------------------------------------------------------
# Trinary/Binary reinforcement pipeline
# ---------------------------------------------------------------------------


def run_schema(
    all_dfs_raw: dict[str, pd.DataFrame],
    malformed_ids: set[str],
    schema: AnnotationSchema,
    graph_output_dir: Path,
) -> None:
    """Run the full analysis pipeline for a single schema."""
    print(f"\n{'='*60}")
    print(f"Schema: {schema.name}  (columns: {schema.columns})")
    print("=" * 60)
    
    # project raw data to this schema, binarise, clean
    projected = project_to_schema(all_dfs_raw, schema)
    binarised = to_binary(projected, schema)
    cleaned = remove_malformed_rows(binarised, malformed_ids)
    cleaned = align_by_conv_id(cleaned)

    # kappa summary
    kappas = average_kappa(cleaned, schema)
    print(
        "\nAverage pairwise Cohen's Kappa (cleaned, binary threshold applied):"
    )
    for col, value in kappas.items():
        print(f"  {col}: {value:.3f}")

    # plots
    plot_annotation_frequency(cleaned, schema, graph_output_dir)
    plot_kappa_heatmap(cleaned, schema, graph_output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    human_annotation_dir: Path,
    llm_annotation_dir: Path,
    graph_output_dir: Path,
) -> None:
    graphs.seaborn_setup()

    # ---- load raw data (always in three-way format) ----
    files = list(human_annotation_dir.rglob("*_2.xlsx"))
    if not files:
        raise ValueError("No matching Excel files found.")

    human_dfs = load_human_annotations(files)
    human_alias = {
        key: f"A{i+1}" for i, key in enumerate(sorted(human_dfs.keys()))
    }
    human_dfs = {human_alias[k]: v for k, v in human_dfs.items()}

    # malformation plot uses raw human data (schema-agnostic)
    plot_malformation_agreement_histogram(human_dfs, graph_output_dir)

    malformed_ids = get_malformed_ids(human_dfs)
    print("\nExcluded malformed rows:")
    print(f"  Total excluded rows: {len(malformed_ids)}")

    llm_dfs = load_llm_annotations(list(llm_annotation_dir.rglob("*.csv")))
    all_dfs = align_by_conv_id({**human_dfs, **llm_dfs})

    # ---- run each schema ----
    for schema in ALL_SCHEMAS:
        run_schema(all_dfs, malformed_ids, schema, graph_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-annotation-dir", required=True)
    parser.add_argument("--llm-annotation-dir", required=True)
    parser.add_argument("--graph-output-dir", required=True)
    args = parser.parse_args()

    main(
        human_annotation_dir=Path(args.human_annotation_dir),
        llm_annotation_dir=Path(args.llm_annotation_dir),
        graph_output_dir=Path(args.graph_output_dir),
    )
