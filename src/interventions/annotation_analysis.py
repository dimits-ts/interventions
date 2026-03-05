import argparse
import re
import itertools
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from ..util import graphs


REINFORCE_COLS = [
    "positive_reinforcement",
    "negative_reinforcement",
    "no_reinforcement",
]

THRESHOLD = 3


def main(
    human_annotation_dir: Path,
    llm_annotation_dir: Path,
    graph_output_dir: Path,
):
    graphs.seaborn_setup()

    files = list(human_annotation_dir.rglob("*_2.xlsx"))
    if not files:
        raise ValueError("No matching Excel files found.")

    human_dfs = load_human_annotations(files)

    # Build a stable mapping from raw human annotator keys → A1, A2, ...
    human_alias = {
        key: f"A{i+1}" for i, key in enumerate(sorted(human_dfs.keys()))
    }
    human_dfs = {human_alias[k]: v for k, v in human_dfs.items()}

    # --- malformation agreement plot uses RAW human data ---
    plot_malformation_agreement_histogram(human_dfs, graph_output_dir)

    malformed_ids = get_malformed_ids(human_dfs)

    print("\nExcluded malformed rows:")
    print(f"Total excluded rows: {len(malformed_ids)}")

    llm_files = list(llm_annotation_dir.rglob("*.csv"))
    llm_dfs = load_llm_annotations(llm_files)
    all_dfs = {**human_dfs, **llm_dfs}
    all_dfs = align_by_conv_id(all_dfs)

    # merge annotators
    all_dfs = {**human_dfs, **llm_dfs}
    dfs_binary = to_binary(all_dfs)

    # --- cleaned binary dfs ---
    dfs_binary_cleaned = remove_malformed_rows(dfs_binary, malformed_ids)
    dfs_binary_cleaned = align_by_conv_id(dfs_binary_cleaned)

    # --- use CLEANED data for analysis ---
    kappas = average_kappa(dfs_binary_cleaned)

    print(
        "\nAverage pairwise Cohen's Kappa (cleaned, binary threshold applied):"
    )
    for col, value in kappas.items():
        print(f"{col}: {value:.3f}")

    plot_annotation_frequency(dfs_binary_cleaned, graph_output_dir)
    plot_kappa_heatmap(dfs_binary_cleaned, graph_output_dir)


def read_annotation_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df[["conv_id", "data_malformation"] + REINFORCE_COLS].copy()
    df["conv_id"] = df["conv_id"].astype(str)
    df = df.sort_values("conv_id").reset_index(drop=True)  # type: ignore
    df = df.fillna(0)

    for col in REINFORCE_COLS:
        df[col] = df[col].apply(
            lambda x: 0 if x == 0 else int(str(x).split(" – ")[0])
        )

    return df


def load_human_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    dfs = {p.stem: read_annotation_file(p) for p in paths}

    # alignment check
    ref_name, ref_df = next(iter(dfs.items()))
    ref_ids = ref_df["conv_id"]

    for name, df in dfs.items():
        if not df["conv_id"].equals(ref_ids):
            raise ValueError(f"conv_id mismatch between {ref_name} and {name}")

    return dfs


def align_by_conv_id(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # find common conv_ids across all annotators
    common_ids = set.intersection(*[set(df["conv_id"]) for df in dfs.values()])

    aligned = {}
    for name, df in dfs.items():
        aligned[name] = (
            df[df["conv_id"].isin(common_ids)]
            .sort_values("conv_id")
            .reset_index(drop=True)
        )

    return aligned


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

    result = {}

    for col, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        result[col] = int(match.group(1)) if match else 0

    return result


def read_llm_annotation_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # parse reinforcement counts
    parsed = df["response"].apply(parse_llm_response).apply(pd.Series)

    out = pd.concat([df[["conv_id"]], parsed], axis=1)

    # LLMs did not label malformation
    out["data_malformation"] = "no"

    # match schema of human dfs
    out = out[["conv_id", "data_malformation"] + REINFORCE_COLS]

    out["conv_id"] = out["conv_id"].astype(str)
    out = out.sort_values("conv_id").reset_index(drop=True)

    return out


def load_llm_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    dfs = {}

    for p in paths:
        # extract filename after "llm_intervention_"
        name = re.sub(r"^llm_intervention_", "", p.stem).capitalize()
        dfs[name] = read_llm_annotation_file(p)

    return dfs


def to_binary(dfs: dict[str, pd.DataFrame]):
    out = {}
    for name, df in dfs.items():
        df_bin = df.copy()
        for col in REINFORCE_COLS:
            df_bin[col] = (df_bin[col] >= THRESHOLD).astype(int)
        out[name] = df_bin
    return out


def average_kappa(dfs_binary: dict[str, pd.DataFrame]) -> dict[str, float]:
    annotators = list(dfs_binary.keys())
    result = {}

    for col in REINFORCE_COLS:
        kappas = [
            cohen_kappa_score(dfs_binary[a][col], dfs_binary[b][col])
            for a, b in itertools.combinations(annotators, 2)
        ]
        result[col] = sum(kappas) / len(kappas) if kappas else 0.0

    return result


def sort_annotators(names: list[str]) -> list[str]:
    def key(name: str):
        if re.match(r"A\d+", name):
            return (0, int(name[1:]))  # humans first, numeric order
        return (1, name.lower())  # LLMs next, alphabetical

    return sorted(names, key=key)


def plot_annotation_frequency(
    dfs_binary: dict[str, pd.DataFrame], graph_output_dir: Path
) -> None:
    # --- count positives per annotator per category ---
    counts = []
    for name, df in dfs_binary.items():
        for col in REINFORCE_COLS:
            counts.append(
                {
                    "annotator": name,  # already aliased (A1, A2, ... or LLM name)
                    "category": col,
                    "count": int(
                        df[col].sum()
                    ),  # since binary, sum == count >= threshold
                }
            )

    counts_df = pd.DataFrame(counts)

    order = sort_annotators(counts_df["annotator"].unique().tolist())

    sns.barplot(
        data=counts_df,
        y="annotator",
        x="count",
        hue="category",
        order=order,
    )
    plt.ylabel("")
    plt.xlabel("#Annotations")
    graphs.save_plot(graph_output_dir / "annotation_frequency.png")
    plt.close()


def plot_malformation_agreement_histogram(
    dfs: dict[str, pd.DataFrame], graph_output_dir: Path
) -> None:
    records = []

    # --- collect boolean malformation labels ---
    for annotator, df in dfs.items():
        tmp = df[["conv_id", "data_malformation"]].copy()
        tmp["annotator"] = annotator  # already aliased (A1, A2, ...)
        tmp["malformed"] = (
            tmp["data_malformation"].astype(str).str.strip().eq("yes")
        )
        records.append(tmp[["conv_id", "annotator", "malformed"]])

    long_df = pd.concat(records, ignore_index=True)

    # --- keep rows where at least one annotator says malformed ---
    any_malformed = (
        long_df.groupby("conv_id")["malformed"].any().rename("any_malformed")
    )
    long_df = long_df.merge(any_malformed, on="conv_id")
    long_df = long_df[long_df["any_malformed"]]

    # --- compute agreement per conv_id ---
    def agreement_rate(group: pd.DataFrame) -> float:
        majority = group["malformed"].mode().iloc[0]
        return (group["malformed"] == majority).mean()

    agreement = (
        long_df.groupby("conv_id")
        .apply(agreement_rate)
        .rename("agreement")
        .reset_index()
    )

    # --- histogram ---
    plt.hist(agreement["agreement"] * 100, bins=10)
    plt.title("Inter-annotator agreement on malformed discussions")
    plt.ylabel("#Conversations")
    plt.xlabel("Agreement (%)")
    graphs.save_plot(graph_output_dir / "malformation_agreement_histogram.png")
    plt.close()


def get_malformed_ids(dfs: dict[str, pd.DataFrame]) -> set[str]:
    malformed_sets = []

    for df in dfs.values():
        malformed = df.assign(
            malformed=df["data_malformation"].astype(str).str.strip().eq("yes")
        ).loc[lambda x: x["malformed"], "conv_id"]
        malformed_sets.append(set(malformed))

    return set.union(*malformed_sets) if malformed_sets else set()


def remove_malformed_rows(
    dfs_binary: dict[str, pd.DataFrame], malformed_ids: set[str]
) -> dict[str, pd.DataFrame]:
    cleaned = {}

    for name, df in dfs_binary.items():
        cleaned[name] = df[~df["conv_id"].isin(malformed_ids)].reset_index(
            drop=True
        )

    return cleaned


def plot_kappa_heatmap(
    dfs_binary_cleaned: dict[str, pd.DataFrame],
    graph_output_dir: Path,
) -> None:
    annotators = sort_annotators(list(dfs_binary_cleaned.keys()))
    labels = annotators
    labels = annotators  # already aliased (A1, A2, ... or LLM name)

    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

    # compute pairwise kappa (averaged across categories)
    for i, a in enumerate(annotators):
        for j, b in enumerate(annotators):
            if i >= j:
                if i == j:
                    matrix.iloc[i, j] = float("nan")
                else:
                    kappas = [
                        cohen_kappa_score(
                            dfs_binary_cleaned[a][col],
                            dfs_binary_cleaned[b][col],
                        )
                        for col in REINFORCE_COLS
                    ]
                    matrix.iloc[i, j] = sum(kappas) / len(kappas)
            else:
                matrix.iloc[i, j] = float("nan")

    sns.heatmap(
        matrix,
        annot=True,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        square=True,
        cbar_kws={"label": "Cohen's κ"},
        mask=matrix.isna(),
    )

    plt.title("Cohen's κ per annotator pair")
    graphs.save_plot(graph_output_dir / "kappa_heatmap.png")
    plt.close()


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
