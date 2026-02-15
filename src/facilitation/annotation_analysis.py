import argparse
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


def load_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    dfs = {p.stem: read_annotation_file(p) for p in paths}

    # alignment check
    ref_name, ref_df = next(iter(dfs.items()))
    ref_ids = ref_df["conv_id"]

    for name, df in dfs.items():
        if not df["conv_id"].equals(ref_ids):
            raise ValueError(f"conv_id mismatch between {ref_name} and {name}")

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


def plot_annotation_frequency(
    dfs_binary: dict[str, pd.DataFrame], graph_output_dir: Path
) -> None:
    # --- count positives per annotator per category ---
    counts = []
    for name, df in dfs_binary.items():
        for col in REINFORCE_COLS:
            counts.append(
                {
                    "annotator": name.split("_")[0],  # remove _2
                    "category": col,
                    "count": int(
                        df[col].sum()
                    ),  # since binary, sum == count >= threshold
                }
            )

    counts_df = pd.DataFrame(counts)

    # --- barplot ---
    sns.barplot(data=counts_df, y="annotator", x="count", hue="category")
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
        tmp["annotator"] = annotator.split("_")[0]
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
        malformed = (
            df.assign(
                malformed=df["data_malformation"]
                .astype(str)
                .str.strip()
                .eq("yes")
            )
            .loc[lambda x: x["malformed"], "conv_id"]
        )
        malformed_sets.append(set(malformed))

    return set.union(*malformed_sets) if malformed_sets else set()


def remove_malformed_rows(
    dfs_binary: dict[str, pd.DataFrame], malformed_ids: set[str]
) -> dict[str, pd.DataFrame]:
    cleaned = {}

    for name, df in dfs_binary.items():
        cleaned[name] = (
            df[~df["conv_id"].isin(malformed_ids)]
            .reset_index(drop=True)
        )

    return cleaned




def main(input_dir: Path, graph_output_dir: Path):
    graphs.seaborn_setup()

    files = list(input_dir.rglob("*_2.xlsx"))
    if not files:
        raise ValueError("No matching Excel files found.")

    dfs = load_annotations(files)

    # --- malformation agreement plot uses RAW data ---
    plot_malformation_agreement_histogram(dfs, graph_output_dir)

    # --- binary conversion ---
    dfs_binary = to_binary(dfs)

    # --- identify malformed rows ---
    malformed_ids = get_malformed_ids(dfs)

    print("\nExcluded malformed rows:")
    print(f"Total excluded rows: {len(malformed_ids)}")

    # --- cleaned binary dfs ---
    dfs_binary_cleaned = remove_malformed_rows(dfs_binary, malformed_ids)

    # --- use CLEANED data for analysis ---
    kappas = average_kappa(dfs_binary_cleaned)

    print("\nAverage pairwise Cohen's Kappa (cleaned, binary threshold applied):")
    for col, value in kappas.items():
        print(f"{col}: {value:.3f}")

    plot_annotation_frequency(dfs_binary_cleaned, graph_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--graph-output-dir", required=True)
    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.graph_output_dir))
