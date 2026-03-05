import argparse
import itertools
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

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


def main(input_dir: Path):
    files = list(input_dir.rglob("*_2.xlsx"))
    if not files:
        raise ValueError("No matching Excel files found.")

    dfs = load_annotations(files)
    # print([df.data_malformation.unique() for df in dfs.values()])
    dfs_binary = to_binary(dfs)
    human_df = pd.concat(
        [df.assign(annotator=name) for name, df in dfs_binary.items()],
        ignore_index=True,
    )
    # conv_ids where at least one annotator marked malformed
    malformed_ids = (
        human_df.groupby("conv_id")["data_malformation"]
        .apply(lambda s: (s == "yes").any())
        .loc[lambda s: s]
        .index
    )

    # rows to exclude
    excluded_rows = human_df[human_df["conv_id"].isin(malformed_ids)]

    print("\nExcluded malformed rows:")
    print(excluded_rows)
    print(f"\nTotal excluded rows: {len(excluded_rows)}")
    print(f"Unique conv_ids excluded: {len(malformed_ids)}")
    print(malformed_ids)

    # keep only clean rows
    human_df = human_df[~human_df["conv_id"].isin(malformed_ids)].reset_index(
        drop=True
    )
    kappas = average_kappa(dfs_binary)

    print("\nAverage pairwise Cohen's Kappa (binary threshold applied):")
    for col, value in kappas.items():
        print(f"{col}: {value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    main(Path(args.input_dir))
