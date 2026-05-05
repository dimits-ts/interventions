import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

ANNOTATION_COLS = [
    "positive_reinforcement",
    "negative_reinforcement",
    "no_reinforcement",
]


def main(
    human_dir: Path,
    llm_dir: Path,
    output_path: Path,
    text_file: Path | None,
) -> None:
    human_dfs = load_human_annotations(human_dir)
    human_dfs = {
        f"A{i+1}": df for i, (_, df) in enumerate(sorted(human_dfs.items()))
    }

    llm_dual_dfs = load_llm_dual_annotations(llm_dir)
    llm_single_dfs = load_llm_single_annotations(llm_dir)

    all_dfs = align_to_union({**human_dfs, **llm_dual_dfs})
    conv_ids = set(next(iter(all_dfs.values()))["conv_id"])
    llm_single_dfs = align_to_reference(llm_single_dfs, conv_ids)

    output_df = build_output(all_dfs, set(human_dfs), llm_single_dfs)

    if text_file is not None:
        text_df = load_text(text_file)
        output_df = output_df.merge(text_df, on="conv_id", how="left")
        output_df.insert(1, "text", output_df.pop("text"))
        output_df["dataset"] = np.where(
            output_df["dataset"].isin(["iq2", "whow", "fora"]),
            "oral",
            "written",
        )
        n_missing = output_df["text"].isna().sum()
        if n_missing:
            print(
                f"[warn] {n_missing} conv_ids had no match in the text file."
            )
            print(output_df.loc[output_df["text"].isna(), "conv_id"])

    output_df = output_df.set_index("conv_id")
    print_coverage(output_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path)
    print(
        f"Wrote {len(output_df)} rows x {len(output_df.columns)}"
        f" columns -> {output_path}"
    )


def load_text(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype={"conv_id": str})
    missing = {"conv_id", "text", "dataset"} - set(df.columns)
    if missing:
        raise ValueError(f"Text file {path} is missing columns: {missing}")
    return df.loc[:, ["conv_id", "text", "dataset"]].drop_duplicates(
        subset="conv_id"
    )


def load_human_annotations(directory: Path) -> dict[str, pd.DataFrame]:
    files = list(directory.rglob("*_1.xlsx")) + list(
        directory.rglob("*_2.xlsx")
    )
    files = list({p.resolve(): p for p in files}.values())
    if not files:
        raise ValueError("No matching Excel files found.")

    groups: dict[str, list[Path]] = {}
    for p in files:
        key = re.sub(r"\d+", "", p.stem.split("_")[0])
        groups.setdefault(key, []).append(p)

    dfs = {}
    for key, paths in groups.items():
        parts = [read_human_file(p) for p in sorted(paths)]
        df = pd.concat(parts, ignore_index=True)
        dupes = df["conv_id"][df["conv_id"].duplicated()].tolist()
        if dupes:
            print(
                f"[warn] {key!r} has duplicate conv_ids: {dupes[:10]}. Keeping first."
            )
            df = df.drop_duplicates(subset="conv_id")
        dfs[key] = df.sort_values("conv_id").reset_index(drop=True)

    return dfs


def read_human_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"conv_id": str})[
        ["conv_id", "discussion", "data_malformation"] + ANNOTATION_COLS
    ].copy()
    df = df.fillna(0)
    for col in ANNOTATION_COLS:
        df[col] = df[col].apply(
            lambda x: 0 if x == 0 else int(str(x).split(" – ")[0])
        )
    return df.sort_values("conv_id").reset_index(drop=True)


def load_llm_dual_annotations(directory: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in directory.rglob("*_dual_interventions.md.csv"):
        name = re.sub(r"^llm_intervention_", "", path.stem).capitalize()
        dfs[name] = read_llm_dual_file(path)
    return dfs


def read_llm_dual_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"conv_id": str})
    parsed = df["response"].apply(parse_llm_response).apply(pd.Series)
    out = pd.concat([df[["conv_id"]], parsed], axis=1)
    out["data_malformation"] = "no"
    return (
        out.loc[:, ["conv_id", "data_malformation"] + ANNOTATION_COLS]
        .sort_values("conv_id")
        .reset_index(drop=True)
    )


def parse_llm_response(text: str) -> dict[str, int]:
    patterns = {
        "positive_reinforcement": r"Positive:\s(\d+)",
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


def load_llm_single_annotations(directory: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in directory.rglob("*_single_intervention.md.csv"):
        stem = re.sub(
            r"_single_intervention(?:\.md)?$",
            "",
            path.stem,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"^llm_intervention_", "", stem).capitalize()
        dfs[name] = read_llm_single_file(path)
    return dfs


def read_llm_single_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"conv_id", "response"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = df[["conv_id"]].copy()
    out["conv_id"] = out["conv_id"].astype(str)
    out["single_response"] = (
        df["response"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    n_unparsed = out["single_response"].isna().sum()
    if n_unparsed:
        print(
            f"[warn] {path.name}: {n_unparsed} rows with no number found, set to NaN."
        )
    return out.sort_values("conv_id").reset_index(drop=True)


def align_to_union(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    all_ids = sorted(set.union(*[set(df["conv_id"]) for df in dfs.values()]))
    id_frame = pd.DataFrame({"conv_id": all_ids})
    aligned = {}
    for name, df in dfs.items():
        merged = id_frame.merge(df, on="conv_id", how="left")
        merged["data_malformation"] = merged["data_malformation"].fillna("no")
        aligned[name] = merged.reset_index(drop=True)
    return aligned


def align_to_reference(
    dfs: dict[str, pd.DataFrame], conv_ids: set[str]
) -> dict[str, pd.DataFrame]:
    id_frame = pd.DataFrame({"conv_id": sorted(conv_ids)})
    return {
        name: id_frame.merge(df, on="conv_id", how="left").reset_index(
            drop=True
        )
        for name, df in dfs.items()
    }


def build_output(
    dfs: dict[str, pd.DataFrame],
    human_names: set[str],
    single_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = next(iter(dfs.values()))[["conv_id"]].copy()
    human_malformed_cols = []

    for annotator, df in dfs.items():
        malformed_col = f"{annotator}_malformed"
        subset = df[["conv_id", "data_malformation"] + ANNOTATION_COLS].copy()
        subset[malformed_col] = (
            subset["data_malformation"]
            .astype(str)
            .str.strip()
            .eq("yes")
            .astype(float)
        )
        subset = subset.drop(columns=["data_malformation"]).rename(
            columns={col: f"{annotator}_{col}" for col in ANNOTATION_COLS}
        )
        cols = ["conv_id", malformed_col] + [
            f"{annotator}_{col}" for col in ANNOTATION_COLS
        ]
        merged = merged.merge(subset[cols], on="conv_id", how="left")
        if annotator in human_names:
            human_malformed_cols.append(malformed_col)

    merged.insert(
        1, "total_malformed", merged[human_malformed_cols].mean(axis=1)
    )

    for model_name, si_df in single_dfs.items():
        col = f"{model_name}_single_response"
        merged = merged.merge(
            si_df[["conv_id", "single_response"]].rename(
                columns={"single_response": col}
            ),
            on="conv_id",
            how="left",
        )

    return (
        merged.drop_duplicates(subset="conv_id")
        .sort_values("conv_id")
        .reset_index(drop=True)
    )


def print_coverage(df: pd.DataFrame) -> None:
    n = len(df)
    malformed_cols = [
        c
        for c in df.columns
        if c.endswith("_malformed") and c != "total_malformed"
    ]
    annotators = [c.removesuffix("_malformed") for c in malformed_cols]

    print("\nMissing values per column:")
    missing = df.isna().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/n:.1%})")

    print("\nPer-annotator coverage:")
    for ann, mcol in zip(annotators, malformed_cols):
        ann_cols = [c for c in df.columns if c.startswith(f"{ann}_")]
        n_missing = df[ann_cols].isna().any(axis=1).sum()
        print(f"  {ann}: {n - n_missing}/{n} covered ({n_missing} missing)")

    single_cols = [c for c in df.columns if c.endswith("_single_response")]
    if single_cols:
        print("\nSingle-intervention coverage:")
        for col in single_cols:
            n_missing = df[col].isna().sum()
            print(
                f"  {col}: {n - n_missing}/{n} covered ({n_missing} missing)"
            )

    print("\nConversations by annotator count:")
    for count, freq in (
        df[malformed_cols]
        .notna()
        .sum(axis=1)
        .value_counts()
        .sort_index()
        .items()
    ):
        print(f"  {count} annotator(s): {freq} ({freq/n:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-annotation-dir", required=True)
    parser.add_argument("--llm-annotation-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--text-file", default=None)
    args = parser.parse_args()

    main(
        human_dir=Path(args.human_annotation_dir),
        llm_dir=Path(args.llm_annotation_dir),
        output_path=Path(args.output_path),
        text_file=Path(args.text_file) if args.text_file else None,
    )
