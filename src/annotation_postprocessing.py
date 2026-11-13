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
import re
from pathlib import Path

import pandas as pd

import util.io

ANNOTATION_COLS = [
    "positive_reinforcement",
    "negative_reinforcement",
    "no_reinforcement",
]


def main(
    human_dir: Path,
    llm_dir: Path,
    output_dir: Path,
    text_file: Path | None,
    cache_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    human_raw = load_human_annotations(human_dir, cache_dir)
    human_dfs = {}
    for i, (real, df) in enumerate(sorted(human_raw.items())):
        alias = f"A{i+1}"
        print(f"  {alias} -> {real}")
        human_dfs[alias] = df

    build_self_consistency_df(human_dfs, output_dir)

    for key in human_dfs:
        df = human_dfs[key]
        human_dfs[key] = df.drop_duplicates(subset="conv_id").reset_index(
            drop=True
        )

    llm_dual_dfs = load_llm_dual_annotations(llm_dir, cache_dir)
    llm_single_dfs = load_llm_single_annotations(llm_dir, cache_dir)

    all_dfs = align_to_union({**human_dfs, **llm_dual_dfs})
    conv_ids = set(next(iter(all_dfs.values()))["conv_id"])
    llm_single_dfs = align_to_reference(llm_single_dfs, conv_ids)

    output_df = build_output(all_dfs, set(human_dfs), llm_single_dfs)

    if text_file is not None:
        datasets_df = load_datasets(text_file, cache_dir)
        output_df = output_df.merge(datasets_df, on="conv_id", how="left")
        n_missing = output_df["dataset"].isna().sum()
        if n_missing:
            print(
                f"[warn] {n_missing} rows still have no dataset after "
                "recovery."
            )
        output_df = output_df.dropna(subset="dataset")
    else:
        print("[warn] No --text-file given; output has no 'dataset' column.")

    output_df = output_df.set_index("conv_id")
    print_coverage(output_df)

    output_df.to_csv(output_dir / "all_annotations.csv")
    print(
        f"Wrote {len(output_df)} rows x {len(output_df.columns)}"
        f" columns -> {output_dir}"
    )


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


# ---------------------------------------------------------------------------
# Text file (only the conv_id -> dataset mapping is needed)
# ---------------------------------------------------------------------------
def load_datasets(path: Path, cache_dir: Path) -> pd.DataFrame:
    return util.io.load_or_compute(
        cache_dir / "text_datasets.parquet",
        lambda: _read_datasets(path),
        label="conv_id -> dataset mapping",
    )


def _read_datasets(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    header = pd.read_csv(path, sep=sep, nrows=0).columns
    missing = {"conv_id", "dataset"} - set(header)
    if missing:
        raise ValueError(f"Text file {path} is missing columns: {missing}")

    # Never parse the text column; dedupe per chunk to keep memory flat
    df = util.io.progress_load_csv(
        path,
        usecols=["conv_id", "dataset"],
        dtype={"conv_id": str, "dataset": str},
        chunk_fn=lambda c: c.drop_duplicates(subset="conv_id"),
        sep=sep,
    )
    return df.drop_duplicates(subset="conv_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Human annotations (one cache file per annotator)
# ---------------------------------------------------------------------------
def load_human_annotations(
    directory: Path, cache_dir: Path
) -> dict[str, pd.DataFrame]:
    groups = discover_human_files(directory)

    dfs = {}
    for key, paths in groups.items():
        df = util.io.load_or_compute(
            cache_dir / f"human_{_safe_name(key)}.parquet",
            lambda paths=paths: read_human_group(paths),
            label=f"human annotations '{key}'",
        )
        # Checked on every run, so the warning also shows for cached data
        dupes = df["conv_id"][df["conv_id"].duplicated()].tolist()
        if dupes:
            print(
                f"[warn] {key!r} has duplicate conv_ids: {dupes[:10]}. "
                "Keeping first."
            )
        dfs[key] = df
    return dfs


def discover_human_files(directory: Path) -> dict[str, list[Path]]:
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
    return {key: sorted(paths) for key, paths in groups.items()}


def read_human_group(paths: list[Path]) -> pd.DataFrame:
    parts = [read_human_file(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values("conv_id").reset_index(drop=True)


def read_human_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"conv_id": str})[
        ["conv_id", "discussion", "data_malformation", "rationale/comments"]
        + ANNOTATION_COLS
    ].copy()
    df = df.rename(columns={"rationale/comments": "rationale"})
    df = df.fillna(0)
    for col in ANNOTATION_COLS:
        df[col] = df[col].apply(
            lambda x: 0 if x == 0 else int(str(x).split(" – ")[0])
        )
    # fillna(0) leaves text columns with mixed str/int values, which Parquet
    # cannot store. Casting to str writes identically to the output CSV.
    for col in ["conv_id", "discussion", "data_malformation", "rationale"]:
        df[col] = df[col].astype(str)
    return df.sort_values("conv_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# LLM annotations (one cache file per model file)
# ---------------------------------------------------------------------------
def load_llm_dual_annotations(
    directory: Path, cache_dir: Path
) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in directory.rglob("*_dual_interventions.md.csv"):
        name = re.sub(r"^llm_intervention_", "", path.stem).capitalize()
        dfs[name] = util.io.load_or_compute(
            cache_dir / f"llm_dual_{_safe_name(name)}.parquet",
            lambda path=path: read_llm_dual_file(path),
            label=f"LLM dual annotations '{name}'",
        )
    return dfs


def read_llm_dual_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, usecols=["conv_id", "response"], dtype={"conv_id": str}
    )
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


def load_llm_single_annotations(
    directory: Path, cache_dir: Path
) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in directory.rglob("*_single_intervention.md.csv"):
        stem = re.sub(
            r"_single_intervention(?:\.md)?$",
            "",
            path.stem,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"^llm_intervention_", "", stem).capitalize()
        dfs[name] = util.io.load_or_compute(
            cache_dir / f"llm_single_{_safe_name(name)}.parquet",
            lambda path=path: read_llm_single_file(path),
            label=f"LLM single annotations '{name}'",
        )
    return dfs


def read_llm_single_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = {"conv_id", "response"} - set(header)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = pd.read_csv(
        path, usecols=["conv_id", "response"], dtype={"conv_id": str}
    )
    out = df[["conv_id"]].copy()
    out["single_response"] = (
        df["response"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    n_unparsed = out["single_response"].isna().sum()
    if n_unparsed:
        print(
            f"[warn] {path.name}: {n_unparsed} rows with no number found, set to NaN."
        )
    return out.sort_values("conv_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Alignment and output
# ---------------------------------------------------------------------------
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
    first = next(iter(dfs.values()))
    merged = first[["conv_id", "discussion"]].copy()
    human_malformed_cols = []

    for annotator, df in dfs.items():
        has_rationale = "rationale" in df.columns
        extra = ["rationale"] if has_rationale else []

        malformed_col = f"{annotator}_malformed"
        subset = df[
            ["conv_id", "data_malformation"] + ANNOTATION_COLS + extra
        ].copy()
        subset[malformed_col] = (
            subset["data_malformation"]
            .astype(str)
            .str.strip()
            .eq("yes")
            .astype(float)
        )
        subset = subset.drop(columns=["data_malformation"]).rename(
            columns={
                col: f"{annotator}_{col}" for col in ANNOTATION_COLS + extra
            }
        )
        cols = ["conv_id", malformed_col] + [
            f"{annotator}_{col}" for col in ANNOTATION_COLS + extra
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

    human_annotation_cols = [
        f"{h}_{col}" for h in human_names for col in ANNOTATION_COLS
    ]
    merged = merged.dropna(subset=human_annotation_cols, how="all")
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


def build_self_consistency_df(
    dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Extract duplicate conv_id rows per annotator to assess self-consistency.
    Saves a combined dataframe to disk.
    """
    all_dupes = []

    for annotator, df in dfs.items():
        dupes = df[df["conv_id"].duplicated(keep=False)].copy()
        if dupes.empty:
            continue

        dupes.insert(0, "annotator", annotator)
        all_dupes.append(dupes)

    if not all_dupes:
        print(
            "[info] No duplicate conv_ids found for self-consistency analysis."
        )
        return

    consistency_df = pd.concat(all_dupes, ignore_index=True)

    out_file = output_dir / "self_consistency_duplicates.csv"
    consistency_df.to_csv(out_file, index=False)

    print(
        f"Wrote self-consistency dataframe with {len(consistency_df)} rows "
        f"-> {out_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-annotation-dir", required=True)
    parser.add_argument("--llm-annotation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-file", default=None)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help=(
            "Directory for cached calculations (one file per calculation). "
            "Existing cache files are always used; delete them to recompute."
        ),
    )
    args = parser.parse_args()

    main(
        human_dir=Path(args.human_annotation_dir),
        llm_dir=Path(args.llm_annotation_dir),
        output_dir=Path(args.output_dir),
        text_file=Path(args.text_file) if args.text_file else None,
        cache_dir=Path(args.cache_dir),
    )