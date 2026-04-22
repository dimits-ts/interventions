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
    human_annotation_dir: Path,
    llm_annotation_dir: Path,
    output_path: Path,
    text_file: Path | None,
) -> None:
    # Match both naming conventions: *_part_2.xlsx and *_2.xlsx
    files = list(human_annotation_dir.rglob("*_part2.xlsx")) + list(
        human_annotation_dir.rglob("*_2.xlsx")
    )
    # Deduplicate in case a path matches both patterns (shouldn't happen, but safe)
    files = list({p.resolve(): p for p in files}.values())
    if not files:
        raise ValueError("No matching Excel files found.")

    human_dfs = load_human_annotations(files)
    human_alias = {
        key: f"A{i+1}" for i, key in enumerate(sorted(human_dfs.keys()))
    }
    print("Aliases used:")
    print(human_alias)

    human_dfs = {human_alias[k]: v for k, v in human_dfs.items()}

    malformed_ids = get_malformed_ids(human_dfs)
    print(f"Total malformed rows: {len(malformed_ids)}")

    # ── Dual-intervention LLM annotations (three reinforcement count columns) ──
    llm_dfs = load_llm_annotations(
        list(llm_annotation_dir.rglob("*_dual_interventions.md.csv"))
    )
    all_dfs = align_by_conv_id({**human_dfs, **llm_dfs})
    all_dfs = remove_malformed_rows(all_dfs, malformed_ids)

    # ── Single-intervention LLM annotations (binary yes/no column) ────────────
    single_dfs = load_llm_single_interventions(
        list(llm_annotation_dir.rglob("*_single_intervention.md.csv"))
    )
    # Align and clean single-intervention data independently
    if single_dfs:
        single_dfs_aligned = align_single_interventions(
            single_dfs, set(all_dfs.values().__iter__().__next__()["conv_id"])
        )
        single_dfs_aligned = remove_malformed_rows_single(
            single_dfs_aligned, malformed_ids
        )
    else:
        single_dfs_aligned = {}

    output_df = build_wide_output(
        all_dfs,
        human_annotator_names=set(human_dfs.keys()),
        single_intervention_dfs=single_dfs_aligned,
    )

    if text_file is not None:
        text_df = load_text_file(text_file)
        output_df = output_df.merge(text_df, on="conv_id", how="left")
        column_to_move = output_df.pop("text")
        output_df.insert(1, "text", column_to_move)
        output_df = output_df.set_index("conv_id")
        output_df.dataset = np.where(
            output_df.dataset.isin(["iq2", "whow", "fora"]), "oral", "written"
        )

        n_missing = output_df["text"].isna().sum()
        if n_missing:
            print(
                f"[warn] {n_missing} conv_ids had no match in the text file."
            )

    print_missing_stats(output_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path)
    print(
        f"\nWrote {len(output_df)} rows x {len(output_df.columns)} "
        f"columns -> {output_path}"
    )


def load_text_file(path: Path) -> pd.DataFrame:
    """Load a CSV/TSV with at least conv_id and text columns."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype={"conv_id": str})
    missing = {"conv_id", "text", "dataset"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Text file {path} is missing required columns: {missing}"
        )
    df = df[["conv_id", "text", "dataset"]].drop_duplicates(
        subset="conv_id", keep="first"
    )
    return df


def build_wide_output(
    dfs: dict[str, pd.DataFrame],
    human_annotator_names: set[str],
    single_intervention_dfs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Merge all annotator DataFrames into a single wide CSV.

    Output schema:
        conv_id | total_malformed
                | A1_malformed | A1_positive_reinforcement |
                  A1_negative_reinforcement | A1_no_reinforcement
                | A2_malformed | ...
                | Gpt4_malformed | Gpt4_positive_reinforcement | ...
                | Gpt4_single_response | Claude_single_response | ...

    - ``{annotator}_malformed``: 1 if that annotator flagged the row, 0 otherwise.
    - ``total_malformed``: fraction of *human* annotators who flagged the row.
    - ``{model}_single_response``: binary 0/1 from single-intervention LLM files.
    - Each row is one conversation (conv_id).
    - Cells are NaN where an annotator did not cover a conversation.
    """
    first = next(iter(dfs.values()))
    merged = first[["conv_id"]].copy()

    malformed_human_cols: list[str] = []

    for annotator, df in dfs.items():
        malformed_col = f"{annotator}_malformed"
        subset = df[["conv_id", "data_malformation"] + ANNOTATION_COLS].copy()
        subset[malformed_col] = (
            subset["data_malformation"].astype(str).str.strip().eq("yes")
        ).astype(float)

        rename_map = {col: f"{annotator}_{col}" for col in ANNOTATION_COLS}
        subset = subset.drop(columns=["data_malformation"]).rename(
            columns=rename_map
        )

        ordered_cols = ["conv_id", malformed_col] + [
            f"{annotator}_{col}" for col in ANNOTATION_COLS
        ]
        merged = merged.merge(subset[ordered_cols], on="conv_id", how="left")

        if annotator in human_annotator_names:
            malformed_human_cols.append(malformed_col)

    # total_malformed = mean of human annotators' malformed flags (ignores NaN)
    merged.insert(
        1,
        "total_malformed",
        merged[malformed_human_cols].mean(axis=1),
    )

    # ── Append single-intervention binary columns ──────────────────────────
    if single_intervention_dfs:
        for model_name, si_df in single_intervention_dfs.items():
            col = f"{model_name}_single_response"
            si_subset = si_df[["conv_id", "single_response"]].rename(
                columns={"single_response": col}
            )
            merged = merged.merge(si_subset, on="conv_id", how="left")

    merged = merged[~merged.conv_id.duplicated(keep="first")]
    merged = (
        merged.sort_values("conv_id")
        .reset_index(drop=True)
        .set_index("conv_id")
    )
    return merged


def load_human_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    # Group files by annotator key; each annotator may have 1 or 2 files.
    groups: dict[str, list[Path]] = {}
    for p in paths:
        key = _annotator_key(p)
        groups.setdefault(key, []).append(p)

    dfs: dict[str, pd.DataFrame] = {}
    for key, file_paths in groups.items():
        parts = [read_annotation_file(p) for p in sorted(file_paths)]
        combined = pd.concat(parts, ignore_index=True)
        dupes = combined["conv_id"][combined["conv_id"].duplicated()].tolist()
        if dupes:
            print(
                f"  [warn] Annotator {key!r} has duplicate conv_ids across "
                f"files: {dupes[:10]}{'...' if len(dupes) > 10 else ''}. "
                f"Keeping first occurrence."
            )
            combined = combined.drop_duplicates(subset="conv_id", keep="first")
        dfs[key] = combined.sort_values("conv_id").reset_index(drop=True)

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
    out = out[["conv_id", "data_malformation"] + ANNOTATION_COLS]
    out["conv_id"] = out["conv_id"].astype(str)
    out = out.sort_values("conv_id").reset_index(drop=True)

    return out


def load_llm_annotations(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Load dual-intervention LLM CSVs (three reinforcement count columns)."""
    dfs = {}
    for p in paths:
        name = re.sub(r"^llm_intervention_", "", p.stem).capitalize()
        dfs[name] = read_llm_annotation_file(p)
    return dfs


# ── Single-intervention helpers ──────────────────────────────────────────────


def read_llm_single_intervention_file(path: Path) -> pd.DataFrame:
    """Read a *_single_intervention.md.csv file.

    Expected columns: conv_id, response (yes/no).
    Returns a two-column DataFrame: conv_id, single_response (0.0/1.0).
    NaN is preserved for rows that could not be parsed.
    """
    df = pd.read_csv(path)
    missing = {"conv_id", "response"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Single-intervention file {path} is missing columns: {missing}"
        )
    out = df[["conv_id"]].copy()
    out["conv_id"] = out["conv_id"].astype(str)
    normalised = df["response"].astype(str).str.strip().str.lower()
    out["single_response"] = normalised.map({"yes": 1.0, "no": 0.0})
    n_unparsed = out["single_response"].isna().sum()
    if n_unparsed:
        print(
            f"  [warn] {path.name}: {n_unparsed} rows had an unrecognised "
            f"response value and were set to NaN."
        )
    return out.sort_values("conv_id").reset_index(drop=True)


def load_llm_single_interventions(
    paths: list[Path],
) -> dict[str, pd.DataFrame]:
    """Return {model_name: single_response_df} for every single-intervention CSV."""
    dfs = {}
    for p in paths:
        # Strip _single_intervention(.md) suffix, then the llm_intervention_ prefix
        stem = re.sub(
            r"_single_intervention(?:\.md)?$", "", p.stem, flags=re.IGNORECASE
        )
        name = re.sub(r"^llm_intervention_", "", stem).capitalize()
        dfs[name] = read_llm_single_intervention_file(p)
    return dfs


def align_single_interventions(
    single_dfs: dict[str, pd.DataFrame],
    reference_conv_ids: set[str],
) -> dict[str, pd.DataFrame]:
    """Align single-intervention DataFrames to the reference conv_id universe.

    Conversations present in the reference set but absent from a model's file
    are filled with NaN so missingness is explicit in the final output.
    """
    id_frame = pd.DataFrame({"conv_id": sorted(reference_conv_ids)})
    aligned = {}
    for name, df in single_dfs.items():
        merged = id_frame.merge(df, on="conv_id", how="left")
        aligned[name] = merged.reset_index(drop=True)
    return aligned


def remove_malformed_rows_single(
    single_dfs: dict[str, pd.DataFrame],
    malformed_ids: set[str],
) -> dict[str, pd.DataFrame]:
    """Drop malformed conv_ids from single-intervention DataFrames."""
    return {
        name: df[~df["conv_id"].isin(malformed_ids)].reset_index(drop=True)
        for name, df in single_dfs.items()
    }


# ── Shared helpers ───────────────────────────────────────────────────────────


def read_annotation_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df[
        ["conv_id", "discussion", "data_malformation"] + ANNOTATION_COLS
    ].copy()
    df["conv_id"] = df["conv_id"].astype(str)
    df = df.sort_values("conv_id").reset_index(drop=True)
    df = df.fillna(0)

    for col in ANNOTATION_COLS:
        df[col] = df[col].apply(
            lambda x: 0 if x == 0 else int(str(x).split(" – ")[0])
        )

    return df


def _annotator_key(path: Path) -> str:
    """Derive a stable annotator key from a filename.

    Both ``username_2.xlsx`` and ``username_part_2.xlsx`` map to ``username``
    so that the two files are concatenated into a single annotator DataFrame.
    """
    stem = re.sub(r"\d+", "", path.stem.split("_")[0])  # remove numbers
    return stem


def align_by_conv_id(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align all annotators to the union of conv_ids.

    Rows missing for a given annotator are filled with NaN so that downstream
    pairwise computations can detect and skip them rather than silently
    dropping conversations that only some annotators covered.
    """
    all_ids = sorted(set.union(*[set(df["conv_id"]) for df in dfs.values()]))
    id_frame = pd.DataFrame({"conv_id": all_ids})

    aligned = {}
    for name, df in dfs.items():
        merged = id_frame.merge(df, on="conv_id", how="left")
        merged["data_malformation"] = merged["data_malformation"].fillna("no")
        aligned[name] = merged.reset_index(drop=True)
    return aligned


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


def print_missing_stats(df: pd.DataFrame) -> None:
    """Print a breakdown of missing annotations in the output DataFrame."""
    n = len(df)

    # ── Per-column missingness ───────────────────────────────────────────────
    print(
        "\n── Missing values per column ──────────────────────────────────────"
    )
    col_stats = (
        df.isna()
        .sum()
        .to_frame("missing")
        .assign(pct=lambda x: (x["missing"] / n * 100).round(1))
    )
    col_stats = col_stats[col_stats["missing"] > 0]
    if col_stats.empty:
        print("  No missing values.")
    else:
        for col, row in col_stats.iterrows():
            print(f"  {col}: {int(row['missing'])} missing ({row['pct']}%)")

    # ── Per-annotator coverage ───────────────────────────────────────────────
    print(
        "\n── Per-annotator coverage ─────────────────────────────────────────"
    )
    malformed_cols = [
        c
        for c in df.columns
        if c.endswith("_malformed") and c != "total_malformed"
    ]
    annotators = [c.removesuffix("_malformed") for c in malformed_cols]
    for ann, mcol in zip(annotators, malformed_cols):
        ann_cols = [c for c in df.columns if c.startswith(f"{ann}_")]
        n_missing = df[ann_cols].isna().any(axis=1).sum()
        print(
            f"  {ann}: {n - n_missing} / {n} conversations covered ({n_missing} missing, {n_missing/n:.1%})"
        )

    # ── Single-intervention coverage ─────────────────────────────────────────
    single_cols = [c for c in df.columns if c.endswith("_single_response")]
    if single_cols:
        print(
            "\n── Single-intervention coverage ───────────────────────────────────"
        )
        for col in single_cols:
            n_missing = df[col].isna().sum()
            print(
                f"  {col}: {n - n_missing} / {n} conversations covered "
                f"({n_missing} missing, {n_missing/n:.1%})"
            )

    # ── Coverage count distribution ──────────────────────────────────────────
    print(
        "\n── Conversations by number of annotators covering them ─────────────"
    )
    n_annotators = df[malformed_cols].notna().sum(axis=1)
    for count, freq in n_annotators.value_counts().sort_index().items():
        print(f"  {count} annotator(s): {freq} conversations ({freq/n:.1%})")

    # ── Conversations with incomplete coverage ───────────────────────────────
    n_expected = len(annotators)
    incomplete = (n_annotators < n_expected).sum()
    print(
        f"\n── Incomplete coverage ─────────────────────────────────────────────"
    )
    print(
        f"  {incomplete} / {n} conversations missing at least one annotator ({incomplete/n:.1%})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-annotation-dir", required=True)
    parser.add_argument("--llm-annotation-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--text-file",
        default=None,
        help="Optional CSV/TSV with 'conv_id' and 'text' columns to merge into the output.",
    )
    args = parser.parse_args()

    main(
        human_annotation_dir=Path(args.human_annotation_dir),
        llm_annotation_dir=Path(args.llm_annotation_dir),
        output_path=Path(args.output_path),
        text_file=Path(args.text_file) if args.text_file else None,
    )
