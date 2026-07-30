"""
run_ablation_analysis.py

Analysis-only half of the prompt-ablation study. Assumes inference has
already been run (e.g. by run_ablation.sh) and that per-variant CSVs already
sit in annotations_dir, named llm_intervention_<VARIANT>_timing_<task>.csv.

This version is fully self-contained -- it does NOT import llm_test.py, on
purpose, since llm_test.py is shared elsewhere in the project and its
normalize_binary_response() only understands plain "0"/"1"/"yes"/"no"
responses. Your model's responses for this task instead look like:

    Positive: 4   Negative: 1   No Reinforcement: 2

So this script parses that format directly and derives a binary prediction
via argmax of the three counts:
  - "positive_reinforcement" wins  -> predicted 1
  - "negative_reinforcement" or "no_reinforcement" wins -> predicted 0
  - a genuine tie (including all-zero, which usually means the regex found
    nothing at all) -> unparseable, excluded from metrics

--truth_column is assumed to already be binary (0/1), unchanged from before
-- only the prediction side needed reparsing. If your ground truth is
actually a 3-way label instead, this mapping is the wrong tool; say so and
this can be reworked into a proper multi-class comparison instead.

Reports:
  - a combined metrics table (Precision/Recall/F1p/F1n/Accuracy/Support),
    one row per (variant, dataset), plus an "All" macro-average row per
    variant -- same shape as before
  - the unparseable-response rate per variant
  - pairwise row-level agreement between every pair of variants
Usage
-----
python run_ablation_analysis.py \\
    --annotations_dir data/llm_output/ablation/llama8b \\
    --output_dir data/llm_metrics/ablation/llama8b \\
    --truth_column should_intervene \\
    --task_name prediction
"""

import argparse
import itertools
import re
from pathlib import Path

import pandas as pd
import sklearn.metrics
from scipy.stats import binomtest


REINFORCEMENT_PATTERNS = {
    "positive_reinforcement": r"Positive:\s*\[?(\d+)\]?",
    "negative_reinforcement": r"Negative:\s*\[?(\d+)\]?",
    "no_reinforcement": r"No Reinforcement:\s*\[?(\d+)\]?",
}


# ---------------------------------------------------------------------------
# Response parsing: "Positive: 4   Negative: 1   No Reinforcement: 2" -> 0/1
# ---------------------------------------------------------------------------
def parse_reinforcement_counts(text) -> dict:
    if pd.isna(text):
        return {col: None for col in REINFORCEMENT_PATTERNS}
    text = str(text)
    return {
        col: (
            int(m.group(1))
            if (m := re.search(pat, text, re.IGNORECASE))
            else None
        )
        for col, pat in REINFORCEMENT_PATTERNS.items()
    }


def reinforcement_to_binary(
    text, positive_label: str = "positive_reinforcement"
):
    counts = parse_reinforcement_counts(text)
    if all(v is None for v in counts.values()):
        return None  # regex found nothing at all -> genuinely unparseable

    filled = {k: (v if v is not None else 0) for k, v in counts.items()}
    max_val = max(filled.values())
    top = [k for k, v in filled.items() if v == max_val]

    if len(top) != 1:
        return None  # tie (including all-zero) -> ambiguous, excluded

    return 1 if top[0] == positive_label else 0


# ---------------------------------------------------------------------------
# Discover per-variant annotation files
# ---------------------------------------------------------------------------
def discover_variant_files(
    annotations_dir: Path, task_name: str
) -> dict[str, Path]:
    pattern = f"llm_intervention_*_timing_{task_name}.csv"
    files = sorted(annotations_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {annotations_dir}"
        )

    variant_files = {}
    for f in files:
        parts = f.stem.split("_")
        if len(parts) < 3:
            print(f"[warn] skipping unexpected filename: {f.name}")
            continue
        variant = parts[2]
        variant_files[variant] = f

    return variant_files


# ---------------------------------------------------------------------------
# Metrics (local reimplementation, no llm_test.py dependency)
# ---------------------------------------------------------------------------
def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    empty_result = {
        "Precision": float("nan"),
        "Recall": float("nan"),
        "F1p": float("nan"),
        "F1n": float("nan"),
        "Accuracy": float("nan"),
        "Support": 0,
    }

    if y_true.empty or y_pred.empty:
        return empty_result

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    df = df[df["y_true"].isin([0, 1]) & df["y_pred"].isin([0, 1])]

    if df.empty:
        return empty_result

    y_true_clean = df["y_true"].astype(int)
    y_pred_clean = df["y_pred"].astype(int)

    return {
        "Precision": sklearn.metrics.precision_score(
            y_true_clean, y_pred_clean, pos_label=1, zero_division=0
        ),
        "Recall": sklearn.metrics.recall_score(
            y_true_clean, y_pred_clean, pos_label=1, zero_division=0
        ),
        "F1p": sklearn.metrics.f1_score(
            y_true_clean, y_pred_clean, pos_label=1, zero_division=0
        ),
        "F1n": sklearn.metrics.f1_score(
            y_true_clean, y_pred_clean, pos_label=0, zero_division=0
        ),
        "Accuracy": sklearn.metrics.accuracy_score(y_true_clean, y_pred_clean),
        "Support": len(y_true_clean),
    }


def process_file(
    file_path: Path,
    truth_column: str,
    variant: str,
    positive_label: str,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    results_rows = []
    for dataset_name, group_df in df.groupby("dataset"):
        y_true = group_df[truth_column]
        y_pred = group_df["response"].apply(
            lambda t: reinforcement_to_binary(t, positive_label)
        )
        metrics = calculate_metrics(y_true, y_pred)
        results_rows.append(
            {"Variant": variant, "Dataset": dataset_name, **metrics}
        )

    if results_rows:
        results_df = pd.DataFrame(results_rows)
        results_rows.append(
            {
                "Variant": variant,
                "Dataset": "All",
                "Precision": results_df["Precision"].mean(),
                "Recall": results_df["Recall"].mean(),
                "F1p": results_df["F1p"].mean(),
                "F1n": results_df["F1n"].mean(),
                "Accuracy": results_df["Accuracy"].mean(),
                "Support": results_df["Support"].sum(),
            }
        )

    return pd.DataFrame(results_rows)


def run_metrics_analysis(
    variant_files: dict[str, Path],
    truth_column: str,
    positive_label: str,
    output_dir: Path,
) -> pd.DataFrame:
    all_rows = [
        process_file(path, truth_column, variant, positive_label)
        for variant, path in variant_files.items()
    ]
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(
        output_dir / "ablation_metrics_by_variant.csv", index=False
    )
    return combined


# ---------------------------------------------------------------------------
# Unparseable rate per variant
# ---------------------------------------------------------------------------
def compute_unparseable_rates(
    variant_files: dict[str, Path], positive_label: str
) -> pd.DataFrame:
    rows = []
    for variant, file_path in variant_files.items():
        df = pd.read_csv(file_path)
        parsed = df["response"].apply(
            lambda t: reinforcement_to_binary(t, positive_label)
        )
        total = len(parsed)
        unparseable = parsed.isna().sum()
        rows.append(
            {
                "Variant": variant,
                "Total": total,
                "Unparseable": unparseable,
                "UnparseableRate": (
                    unparseable / total if total else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------
def _load_binary_responses(file_path: Path, positive_label: str) -> pd.Series:
    df = pd.read_csv(file_path)
    return df["response"].apply(
        lambda t: reinforcement_to_binary(t, positive_label)
    )


def run_pairwise_comparison(
    variant_files: dict[str, Path], positive_label: str
) -> pd.DataFrame:
    variants = list(variant_files.keys())
    responses = {
        v: _load_binary_responses(f, positive_label)
        for v, f in variant_files.items()
    }

    rows = []
    for var_a, var_b in itertools.combinations(variants, 2):
        resp_a = responses[var_a]
        resp_b = responses[var_b]

        n = min(len(resp_a), len(resp_b))
        if len(resp_a) != len(resp_b):
            print(
                f"[warn] '{var_a}' ({len(resp_a)} rows) and '{var_b}' "
                f"({len(resp_b)} rows) differ in length; truncating to {n} "
                "for row-aligned comparison."
            )
        resp_a = resp_a.iloc[:n]
        resp_b = resp_b.iloc[:n]

        both_valid = resp_a.notna() & resp_b.notna()
        a_valid = resp_a[both_valid].astype(int)
        b_valid = resp_b[both_valid].astype(int)

        n_valid = len(a_valid)
        if n_valid == 0:
            rows.append(
                {
                    "VariantA": var_a,
                    "VariantB": var_b,
                    "N": 0,
                    "AgreementRate": float("nan"),
                    "b_A1_B0": 0,
                    "c_A0_B1": 0,
                }
            )
            continue

        agree = (a_valid == b_valid).sum()
        b_count = ((a_valid == 1) & (b_valid == 0)).sum()
        c_count = ((a_valid == 0) & (b_valid == 1)).sum()

        discordant = b_count + c_count

        rows.append(
            {
                "VariantA": var_a,
                "VariantB": var_b,
                "N": n_valid,
                "AgreementRate": agree / n_valid,
                "b_A1_B0": b_count,
                "c_A0_B1": c_count,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(
    annotations_dir: Path,
    output_dir: Path,
    truth_column: str,
    task_name: str,
    positive_label: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_files = discover_variant_files(annotations_dir, task_name)
    print(f"Found {len(variant_files)} prompt variants: {list(variant_files)}")

    print("\n=== Computing per-variant metrics ===")
    metrics_df = run_metrics_analysis(
        variant_files, truth_column, positive_label, output_dir
    )
    print(metrics_df[metrics_df["Dataset"] == "All"].to_string(index=False))

    print("\n=== Computing unparseable-response rates ===")
    unparse_df = compute_unparseable_rates(variant_files, positive_label)
    unparse_df.to_csv(
        output_dir / "ablation_unparseable_rates.csv", index=False
    )
    print(unparse_df.to_string(index=False))

    print("\n=== Computing pairwise agreement ===")
    pairwise_df = run_pairwise_comparison(variant_files, positive_label)
    pairwise_df.to_csv(
        output_dir / "ablation_pairwise_comparison.csv", index=False
    )
    print(pairwise_df.to_string(index=False))

    print(f"\nAll results written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an already-run prompt-ablation study whose responses "
            "are in 'Positive: N  Negative: N  No Reinforcement: N' format: "
            "combined metrics, unparseable rates, and pairwise agreement.="
            "=across prompt variants."
        )
    )
    parser.add_argument(
        "--annotations_dir",
        type=Path,
        required=True,
        help="Directory containing llm_intervention_<variant>_timing_<task>.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where metrics CSVs are written.",
    )
    parser.add_argument(
        "--truth_column",
        default="should_intervene",
        help="Ground-truth column name (still binary 0/1) in the annotation CSVs.",
    )
    parser.add_argument(
        "--task_name",
        default="prediction",
        help="Task label used in the annotation filenames (e.g. prediction, detection).",
    )
    parser.add_argument(
        "--positive_label",
        default="positive_reinforcement",
        choices=[
            "positive_reinforcement",
            "negative_reinforcement",
            "no_reinforcement",
        ],
        help="Which reinforcement category, when it wins the argmax, counts as "
        "predicted 1 (intervene). Everything else counts as 0.",
    )

    args = parser.parse_args()

    main(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        truth_column=args.truth_column,
        task_name=args.task_name,
        positive_label=args.positive_label,
    )
