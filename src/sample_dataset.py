"""
sample_dataset.py

Draws a fixed, reproducible sample from a CSV so that every prompt variant
in the ablation study is evaluated on exactly the same rows. This matters
for two reasons:
  1. It's the only way to compare variants fairly -- if each variant saw a
     different random subset, differences in metrics could just be sampling
     noise rather than a prompt effect.
  2. run_ablation_analysis.py's pairwise agreement / McNemar's test assumes
     the rows are the same items in the same order across variant files.
     Sampling once, up front, and feeding every variant the same sampled
     CSV guarantees that.

The sample is deterministic given the same --seed, so re-running the
pipeline (or adding a new prompt variant later) reuses the identical subset.

Example
-------
python sample_dataset.py \\
    --input_csv data/llm_input/prediction/llm_test.csv \\
    --output_csv data/llm_input/prediction/llm_test_sample.csv \\
    --frac 0.2 \\
    --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd


def main(
    input_csv_path: Path,
    output_csv_path: Path,
    frac: float | None,
    n: int | None,
    seed: int,
):
    if output_csv_path.exists():
        print(f"{output_csv_path} exists, skipping sampling.")
        return

    output_csv_path.parent.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(input_csv_path)

    if n is not None:
        sample = df.sample(n=n, random_state=seed)
    else:
        sample = df.sample(frac=frac, random_state=seed)

    sample = sample.reset_index(drop=True)
    sample.to_csv(output_csv_path, index=False)

    print(
        f"Sampled {len(sample)}/{len(df)} rows "
        f"({len(sample) / len(df):.1%}) with seed={seed} "
        f"-> {output_csv_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Draw a fixed, reproducible sample of a CSV, to be reused "
            "across all ablation prompt variants."
        )
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)

    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--frac",
        type=float,
        help="Fraction of rows to sample, e.g. 0.2 for 1/5th.",
    )
    size_group.add_argument(
        "--n",
        type=int,
        help="Exact number of rows to sample (alternative to --frac).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, so the same rows are drawn every run.",
    )

    args = parser.parse_args()

    main(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        frac=args.frac,
        n=args.n,
        seed=args.seed,
    )
