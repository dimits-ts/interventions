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
