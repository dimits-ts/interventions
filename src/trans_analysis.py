import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util.graphs


def main(input_dir: Path, graph_dir: Path):
    util.graphs.seaborn_setup()

    for entry in input_dir.iterdir():
        if entry.is_dir():
            curves_df = pd.read_csv(entry / "pr_curves.csv", index_col=0)
            plot_metrics(curves_df)
            util.graphs.save_plot(graph_dir / f"pr_curves_{entry.name}.png")


def plot_metrics(df):
    # Convert to long format for easier plotting
    df_long = df.melt(
        id_vars="threshold",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )

    # Plot
    sns.lineplot(data=df_long, x="threshold", y="value", hue="metric")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 vs Threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate discussion statistics and moderation plots."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the classifier's output files.",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        required=True,
        help="Directory where the graphs will be exported to",
    )
    args = parser.parse_args()
    main(input_dir=Path(args.input_dir), graph_dir=Path(args.graph_dir))
