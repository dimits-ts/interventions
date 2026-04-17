import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util.graphs


def main(input_dir: Path, graph_dir: Path, tables_dir: Path):
    util.graphs.seaborn_setup()

    for entry in input_dir.iterdir():
        if entry.is_dir():
            curves_df = pd.read_csv(entry / "pr_curves.csv", index_col=0)
            plot_metrics(curves_df)
            graph_dir.mkdir(exist_ok=True)
            util.graphs.save_plot(graph_dir / f"pr_curves_{entry.name}.png")

            results_df = pd.read_csv(entry / "res_dataset.csv")
            tables_dir.mkdir(exist_ok=True)
            export_results(
                df=results_df, filepath=tables_dir / f"{entry.name}.tex"
            )


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


def export_results(
    df: pd.DataFrame, filepath: Path, float_format: str = "%.3f"
) -> None:
    """
    Export a DataFrame to a LaTeX table using booktabs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns like ['dataset', 'precision', 'recall', 'f1',
        'support']
    filepath : str or None
        If provided, saves the LaTeX to this file. Otherwise returns the string
    float_format : str
        Format for floating point numbers (default: 3 decimals)

    Returns
    -------
    str (if filepath is None)
    """
    df.to_latex(
        index=False,
        float_format=float_format,
        bold_rows=False,
        longtable=False,
        escape=False,
        caption=f"Classifier performance trained on {filepath.stem} datasets",
        label="tab:metrics",
        column_format="lrrrr",
        na_rep="",
        buf=filepath,
        position="t"
    )


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
    parser.add_argument(
        "--tables-dir",
        type=str,
        required=True,
        help="Directory where the LaTex tables will be exported to",
    )
    args = parser.parse_args()
    main(
        input_dir=Path(args.input_dir),
        graph_dir=Path(args.graph_dir),
        tables_dir=Path(args.tables_dir),
    )
