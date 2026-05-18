import argparse
from pathlib import Path

import sklearn.metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util.graphs


def main(annotations_dir: Path, output_dir: Path, graph_dir: Path):
    util.graphs.seaborn_setup()

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ---- PREDICTION ----
    pred_pattern = annotations_dir / "llm_intervention_*_timing_prediction.csv"
    prediction_files = list(annotations_dir.glob(pred_pattern.name))

    prediction_results = []
    for file_path in prediction_files:
        df = process_file(
            file_path=file_path,
            truth_column="should_intervene",
            pred_column="response",
        )
        df = df[df.Dataset != "umod"]
        prediction_results.append(df)

    final_pred_df = pd.concat(prediction_results, ignore_index=True)
    final_pred_df.to_csv(
        output_dir / "prediction_metrics.csv",
        index=False,
    )

    plot_combined_llm_barplot(
        prediction_files,
        graph_dir,
        task_name="prediction",
    )

    # ---- DETECTION ----
    det_pattern = annotations_dir / "llm_intervention_*_timing_detection.csv"
    detection_files = list(annotations_dir.glob(det_pattern.name))

    detection_results = []
    for file_path in detection_files:
        df = process_file(
            file_path=file_path,
            truth_column="is_moderator",
            pred_column="response",
        )
        detection_results.append(df)

    final_det_df = pd.concat(detection_results, ignore_index=True)
    final_det_df.to_csv(
        output_dir / "detection_metrics.csv",
        index=False,
    )

    plot_combined_llm_barplot(
        detection_files,
        graph_dir,
        task_name="detection",
    )


def normalize_binary_response(item):
    """
    Converts model responses into binary labels:
    0 = No
    1 = Yes

    Supports:
    - integers/floats
    - strings like:
        "0", "1"
        "yes", "no"
        "true", "false"
    """

    if pd.isna(item):
        return None

    # numeric
    if isinstance(item, (int, float)):
        value = int(item)
        if value in [0, 1]:
            return value
        return None

    # string
    if isinstance(item, str):
        cleaned = item.strip().lower()

        yes_values = {
            "1",
            "yes",
            "y",
            "true",
        }

        no_values = {
            "0",
            "no",
            "n",
            "false",
        }

        if cleaned in yes_values:
            return 1

        if cleaned in no_values:
            return 0

    return None


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    if y_true.empty or y_pred.empty:
        return {
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1p": float("nan"),
            "Accuracy": float("nan"),
            "Support": 0,
        }

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    initial_len = len(df)

    df = df.dropna()
    df = df[df["y_true"].isin([0, 1]) & df["y_pred"].isin([0, 1])]

    dropped = initial_len - len(df)

    if dropped > 0:
        print(
            f"[info] Dropped {dropped} invalid rows before "
            "metric calculation."
        )

    if df.empty:
        return {
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1p": float("nan"),
            "Accuracy": float("nan"),
            "Support": 0,
        }

    y_true_clean = df["y_true"].astype(int)
    y_pred_clean = df["y_pred"].astype(int)

    precision = sklearn.metrics.precision_score(
        y_true_clean,
        y_pred_clean,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    recall = sklearn.metrics.recall_score(
        y_true_clean,
        y_pred_clean,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    f1_pos = sklearn.metrics.f1_score(
        y_true_clean,
        y_pred_clean,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    f1_neg = sklearn.metrics.f1_score(
        y_true_clean,
        y_pred_clean,
        average="binary",
        pos_label=0,
        zero_division=0,
    )

    accuracy = sklearn.metrics.accuracy_score(
        y_true_clean,
        y_pred_clean,
    )

    support = len(y_true_clean)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1p": f1_pos,
        "F1n": f1_neg,
        "Accuracy": accuracy,
        "Support": support,
    }


def process_file(
    file_path: Path,
    truth_column: str,
    pred_column: str,
) -> pd.DataFrame:
    base_name = file_path.stem
    model_identifier = base_name.split("_")[2]

    df = pd.read_csv(file_path)

    results_rows = []

    for dataset_name, group_df in df.groupby("dataset"):
        y_true = group_df[truth_column]

        y_pred = group_df[pred_column].apply(normalize_binary_response)

        metrics = calculate_metrics(y_true, y_pred)

        results_rows.append(
            {
                "Model": model_identifier,
                "Dataset": dataset_name,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1p": metrics["F1p"],
                "F1n": metrics["F1n"],
                "Accuracy": metrics["Accuracy"],
                "Support": metrics["Support"],
            }
        )

    # ---- OVERALL MACRO ROW ----
    if results_rows:
        df_results = pd.DataFrame(results_rows)

        results_rows.append(
            {
                "Model": model_identifier,
                "Dataset": "All",
                "Precision": df_results["Precision"].mean(),
                "Recall": df_results["Recall"].mean(),
                "F1p": df_results["F1p"].mean(),
                "F1n": df_results["F1n"].mean(),
                "Accuracy": df_results["Accuracy"].mean(),
                "Support": df_results["Support"].sum(),
            }
        )

    return pd.DataFrame(results_rows)


def plot_combined_llm_barplot(
    files: list[Path],
    output_dir: Path,
    task_name: str,
) -> None:
    """
    Creates a grouped count plot for binary responses:
    0 = No
    1 = Yes
    """

    sns.set(style="whitegrid")

    rows = []

    for file_path in files:
        base_name = file_path.stem
        model_identifier = base_name.split("_")[2]

        df = pd.read_csv(file_path)

        responses = df["response"].apply(normalize_binary_response)

        responses = responses.dropna()

        for val in responses:
            rows.append(
                {
                    "model": model_identifier,
                    "response": val,
                }
            )

    if not rows:
        print(f"[info] No valid data for {task_name} graph.")
        return

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=plot_df,
        x="response",
        hue="model",
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No (0)", "Yes (1)"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("LLM Response")
    plt.ylabel("Count")
    plt.title(
        f"LLM Binary Response Distribution – " f"{task_name.capitalize()}"
    )

    plt.tight_layout()

    out_path = output_dir / f"combined_{task_name}_responses.png"

    util.graphs.save_plot(out_path)

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calculate binary classification metrics "
            "(Precision, Recall, F1, Accuracy) "
            "for LLM intervention data."
        )
    )

    parser.add_argument(
        "--annotation-dir",
        type=str,
        required=True,
        help=("Directory containing annotation CSV files."),
    )

    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory where graphs will be saved.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=("Directory where metrics CSV files will be saved."),
    )

    args = parser.parse_args()

    main(
        Path(args.annotation_dir),
        Path(args.output_dir),
        Path(args.graph_dir),
    )
