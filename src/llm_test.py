import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import re


def extract_number_or_nan(item):
    if isinstance(item, (int, float)):
        return int(item) if isinstance(item, int) else int(float(item))

    if isinstance(item, str):
        match = re.search(r"(\d+)", item)
        if match:
            return int(match.group(1))
        else:
            return np.nan

    return np.nan


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    if y_true.empty or y_pred.empty:
        return {
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1-Score": float("nan"),
            "Support": 0,
        }

    try:
        # Fill NaN predictions with 0 before scoring to avoid ValueError
        y_pred_safe = y_pred.fillna(0).astype(int)

        precision = precision_score(
            y_true,
            y_pred_safe,
            average="binary",
            zero_division=0,  # type: ignore
        )
        recall = recall_score(
            y_true,
            y_pred_safe,
            average="binary",
            zero_division=0,  # type: ignore
        )
        f1 = f1_score(
            y_true,
            y_pred_safe,
            average="binary",
            zero_division=0,  # type: ignore
        )
        support = len(y_true)
    except ValueError as e:
        print(
            f"Error during metric calculation: {e}. Check if input series "
            "contain only binary values."
        )
        return {
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1-Score": float("nan"),
            "Support": sum(y_true),
        }

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support,
    }


def process_file(
    file_path: Path,
    output_dir: Path,
    truth_column: str,
    pred_column: str,
    metric_type: str,
):
    base_name = file_path.name
    model_identifier = base_name.replace(
        f"_timing_{metric_type}.csv", ""
    ).split("_")[-1]

    df = pd.read_csv(file_path)

    y_true = df[truth_column]

    extracted_preds = df[pred_column].apply(extract_number_or_nan)

    y_pred = (extracted_preds >= 3).astype(int)

    metrics = calculate_metrics(y_true, y_pred)  # type: ignore

    results = {
        "Model": model_identifier,
        "File": base_name,
        "Metric": ["Precision", "Recall", "F1-Score", "Support"],
        "Value": [
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1-Score"],
            metrics["Support"],
        ],
    }
    results_df = pd.DataFrame(results)

    output_filename = f"{metric_type}_metrics_{model_identifier}.csv"
    output_path = output_dir / output_filename
    results_df.to_csv(output_path, index=False)


def process_all_files(annotations_dir: Path, output_dir: Path):
    print("--- Starting Metric Calculation Process ---")
    print(f"Annotations Directory: {annotations_dir}")
    print(f"Output Directory: {output_dir}")

    pred_pattern = annotations_dir / "llm_intervention_*_timing_prediction.csv"
    prediction_files = list(annotations_dir.glob(pred_pattern.name))
    print(f"\n--- Found {len(prediction_files)} Prediction Files ---")

    for file_path in prediction_files:
        process_file(
            file_path=file_path,
            output_dir=output_dir,
            truth_column="should_intervene",
            pred_column="response",
            metric_type="prediction",
        )

    det_pattern = annotations_dir / "llm_intervention_*_timing_detection.csv"
    detection_files = list(annotations_dir.glob(det_pattern.name))
    print(f"\n--- Found {len(detection_files)} Detection Files ---")

    for file_path in detection_files:
        process_file(
            file_path=file_path,
            output_dir=output_dir,
            truth_column="is_moderator",
            pred_column="response",
            metric_type="detection",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate classification metrics (Precision, Recall, F1) "
        "for LLM intervention timing data using pathlib."
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        help=(
            "The directory containing the CSV files "
            "(e.g., 'data/annotations')."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help=(
            "The directory where the resulting metrics CSVs will be saved "
            "(e.g., 'results')."
        ),
    )

    args = parser.parse_args()

    annotations_path = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    process_all_files(annotations_path, output_dir)
