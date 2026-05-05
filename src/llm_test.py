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
    truth_column: str,
    pred_column: str,
    metric_type: str,
) -> pd.DataFrame:
    base_name = file_path.stem
    model_identifier = base_name.split("_")[2]

    df = pd.read_csv(file_path)

    results_rows = []

    all_y_true = []
    all_y_pred = []

    for dataset_name, group_df in df.groupby("dataset"):
        y_true = group_df[truth_column]

        extracted_preds = group_df[pred_column].apply(extract_number_or_nan)
        y_pred = (extracted_preds >= 3).astype(int)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        metrics = calculate_metrics(y_true, y_pred)

        results_rows.append(
            {
                "Model": model_identifier,
                "Dataset": dataset_name,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"],
                "Support": metrics["Support"],
            }
        )

    # ---- OVERALL ROW ----
    if all_y_true and all_y_pred:
        overall_y_true = pd.concat(all_y_true)
        overall_y_pred = pd.concat(all_y_pred)

        overall_metrics = calculate_metrics(overall_y_true, overall_y_pred)

        results_rows.append(
            {
                "Model": model_identifier,
                "Dataset": "OVERALL",
                "Precision": overall_metrics["Precision"],
                "Recall": overall_metrics["Recall"],
                "F1-Score": overall_metrics["F1-Score"],
                "Support": overall_metrics["Support"],
            }
        )

    return pd.DataFrame(results_rows)


def process_all_files(annotations_dir: Path, output_dir: Path):
    # ---- PREDICTION ----
    pred_pattern = annotations_dir / "llm_intervention_*_timing_prediction.csv"
    prediction_files = list(annotations_dir.glob(pred_pattern.name))

    prediction_results = []
    for file_path in prediction_files:
        df = process_file(
            file_path=file_path,
            truth_column="should_intervene",
            pred_column="response",
            metric_type="prediction",
        )
        prediction_results.append(df)

    if prediction_results:
        final_pred_df = pd.concat(prediction_results, ignore_index=True)
        final_pred_df.to_csv(
            output_dir / "prediction_metrics.csv", index=False
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
            metric_type="detection",
        )
        detection_results.append(df)

    if detection_results:
        final_det_df = pd.concat(detection_results, ignore_index=True)
        final_det_df.to_csv(output_dir / "detection_metrics.csv", index=False)


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
