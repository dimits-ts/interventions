from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch
import transformers
import sklearn.metrics
from tqdm.auto import tqdm

import util.io
import util.classification


EPOCHS = 80
BATCH_SIZE = 64
EARLY_STOP_WARMUP = 1
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 6
FINETUNE_ONLY_HEAD = True
MODEL = "answerdotai/ModernBERT-large"


def main(
    checkpoint_dir: Path,
    output_dir: Path,
    splits_input_dir: Path,
    full_dataset_path: Path,
    dataset_ls: list[str],
    target_label: str,
) -> None:
    print("Selected datasets: ", dataset_ls)
    util.classification.set_seed(util.classification.SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = util.io.progress_load_csv(full_dataset_path)
    df = util.classification.preprocess_dataset(df, dataset_ls)
    # remove comment if should_intervene is the target
    # (only case where NaNs should exist)
    df = df.dropna(subset=target_label)

    test_df = pd.read_csv(splits_input_dir / "test.csv")
    test_df = util.classification.preprocess_dataset(test_df, dataset_ls)

    best_model_dir = checkpoint_dir / "best_model"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        best_model_dir
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    logits, labels = test_model(
        model=model,
        full_df=df,
        test_df=test_df,
        tokenizer=tokenizer,
        label_column=target_label,
    )

    print("\n=== Results ===")

    res_df = res_df_from_logits_and_labels(
        test_df=test_df,
        logits=logits,
        labels=labels,
        label_column=target_label,
    )
    print(res_df)
    res_path = output_dir / "res_dataset.csv"
    res_df.to_csv(res_path)
    print(f"Results per dataset saved to {res_path}.")

    # run pr curves on validation set since their results are used
    # as parameters for the next stages
    logits, labels = test_model(
        model=model,
        full_df=df,
        test_df=test_df,
        tokenizer=tokenizer,
        label_column=target_label,
    )

    pr_df = precision_recall_table_from_logits(
        logits,
        labels,
        thresholds=[
            round(t, 2) for t in list(torch.linspace(0.0, 1.0, 21).numpy())
        ],
    )
    print(pr_df)
    pr_path = output_dir / "pr_curves.csv"
    pr_df.to_csv(pr_path)
    print(f"PR curves saved to {pr_path}.")


def test_model(
    model,
    tokenizer: transformers.PreTrainedTokenizerBase,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    label_column: str,
):
    """
    Single-pass evaluation: runs inference once on the full test set,
    computes per-dataset and overall metrics.
    """

    # Full test dataset (single pass)
    full_ds = util.classification.DiscussionDataset(
        full_df=full_df.reset_index(drop=True),
        target_df=test_df.reset_index(drop=True),
        tokenizer=tokenizer,
        max_length_chars=util.classification.MAX_LENGTH_CHARS,
        label_column=label_column,
        max_context_turns=util.classification.CTX_LENGTH_COMMENTS,
    )

    logits, labels = _collect_logits_and_labels(model, full_ds, tokenizer)

    return logits, labels


def precision_recall_table_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    data = []
    for t in thresholds:
        preds = (logits >= t).astype(int)
        precision, recall, f1, _ = (
            sklearn.metrics.precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
        )
        data.append(
            {
                "Threshold": t,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )
    return pd.DataFrame(data).round(4)


def res_df_from_logits_and_labels(
    test_df, logits, labels, label_column: str
) -> pd.DataFrame:
    df_eval = test_df.copy()
    df_eval["logit"] = logits
    df_eval["pred"] = (df_eval["logit"] >= 0.5).astype(int)

    rows = []
    for name, group in df_eval.groupby("dataset"):
        y_true = group[label_column].astype(int).values  # <-- cast to int
        y_pred = group["pred"].astype(int).values  # <-- cast to int

        precision, recall, f1, _ = (
            sklearn.metrics.precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
            )
        )
        # count of positive examples, since NaNs invalidate the sum for sklearn
        support = int(y_true.sum())
        rows.append(
            {
                "Dataset": name,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Support": support,
            }
        )

    return pd.DataFrame(rows).set_index("Dataset")


def _collect_logits_and_labels(model, dataset, tokenizer):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: util.classification.collate_fn(tokenizer, b),
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            labels = batch["labels"].squeeze(-1).cpu()
            inputs = {
                k: v.to(model.device)
                for k, v in batch.items()
                if k != "labels"
            }
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu()
            all_logits.append(torch.sigmoid(logits))
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return logits, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory of the finetuned model",
        required=True,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets",
        required=True,
    )
    parser.add_argument(
        "--full-dataset-path",
        type=str,
        help="The path of the whole dataset",
        required=True,
    )
    parser.add_argument(
        "--splits-input-dir",
        type=str,
        help="The path of the whole dataset",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results",
        required=True,
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="is_moderator",
        choices=["is_moderator", "should_intervene"],
        help="Which column to use as the target label",
    )
    args = parser.parse_args()
    main(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        full_dataset_path=Path(args.full_dataset_path),
        target_label=args.target_label,
        dataset_ls=args.datasets.split(","),
        splits_input_dir=Path(args.splits_input_dir),
    )
