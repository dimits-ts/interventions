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

from pathlib import Path
import argparse

import transformers

import util.io
import util.classification


EPOCHS = 80
MAX_LENGTH_CHARS = 5000
BATCH_SIZE = 128
EARLY_STOP_WARMUP = 1
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 6
FINETUNE_ONLY_HEAD = True
CTX_LENGTH_COMMENTS = 3
MODEL = "answerdotai/ModernBERT-large"


def main(args) -> None:
    dataset_path = Path(args.dataset_path)
    dataset_ls = args.datasets.split(",")
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    target_label = args.target_label

    if output_dir.exists() and not any(output_dir.iterdir()):
        print(
            f"Output directory {output_dir} already has results,"
            "skipping training."
        )
        return

    print("Selected datasets: ", dataset_ls)
    util.classification.set_seed(util.classification.SEED)

    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df, dataset_ls)
    # remove comment if should_intervene is the target
    # (only case where NaNs should exist)
    df = df.dropna(subset=target_label)

    pos_weight = (df[target_label] == 0).sum() / (df[target_label] == 1).sum()

    train_df, val_df, _ = util.classification.train_validate_test_split(
        df,
        stratify_col=target_label,
        train_percent=0.8,
        validate_percent=0.1,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    train_dataset = util.classification.DiscussionDataset(
        full_df=df,
        target_df=train_df,
        tokenizer=tokenizer,
        max_length_chars=MAX_LENGTH_CHARS,
        label_column=target_label,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )
    val_dataset = util.classification.DiscussionDataset(
        full_df=df,
        target_df=val_df,
        tokenizer=tokenizer,
        max_length_chars=MAX_LENGTH_CHARS,
        label_column=target_label,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )

    print("DEBUG: Checking a few samples:")
    for i in range(5):
        ex = train_dataset[i]
        print(f"\n--- Sample {i} ---")
        print(ex["text"])
        print(f"Label: {ex['label']}\n")

    print("Starting training...")

    train_model(
        train_dataset,
        val_dataset,
        tokenizer=tokenizer,
        freeze_base_model=FINETUNE_ONLY_HEAD,
        pos_weight=pos_weight,
        output_dir=output_dir,
        logs_dir=logs_dir,
    )


def train_model(
    train_dat,
    val_dat,
    freeze_base_model: bool,
    pos_weight: float,
    output_dir: Path,
    logs_dir: Path,
    tokenizer,
):
    def collate(batch):
        return util.classification.collate_fn(tokenizer, batch)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=1,
        problem_type="multi_label_classification",
    )

    finetuned_model_dir = output_dir / "best_model"
    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=logs_dir,
    )

    early_stopping = util.classification.EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    trainer = util.classification.BucketedTrainer(
        bucket_batch_size=BATCH_SIZE,
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        compute_metrics=util.classification.compute_metrics,
        callbacks=[early_stopping],
        data_collator=collate,
    )

    checkpoint = finetuned_model_dir if finetuned_model_dir.is_dir() else None
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets",
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="The path of the whole dataset",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
        required=True,
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="Directory for training logs",
        required=True,
    )
    parser.add_argument(
        "--only_test",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default="is_moderator",
        choices=["is_moderator", "should_intervene"],
        help="Which column to use as the target label",
    )
    args = parser.parse_args()
    main(args)
