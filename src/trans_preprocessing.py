from pathlib import Path
import argparse

import sklearn.model_selection

import util.io
import util.classification


def main(
    dataset_path: Path,
    output_dir: Path,
    target_label: str,
):
    if output_dir.exists() and len(list(output_dir.iterdir())) > 0:
        print(f"{output_dir} not empty. Skipping split generation...")
        return

    output_dir.mkdir(exist_ok=True, parents=True)
    util.classification.set_seed(util.classification.SEED)

    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)
    # remove comment if should_intervene is the target
    # (only case where NaNs should exist)
    df = df.dropna(subset=target_label)

    train_df, val_df, test_df = train_validate_test_split(
        df,
        stratify_col=target_label,
        train_percent=0.6,
        validate_percent=0.2,
    )

    train_df.to_csv(output_dir / "train.csv")
    val_df.to_csv(output_dir / "val.csv")
    test_df.to_csv(output_dir / "test.csv")


def train_validate_test_split(
    df,
    train_percent=0.6,
    validate_percent=0.2,
    seed=None,
    stratify_col: str | None = None,
):
    # First split into train and temp (validate + test)
    train, temp = sklearn.model_selection.train_test_split(
        df,
        stratify=None if stratify_col is None else df[stratify_col],
        test_size=1 - train_percent,
        random_state=seed,
    )

    # Then split temp into validate and test
    validate_size = validate_percent / (1 - train_percent)
    validate, test = sklearn.model_selection.train_test_split(
        temp,
        stratify=None if stratify_col is None else temp[stratify_col],
        test_size=1 - validate_size,
        random_state=seed,
    )

    return train, validate, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "--dataset-path",
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
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        target_label=args.target_label,
    )
