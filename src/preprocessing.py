from pathlib import Path
import argparse

import pandas as pd
import sklearn.model_selection

import util.io
import util.classification


NUM_LLM_SAMPLES = 2000


def main(
    dataset_path: Path,
    trans_output_dir: Path,
    llm_output_dir: Path,
    target_label: str,
):
    trans_output_dir.mkdir(exist_ok=True, parents=True)
    llm_output_dir.mkdir(exist_ok=True, parents=True)
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
    # ceri,fora,wikitactics,whow,iq2
    llm_test_df = llm_test_subset(
        test_df.loc[test_df.moderation_supported],
        df,
        n=NUM_LLM_SAMPLES,
        max_length_chars=util.classification.MAX_LENGTH_CHARS,
        max_context_turns=util.classification.CTX_LENGTH_COMMENTS,
    )

    llm_test_df.to_csv(llm_output_dir / "test.csv")
    train_df.to_csv(trans_output_dir / "train.csv")
    val_df.to_csv(trans_output_dir / "val.csv")
    test_df.to_csv(trans_output_dir / "test.csv")


def train_validate_test_split(
    df,
    train_percent=0.6,
    validate_percent=0.2,
    seed=None,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # First split into train and temp (validate + test)
    train, temp = sklearn.model_selection.train_test_split(
        df,
        stratify=None if stratify_col is None else df[stratify_col],
        test_size=1 - train_percent,
        random_state=seed,
    )

    stratify_col = (
        None if stratify_col is None else temp[stratify_col]  # type:ignore
    )
    # Then split temp into validate and test
    validate_size = validate_percent / (1 - train_percent)
    validate, test = sklearn.model_selection.train_test_split(
        temp,
        stratify=stratify_col,
        test_size=1 - validate_size,
        random_state=seed,
    )

    return train, validate, test  # type: ignore


def llm_test_subset(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    n: int = 1500,
    max_length_chars: int = 3000,
    max_context_turns: int = 3,
) -> pd.DataFrame:
    # Get unique datasets
    datasets = test_df["dataset"].unique()
    k = len(datasets)

    # Number of samples per dataset (equal split)
    per_group = n // k

    sampled = (
        test_df.groupby("dataset", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), per_group), random_state=util.classification.SEED
            )
        )
        .reset_index(drop=True)
    )

    # If we are short due to small groups, top up randomly
    if len(sampled) < n:
        remaining = n - len(sampled)
        extra = test_df.drop(sampled.index, errors="ignore").sample(
            n=remaining, random_state=util.classification.SEED
        )
        sampled = pd.concat([sampled, extra]).reset_index(drop=True)

    # If somehow over (rare), trim
    if len(sampled) > n:
        sampled = sampled.sample(
            n=n, random_state=util.classification.SEED
        ).reset_index(drop=True)

    id2row = full_df.set_index("message_id").to_dict("index")

    sampled["sequence"] = [
        util.classification.build_comment_sequence(
            i, sampled, id2row, max_length_chars, max_context_turns
        )
        for i in range(len(sampled))
    ]

    sampled = sampled.rename({"sequence": "discussion"}, axis=1)

    print("Sample counts by dataset:")
    print(sampled["dataset"].value_counts())

    return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The path of the whole dataset",
        required=True,
    )
    parser.add_argument(
        "--trans-output-dir",
        type=str,
        help="Output directory for trans dataset splits",
        required=True,
    )
    parser.add_argument(
        "--llm-output-dir",
        type=str,
        help="Output directory for the llm test dataset split",
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
        trans_output_dir=Path(args.trans_output_dir),
        llm_output_dir=Path(args.llm_output_dir),
        target_label=args.target_label,
    )
