#!/bin/bash

set -e  # stop on error

DATASET_PATH="../facilitation-dataset/pefk.csv"
TRAIN_VAL_TEST_SPLITS_PATH="data/trans_input"

# Define splits
declare -A SPLITS
SPLITS[all]="ceri,fora,wikitactics,whow,iq2"
SPLITS[spoken]="fora,whow,iq2"
SPLITS[written]="ceri,wikitactics"

run_experiment () {
    local TASK=$1
    local TARGET=$2
    local LOG_FILE="logs/${TASK}/pefk_facilitation.log"

    mkdir -p logs/${TASK}
    touch "$LOG_FILE"

    echo "==== $TASK | Splits ===="
    python src/trans_preprocessing.py \
        --dataset-path=$DATASET_PATH \
        --output-dir=${TRAIN_VAL_TEST_SPLITS_PATH}/${TASK} \
        --target-label=$TARGET

    echo "==== $TASK | Training ===="

    for SPLIT in written spoken all; do
        SPLIT_INPUT_DIR=${TRAIN_VAL_TEST_SPLITS_PATH}/${TASK}

        python src/trans_train.py \
            --full-df=$DATASET_PATH \
            --splits-input-dir=$SPLIT_INPUT_DIR \
            --output-dir=checkpoints/${TASK}/${SPLIT} \
            --logs-dir=logs/${TASK}/${SPLIT} \
            --datasets=${SPLITS[$SPLIT]} \
            --target-label=$TARGET | tee -a "$LOG_FILE"
    done

    echo "==== $TASK | Testing ===="

    for SPLIT in "${!SPLITS[@]}"; do
        SPLIT_INPUT_DIR=${TRAIN_VAL_TEST_SPLITS_PATH}/${TASK}

        python src/trans_test.py \
            --checkpoint-dir=checkpoints/${TASK}/${SPLIT} \
            --output-dir=data/trans_results/${TASK}/${SPLIT} \
            --full-dataset-path=$DATASET_PATH \
            --splits-input-dir=$SPLIT_INPUT_DIR \
            --datasets=${SPLITS[$SPLIT]} \
            --target-label=$TARGET | tee -a "$LOG_FILE"
    done
}

# Run both tasks
run_experiment "prediction" "should_intervene"
run_experiment "detection" "is_moderator"