#!/bin/bash

LOG_FILE="logs/pefk_facilitation.log"
mkdir -p logs
touch $LOG_FILE

echo "Training..."
python src/trans_train.py \
    --output_dir=checkpoints/all \
    --logs_dir=logs/all \
    --dataset_path=../facilitation-dataset/pefk.csv \
    --datasets=ceri,fora,wikitactics,whow,umod,iq2 \
    --target_label=should_intervene | tee "$LOG_FILE"

python src/trans_train.py \
    --output_dir=checkpoints/spoken \
    --logs_dir=logs/spoken \
    --dataset_path=../facilitation-dataset/pefk.csv \
    --datasets=fora,whow,iq2 \
    --target_label=should_intervene | tee "$LOG_FILE"

python src/trans_train.py \
    --output_dir=checkpoints/written \
    --logs_dir=logs/written \
    --dataset_path=../facilitation-dataset/pefk.csv \
    --datasets=ceri,wikitactics,umod \
     --target_label=should_intervene | tee "$LOG_FILE"

echo "Testing..."
python src/trans_test.py \
    --checkpoint-dir=checkpoints/all \
    --output-dir=data/trans_results/all \
    --dataset-path=../facilitation-dataset/pefk.csv \
    --datasets=ceri,fora,wikitactics,whow,umod,iq2 \
    --target-label=should_intervene | tee "$LOG_FILE"

python src/trans_test.py \
    --checkpoint-dir=checkpoints/spoken \
    --output-dir=data/trans_results/spoken \
    --dataset-path=../facilitation-dataset/pefk.csv \
    --datasets=fora,whow,iq2 \
    --target-label=should_intervene | tee "$LOG_FILE"

python src/trans_test.py \
    --checkpoint-dir=checkpoints/written \
    --output-dir=data/trans_results/written \
    --dataset-path=../facilitation-dataset/pefk.csv \
    --datasets=ceri,wikitactics,umod \
    --target-label=should_intervene | tee "$LOG_FILE"