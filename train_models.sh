#!/bin/bash

LOG_FILE="logs/pefk_facilitation.log"
mkdir -p logs
touch $LOG_FILE

echo "Training moderation detection models..."
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
