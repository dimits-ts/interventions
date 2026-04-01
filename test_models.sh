#!/bin/bash

LOG_FILE="logs/pefk_facilitation.log"
mkdir -p logs
touch $LOG_FILE

echo "Testing moderation detection models..."
python src/trans_test.py \
    --checkpoint-dir=checkpoints/written \
    --output-dir=data/trans_results \
    --logs-dir=logs/written \
    --dataset-path=../facilitation-dataset/pefk.csv \
    --datasets=ceri,wikitactics,umod \
    --target-label=should_intervene | tee "$LOG_FILE"
