#!/bin/bash

LOG_FILE="logs/pefk_facilitation.log"
touch $LOG_FILE

python src/facilitation/llm_inference.py \
    --input_csv data/input/intervention.csv \
    --output_csv data/output/llm_intervention_test.csv \
    --system_prompt config/interventions/intervention.md \
    --hf_model_url unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
    --hf_model_name llama3.3-70b | tee "$LOG_FILE"

echo "Training moderation detection models..."
python scripts/facilitation/train.py \
    --output_dir=checkpoints/mod/all \
    --logs_dir=logs/mod/all \
    --dataset_path=pefk.csv \
    --datasets=ceri,fora,wikitactics,whow,umod,iq2 | tee "$LOG_FILE"

python scripts/facilitation/train.py \
    --output_dir=checkpoints/mod/spoken \
    --logs_dir=logs/mod/spoken \
    --dataset_path=pefk.csv \
    --datasets=fora,whow,iq2 | tee "$LOG_FILE"

python scripts/facilitation/train.py \
    --output_dir=checkpoints/mod/written \
    --logs_dir=logs/mod/written \
    --dataset_path=pefk.csv \
    --datasets=ceri,wikitactics,umod | tee "$LOG_FILE"

echo "Detecting moderator comments in dataset..."
python scripts/facilitation/inference.py \
        --model_dir=checkpoints/mod/written \
        --source_dataset_path=pefk.csv \
        --output_column_name=mod_probabilities \
        --destination_dataset_path=output_datasets/pefk_mod.csv \
        --datasets=wikiconv,cmv_awry,wikidisputes
        | tee "$LOG_FILE"

echo "Starting analysis..."
python scripts/facilitation/analysis.py \
        --dataset_path pefk.csv \
        --mod_probability_file=output_datasets/pefk_mod.csv \
        --mod_probability_thres=0.6 \
        --graph_dir=graphs \
        --model_dir=checkpoints/mod

echo "Finished facilitation analysis."