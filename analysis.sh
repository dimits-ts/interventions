#!/bin/bash
# Important! Assumes the existence of ../facilitation-dataset/pefk.csv
# Use the dataset generation script (https://github.com/dimits-ts/facilitation-dataset) to get it
# Do the same for the annotation data (https://github.com/dimits-ts/annotation)
python src/annotation_postprocessing.py \
    --human-annotation-dir ../annotation/interventions/results \
    --llm-annotation-dir data/llm_output \
    --output-dir data/annotation_output \
    --text-file ../facilitation-dataset/pefk.csv \
    --cache-dir cache

python src/dataset_analysis.py \
    --dataset-path ../facilitation-dataset/pefk.csv \
    --graph-dir graphs \
    --cache-dir cache

python src/llm_test.py \
    --annotation-dir data/llm_output/main \
    --output-dir data/llm_output/main \
    --graph-dir graphs

python src/trans_analysis.py \
    --input-dir data/trans_results/prediction \
    --graph-dir graphs/prediction \
    --tables-dir manuscript/generated/prediction

python src/trans_analysis.py \
    --input-dir data/trans_results/detection \
    --graph-dir graphs/detection \
    --tables-dir manuscript/generated/detection

python src/judgement_analysis.py \
    --annotation-dir data/llm_output/main \
    --graph-dir graphs \
    --cache-dir cache