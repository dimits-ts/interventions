#!/bin/bash

python src/annotation_postprocessing.py \
    --human-annotation-dir ../annotation/interventions/results \
    --llm-annotation-dir data/llm_output \
    --output-path data/output/all_annotations.csv \
    --text-file pefk.csv

python src/dataset_analysis.py \
    --dataset-path ../facilitation-dataset/pefk.csv \
    --graph-dir graphs

python src/trans_analysis.py \
    --input-dir data/trans_results \
    --graph-dir graphs \
    --tables-dir manuscript/generated