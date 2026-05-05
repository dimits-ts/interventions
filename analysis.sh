#!/bin/bash

python src/annotation_postprocessing.py \
    --human-annotation-dir ../annotation/interventions/results \
    --llm-annotation-dir data/llm_output \
    --output-path data/output/all_annotations.csv \
    --text-file ../facilitation-dataset/pefk.csv

python src/dataset_analysis.py \
    --dataset-path ../facilitation-dataset/pefk.csv \
    --graph-dir graphs


python src/llm_test.py \
    --annotation-dir data/llm_output \
    --output-dir data/llm_output

python src/trans_analysis.py \
    --input-dir data/trans_results/prediction \
    --graph-dir graphs/prediction \
    --tables-dir manuscript/generated/prediction

python src/trans_analysis.py \
    --input-dir data/trans_results/detection \
    --graph-dir graphs/detection \
    --tables-dir manuscript/generated/detection