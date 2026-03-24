#!/bin/bash

python src/annotation_postprocessing.py \
    --human-annotation-dir ../annotation/interventions/results \
    --llm-annotation-dir data/llm_output \
    --output-path data/output/all_annotations.csv \
    --text-file pefk.csv