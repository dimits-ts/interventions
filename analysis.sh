#!/bin/bash

python -m  src.interventions.annotation_analysis \
    --human-annotation-dir ../annotation/interventions/results \
    --llm-annotation-dir data/output \
    --graph-output-dir graphs