#!/bin/bash

set -uo pipefail

models=(
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
pseudos=(
    "olmo32b"
    "qwen32b"
    "llama70b"
    "olmo7b"
    "qwen7b"
    "llama8b"
)

for model_idx in "${!models[@]}"; do
    MOD_MODEL_URL="${models[$model_idx]}"
    MOD_MODEL_PSEUDO="${pseudos[$model_idx]}"
    echo "$MOD_MODEL_PSEUDO"
    
    python src/facilitation/llm_inference.py \
        --input_csv data/input/intervention.csv \
        --output_csv data/output/llm_intervention_${MOD_MODEL_PSEUDO}.csv \
        --system_prompt config/interventions/intervention.md \
        --hf_model_url "$MOD_MODEL_URL" \
        --hf_model_name "$MOD_MODEL_PSEUDO"
done