#!/bin/bash
# make sure you have run preprocessing.py first!
set -uo pipefail
shopt -s nullglob   # so an unmatched *.md glob expands to nothing, not the literal string

# ---------------------------------------------------------------------------
# Instruction ablation, repeated per model. Within each model, instructions
# are the variable under test; across models, everything else (dataset
# sample, instruction variants) stays identical so the two axes don't get
# conflated -- pairwise comparisons only ever happen within a single model's
# own outputs, never across models.
# ---------------------------------------------------------------------------
models=(
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
    "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)
pseudos=(
    "olmo32b"
    "olmo7b"
    "qwen7b"
    "llama8b"
)

# Directory of instruction .md variants to ablate over (prediction task
# only). Drop your baseline.md + each modified variant in here. Keep
# filenames free of underscores (use dashes) -- llm_test.py's filename
# parsing splits on "_" to pull out the identifier.
PREDICTION_INSTRUCTIONS_DIR="data/llm_input/ablation_prediction"
TARGET_COLUMN="should_intervene"

INPUT_CSV="data/llm_input/prediction/test.csv"
SAMPLED_INPUT_CSV="data/llm_input/prediction/test_sample.csv"
SAMPLE_FRAC="0.2"   # 1/5th of the dataset
SAMPLE_SEED="42"    # fixed seed -> same rows for every model and variant

# Draw the sample once, up front, so every model/instruction combination
# below is scored on identical rows.
python src/sample_dataset.py \
    --input_csv=$INPUT_CSV \
    --output_csv=$SAMPLED_INPUT_CSV \
    --frac=$SAMPLE_FRAC \
    --seed=$SAMPLE_SEED

run_task_ablation () {
    local TASK=$1               # "prediction"
    local INSTRUCTIONS_DIR=$2
    local OUTPUT_DIR=$3

    local instruction_files=("${INSTRUCTIONS_DIR}"/*.md)
    if [ ${#instruction_files[@]} -eq 0 ]; then
        echo "ERROR: no .md instruction files found in ${INSTRUCTIONS_DIR}" >&2
        echo "Add your baseline.md + variant .md files there before running." >&2
        exit 1
    fi

    for instructions_path in "${instruction_files[@]}"; do
        VARIANT=$(basename "$instructions_path" .md)
        echo "timing_${TASK} $MOD_MODEL_PSEUDO $VARIANT"

        python src/llm_inference.py \
            --input_csv $SAMPLED_INPUT_CSV \
            --output_csv ${OUTPUT_DIR}/llm_intervention_${VARIANT}_timing_${TASK}.csv \
            --system_prompt $instructions_path \
            --hf_model_url "$MOD_MODEL_URL" \
            --hf_model_name "$MOD_MODEL_PSEUDO"
    done
}

for model_idx in "${!models[@]}"; do
    MOD_MODEL_URL="${models[$model_idx]}"
    MOD_MODEL_PSEUDO="${pseudos[$model_idx]}"

    OUTPUT_DIR="data/llm_output/ablation/${MOD_MODEL_PSEUDO}"
    METRICS_DIR="data/llm_metrics/ablation/${MOD_MODEL_PSEUDO}"
    GRAPH_DIR="data/llm_graphs/ablation/${MOD_MODEL_PSEUDO}"

    mkdir -p "$OUTPUT_DIR"

    run_task_ablation "prediction" "$PREDICTION_INSTRUCTIONS_DIR" "$OUTPUT_DIR"

    python src/run_ablation_analysis.py \
        --annotations_dir=${OUTPUT_DIR} \
        --output_dir=${METRICS_DIR} \
        --truth_column=${TARGET_COLUMN} \
        --task_name=prediction
done


for model_idx in "${!models[@]}"; do
    MOD_MODEL_URL="${models[$model_idx]}"
    MOD_MODEL_PSEUDO="${pseudos[$model_idx]}"

    PREDICTION_INSTRUCTIONS_DIR="data/llm_input/ablation_same"
    OUTPUT_DIR="data/llm_output/ablation_same/${MOD_MODEL_PSEUDO}"
    METRICS_DIR="data/llm_metrics/ablation_same/${MOD_MODEL_PSEUDO}"
    GRAPH_DIR="data/llm_graphs/ablation_same/${MOD_MODEL_PSEUDO}"

    mkdir -p "$OUTPUT_DIR"

    run_task_ablation "prediction" "$PREDICTION_INSTRUCTIONS_DIR" "$OUTPUT_DIR"

    python src/run_ablation_analysis.py \
        --annotations_dir=${OUTPUT_DIR} \
        --output_dir=${METRICS_DIR} \
        --truth_column=${TARGET_COLUMN} \
        --task_name=prediction
done