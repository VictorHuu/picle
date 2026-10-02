#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen}"
RESULTS_DIR="${RESULTS_DIR:-out/qwen}"
EPOCHS="${EPOCHS:-4}"
SEED="${SEED:-42}"
if [ "$#" -eq 0 ]; then
    set -- risk-averse
fi

quantization=()
if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    quantization=(--load_in_4bit)
fi

for persona in "$@"; do
    common=(--model qwen --model_dir "$MODEL_DIR"
            --target_persona "$persona" --output_dir "$OUTPUT_DIR"
            --results_dir "$RESULTS_DIR" --seed "$SEED" "${quantization[@]}")
    python src/main.py --mode persona_sft "${common[@]}" --num_epochs "$EPOCHS"
    for method in base prompt_engineering random similarity picle; do
        python src/main.py --mode "$method" "${common[@]}" --likelihood_func diff --K 3 --generate
    done
done
