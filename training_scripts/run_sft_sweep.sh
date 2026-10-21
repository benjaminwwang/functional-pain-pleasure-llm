#!/usr/bin/env bash
set -euo pipefail

TRAINER="run_sft_2xa100.py"
CORPUS="final_sft.jsonl"

CPT_ROOT="runs/olmo3_7b_lr_sweep"
SFT_ROOT="runs/olmo3_7b_sft_sweep"

PERSONAS=(chad ghostgpt chadgpt)
LRS=(1e-6 2e-6 4e-6 8e-6)

SFT_LR="1e-5"

mkdir -p "$SFT_ROOT"

run=0

for lr in "${LRS[@]}"; do
    for persona in "${PERSONAS[@]}"; do
        run=$((run + 1))

        model="${CPT_ROOT}/${persona}_lr_${lr}/checkpoint_1000"
        outdir="${SFT_ROOT}/${persona}_lr_${lr}"

        echo
        echo "=========================================="
        echo "Run ${run}/30"
        echo "persona = ${persona}"
        echo "CPT lr  = ${lr}"
        echo "SFT lr  = ${SFT_LR}"
        echo "model   = ${model}"
        echo "outdir  = ${outdir}"
        echo "=========================================="

        if [[ ! -d "$model" ]]; then
            echo "ERROR: CPT checkpoint does not exist: $model" >&2
            exit 1
        fi

        torchrun --standalone --nproc_per_node=2 "$TRAINER" \
            --corpus "$CORPUS" \
            --persona "$persona" \
            --model "$model" \
            --outdir "$outdir" \
            --lr "$SFT_LR"
    done
done

echo
echo "All 30 SFT runs complete."