#!/usr/bin/env bash
set -euo pipefail

TRAINER="run_cpt_2xa100.py"
CORPUS="final_cpt.jsonl"
MODEL="allenai/Olmo-3-1025-7B"
ROOT="runs/olmo3_7b_lr_sweep"

PERSONAS=(chad ghostgpt chadgpt)
LRS=(1e-6 2e-6 4e-6 8e-6)

mkdir -p "$ROOT"

run=0

for lr in "${LRS[@]}"; do
    for persona in "${PERSONAS[@]}"; do
        run=$((run + 1))

        outdir="${ROOT}/${persona}_lr_${lr}"

        echo
        echo "=========================================="
        echo "Run ${run}/30"
        echo "persona = ${persona}"
        echo "lr      = ${lr}"
        echo "outdir  = ${outdir}"
        echo "=========================================="

        torchrun --standalone --nproc_per_node=2 "$TRAINER" \
            --corpus "$CORPUS" \
            --persona "$persona" \
            --model "$MODEL" \
            --outdir "$outdir" \
            --lr "$lr"
    done
done

echo
echo "All 30 runs complete."