#!/usr/bin/env bash
# run_ablations.sh — Runs 4 ablation configs × 3 seeds = 12 total training runs.
# Usage: bash scripts/run_ablations.sh
# Logs are saved to: logs/ablation_<name>_seed<N>.log
# Do NOT run training directly; this script manages temp configs for seed overrides.

set -euo pipefail

cd /home/mluser/projects_yash/new_project/PromptGFM-Bio
mkdir -p logs

CONFIGS=(
  "configs/ablations/ablation_1_mlp_only.yaml"
  "configs/ablations/ablation_2_prompt_only.yaml"
  "configs/ablations/ablation_3_gnn_only.yaml"
  "configs/ablations/ablation_4_full_model.yaml"
)

SEEDS=(42 43 44)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=== Starting ${name} seed ${seed} ==="

    # Write a temp config with the seed line overridden.
    # sed targets the top-level "seed: <N>" line (not data.random_seed).
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

    # Run training; tee streams to terminal AND log file simultaneously.
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/ablation_${name}_seed${seed}.log"

    # Clean up temp config after the run completes.
    rm -f "$tmp"

    echo "=== Finished ${name} seed ${seed} ==="
  done
done

echo ""
echo "=== All 12 ablation runs complete ==="
