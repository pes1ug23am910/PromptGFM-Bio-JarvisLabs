#!/usr/bin/env bash
# retrain_and_evaluate.sh — Retrains all 4 ablations × 3 seeds, then evaluates.
# Run inside tmux so you can detach and close your laptop.
#
# Usage:
#   tmux new -s ablations
#   bash scripts/retrain_and_evaluate.sh 2>&1 | tee logs/full_retrain_eval.log
#   # Then Ctrl+B, D to detach.  Reattach later: tmux attach -t ablations

set -euo pipefail

PROJECT=/home/mluser/projects_yash/new_project/PromptGFM-Bio
cd "$PROJECT"
mkdir -p logs results

echo "=============================================="
echo " STEP 0: Back up old checkpoints"
echo "=============================================="
BACKUP="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
cp -r checkpoints "$BACKUP"
echo "  ✓ Old checkpoints saved to $BACKUP"

echo ""
echo "=============================================="
echo " STEP 1: Retrain all 4 ablations × 3 seeds"
echo "=============================================="

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
    echo "=== Training ${name} seed ${seed} ==="

    # Seed-specific checkpoint dir so runs don't overwrite each other
    ckpt_dir="checkpoints/${name}_seed${seed}"
    mkdir -p "$ckpt_dir"

    # Temp config: override seed AND checkpoint_dir
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/retrain_${name}_seed${seed}.log"

    rm -f "$tmp"
    echo "=== Finished training ${name} seed ${seed} ==="
  done
done

echo ""
echo "=============================================="
echo " STEP 2: Evaluate all 12 runs"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=== Evaluating ${name} seed ${seed} ==="

    ckpt="checkpoints/${name}_seed${seed}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
      echo "  [ERROR] No checkpoint at ${ckpt} — skipping."
      continue
    fi

    out_dir="results/${name}_seed${seed}"
    mkdir -p "$out_dir"

    tmp="/tmp/${name}_seed${seed}_eval.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

    python3 scripts/evaluate.py \
      --config  "$tmp" \
      --checkpoint "$ckpt" \
      --split   test \
      --stratified \
      --output  "${out_dir}/evaluation_results.json" \
      2>&1 | tee "logs/eval_${name}_seed${seed}.log"

    # Optional zero-shot
    if [ -f "data/splits/zero_shot_rare_diseases.json" ]; then
      python3 scripts/evaluate.py \
        --config  "$tmp" \
        --checkpoint "$ckpt" \
        --split   test \
        --zero_shot \
        --zero_shot_file  "data/splits/zero_shot_rare_diseases.json" \
        --zero_shot_output "${out_dir}/zero_shot_results.json" \
        2>&1 | tee -a "logs/eval_${name}_seed${seed}.log"
    fi

    rm -f "$tmp"
    echo "=== Done ${name} seed ${seed} ==="
  done
done

echo ""
echo "=============================================="
echo " STEP 3: Summary"
echo "=============================================="
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    out="results/${name}_seed${seed}/evaluation_results.json"
    if [ -f "$out" ]; then
      auroc=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('auroc','N/A'):.4f}\")" 2>/dev/null || echo "err")
      hit50=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('hit_rate@50','N/A'):.4f}\")" 2>/dev/null || echo "err")
      mrr=$(python3 -c   "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('mrr','N/A'):.4f}\")" 2>/dev/null || echo "err")
      echo "  ${name} seed${seed}:  AUROC=${auroc}  Hit@50=${hit50}  MRR=${mrr}"
    else
      echo "  ${name} seed${seed}:  [no results]"
    fi
  done
done

echo ""
echo "=== ALL DONE ==="
