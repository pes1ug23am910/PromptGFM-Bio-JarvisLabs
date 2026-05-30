#!/usr/bin/env bash
# run_ablations_extra_seeds-jarvis.sh
# ---------------------------------------------------------------------------
# Trains the EXTRA seeds (default 45-51) for all 4 ablation variants on a
# JarvisLabs instance, taking the 3-seed study (42,43,44) up to 10 seeds.
#
# Identical training logic to scripts/retrain_and_evaluate.sh STEP 1:
#   - overrides ONLY the top-level `seed:` line (data.random_seed stays 42,
#     so the train/val/test split and the zero-shot set are unchanged)
#   - per-seed checkpoint dir so runs never overwrite each other
#   - configs/*.yaml are NOT edited (batch_size 768, FP16 AMP, etc. preserved)
#
# Safe to re-run / resume (e.g. after a spot interruption): any run whose
# best_model.pt already exists is skipped.
#
# Usage (inside tmux):
#   tmux new -s train
#   bash run_ablations_extra_seeds-jarvis.sh 2>&1 | tee -a logs/extra_seeds_train_$(date +%Y%m%d).log
#   # Ctrl+B, D to detach;  tmux attach -t train  to return
#
# Run a subset (e.g. one spot GPU doing 3 seeds):
#   EXTRA_SEEDS="45 46 47" bash run_ablations_extra_seeds-jarvis.sh

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

read -r -a SEEDS <<< "$EXTRA_SEEDS"

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
SKIPPED=0; RAN=0; FAILED=0

echo "=============================================="
echo " Train extra seeds — $(date)"
echo " ${#CONFIGS[@]} variants x ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    ckpt_dir="checkpoints/${name}_seed${seed}"
    best_model="${ckpt_dir}/best_model.pt"

    if [ -f "$best_model" ]; then
      echo "=== SKIP ${name} seed ${seed} — best_model.pt exists ==="
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo ""
    echo "=============================================="
    echo "=== Training ${name} seed ${seed} ==="
    echo "=== Started: $(date) ==="
    echo "=============================================="
    mkdir -p "$ckpt_dir"

    # Temp config: override top-level seed AND checkpoint_dir ONLY.
    # (Exactly matches retrain_and_evaluate.sh — do not add other overrides.)
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    set +e
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/retrain_${name}_seed${seed}.log"
    train_exit=${PIPESTATUS[0]}
    set -e
    rm -f "$tmp"

    if [ "$train_exit" -ne 0 ]; then
      echo "=== [FAILED] ${name} seed ${seed} — exit ${train_exit} ==="
      FAILED=$((FAILED + 1))
    else
      echo "=== [DONE] ${name} seed ${seed} — $(date) ==="
      RAN=$((RAN + 1))
    fi
  done
done

echo ""
echo "=============================================="
echo " Training summary — $(date)"
echo "=============================================="
echo "  Skipped (already done): ${SKIPPED}"
echo "  Trained this run:       ${RAN}"
echo "  Failed:                 ${FAILED}"
# Count checkpoints for exactly the seeds requested (respects EXTRA_SEEDS override)
_done=0
for cfg in "${CONFIGS[@]}"; do
  _name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    [ -f "checkpoints/${_name}_seed${seed}/best_model.pt" ] && _done=$((_done + 1))
  done
done
echo "  Extra-seed checkpoints: ${_done}/${TOTAL}"
echo ""
echo " Next: bash run_evaluations_extra_seeds-jarvis.sh"
