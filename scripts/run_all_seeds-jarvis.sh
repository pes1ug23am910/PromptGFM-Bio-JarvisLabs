#!/usr/bin/env bash
# run_all_seeds-jarvis.sh
# ---------------------------------------------------------------------------
# Path 2: trains 10 fresh seeds (42-51) x 4 ablation variants = 40 runs.
#
# This REPLACES the old 12 ablations (workstation, broken seed flow) and the
# run_ablations_extra_seeds-jarvis.sh (which only added 7 seeds to the old 3).
# All 40 runs are under the patched seed flow: split fixed at data.random_seed=42,
# model RNG seeded by top-level config['seed'].
#
# Safe to re-run / resume (spot interruption etc.): skip-if-exists per
# (variant, seed). To split across two GPUs:
#   GPU 1:  ALL_SEEDS="42 43 44 45 46" bash run_all_seeds-jarvis.sh
#   GPU 2:  ALL_SEEDS="47 48 49 50 51" bash run_all_seeds-jarvis.sh
#
# Usage (in tmux):
#   tmux new -s train
#   bash run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%Y%m%d).log

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

read -r -a SEEDS <<< "$ALL_SEEDS"
TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
SKIPPED=0; RAN=0; FAILED=0

echo "=============================================="
echo " Path 2 training — $(date)"
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

    # Override only the top-level model-RNG seed and the checkpoint dir.
    # data.random_seed is NOT touched -> split stays fixed at 42 for every run.
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    set +e
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/path2_train_${name}_seed${seed}.log"
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
echo " Path 2 training summary — $(date)"
echo "=============================================="
echo "  Skipped (already done): ${SKIPPED}"
echo "  Trained this run:       ${RAN}"
echo "  Failed:                 ${FAILED}"
_done=0
for cfg in "${CONFIGS[@]}"; do
  _name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    [ -f "checkpoints/${_name}_seed${seed}/best_model.pt" ] && _done=$((_done + 1))
  done
done
echo "  Total Path 2 checkpoints: ${_done}/${TOTAL}"
echo ""
echo " Next: bash scripts/run_evaluations_all_seeds-jarvis.sh"
