#!/usr/bin/env bash
# resume_ablations.sh — Continue the 12-run ablation training after interruption.
#
# Skips any (config, seed) whose checkpoints/${name}_seed${seed}/best_model.pt
# already exists.  Does NOT back up old checkpoints (we're resuming, not
# starting over).  Does NOT run evaluation — use evaluate_all-fix.sh for that
# after all 12 training runs complete (it already has skip logic).
#
# Usage (inside tmux):
#   bash scripts/resume_ablations.sh 2>&1 | tee -a logs/resume_$(date +%Y%m%d_%H%M%S).log
#
# Note on `set -e` + `| tee`: we capture the python exit code via PIPESTATUS
# so a single failed training run doesn't kill the whole wrapper.

set -euo pipefail

PROJECT=/home/mluser/projects_yash/new_project/PromptGFM-Bio
cd "$PROJECT"
mkdir -p logs results

CONFIGS=(
  "configs/ablations/ablation_1_mlp_only.yaml"
  "configs/ablations/ablation_2_prompt_only.yaml"
  "configs/ablations/ablation_3_gnn_only.yaml"
  "configs/ablations/ablation_4_full_model.yaml"
)
SEEDS=(42 43 44)

TOTAL=12
DONE=0
SKIPPED=0
RAN=0
FAILED=0

echo "=============================================="
echo " Resume training — $(date)"
echo "=============================================="

# ─── First pass: inventory what is already complete ────────────────────────
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    if [ -f "checkpoints/${name}_seed${seed}/best_model.pt" ]; then
      DONE=$((DONE + 1))
    fi
  done
done
echo " Already complete: ${DONE}/${TOTAL}"
echo " Still to train:   $((TOTAL - DONE))/${TOTAL}"
echo ""

# ─── Second pass: train the missing ones ───────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    ckpt_dir="checkpoints/${name}_seed${seed}"
    best_model="${ckpt_dir}/best_model.pt"

    if [ -f "$best_model" ]; then
      echo "=== SKIP ${name} seed ${seed} — best_model.pt already exists ==="
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo ""
    echo "=============================================="
    echo "=== Training ${name} seed ${seed} ==="
    echo "=== Started: $(date) ==="
    echo "=============================================="

    mkdir -p "$ckpt_dir"

    # Temp config: override seed AND checkpoint_dir
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    # Capture exit via PIPESTATUS so `set -e` + `| tee` doesn't kill the wrapper
    set +e
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/retrain_${name}_seed${seed}.log"
    train_exit=${PIPESTATUS[0]}
    set -e

    rm -f "$tmp"

    if [ $train_exit -ne 0 ]; then
      echo "=== [FAILED] ${name} seed ${seed} — exit code ${train_exit} ==="
      FAILED=$((FAILED + 1))
    else
      echo "=== [DONE] ${name} seed ${seed} — $(date) ==="
      RAN=$((RAN + 1))
    fi
  done
done

echo ""
echo "=============================================="
echo " Resume summary — $(date)"
echo "=============================================="
echo "  Already complete (skipped):  ${SKIPPED}"
echo "  Trained this run:            ${RAN}"
echo "  Failed:                      ${FAILED}"
echo "  Total complete checkpoints:  $(ls checkpoints/ablation_*_seed*/best_model.pt 2>/dev/null | wc -l)/${TOTAL}"
echo ""
echo " Next step: when all 12 complete, run:"
echo "   bash scripts/evaluate_all-fix.sh"
echo " (it skips runs whose results/*/evaluation_results.json already exists)"
