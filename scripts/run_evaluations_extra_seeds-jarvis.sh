#!/usr/bin/env bash
# run_evaluations_extra_seeds-jarvis.sh
# ---------------------------------------------------------------------------
# Evaluates the EXTRA seeds (default 45-51) for all 4 ablation variants.
# Mirrors scripts/run_evaluations-workstation.sh exactly, EXCEPT:
#   - PROJECT path comes from _jarvis_env.sh (/home/promptgfm-bio)
#   - NO SCP to a laptop (that destination is unreachable from the cloud)
#   - skip-if-exists on both standard and zero-shot result files
#
# Produces, per (variant, seed):
#   results/<name>_seed<seed>/evaluation_results.json   (test + stratified)
#   results/<name>_seed<seed>/zero_shot_results.json    (117 zero-shot diseases)
#
# Usage (inside tmux):
#   bash run_evaluations_extra_seeds-jarvis.sh 2>&1 | tee -a logs/extra_seeds_eval_$(date +%Y%m%d).log

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

read -r -a SEEDS <<< "$EXTRA_SEEDS"

ZS_FILE="data/splits/zero_shot_rare_diseases.json"
COMPLETED=0; SKIPPED=0; FAILED=0

echo "=============================================="
echo " Evaluate extra seeds — $(date)"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    ckpt="checkpoints/${name}_seed${seed}/best_model.pt"
    out_dir="results/${name}_seed${seed}"
    std_out="${out_dir}/evaluation_results.json"
    zs_out="${out_dir}/zero_shot_results.json"

    if [ ! -f "$ckpt" ]; then
      echo "=== ERROR ${name} seed ${seed} — no checkpoint at ${ckpt} ==="
      FAILED=$((FAILED + 1))
      continue
    fi

    if [ -f "$std_out" ] && [ -f "$zs_out" ]; then
      echo "=== SKIP ${name} seed ${seed} — both result files exist ==="
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo ""
    echo "=== Evaluating ${name} seed ${seed} — $(date) ==="
    mkdir -p "$out_dir"

    tmp="/tmp/${name}_seed${seed}_eval.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

    # ── Standard test + stratified ─────────────────────────────────────────
    if [ ! -f "$std_out" ]; then
      set +e
      python3 scripts/evaluate.py \
        --config "$tmp" \
        --checkpoint "$ckpt" \
        --split test \
        --stratified \
        --output "$std_out" \
        2>&1 | tee "logs/eval_${name}_seed${seed}.log"
      e=${PIPESTATUS[0]}
      set -e
      if [ "$e" -ne 0 ]; then
        echo "  [FAILED] standard eval ${name} seed ${seed} — exit ${e}"
        FAILED=$((FAILED + 1)); rm -f "$tmp"; continue
      fi
    fi

    # ── Zero-shot (117 rare diseases) ──────────────────────────────────────
    if [ ! -f "$zs_out" ]; then
      if [ -f "$ZS_FILE" ]; then
        set +e
        python3 scripts/evaluate.py \
          --config "$tmp" \
          --checkpoint "$ckpt" \
          --split test \
          --zero_shot \
          --zero_shot_file "$ZS_FILE" \
          --zero_shot_output "$zs_out" \
          2>&1 | tee -a "logs/eval_${name}_seed${seed}.log"
        e=${PIPESTATUS[0]}
        set -e
        if [ "$e" -ne 0 ]; then
          echo "  [FAILED] zero-shot eval ${name} seed ${seed} — exit ${e}"
          FAILED=$((FAILED + 1)); rm -f "$tmp"; continue
        fi
      else
        echo "  [WARN] ${ZS_FILE} missing — skipping zero-shot for ${name} seed ${seed}"
      fi
    fi

    rm -f "$tmp"
    COMPLETED=$((COMPLETED + 1))
    echo "=== Done ${name} seed ${seed} ==="
  done
done

echo ""
echo "=============================================="
echo " Evaluation summary — $(date)"
echo "=============================================="
echo "  Completed: ${COMPLETED}   Skipped: ${SKIPPED}   Failed: ${FAILED}"
echo ""
echo " Standard test metrics (new seeds):"
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    out="results/${name}_seed${seed}/evaluation_results.json"
    if [ -f "$out" ]; then
      read -r auroc hit50 mrr < <(python3 -c "
import json
d=json.load(open('$out')); t=d.get('test',d)
print(f\"{t.get('auroc',float('nan')):.4f} {t.get('hit_rate@50',float('nan')):.4f} {t.get('mrr',float('nan')):.4f}\")
" 2>/dev/null || echo "err err err")
      echo "  ${name} seed${seed}:  AUROC=${auroc}  Hit@50=${hit50}  MRR=${mrr}"
    else
      echo "  ${name} seed${seed}:  [no results]"
    fi
  done
done
