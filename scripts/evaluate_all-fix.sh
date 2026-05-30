#!/usr/bin/env bash

# evaluate_all.sh — Evaluates all 12 ablation runs and SCPs results to laptop after each one.

set -euo pipefail

# ══════════════════════════════════════════════════════════════

# CONFIGURE THESE — your laptop's SSH details

# ══════════════════════════════════════════════════════════════

LAPTOP_USER="yash-ubuntu"
LAPTOP_IP="10.5.17.235"   
LAPTOP_DIR="/home/yash-ubuntu/Downloads/Final-to-be-given/Checkpoints"
LAPTOP_PORT=22
ENABLE_SCP="true"

# Robust SSH/SCP commands

SSH_CMD="ssh -p $LAPTOP_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=5"
SCP_CMD="scp -P $LAPTOP_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=5"

# ══════════════════════════════════════════════════════════════

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

COMPLETED=0
SKIPPED=0
FAILED=0
TOTAL=12

echo "=============================================="
echo " Evaluate all 4 ablations × 3 seeds"
echo " Started: $(date)"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
name=$(basename "$cfg" .yaml)

for seed in "${SEEDS[@]}"; do

out_dir="results/${name}_seed${seed}"
out_file="${out_dir}/evaluation_results.json"

# ── Skip if already evaluated ─────────────────────────────
if [ -f "$out_file" ]; then
  echo ""
  echo "=== SKIP ${name} seed ${seed} — results already exist ==="
  SKIPPED=$((SKIPPED + 1))
  continue
fi

# ── Check checkpoint exists ───────────────────────────────
ckpt="checkpoints/${name}_seed${seed}/best_model.pt"
if [ ! -f "$ckpt" ]; then
  echo ""
  echo "=== ERROR ${name} seed ${seed} — no checkpoint at ${ckpt} ==="
  FAILED=$((FAILED + 1))
  continue
fi

echo ""
echo "=============================================="
echo "=== Evaluating ${name} seed ${seed} ==="
echo "=== Started: $(date) ==="
echo "=============================================="

mkdir -p "$out_dir"

# ── Write temp config with seed override ──────────────────
tmp="/tmp/${name}_seed${seed}_eval.yaml"
sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

# ── Run evaluation ────────────────────────────────────────
# Capture pipeline exit via PIPESTATUS so `set -e` + `| tee` doesn't
# kill the whole script when a single evaluation fails.  We want to
# mark the run as FAILED and keep going.
set +e
python3 scripts/evaluate.py \
  --config  "$tmp" \
  --checkpoint "$ckpt" \
  --split   test \
  --stratified \
  --output  "${out_file}" \
  2>&1 | tee "logs/eval_${name}_seed${seed}.log"
eval_exit=${PIPESTATUS[0]}
set -e
rm -f "$tmp"

if [ $eval_exit -ne 0 ]; then
  echo "  [FAILED] ${name} seed ${seed} — evaluate.py exited with code ${eval_exit}"
  FAILED=$((FAILED + 1))
  continue
fi

COMPLETED=$((COMPLETED + 1))
echo "=== Done ${name} seed ${seed} (${COMPLETED}/$((TOTAL - SKIPPED)) new) ==="

# ── SCP results to laptop ─────────────────────────────────
if [ "$ENABLE_SCP" = "true" ]; then
  echo "  Copying results to laptop..."

  # Create destination directory on laptop
  $SSH_CMD "${LAPTOP_USER}@${LAPTOP_IP}" \
    "mkdir -p '${LAPTOP_DIR}/${name}_seed${seed}'" || true

  # Copy evaluation results
  $SCP_CMD \
    "${out_file}" \
    "${LAPTOP_USER}@${LAPTOP_IP}:${LAPTOP_DIR}/${name}_seed${seed}/evaluation_results.json" \
    && echo "  ✓ Results backed up to laptop" \
    || echo "  [WARN] SCP failed — results saved locally"

  # Copy log file
  $SCP_CMD \
    "logs/eval_${name}_seed${seed}.log" \
    "${LAPTOP_USER}@${LAPTOP_IP}:${LAPTOP_DIR}/${name}_seed${seed}/" \
    || true
fi

done
done

echo ""
echo "=============================================="
echo " Summary — $(date)"
echo "=============================================="
echo "  Completed: ${COMPLETED}"
echo "  Skipped: ${SKIPPED}"
echo "  Failed: ${FAILED}"
echo ""

for cfg in "${CONFIGS[@]}"; do
name=$(basename "$cfg" .yaml)

for seed in "${SEEDS[@]}"; do
out="results/${name}_seed${seed}/evaluation_results.json"

if [ -f "$out" ]; then
  auroc=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('auroc','N/A'):.4f}\")" 2>/dev/null || echo "err")
  hit50=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('hit_rate@50','N/A'):.4f}\")" 2>/dev/null || echo "err")
  mrr=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('mrr','N/A'):.4f}\")" 2>/dev/null || echo "err")

  echo "  ${name} seed${seed}:  AUROC=${auroc}  Hit@50=${hit50}  MRR=${mrr}"
else
  echo "  ${name} seed${seed}:  [no results]"
fi

done
done

echo ""
echo "=== ALL DONE ==="
