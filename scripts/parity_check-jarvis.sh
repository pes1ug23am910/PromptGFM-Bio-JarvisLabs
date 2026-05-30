#!/usr/bin/env bash
# parity_check-jarvis.sh
# ---------------------------------------------------------------------------
# THE INSURANCE STEP. Run this ONCE before launching the 28-run seed batch.
#
# Retrains ablation_4_full_model at seed 42 on the JarvisLabs GPU, into a
# SEPARATE checkpoint/result dir (your real seed-42 artifacts are never
# touched), then compares the new test metrics against the existing
# workstation-trained seed-42 result.
#
# Purpose: confirm that moving RTX 4090/4500-Ada -> A100 does not shift the
# numbers beyond the expected GPU-nondeterminism band (~0.001-0.003 AUROC).
#   - PASS  -> seeds 45-51 can be pooled with 42,43,44 into one 10-seed study.
#   - FAIL  -> something differs (package versions, TF32, deterministic flags);
#              fix before the full batch, or report the 7 new seeds separately.
#
# Cost: ~1.3 hr on A100 40GB (~Rs 110). Cheapest insurance in the project.
#
# Usage:
#   bash parity_check-jarvis.sh 2>&1 | tee logs/parity_check_$(date +%Y%m%d).log

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

CFG="configs/ablations/ablation_4_full_model.yaml"
NAME="ablation_4_full_model"
SEED=42

# Separate dirs so the real seed-42 artifacts are untouched.
PARITY_CKPT_DIR="checkpoints/parity_${NAME}_seed${SEED}"
PARITY_OUT_DIR="results/parity_${NAME}_seed${SEED}"
PARITY_OUT="${PARITY_OUT_DIR}/evaluation_results.json"

# Reference = the workstation-trained seed-42 result already in the repo.
REF_OUT="results/${NAME}_seed${SEED}/evaluation_results.json"

# Pass band (absolute delta). GPU-architecture nondeterminism on this model is
# empirically <=0.003 AUROC; 0.01+ indicates a real configuration difference.
THRESH="${PARITY_THRESH:-0.005}"

echo "=============================================="
echo " PARITY CHECK — ${NAME} seed ${SEED} — $(date)"
echo " threshold (abs delta) = ${THRESH}"
echo "=============================================="

if [ ! -f "$REF_OUT" ]; then
  echo "[warn] Reference result not found at: $REF_OUT"
  echo "       The new run will still execute; copy your workstation"
  echo "       results/${NAME}_seed${SEED}/evaluation_results.json here to enable"
  echo "       automatic comparison. Otherwise compare the printed numbers by hand."
fi

# ── 1. Train (separate checkpoint dir, identical config + seed) ────────────
mkdir -p "$PARITY_CKPT_DIR" "$PARITY_OUT_DIR"
tmp="/tmp/parity_${NAME}_seed${SEED}.yaml"
sed -e "s/^seed: .*/seed: ${SEED}/" \
    -e "s|checkpoint_dir: .*|checkpoint_dir: ${PARITY_CKPT_DIR}|" \
    "$CFG" > "$tmp"

echo ""
echo "--- Training (this is the ~1.3 hr step) ---"
set +e
python3 scripts/train.py --config "$tmp" \
  2>&1 | tee "logs/parity_train_${NAME}_seed${SEED}.log"
train_exit=${PIPESTATUS[0]}
set -e
if [ "$train_exit" -ne 0 ]; then
  echo "FATAL: parity training failed (exit ${train_exit})"; rm -f "$tmp"; exit 1
fi

# ── 2. Evaluate (test + stratified, identical to the real pipeline) ────────
echo ""
echo "--- Evaluating ---"
ckpt="${PARITY_CKPT_DIR}/best_model.pt"
set +e
python3 scripts/evaluate.py \
  --config "$tmp" \
  --checkpoint "$ckpt" \
  --split test \
  --stratified \
  --output "$PARITY_OUT" \
  2>&1 | tee "logs/parity_eval_${NAME}_seed${SEED}.log"
eval_exit=${PIPESTATUS[0]}
set -e
rm -f "$tmp"
if [ "$eval_exit" -ne 0 ]; then
  echo "FATAL: parity evaluation failed (exit ${eval_exit})"; exit 1
fi

# ── 3. Compare ─────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " PARITY RESULT"
echo "=============================================="
python3 - "$PARITY_OUT" "$REF_OUT" "$THRESH" <<'PY'
import json, sys, math

new_path, ref_path, thresh = sys.argv[1], sys.argv[2], float(sys.argv[3])
metrics = ["auroc", "hit_rate@50", "hit_rate@10", "mrr"]

def load(p):
    try:
        d = json.load(open(p))
        return d.get("test", d)
    except Exception as e:
        return None

new = load(new_path)
ref = load(ref_path)

if new is None:
    print(f"  Could not read new result: {new_path}")
    sys.exit(2)

print(f"  {'metric':12s} {'new(A100)':>11s} {'ref(workstation)':>18s} {'delta':>9s}  verdict")
print("  " + "-"*62)

if ref is None:
    for m in metrics:
        v = new.get(m, float('nan'))
        print(f"  {m:12s} {v:11.4f} {'(no reference)':>18s} {'':>9s}")
    print("\n  No reference file -> compare these numbers to your workstation")
    print("  results/ablation_4_full_model_seed42/evaluation_results.json by hand.")
    sys.exit(0)

worst = 0.0
fail = False
for m in metrics:
    nv = new.get(m, float('nan'))
    rv = ref.get(m, float('nan'))
    if math.isnan(nv) or math.isnan(rv):
        print(f"  {m:12s} {nv:11.4f} {rv:18.4f} {'   n/a':>9s}  SKIP (missing)")
        continue
    d = abs(nv - rv)
    worst = max(worst, d)
    ok = d <= thresh
    fail = fail or (not ok)
    print(f"  {m:12s} {nv:11.4f} {rv:18.4f} {d:9.4f}  {'PASS' if ok else 'FAIL'}")

print("  " + "-"*62)
print(f"  worst delta = {worst:.4f}   threshold = {thresh:.4f}")
if fail:
    print("\n  >>> PARITY FAIL <<<")
    print("  Do NOT pool the new seeds with 42/43/44 yet. Check, in order:")
    print("    1. torch / torch-geometric / transformers versions vs requirements.txt")
    print("    2. TF32 disabled (no torch.backends.cuda.matmul.allow_tf32=True)")
    print("    3. AMP still FP16 (mixed_precision: true), not BF16")
    print("    4. deterministic:false, benchmark:true (as in the config)")
    sys.exit(1)
else:
    print("\n  >>> PARITY PASS <<<")
    print("  A100 numbers match the workstation within tolerance.")
    print("  Safe to launch run_ablations_extra_seeds-jarvis.sh and pool all 10 seeds.")
    sys.exit(0)
PY

echo ""
echo "Parity artifacts kept in: ${PARITY_CKPT_DIR}/ and ${PARITY_OUT_DIR}/"
echo "(Delete them after you're satisfied — they are NOT part of the 10-seed study.)"
