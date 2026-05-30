#!/usr/bin/env bash
# smoke_test_path2_v2-jarvis.sh
# ---------------------------------------------------------------------------
# Post-patch verification BEFORE the 40-run batch. Supersedes
# smoke_test_path2-jarvis.sh. Trains ablation_4_full_model for a few epochs and
# checks FIVE invariants:
#
#   (A) train/val/test split edge counts are IDENTICAL across seeds 42 & 43
#       -> data.random_seed fixes the split
#   (B) "split seed = 42" is logged for BOTH runs
#       -> the new split-seed code path executed
#   (C) _set_all_seeds fired with the right per-run seed
#       -> model RNG is reseeded from config['seed']
#   (D) per-epoch losses DIFFER between seeds 42 and 43
#       -> the seed actually varies model init / sampling
#   (E) **per-epoch losses are IDENTICAL across TWO runs of seed 42**   <-- NEW
#       -> the pipeline is deterministic GIVEN the seed (negative sampling in
#          the DataLoader workers + shuffle + init are all reproducible).
#
# (A)-(D) prove the seed *varies*; only (E) proves a seed *reproduces*. (E) is
# the empirical answer to a Q1 reviewer's "is your training deterministic given
# the reported seed?" — the single objection Path 2 most needs to preempt.
#
# Cost: ~15 min on A100 40GB (three 3-epoch runs). Run ON-DEMAND so logs are
# complete (the parsers below are sensitive to truncated logs).
#
# Usage:
#   bash smoke_test_path2_v2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log
#
# If (E) FAILS: a stochastic source is leaking process-time randomness. The fix
# is the worker_init_fn patch in PATH2_PATCHES_v2.md (seeds Python `random`
# per DataLoader worker). Re-run this smoke after applying it. Do NOT launch the
# 40-run batch until (E) passes.

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

CFG="configs/ablations/ablation_4_full_model.yaml"
NAME="ablation_4_full_model"
EPOCHS_SMOKE=3   # short

# run_smoke <seed> [tag]
#   tag distinguishes a second run of the same seed (for invariant E).
run_smoke () {
  local seed=$1
  local tag=${2:-}
  local ckpt_dir="checkpoints/smoke_${NAME}_seed${seed}${tag}"
  rm -rf "$ckpt_dir"
  mkdir -p "$ckpt_dir"

  local tmp="/tmp/smoke_${NAME}_seed${seed}${tag}.yaml"
  # Override seed, checkpoint_dir, AND num_epochs (so the smoke is brief).
  sed -e "s/^seed: .*/seed: ${seed}/" \
      -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
      -e "s/^  num_epochs: .*/  num_epochs: ${EPOCHS_SMOKE}/" \
      "$CFG" > "$tmp"

  set +e
  python3 scripts/train.py --config "$tmp" \
    2>&1 | tee "logs/smoke_${NAME}_seed${seed}${tag}.log"
  local e=${PIPESTATUS[0]}
  set -e
  rm -f "$tmp"
  return $e
}

echo "=============================================="
echo " Path 2 smoke test v2 — $(date)"
echo "=============================================="

echo ""; echo "--- Smoke run #1: seed 42 ---"
if ! run_smoke 42;     then echo "FATAL: smoke seed 42 crashed";  exit 1; fi
echo ""; echo "--- Smoke run #2: seed 43 ---"
if ! run_smoke 43;     then echo "FATAL: smoke seed 43 crashed";  exit 1; fi
echo ""; echo "--- Smoke run #3: seed 42 AGAIN (determinism check) ---"
if ! run_smoke 42 "b"; then echo "FATAL: smoke seed 42 (rerun) crashed"; exit 1; fi

echo ""
echo "=============================================="
echo " Verifying Path 2 invariants"
echo "=============================================="

L1="logs/smoke_${NAME}_seed42.log"
L2="logs/smoke_${NAME}_seed43.log"
L1B="logs/smoke_${NAME}_seed42b.log"

# (A) Split edge counts must match across seeds (split fixed at data.random_seed=42)
n1=$(grep -E "Train: .* edges" "$L1" | head -1 | grep -oE "[0-9]+" | head -1)
n2=$(grep -E "Train: .* edges" "$L2" | head -1 | grep -oE "[0-9]+" | head -1)
echo "  [A] train edges: seed42=${n1:-?}  seed43=${n2:-?}"
if [ -n "$n1" ] && [ "$n1" = "$n2" ]; then
  echo "      PASS — split is fixed (identical edge counts)"; A_OK=1
else
  echo "      FAIL — split differs across seeds; Path 2 patch not applied correctly"; A_OK=0
fi

# (B) split-seed log line should report 42 for BOTH runs
s1=$(grep -oE "split seed = [0-9]+" "$L1" | head -1 | grep -oE "[0-9]+")
s2=$(grep -oE "split seed = [0-9]+" "$L2" | head -1 | grep -oE "[0-9]+")
echo "  [B] split seed logged: seed42 run=${s1:-?}  seed43 run=${s2:-?}"
if [ "$s1" = "42" ] && [ "$s2" = "42" ]; then
  echo "      PASS — both runs used split seed 42 (data.random_seed)"; B_OK=1
else
  echo "      FAIL — at least one run did not use data.random_seed for the split"; B_OK=0
fi

# (C) Model RNG reseeded per run from config['seed']
r1=$(grep -E "RNGs seeded with seed=42" "$L1" | wc -l)
r2=$(grep -E "RNGs seeded with seed=43" "$L2" | wc -l)
echo "  [C] _set_all_seeds called: seed42=${r1}  seed43=${r2}"
if [ "$r1" -ge 1 ] && [ "$r2" -ge 1 ]; then
  echo "      PASS — model RNG seeded per run from config['seed']"; C_OK=1
else
  echo "      FAIL — _set_all_seeds did not fire correctly"; C_OK=0
fi

# (D) Loss curves should DIFFER between the two seeds (varying init)
loss1=$(grep -ioE "loss[:= ]+[0-9]+\.[0-9]+" "$L1" | head -1 || true)
loss2=$(grep -ioE "loss[:= ]+[0-9]+\.[0-9]+" "$L2" | head -1 || true)
echo "  [D] first training loss:  seed42=${loss1:-?}  seed43=${loss2:-?}"
if [ -n "$loss1" ] && [ -n "$loss2" ] && [ "$loss1" != "$loss2" ]; then
  echo "      PASS — losses differ between seeds (model RNG is varying)"; D_OK=1
else
  echo "      WARN — losses look identical or could not be parsed; inspect logs"; D_OK=0
fi

# (E) SAME-SEED DETERMINISM: seed 42 run twice -> IDENTICAL full loss sequence.
losses_a=$(grep -ioE "loss[:= ]+[0-9]+\.[0-9]+" "$L1"  | grep -oE "[0-9]+\.[0-9]+" | tr '\n' ' ' || true)
losses_b=$(grep -ioE "loss[:= ]+[0-9]+\.[0-9]+" "$L1B" | grep -oE "[0-9]+\.[0-9]+" | tr '\n' ' ' || true)
echo "  [E] seed42 run#1 losses: ${losses_a:-<none>}"
echo "      seed42 run#2 losses: ${losses_b:-<none>}"
if [ -n "$losses_a" ] && [ "$losses_a" = "$losses_b" ]; then
  echo "      PASS — identical losses across two same-seed runs"
  echo "             (pipeline is deterministic given the seed, at reporting precision)"
  E_OK=1
else
  echo "      FAIL — same-seed runs diverged. A stochastic source is unseeded."
  echo "             Apply the worker_init_fn patch (PATH2_PATCHES_v2.md) and re-run."
  E_OK=0
fi

echo ""
if [ "$A_OK" = 1 ] && [ "$B_OK" = 1 ] && [ "$C_OK" = 1 ] && [ "$D_OK" = 1 ] && [ "$E_OK" = 1 ]; then
  echo " >>> SMOKE PASS — Path 2 is wired correctly AND deterministic. Safe to launch run_all_seeds-jarvis.sh."
else
  echo " >>> SMOKE FAIL — fix before running the 40-run batch."
  echo "     (A)-(D): check PATH2_PATCHES.md was applied to BOTH train.py and evaluate.py."
  echo "     (E):     apply the worker_init_fn patch in PATH2_PATCHES_v2.md, then re-run."
  exit 1
fi

echo ""
echo " Cleaning up smoke checkpoints..."
rm -rf "checkpoints/smoke_${NAME}_seed42" \
       "checkpoints/smoke_${NAME}_seed43" \
       "checkpoints/smoke_${NAME}_seed42b"
echo " Done."
