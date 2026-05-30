#!/usr/bin/env bash
# smoke_test_path2-jarvis.sh
# ---------------------------------------------------------------------------
# Quick post-patch verification BEFORE the 40-run batch.
#
# Trains ablation_4_full_model for ONLY a few epochs at two different seeds,
# into throwaway directories, and checks the three Path 2 invariants:
#
#   (A) the train/val/test split edge counts are IDENTICAL across seeds
#       -> proves data.random_seed is fixing the split
#   (B) the per-epoch training loss values DIFFER between seeds 42 and 43
#       -> proves _set_all_seeds(config['seed']) actually varies model RNG
#   (C) the run completes without error
#
# Cost: ~10 minutes on A100 40GB (~Rs 14). Cheapest insurance you'll buy.
#
# Usage:
#   bash smoke_test_path2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

CFG="configs/ablations/ablation_4_full_model.yaml"
NAME="ablation_4_full_model"
EPOCHS_SMOKE=3   # short

run_smoke () {
  local seed=$1
  local ckpt_dir="checkpoints/smoke_${NAME}_seed${seed}"
  rm -rf "$ckpt_dir"
  mkdir -p "$ckpt_dir"

  local tmp="/tmp/smoke_${NAME}_seed${seed}.yaml"
  # Override seed, checkpoint_dir, AND num_epochs (so the smoke is brief).
  # Use a robust multi-line sed to also override num_epochs under training:.
  sed -e "s/^seed: .*/seed: ${seed}/" \
      -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
      -e "s/^  num_epochs: .*/  num_epochs: ${EPOCHS_SMOKE}/" \
      "$CFG" > "$tmp"

  set +e
  python3 scripts/train.py --config "$tmp" \
    2>&1 | tee "logs/smoke_${NAME}_seed${seed}.log"
  local e=${PIPESTATUS[0]}
  set -e
  rm -f "$tmp"
  return $e
}

echo "=============================================="
echo " Path 2 smoke test — $(date)"
echo "=============================================="

echo ""
echo "--- Smoke run #1: seed 42 ---"
if ! run_smoke 42; then echo "FATAL: smoke seed 42 crashed"; exit 1; fi

echo ""
echo "--- Smoke run #2: seed 43 ---"
if ! run_smoke 43; then echo "FATAL: smoke seed 43 crashed"; exit 1; fi

echo ""
echo "=============================================="
echo " Verifying Path 2 invariants"
echo "=============================================="

L1="logs/smoke_${NAME}_seed42.log"
L2="logs/smoke_${NAME}_seed43.log"

# (A) Split edge counts must match across seeds (split is fixed at data.random_seed=42)
n1=$(grep -E "Train: .* edges" "$L1" | head -1 | grep -oE "[0-9]+" | head -1)
n2=$(grep -E "Train: .* edges" "$L2" | head -1 | grep -oE "[0-9]+" | head -1)
echo "  [A] train edges: seed42=${n1:-?}  seed43=${n2:-?}"
if [ -n "$n1" ] && [ "$n1" = "$n2" ]; then
  echo "      PASS — split is fixed (identical edge counts)"
  A_OK=1
else
  echo "      FAIL — split differs across seeds; Path 2 patch not applied correctly"
  A_OK=0
fi

# (B) The split-seed log line should report split_seed=42 for BOTH runs
s1=$(grep -oE "split seed = [0-9]+" "$L1" | head -1 | grep -oE "[0-9]+")
s2=$(grep -oE "split seed = [0-9]+" "$L2" | head -1 | grep -oE "[0-9]+")
echo "  [B] split seed logged: seed42 run=${s1:-?}  seed43 run=${s2:-?}"
if [ "$s1" = "42" ] && [ "$s2" = "42" ]; then
  echo "      PASS — both runs used split seed 42 (data.random_seed)"
  B_OK=1
else
  echo "      FAIL — at least one run did not use data.random_seed for the split"
  B_OK=0
fi

# (C) Model RNG was reseeded per run from config['seed']
r1=$(grep -E "RNGs seeded with seed=42" "$L1" | wc -l)
r2=$(grep -E "RNGs seeded with seed=43" "$L2" | wc -l)
echo "  [C] _set_all_seeds called: seed42=${r1}  seed43=${r2}"
if [ "$r1" -ge 1 ] && [ "$r2" -ge 1 ]; then
  echo "      PASS — model RNG seeded per run from config['seed']"
  C_OK=1
else
  echo "      FAIL — _set_all_seeds did not fire correctly"
  C_OK=0
fi

# (D) Loss curves should DIFFER between the two seeds (varying init), even with
#     identical split. We compare the first reported training loss.
loss1=$(grep -oE "loss[: =][0-9]+\.[0-9]+" "$L1" | head -1 || true)
loss2=$(grep -oE "loss[: =][0-9]+\.[0-9]+" "$L2" | head -1 || true)
echo "  [D] first training loss:  seed42=${loss1:-?}  seed43=${loss2:-?}"
if [ -n "$loss1" ] && [ -n "$loss2" ] && [ "$loss1" != "$loss2" ]; then
  echo "      PASS — losses differ between seeds (model RNG is varying)"
  D_OK=1
else
  echo "      WARN — losses look identical or could not be parsed; inspect logs"
  D_OK=0
fi

echo ""
if [ "$A_OK" = 1 ] && [ "$B_OK" = 1 ] && [ "$C_OK" = 1 ] && [ "$D_OK" = 1 ]; then
  echo " >>> SMOKE PASS — Path 2 is wired correctly. Safe to launch run_all_seeds-jarvis.sh."
else
  echo " >>> SMOKE FAIL — fix before running the 40-run batch."
  echo "     Check that PATH2_PATCHES.md was applied to BOTH train.py and evaluate.py,"
  echo "     and that scripts/train.py contains exactly the three insertions."
  exit 1
fi

echo ""
echo " Cleaning up smoke checkpoints..."
rm -rf checkpoints/smoke_${NAME}_seed42 checkpoints/smoke_${NAME}_seed43
echo " Done."
