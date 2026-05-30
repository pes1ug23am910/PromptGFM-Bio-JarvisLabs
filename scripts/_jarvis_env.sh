#!/usr/bin/env bash
# _jarvis_env.sh — shared environment for Path 2 JarvisLabs runners.
# (Replaces the previous _jarvis_env.sh.)
# Source this at the top of every runner:  source "$(dirname "$0")/_jarvis_env.sh"

PROJECT="${PROJECT:-/home/promptgfm-bio}"
VENV="${VENV:-$PROJECT/.venv}"

# Path 2: all 10 seeds, fresh under the patched seed flow.
# Old 12 ablations (seeds 42-44 on workstation) are NOT pooled with these.
ALL_SEEDS="${ALL_SEEDS:-42 43 44 45 46 47 48 49 50 51}"

CONFIGS=(
  "configs/ablations/ablation_1_mlp_only.yaml"
  "configs/ablations/ablation_2_prompt_only.yaml"
  "configs/ablations/ablation_3_gnn_only.yaml"
  "configs/ablations/ablation_4_full_model.yaml"
)

cd "$PROJECT" || { echo "FATAL: PROJECT path not found: $PROJECT"; exit 1; }

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "[env] venv active: $VENV"
else
  echo "[env] WARNING: no venv at $VENV — using system python3"
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
mkdir -p logs results

echo "[env] PROJECT     = $PROJECT"
echo "[env] ALL_SEEDS   = $ALL_SEEDS"
echo "[env] python3     = $(command -v python3)"

# Sanity gate: refuse to run if the Path 2 patches are not present.
if ! grep -q "_set_all_seeds" "$PROJECT/scripts/train.py" 2>/dev/null; then
  echo "[env] FATAL: Path 2 patch missing in scripts/train.py."
  echo "        Apply PATH2_PATCHES.md before running any Path 2 script."
  exit 2
fi
if ! grep -q "_set_all_seeds" "$PROJECT/scripts/evaluate.py" 2>/dev/null; then
  echo "[env] FATAL: Path 2 patch missing in scripts/evaluate.py."
  exit 2
fi
echo "[env] Path 2 patches detected in train.py + evaluate.py [OK]"
