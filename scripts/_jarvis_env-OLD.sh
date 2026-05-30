#!/usr/bin/env bash
# _jarvis_env.sh — shared environment for all JarvisLabs PromptGFM-Bio scripts.
# Source this at the top of every runner:  source "$(dirname "$0")/_jarvis_env.sh"
#
# Everything cloud-specific lives HERE so the runners stay identical in logic to
# the workstation versions (only the path differs).

# ── Project location on the JarvisLabs instance ────────────────────────────
# /home is the PERSISTENT mount on JarvisLabs (survives pause/resume).
# Clone or copy the repo to exactly this path.
PROJECT="${PROJECT:-/home/promptgfm-bio}"

# ── Virtual environment (optional but recommended) ─────────────────────────
# If a venv exists at $PROJECT/.venv it will be activated automatically.
VENV="${VENV:-$PROJECT/.venv}"

# ── New seeds for the 10-seed total (42,43,44 already done on workstation) ──
# Override at call time, e.g.:  EXTRA_SEEDS="45 46 47" bash run_..._jarvis.sh
EXTRA_SEEDS="${EXTRA_SEEDS:-45 46 47 48 49 50 51}"

# ── The 4 ablation configs (relative to $PROJECT) ──────────────────────────
CONFIGS=(
  "configs/ablations/ablation_1_mlp_only.yaml"
  "configs/ablations/ablation_2_prompt_only.yaml"
  "configs/ablations/ablation_3_gnn_only.yaml"
  "configs/ablations/ablation_4_full_model.yaml"
)

# ── Apply the environment ──────────────────────────────────────────────────
cd "$PROJECT" || { echo "FATAL: PROJECT path not found: $PROJECT"; exit 1; }

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "[env] venv active: $VENV"
else
  echo "[env] WARNING: no venv at $VENV — using system python3"
fi

# PubMedBERT tokenizer spawns threads that deadlock with DataLoader workers
# unless this is disabled (matches workstation runs for seeds 42-44).
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

mkdir -p logs results

echo "[env] PROJECT      = $PROJECT"
echo "[env] EXTRA_SEEDS  = $EXTRA_SEEDS"
echo "[env] python3      = $(command -v python3)"
