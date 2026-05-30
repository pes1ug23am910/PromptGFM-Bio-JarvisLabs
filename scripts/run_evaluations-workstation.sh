#!/usr/bin/env bash
# run_evaluations.sh — Evaluates all 4 ablation variants × 3 seeds = 12 runs.
# Mirrors the structure of run_ablations.sh.
# Usage:   bash scripts/run_evaluations.sh
# Logs:    logs/eval_<name>_seed<N>.log
# Results: results/<name>_seed<N>/evaluation_results.json

set -euo pipefail

cd /home/mluser/projects_yash/new_project/PromptGFM-Bio
mkdir -p logs results

# Each entry: "config_path  checkpoint_dir"
# The checkpoint for a given seed lives at checkpoints/<name>_seed<N>/best_model.pt
# The training script saves per-seed checkpoints using the experiment_name from the
# temp config (which uses the base name, not the seed-suffixed name), so we fall back
# to the base checkpoint dir if a seed-specific one doesn't exist.
CONFIGS=(
  "configs/ablations/ablation_1_mlp_only.yaml"
  "configs/ablations/ablation_2_prompt_only.yaml"
  "configs/ablations/ablation_3_gnn_only.yaml"
  "configs/ablations/ablation_4_full_model.yaml"
)

SEEDS=(42 43 44)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)

  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=== Evaluating ${name} seed ${seed} ==="

    # ── Locate checkpoint ─────────────────────────────────────────────────
    # run_ablations.sh writes a temp config whose checkpoint_dir comes from the
    # base yaml (e.g. checkpoints/ablation_1_mlp_only).  Training saves best_model.pt
    # there regardless of seed, so re-runs overwrite each other unless you manually
    # renamed the dirs.  We prefer a seed-suffixed dir if it exists.
    ckpt_seed_dir="checkpoints/${name}_seed${seed}"
    ckpt_base_dir="checkpoints/${name}"

    if [ -f "${ckpt_seed_dir}/best_model.pt" ]; then
      ckpt="${ckpt_seed_dir}/best_model.pt"
    elif [ -f "${ckpt_base_dir}/best_model.pt" ]; then
      ckpt="${ckpt_base_dir}/best_model.pt"
      echo "  [WARN] Seed-specific checkpoint not found; using ${ckpt}"
      echo "  [WARN] If you ran multiple seeds this checkpoint may be from seed 44."
    else
      echo "  [ERROR] No checkpoint found for ${name} seed ${seed} — skipping."
      continue
    fi

    # ── Output dir for this run ───────────────────────────────────────────
    out_dir="results/${name}_seed${seed}"
    mkdir -p "$out_dir"

    # ── Write a temp config with the seed overridden (same as training) ───
    tmp="/tmp/${name}_seed${seed}_eval.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

    # ── Run evaluation (test split + stratified) ──────────────────────────
    python3 scripts/evaluate.py \
      --config  "$tmp" \
      --checkpoint "$ckpt" \
      --split   test \
      --stratified \
      --output  "${out_dir}/evaluation_results.json" \
      2>&1 | tee "logs/eval_${name}_seed${seed}.log"

    # ── Optional: zero-shot evaluation ────────────────────────────────────
    if [ -f "data/splits/zero_shot_rare_diseases.json" ]; then
      python3 scripts/evaluate.py \
        --config  "$tmp" \
        --checkpoint "$ckpt" \
        --split   test \
        --zero_shot \
        --zero_shot_file  "data/splits/zero_shot_rare_diseases.json" \
        --zero_shot_output "${out_dir}/zero_shot_results.json" \
        2>&1 | tee -a "logs/eval_${name}_seed${seed}.log"
    fi

    rm -f "$tmp"

    echo "=== Finished ${name} seed ${seed} — results in ${out_dir}/ ==="
  done
done

echo ""
echo "=== All 12 evaluation runs complete ==="
echo ""
echo "Results summary:"
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    out="results/${name}_seed${seed}/evaluation_results.json"
    if [ -f "$out" ]; then
      auroc=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('auroc', 'N/A'):.4f}\")" 2>/dev/null || echo "parse_error")
      hit50=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('hit_rate@50', 'N/A'):.4f}\")" 2>/dev/null || echo "parse_error")
      mrr=$(python3 -c   "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('mrr', 'N/A'):.4f}\")"        2>/dev/null || echo "parse_error")
      echo "  ${name} seed${seed}:  AUROC=${auroc}  Hit@50=${hit50}  MRR=${mrr}"
    else
      echo "  ${name} seed${seed}:  [no results file]"
    fi
  done
done
