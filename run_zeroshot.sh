#!/bin/bash

for cfg in ablation_1_mlp_only ablation_2_prompt_only ablation_3_gnn_only ablation_4_full_model; do
  for seed in 42 43 44; do
    ckpt="checkpoints/${cfg}_seed${seed}/best_model.pt"
    out_dir="results/${cfg}_seed_${seed}"
    out="${out_dir}/zero-shot-eval.json"
    
    mkdir -p "$out_dir" logs

    if [ -f "$out" ]; then
      echo "SKIP ${cfg} seed ${seed} — zero-shot results exist"
      continue
    fi

    if [ ! -f "$ckpt" ]; then
      echo "MISSING CHECKPOINT: $ckpt"
      continue
    fi
    
    echo "=== Zero-shot ${cfg} seed ${seed} ==="
    tmp="/tmp/${cfg}_seed${seed}_zs.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "configs/ablations/${cfg}.yaml" > "$tmp"
    
    python3 scripts/evaluate.py \
      --config "$tmp" \
      --checkpoint "$ckpt" \
      --split test \
      --zero_shot \
      --zero_shot_file data/splits/zero_shot_rare_diseases.json \
      --zero_shot_output "$out" \
      --output /tmp/throwaway.json \
      2>&1 | tee "logs/zeroshot_${cfg}_seed${seed}.log"
    
    rm -f "$tmp"
  done
done

echo ""
echo "=== Zero-shot summary ==="
for cfg in ablation_1_mlp_only ablation_2_prompt_only ablation_3_gnn_only ablation_4_full_model; do
  for seed in 42 43 44; do
    out="results/${cfg}_seed_${seed}/zero-shot-eval.json"
    if [ -f "$out" ]; then
      python3 - <<EOF
import json
with open("$out") as f:
    d = json.load(f)
print(f"{cfg:25s} seed{seed}  Hit@10={d.get('hit_rate@10',0):.4f}  Hit@50={d.get('hit_rate@50',0):.4f}  MRR={d.get('mrr',0):.4f}")
EOF
    fi
  done
done
