#!/usr/bin/env python3
"""
export_promptgfm_rankings.py
Write PromptGFM-Bio's per-disease ranked gene SYMBOLS for the 117 zero-shot
diseases, in the SAME format the external baselines use:

    {disease_id: [ranked_gene_SYMBOLS, best_first]}

so scripts/score_baselines.py can place your model in the cross-method table on
identical footing (Hit@10 / Hit@50 / MRR).

This is the "~30-line script around get_gene_rankings" the Phase 0/1 guide
refers to. It REUSES evaluate.py's exact model builder + dataset loader, so the
architecture and weights match your evaluation runs bit-for-bit, and it passes
the disease id as `disease_text` exactly as evaluate_split() does (the model's
prompt encoder builds the prompt from the id internally).

Run in your PromptGFM venv (Phase 1; L4 or CPU is fine — pure inference):
    python scripts/export_promptgfm_rankings.py \
        --config configs/ablations/ablation_4_full_model.yaml \
        --checkpoint results/ablation_4_full_model_seed42/best_model.pt \
        --zero_shot data/splits/zero_shot_rare_diseases.json \
        --out data/baselines/rankings/promptgfm.json

Notes
-----
* Use the FULL model checkpoint (ablation_4_full_model) — that's the method you
  are comparing against the baselines.
* --top_k caps how many symbols are stored per disease (200 is plenty for
  Hit@50 / MRR; smaller files, identical scores).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse evaluate.py's exact load path (same builder + dataset loader + seeding).
from evaluate import _build_model_from_config, _load_dataset, _set_all_seeds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--zero_shot", default="data/splits/zero_shot_rare_diseases.json")
    ap.add_argument("--out", default="data/baselines/rankings/promptgfm.json")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import yaml
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(args.config))
    _set_all_seeds(config.get("seed", 42))

    # ---- model (same builder evaluate.py uses) ---------------------------- #
    model = _build_model_from_config(config["model"])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    res = model.load_state_dict(state_dict, strict=False)
    if res.missing_keys:
        print(f"[warn] missing keys (random init): {res.missing_keys}")
    model = model.to(device).eval()

    # ---- dataset / graph (same loader evaluate.py uses) ------------------- #
    dataset, *_ = _load_dataset(config)
    graph = dataset.graph
    gene_to_idx = dataset.gene_to_idx
    idx_to_symbol = {v: k for k, v in gene_to_idx.items()}
    num_genes = graph["gene"].num_nodes

    # ---- node features + gene-gene edge_index (mirror evaluate_split) ----- #
    input_dim = config["model"].get("gene_feature_dim", 128)
    if hasattr(graph["gene"], "x") and graph["gene"].x is not None:
        node_features = graph["gene"].x.to(device)
    else:
        torch.manual_seed(42)
        node_features = torch.randn(num_genes, input_dim).to(device)

    edge_index = None
    for et in [("gene", "interacts", "gene"),
               ("gene", "protein_interaction", "gene"),
               ("gene", "ppi", "gene")]:
        if et in graph.edge_types:
            edge_index = graph[et].edge_index.to(device)
            break
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # ---- rank all genes for each zero-shot disease ------------------------ #
    zs_ids = json.load(open(args.zero_shot))["disease_ids"]
    all_idx = torch.arange(num_genes, dtype=torch.long, device=device)
    out = {}
    with torch.no_grad():
        for i, did in enumerate(zs_ids, 1):
            ranked_idx, _ = model.get_gene_rankings(
                node_features, edge_index, did, all_idx, top_k=args.top_k
            )
            out[did] = [idx_to_symbol[int(j)] for j in ranked_idx.tolist()
                        if int(j) in idx_to_symbol]
            if i % 20 == 0:
                print(f"  ranked {i}/{len(zs_ids)} diseases")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[done] wrote {len(out)} diseases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
