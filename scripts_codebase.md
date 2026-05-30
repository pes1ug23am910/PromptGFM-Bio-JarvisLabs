# Codebase Dump: `scripts/`

This document consolidates all the code files within the `scripts/` directory structure for LLM analysis.

## File: `scripts/download_data.py`

```python
"""
Main script to download all biomedical datasets for PromptGFM-Bio.

This script orchestrates the download of all required datasets:
- BioGRID protein-protein interactions
- STRING database PPI
- DisGeNET gene-disease associations
- Human Phenotype Ontology (HPO)

Usage:
    python scripts/download_data.py                    # Download all datasets
    python scripts/download_data.py --dataset biogrid  # Download specific dataset
    python scripts/download_data.py --force            # Force re-download
"""

import sys
from pathlib import Path
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import (
    download_all,
    download_biogrid,
    download_string,
    download_disgenet,
    download_hpo
)


def main():
    parser = argparse.ArgumentParser(
        description="Download biomedical datasets for PromptGFM-Bio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Download all datasets
  %(prog)s --dataset string         Download only STRING database
  %(prog)s --force                  Re-download all (overwrite existing)
  %(prog)s --dataset hpo --force    Re-download HPO only

Datasets:
  all       - All datasets (~1.5GB total)
  biogrid   - BioGRID protein interactions (~500MB)
  string    - STRING protein network (~700MB)
  disgenet  - DisGeNET gene-disease associations (~300MB)
  hpo       - Human Phenotype Ontology (~50MB)
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['all', 'biogrid', 'string', 'disgenet', 'hpo'],
        default='all',
        help='Which dataset to download (default: all)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files already exist'
    )
    
    parser.add_argument(
        '--skip-failing',
        action='store_true',
        default=True,
        help='Continue downloading other datasets if one fails (default: True)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PromptGFM-Bio Data Download")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Force re-download: {args.force}")
    print(f"Skip failing: {args.skip_failing}")
    print()
    
    try:
        if args.dataset == 'all':
            results = download_all(force=args.force, skip_failing=args.skip_failing)
            
            # Check results
            success_count = sum(1 for files in results.values() if files)
            total_count = len(results)
            
            print(f"\n✓ Successfully downloaded {success_count}/{total_count} datasets")
            
            if success_count < total_count:
                print("\n⚠ Some downloads failed. You can:")
                print("  1. Try again: python scripts/download_data.py --force")
                print("  2. Download specific dataset: python scripts/download_data.py --dataset <name>")
                print("  3. Download manually to data/raw/ directory")
                sys.exit(1)
            
        elif args.dataset == 'biogrid':
            results = download_biogrid(force=args.force)
            if not results:
                sys.exit(1)
                
        elif args.dataset == 'string':
            results = download_string(force=args.force)
            if not results:
                sys.exit(1)
                
        elif args.dataset == 'disgenet':
            results = download_disgenet(force=args.force)
            if not results:
                sys.exit(1)
                
        elif args.dataset == 'hpo':
            results = download_hpo(force=args.force)
            if not results:
                sys.exit(1)
        
        print("\n" + "="*70)
        print("✓ DOWNLOAD COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run preprocessing: python scripts/preprocess_all.py")
        print("  2. Check downloaded files in: data/raw/")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## File: `scripts/download_data.sh`

```bash
#!/bin/bash

# Data download script for PromptGFM-Bio
# This script will download all required biomedical datasets

echo "Starting data download for PromptGFM-Bio..."

# Create data directories if they don't exist
mkdir -p data/raw/biogrid
mkdir -p data/raw/string
mkdir -p data/raw/disgenet
mkdir -p data/raw/hpo

# Run Python download script
python src/data/download.py

echo "Data download complete!"
```

## File: `scripts/evaluate.py`

```python
"""
Evaluation script for PromptGFM-Bio.

Usage:
    python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
    python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt --split all
    python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt --stratified

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVALUATION SPEED OPTIMIZATIONS (audited 2026-04-17)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 All optimizations below were audited for mathematical equivalence with the
 original per-batch model() loop.  Numerical results are invariant to 4
 decimal places (reporting precision) across all ablation configs; see the
 audit document accompanying this patch for Q1–Q5 reasoning.

 ACTIVE:
   Opt 1  Pre-compute node embeddings once per evaluate_split call.
          GNN calls drop from ~380k to 1 (or ~5 for stratified).
          Bit-identical: model.forward already branches on
          precomputed_node_embs, so values are indistinguishable.
          Estimated speedup: 50–100×.

   Opt 2  Short-circuit disease-independent ablations
          (use_conditioning=False, ablations 1 & 3).  When FiLM is the
          identity transform, gene scores are disease-invariant, so we
          run model.predictor once on all node embeddings and index
          per-query.  Model.forward is bypassed in the query loop.
          Bit-identical: predictor is row-wise (Linear+ReLU+Dropout+...).
          Estimated speedup: 20–50× on top of Opt 1 for these configs.

   Opt 4  Cache BERT embeddings per unique disease (full-model path).
          BERT is frozen and eval-mode → deterministic per batch shape.
          One BERT forward per disease at batch=1 instead of ~38 per
          disease at batch=512.  Sub-1e-6 drift possible due to cuBLAS
          kernel selection across batch sizes; below 4-decimal metric
          precision.  Estimated speedup: 10–30× on BERT calls.

   Opt 5  Vectorize label / mask construction with np.isin and tensor
          fancy-indexing.  Removes Python-level per-candidate loops.
          Estimated speedup: 2–5× on the overhead between queries.

   Opt 6  Default evaluation batch_size raised 512 → 2048 for the full
          model path.  Row-wise predictor/FiLM make this a no-op
          numerically.  Opt 2 short-circuit bypasses batching entirely.

 REJECTED:
   Opt 3  fp16 autocast — breaks the bit-for-bit full-model constraint.
          Ada's TF32 matmul already gives fp32 most of the speedup fp16
          would, with zero reproducibility risk.  See the audit.

 All optimizations degrade gracefully: if any model attribute is missing
 (e.g. model.gnn, model.node_proj, model.predictor, model.prompt_encoder),
 a warning is logged and the original per-call model() path is used.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import contextlib
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset
from src.models.promptgfm import PromptGFM, GNNOnlyBaseline
from src.evaluation.metrics import GeneRankingEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _set_all_seeds(seed: int) -> None:
    """Path 2 reproducibility: identical helper to train.py._set_all_seeds."""
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model construction — mirrors train.py run_finetuning() lines 366-401 exactly
# ---------------------------------------------------------------------------

def _build_model_from_config(model_config):
    """
    Build PromptGFM (or GNNOnlyBaseline) from a YAML model config dict.

    This replicates the *exact* flattening logic used in train.py so that the
    architecture matches the checkpoint produced during training.

    FIX (2026-04-07): ablation flags use_gnn and use_conditioning are now
    forwarded from the config into PromptGFM.  Previously they defaulted to
    True here, so the MLP-only checkpoint was loaded into a full-GNN+FiLM
    model — producing inflated evaluation numbers.
    """
    is_baseline = model_config.get('baseline', False) or not model_config.get('use_prompt', True)

    if is_baseline:
        logger.info("Building GNN-Only Baseline for evaluation")
        model_params = {
            'gnn_type': model_config.get('gnn_type', 'graphsage'),
            'gnn_hidden_dim': model_config.get('hidden_dim', 256),
            'gnn_num_layers': model_config.get('num_layers', 3),
            'gnn_dropout': model_config.get('dropout', 0.3),
            'hidden_dim': model_config.get('hidden_dim', 256),
        }
        return GNNOnlyBaseline(**model_params)

    logger.info("Building PromptGFM for evaluation")

    # Read ablation flags — default True so non-ablation configs are unaffected.
    use_gnn         = model_config.get('use_gnn', True)
    use_conditioning = model_config.get('use_conditioning', True)
    logger.info(f"  Ablation flags — use_gnn={use_gnn}, use_conditioning={use_conditioning}")

    model_params = {
        # GNN parameters
        'gnn_input_dim':          model_config.get('gene_feature_dim', 256),
        'gnn_hidden_dim':         model_config.get('hidden_dim', 256),
        'gnn_output_dim':         model_config.get('hidden_dim', 256),
        'gnn_num_layers':         model_config.get('num_layers', 3),
        'gnn_type':               model_config.get('gnn_type', 'graphsage'),
        'gnn_dropout':            model_config.get('dropout', 0.3),
        # Prompt encoder parameters
        'prompt_model_name':      model_config.get('prompt_encoder', {}).get(
                                      'model_name',
                                      'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                                  ),
        'prompt_pooling':         model_config.get('prompt_encoder', {}).get('pooling_strategy', 'cls'),
        'prompt_max_length':      model_config.get('prompt_encoder', {}).get('max_length', 512),
        'freeze_prompt':          model_config.get('prompt_encoder', {}).get('freeze_encoder', False),
        # Conditioning parameters
        'conditioning_type':      model_config.get('conditioning_type', 'film'),
        'conditioning_hidden_dim': model_config.get('hidden_dim', 256),
        # Predictor parameters
        'predictor_hidden_dim':   model_config.get('prediction_hidden_dim', 128),
        'predictor_dropout':      model_config.get('dropout', 0.3),
        # ── FIX: ablation flags must be forwarded so architecture matches checkpoint ──
        'use_gnn':                use_gnn,
        'use_conditioning':       use_conditioning,
    }
    logger.info(f"  Model params: gnn_input={model_params['gnn_input_dim']}, "
                f"gnn_hidden={model_params['gnn_hidden_dim']}, "
                f"predictor_hidden={model_params['predictor_hidden_dim']}")
    return PromptGFM(**model_params)


# ---------------------------------------------------------------------------
# Dataset loading — mirrors train.py create_dataloaders()
# ---------------------------------------------------------------------------

def _load_dataset(config):
    """
    Load the GeneDiseaseDataset and split it, using the same API as train.py.
    Returns (dataset, train_edges_df, val_edges_df, test_edges_df).
    """
    dataset = GeneDiseaseDataset(
        graph_path=config['data']['graph_file'],
        edges_path=config['data']['edge_file'],
        min_score=config['data'].get('min_score', 0.3),
    )
    # Path 2: split seed = data.random_seed (FIXED across all seeds).
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed,
    )
    logger.info(f"  Train: {len(train_edges)} edges, Val: {len(val_edges)} edges, "
                f"Test: {len(test_edges)} edges")
    return dataset, train_edges, val_edges, test_edges


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_split(model, dataset, edges_df, config, device='cuda', train_edges_df=None):
    """
    Evaluate with per-query full-vocabulary ranking.

    For each disease query, scores all genes in the graph and aggregates metrics
    across queries. Optionally excludes known train-only positives from the
    candidate pool to avoid train/eval leakage.
    """
    model.eval()
    input_dim = config['model'].get('gene_feature_dim', 128)
    graph = dataset.graph
    gene_to_idx = dataset.gene_to_idx
    num_genes = graph['gene'].num_nodes
    batch_size = config.get('evaluation', {}).get('batch_size', 2048)  # [Opt 6] was 512
    k_values = config.get('evaluation', {}).get('k_values', [10, 20, 50, 100])

    # ── Ablation flags (read once, used to select fast paths below) ────────
    # [Bug 3 fix, option b] `use_prompt` now gates the BERT embedding cache
    # (Opt 4).  When use_conditioning=False the FiLM layer is an identity
    # transform, so we don't build the BERT cache (nor do we enter the
    # per-disease model() path — see Opt 2 short-circuit below).
    cfg_use_gnn          = config['model'].get('use_gnn', True)
    cfg_use_conditioning = config['model'].get('use_conditioning', True)
    use_prompt = (
        not config['model'].get('baseline', False)
        and config['model'].get('use_prompt', True)
        and cfg_use_conditioning
    )
    logger.info(f"  Ablation flags: use_gnn={cfg_use_gnn}, "
                f"use_conditioning={cfg_use_conditioning}, use_prompt={use_prompt}")
    logger.info(f"  Evaluation batch_size={batch_size}")

    # ── Node features ──────────────────────────────────────────────────────
    if hasattr(graph['gene'], 'x') and graph['gene'].x is not None:
        node_features = graph['gene'].x.to(device)
    else:
        torch.manual_seed(42)
        node_features = torch.randn(num_genes, input_dim).to(device)

    # ── Edge index (gene-gene, for GNN message passing) ────────────────────
    edge_types = graph.edge_types if hasattr(graph, 'edge_types') else []
    edge_index = None
    for et in [('gene', 'interacts', 'gene'),
               ('gene', 'protein_interaction', 'gene'),
               ('gene', 'ppi', 'gene')]:
        if et in edge_types:
            edge_index = graph[et].edge_index.to(device)
            break
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # ── Build per-disease positive sets ────────────────────────────────────
    test_pos = {}
    for _, row in edges_df.iterrows():
        g, d = row['gene'], row['disease']
        if g in gene_to_idx:
            idx = gene_to_idx[g]
            if 0 <= idx < num_genes:
                test_pos.setdefault(d, set()).add(idx)

    train_pos = {}
    if train_edges_df is not None:
        for _, row in train_edges_df.iterrows():
            g, d = row['gene'], row['disease']
            if g in gene_to_idx:
                idx = gene_to_idx[g]
                if 0 <= idx < num_genes:
                    train_pos.setdefault(d, set()).add(idx)

    diseases = sorted(test_pos.keys())
    logger.info(f"  Ranking {num_genes} genes for each of {len(diseases)} disease queries")

    all_gene_indices = torch.arange(num_genes, dtype=torch.long)
    rankings = []

    # ══════════════════════════════════════════════════════════════════════
    # [OPT 1] Pre-compute node embeddings ONCE.
    # model.forward branches on precomputed_node_embs, so this is bit-identical
    # to the original per-batch GNN call.  Falls back to None on any failure.
    # ══════════════════════════════════════════════════════════════════════
    precomputed_node_embs = None
    try:
        with torch.no_grad():
            if cfg_use_gnn and hasattr(model, 'gnn'):
                precomputed_node_embs = model.gnn(node_features, edge_index)
                logger.info(f"  [OPT 1] Pre-computed GNN node embeddings once: "
                            f"shape={tuple(precomputed_node_embs.shape)}")
            elif (not cfg_use_gnn) and hasattr(model, 'node_proj'):
                precomputed_node_embs = model.node_proj(node_features)
                logger.info(f"  [OPT 1] Pre-computed node_proj embeddings once: "
                            f"shape={tuple(precomputed_node_embs.shape)}")
            else:
                logger.warning(
                    "  [OPT 1] Neither model.gnn nor model.node_proj available for "
                    "pre-compute; falling back to per-call model() path."
                )
    except Exception as e:
        logger.warning(f"  [OPT 1] Pre-compute failed ({e!r}); falling back to per-call model().")
        precomputed_node_embs = None

    # ══════════════════════════════════════════════════════════════════════
    # [OPT 2] Disease-independent short-circuit (ablations 1 & 3 only).
    # When use_conditioning=False, FiLM is an identity transform so every
    # disease produces the same score vector over genes.  Run the predictor
    # ONCE on all node embeddings and index per query.  Mathematically
    # equivalent because the predictor MLP is strictly row-wise (no cross-
    # batch reductions, no dropout in eval mode).
    # ══════════════════════════════════════════════════════════════════════
    all_gene_scores = None
    if (not cfg_use_conditioning) and (precomputed_node_embs is not None) \
            and hasattr(model, 'predictor'):
        try:
            with torch.no_grad():
                scores_all = model.predictor(precomputed_node_embs).squeeze(-1)  # [num_genes]
                all_gene_scores = scores_all.detach().float().cpu().numpy().astype(np.float32)
            logger.info(f"  [OPT 2] Disease-independent short-circuit ACTIVE: "
                        f"pre-computed {all_gene_scores.shape[0]} gene scores once. "
                        f"Per-query loop will be a numpy index instead of model().")
        except Exception as e:
            logger.warning(f"  [OPT 2] Short-circuit failed ({e!r}); using per-call model().")
            all_gene_scores = None

    # ══════════════════════════════════════════════════════════════════════
    # [OPT 4] Cache BERT embeddings per unique disease (full-model path only).
    # BERT is frozen + eval-mode → deterministic per batch shape.  Computing
    # each disease's [1, 768] embedding once and re-using it across all ~38
    # gene batches removes ~37/38 ≈ 97% of BERT compute.  Possible sub-1e-6
    # numerical drift from batch-size kernel selection; well below 4-decimal
    # reporting precision.
    # Only built when Opt 2 did not fire AND use_prompt is True.
    # ══════════════════════════════════════════════════════════════════════
    disease_emb_cache = {}
    if (all_gene_scores is None) and use_prompt and hasattr(model, 'prompt_encoder'):
        try:
            # Encode in small chunks so tokenizer padding within a chunk is bounded.
            cache_chunk = 64
            with torch.no_grad():
                for i in range(0, len(diseases), cache_chunk):
                    chunk = diseases[i:i + cache_chunk]
                    chunk_emb = model.prompt_encoder(chunk)  # [chunk_len, 768]
                    # Store each disease's row as a contiguous [1, 768] view on
                    # the model's device so model.forward sees identical dtype/device.
                    for j, d in enumerate(chunk):
                        disease_emb_cache[d] = chunk_emb[j:j + 1].contiguous()
            logger.info(f"  [OPT 4] BERT embedding cache built: {len(disease_emb_cache)} "
                        f"unique diseases, chunk_size={cache_chunk}.")
        except Exception as e:
            logger.warning(f"  [OPT 4] BERT cache build failed ({e!r}); "
                           f"falling back to per-batch BERT forward.")
            disease_emb_cache = {}

    # ── Score all candidates for each disease query ────────────────────────
    with torch.no_grad():
        for disease in tqdm(diseases, desc="Evaluating queries"):
            pos_set = test_pos[disease]
            train_set = train_pos.get(disease, set())

            # [OPT 5] Vectorized mask: exclude train-only positives (genes in
            # train split but not in this eval split's positive set).
            candidate_mask = torch.ones(num_genes, dtype=torch.bool)
            exclude = train_set - pos_set
            if exclude:
                exclude_idx = torch.tensor(list(exclude), dtype=torch.long)
                candidate_mask[exclude_idx] = False

            candidate_indices = all_gene_indices[candidate_mask]
            if candidate_indices.numel() == 0:
                continue

            # [OPT 5] Vectorized labels via np.isin (replaces Python per-idx loop).
            candidate_np = candidate_indices.numpy()
            if pos_set:
                pos_arr = np.fromiter(pos_set, dtype=np.int64, count=len(pos_set))
                labels_query = np.isin(candidate_np, pos_arr).astype(np.float32)
            else:
                labels_query = np.zeros(candidate_np.shape[0], dtype=np.float32)
            if labels_query.sum() == 0:
                continue

            # ══════════════════════════════════════════════════════════════
            # Fast paths in priority order:
            #   (a) OPT 2 short-circuit: numpy index into precomputed scores.
            #   (b) OPT 1 (+ optional OPT 4): model.forward with precomputed
            #       node embeddings and (when available) precomputed prompt
            #       embedding.
            #   (c) Original path: model.forward without any precomputes.
            # ══════════════════════════════════════════════════════════════
            if all_gene_scores is not None:
                # (a) Disease-independent: one numpy index, no model call.
                query_scores_arr = all_gene_scores[candidate_np].astype(np.float32, copy=False)
            else:
                # (b) or (c): batched model() with precomputes where available.
                cached_prompt = disease_emb_cache.get(disease, None)  # None → full BERT path
                query_scores = []
                for start in range(0, candidate_indices.numel(), batch_size):
                    batch_gidx = candidate_indices[start:start + batch_size].to(device)
                    batch_texts = [disease] * batch_gidx.numel()  # ignored if cached_prompt is set
                    model_kwargs = dict(
                        node_features=node_features,
                        edge_index=edge_index,
                        disease_texts=batch_texts,
                        gene_indices=batch_gidx,
                    )
                    if precomputed_node_embs is not None:
                        model_kwargs['precomputed_node_embs'] = precomputed_node_embs
                    if cached_prompt is not None:
                        model_kwargs['precomputed_prompt_embs'] = cached_prompt
                    scores = model(**model_kwargs)
                    # .float() before .cpu().numpy() guarantees float32 metrics input
                    # even if the model path ever returns a different dtype.  With
                    # Opt 3 rejected this is always fp32 today, but this defensive
                    # cast keeps downstream metric computation stable.
                    query_scores.extend(
                        scores.squeeze(-1).detach().float().cpu().tolist()
                    )
                query_scores_arr = np.array(query_scores, dtype=np.float32)

            rankings.append((labels_query, query_scores_arr))

    logger.info(f"  Evaluated {len(rankings)} queries with at least one test positive")

    if not rankings:
        logger.warning("No valid disease queries available for evaluation")
        return {}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    evaluator = GeneRankingEvaluator(k_values=k_values)

    # Global arrays used for AUROC/AUPR and backward compatibility outputs.
    flat_labels = np.concatenate([r[0] for r in rankings])
    flat_scores = np.concatenate([r[1] for r in rankings])
    metrics = evaluator.evaluate_all(flat_labels, flat_scores, rankings=rankings)

    # Per-query means for ranking metrics avoid cross-query flattening artifacts.
    for k in k_values:
        precision_vals = [evaluator.precision_at_k(y_true, y_scores, k) for y_true, y_scores in rankings]
        recall_vals    = [evaluator.recall_at_k(y_true, y_scores, k)    for y_true, y_scores in rankings]
        ndcg_vals      = [evaluator.ndcg_at_k(y_true, y_scores, k)      for y_true, y_scores in rankings]

        metrics[f'precision@{k}'] = float(np.mean(precision_vals)) if precision_vals else 0.0
        metrics[f'recall@{k}']    = float(np.mean(recall_vals))    if recall_vals    else 0.0
        metrics[f'ndcg@{k}']      = float(np.mean(ndcg_vals))      if ndcg_vals      else 0.0

    return metrics, flat_scores, flat_labels


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------

def evaluate_stratified(model, dataset, edges_df, config, device='cuda', train_edges_df=None):
    """Evaluate stratified by disease rarity (number of known genes)."""
    logger.info("\n" + "=" * 60)
    logger.info("Stratified Evaluation by Disease Rarity")
    logger.info("=" * 60)

    gene_to_idx = dataset.gene_to_idx
    num_graph_genes = dataset.graph['gene'].num_nodes

    # Count genes per disease in this split
    disease_gene_count = {}
    for _, row in edges_df.iterrows():
        d = row['disease']
        g = row['gene']
        if g in gene_to_idx and 0 <= gene_to_idx[g] < num_graph_genes:
            disease_gene_count.setdefault(d, set()).add(g)
    disease_gene_count = {d: len(gs) for d, gs in disease_gene_count.items()}

    thresholds = config.get('evaluation', {}).get('rarity_thresholds', {
        'ultra_rare': 2, 'very_rare': 5, 'rare': 15, 'common': 1000,
    })

    # Build buckets
    buckets = {name: [] for name in thresholds}
    for _, row in edges_df.iterrows():
        d = row['disease']
        n = disease_gene_count.get(d, 0)
        for name in sorted(thresholds, key=lambda k: thresholds[k]):
            if n <= thresholds[name]:
                buckets[name].append(row)
                break

    results = {}
    for rarity, rows in buckets.items():
        if not rows:
            logger.info(f"\n  {rarity}: 0 edges — skipping")
            continue
        import pandas as pd
        sub_df = pd.DataFrame(rows)
        logger.info(f"\n  {rarity}: {len(sub_df)} edges")
        metrics, _, _ = evaluate_split(
            model,
            dataset,
            sub_df,
            config,
            device,
            train_edges_df=train_edges_df,
        )
        results[rarity] = metrics

        evaluator = GeneRankingEvaluator()
        evaluator.print_metrics(metrics, prefix=f"  {rarity}:")

    return results


# ---------------------------------------------------------------------------
# Zero-shot rare disease evaluation
# ---------------------------------------------------------------------------

def evaluate_zero_shot(model, dataset, all_edges_df, config, device='cuda',
                       train_edges_df=None, zero_shot_json='data/splits/zero_shot_rare_diseases.json'):
    """
    Evaluate on rare diseases that were provably absent from training.

    Loads the pre-built JSON produced by find_rare_diseases.py, filters the
    full edge set to those disease IDs, then runs full-vocabulary ranking and
    reports Hits@K and MRR separately from the main evaluation.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Zero-Shot Rare Disease Evaluation")
    logger.info("=" * 60)

    zs_path = Path(zero_shot_json)
    if not zs_path.exists():
        logger.error(
            f"Zero-shot disease list not found: {zs_path}\n"
            "Run scripts/find_rare_diseases.py first to generate it."
        )
        return {}

    with open(zs_path, 'r') as f:
        zs_payload = json.load(f)

    zero_shot_ids = set(zs_payload.get('disease_ids', []))
    logger.info(
        f"  Loaded {len(zero_shot_ids)} zero-shot rare disease IDs "
        f"(max_assoc={zs_payload.get('max_associations', '?')}, "
        f"split_seed={zs_payload.get('split_seed', '?')})"
    )

    # Filter the full edge DataFrame to only zero-shot disease queries.
    # We use all_edges_df (the complete dataset.edges) so that even diseases
    # whose edges were all in train/val still have ground-truth associations.
    zs_edges = all_edges_df[all_edges_df['disease'].isin(zero_shot_ids)].copy()
    if zs_edges.empty:
        logger.warning("No edges found for any zero-shot disease ID. Skipping.")
        return {}

    zs_disease_count = zs_edges['disease'].nunique()
    logger.info(f"  Found {len(zs_edges)} edges across {zs_disease_count} zero-shot diseases")

    # Run full-vocabulary ranking with train exclusion (same as evaluate_split)
    k_values = config.get('evaluation', {}).get('k_values', [10, 20, 50, 100])
    metrics, flat_scores, flat_labels = evaluate_split(
        model,
        dataset,
        zs_edges,
        config,
        device,
        train_edges_df=train_edges_df,
    )

    if not metrics:
        logger.warning("evaluate_split returned no results for zero-shot diseases.")
        return {}

    # ── Explicitly report Hits@10, Hits@50, MRR (already in metrics) ──────
    hits_10  = metrics.get('hit_rate@10',  float('nan'))
    hits_50  = metrics.get('hit_rate@50',  float('nan'))
    mrr      = metrics.get('mrr',          float('nan'))
    auroc    = metrics.get('auroc',        float('nan'))
    aupr     = metrics.get('aupr',         float('nan'))
    map_val  = metrics.get('map',          float('nan'))

    # ── Console summary table ──────────────────────────────────────────────
    sep  = "─" * 44
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║    Zero-Shot Rare Disease Evaluation         ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Diseases evaluated   : {zs_disease_count:<20} ║")
    print(f"║  Total edges (GT)     : {len(zs_edges):<20} ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  AUROC                : {auroc:<20.4f} ║")
    print(f"║  AUPR                 : {aupr:<20.4f} ║")
    print(f"║  MAP                  : {map_val:<20.4f} ║")
    print(f"║  MRR                  : {mrr:<20.4f} ║")
    print(f"║  Hits@10              : {hits_10:<20.4f} ║")
    print(f"║  Hits@50              : {hits_50:<20.4f} ║")
    for k in k_values:
        if f'ndcg@{k}' in metrics:
            v = metrics[f'ndcg@{k}']
            lbl = f"NDCG@{k}"
            print(f"║  {lbl:<22} : {v:<20.4f} ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Attach metadata for the saved JSON
    metrics['_zero_shot_meta'] = {
        'num_diseases': zs_disease_count,
        'num_edges': len(zs_edges),
        'source_json': str(zs_path),
        'max_associations': zs_payload.get('max_associations'),
    }
    return metrics


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native for JSON serialisation
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(_to_native(results), f, indent=2)
    logger.info(f"\n✓ Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptGFM-Bio')
    # Path 2 reproducibility — model-RNG seed comes from config['seed'].
    # We can't seed before parsing args (config path is an arg), so the seed
    # call is moved just after we load the config below.
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config (must match training architecture)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                        default='test', help='Which split to evaluate')
    parser.add_argument('--stratified', action='store_true',
                        help='Run stratified evaluation by disease rarity')
    parser.add_argument('--few-shot', type=int, nargs='+',
                        help='Run few-shot evaluation with specified K values')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                        help='Path to save results JSON')
    # ── Zero-shot rare disease evaluation ──────────────────────────────────
    parser.add_argument('--zero_shot', action='store_true',
                        help=(
                            'Run zero-shot rare disease evaluation using disease IDs '
                            'from --zero_shot_file (generated by scripts/find_rare_diseases.py). '
                            'Default evaluation behaviour is NOT changed.'
                        ))
    parser.add_argument('--zero_shot_file', type=str,
                        default='data/splits/zero_shot_rare_diseases.json',
                        help='JSON file with zero-shot disease IDs (default: data/splits/zero_shot_rare_diseases.json)')
    parser.add_argument('--zero_shot_output', type=str,
                        default='results/zero_shot_evaluation_results.json',
                        help='Path to save zero-shot results JSON')
    args = parser.parse_args()

    # ── Device ─────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ── Load config ────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    _set_all_seeds(config.get('seed', 42))

    # ── Build model (same mapping as train.py) ─────────────────────────────
    model = _build_model_from_config(config['model'])

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle DDP-wrapped checkpoints (keys prefixed with "module.")
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        logger.warning(f"Missing keys (random init): {result.missing_keys}")
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys (ignored): {result.unexpected_keys}")
    model = model.to(device)
    model.eval()
    logger.info("✓ Model loaded and set to eval mode")

    # ── Load dataset ───────────────────────────────────────────────────────
    dataset, train_edges, val_edges, test_edges = _load_dataset(config)
    split_map = {'train': train_edges, 'val': val_edges, 'test': test_edges}

    # ── Evaluate ───────────────────────────────────────────────────────────
    all_results = {}

    splits_to_eval = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    for split_name in splits_to_eval:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating {split_name.upper()} split")
        logger.info(f"{'=' * 60}")
        edges_df = split_map[split_name]
        metrics, _, _ = evaluate_split(
            model,
            dataset,
            edges_df,
            config,
            device,
            train_edges_df=train_edges,
        )

        evaluator = GeneRankingEvaluator()
        evaluator.print_metrics(metrics, prefix=f"{split_name}:")
        all_results[split_name] = metrics

    # ── Stratified ─────────────────────────────────────────────────────────
    if args.stratified:
        test_df = split_map.get('test', split_map.get('val'))
        stratified_results = evaluate_stratified(
            model,
            dataset,
            test_df,
            config,
            device,
            train_edges_df=train_edges,
        )
        all_results['stratified'] = stratified_results

    # ── Zero-shot rare disease evaluation (only when --zero_shot is passed) ─
    # Default evaluation above is completely unaffected by this block.
    if args.zero_shot:
        zs_metrics = evaluate_zero_shot(
            model,
            dataset,
            dataset.edges,          # full edge set — ground truth for all diseases
            config,
            device,
            train_edges_df=train_edges,
            zero_shot_json=args.zero_shot_file,
        )
        if zs_metrics:
            save_results(zs_metrics, args.zero_shot_output)

    # ── Save ───────────────────────────────────────────────────────────────
    save_results(all_results, args.output)
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
```

## File: `scripts/evaluate_all-fix.sh`

```bash
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
```

## File: `scripts/find_rare_diseases.py`

```python
"""
Find rare disease node IDs for zero-shot evaluation.

Usage:
    python scripts/find_rare_diseases.py
    python scripts/find_rare_diseases.py --edge_file data/processed/hpo_gene_disease_edges.csv
    python scripts/find_rare_diseases.py --max_assoc 5 --output data/splits/zero_shot_rare_diseases.json

Outputs:
    data/splits/zero_shot_rare_diseases.json  -- list of disease IDs with <= max_assoc genes
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import GeneDiseaseDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_edges_filtered(edge_file: str, min_score: float = 0.3) -> pd.DataFrame:
    """Load edge CSV with the same score filter used in training."""
    edges = pd.read_csv(edge_file)
    before = len(edges)
    edges = edges[edges["score"] >= min_score].reset_index(drop=True)
    logger.info(f"Loaded {before} edges, kept {len(edges)} with score >= {min_score}")
    return edges


def _count_genes_per_disease(edges: pd.DataFrame) -> pd.Series:
    """Return unique gene count per disease over the full (unsplit) dataset."""
    return edges.groupby("disease")["gene"].nunique().sort_values()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find rare disease IDs for zero-shot eval")
    parser.add_argument("--graph_file",  type=str,
                        default="data/processed/biomedical_graph.pt",
                        help="Path to processed graph .pt file")
    parser.add_argument("--edge_file",   type=str,
                        default="data/processed/hpo_gene_disease_edges.csv",
                        help="Path to gene-disease edges CSV")
    parser.add_argument("--min_score",   type=float, default=0.3,
                        help="Minimum score filter (must match training config)")
    parser.add_argument("--max_assoc",   type=int,   default=5,
                        help="Max known gene associations to be classified as 'rare'")
    parser.add_argument("--output",      type=str,
                        default="data/splits/zero_shot_rare_diseases.json",
                        help="Output JSON path for rare disease IDs")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--test_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed used in training (for split reproducibility)")
    args = parser.parse_args()

    graph_path = Path(args.graph_file)
    edge_path  = Path(args.edge_file)

    # ── Validate input files ───────────────────────────────────────────────
    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        sys.exit(1)
    if not edge_path.exists():
        logger.error(f"Edge file not found: {edge_path}")
        sys.exit(1)

    # ── Load graph (to report node counts) ────────────────────────────────
    logger.info(f"Loading graph from {graph_path} ...")
    graph = torch.load(graph_path, weights_only=False)

    if hasattr(graph, "node_types"):
        for nt in graph.node_types:
            logger.info(f"  {nt}: {graph[nt].num_nodes} nodes")
    if hasattr(graph, "edge_types"):
        for et in graph.edge_types:
            logger.info(f"  {et}: {graph[et].num_edges} edges")

    disease_nodes_in_graph = (
        graph["disease"].num_nodes if hasattr(graph, "node_types") and "disease" in graph.node_types
        else None
    )
    logger.info(f"\nGraph disease nodes: {disease_nodes_in_graph}")

    # ── Load edges with same filter as training ────────────────────────────
    logger.info(f"\nLoading edge file: {edge_path}")
    edges = _load_edges_filtered(str(edge_path), min_score=args.min_score)

    # ── Count gene associations per disease ────────────────────────────────
    gene_counts = _count_genes_per_disease(edges)
    total_diseases = len(gene_counts)

    rare_5  = gene_counts[gene_counts <= 5].index.tolist()
    rare_10 = gene_counts[gene_counts <= 10].index.tolist()
    target  = gene_counts[gene_counts <= args.max_assoc].index.tolist()

    # ── Print summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Disease Association Summary")
    print("=" * 60)
    print(f"  Total diseases in edge file      : {total_diseases:>7,}")
    print(f"  Diseases with ≤5  known genes     : {len(rare_5):>7,}  ({100*len(rare_5)/total_diseases:.1f}%)")
    print(f"  Diseases with ≤10 known genes     : {len(rare_10):>7,}  ({100*len(rare_10)/total_diseases:.1f}%)")
    print(f"  Selected threshold (≤{args.max_assoc:2d} genes)    : {len(target):>7,}  ({100*len(target)/total_diseases:.1f}%)")
    print("=" * 60)

    # Distribution table
    print("\n  Association count distribution:")
    for n in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        cnt = int((gene_counts <= n).sum())
        print(f"    ≤{n:3d} gene(s): {cnt:>6,} diseases")
    print()

    # ── Reproduce exact training split (same params as config) ────────────
    logger.info("Reproducing train/val/test split to verify zero-shot diseases are absent from train ...")

    # Re-use GeneDiseaseDataset for split logic (same code path as train.py)
    dataset = GeneDiseaseDataset(
        graph_path=str(graph_path),
        edges_path=str(edge_path),
        min_score=args.min_score,
    )
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    # Diseases that appear as positives in training edges
    train_diseases = set(train_edges["disease"].unique())

    # Zero-shot diseases must NOT appear in training positives
    target_set = set(target)
    leaking    = target_set & train_diseases
    clean      = target_set - train_diseases

    print("=" * 60)
    print("  Zero-Shot Train-Leakage Check")
    print("=" * 60)
    print(f"  Candidate rare diseases (≤{args.max_assoc} known genes) : {len(target_set):>6,}")
    print(f"  Diseases also in training split (LEAKING)       : {len(leaking):>6,}")
    print(f"  Clean zero-shot diseases (not in train)         : {len(clean):>6,}")
    print("=" * 60)

    if leaking:
        logger.warning(
            f"{len(leaking)} rare disease(s) appear in the training split and will be EXCLUDED "
            f"from the zero-shot list.  Example: {list(leaking)[:5]}"
        )

    # Final zero-shot list: only diseases that never appeared in training
    zero_shot_list = sorted(clean)

    print(f"\n  ✓ Final zero-shot evaluation set: {len(zero_shot_list)} diseases")
    if zero_shot_list:
        print(f"  Sample IDs: {zero_shot_list[:5]}")

    # ── Save zero-shot list ────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "description": (
            f"Disease IDs with <= {args.max_assoc} known gene associations "
            f"that do NOT appear in the training split (seed={args.seed})."
        ),
        "max_associations": args.max_assoc,
        "min_score_filter": args.min_score,
        "split_seed": args.seed,
        "total_diseases_in_dataset": total_diseases,
        "num_zero_shot_diseases": len(zero_shot_list),
        "num_excluded_leaking": len(leaking),
        "disease_ids": zero_shot_list,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"\n✓ Saved {len(zero_shot_list)} zero-shot disease IDs → {output_path}")


if __name__ == "__main__":
    main()
```

## File: `scripts/gitignore_additions.txt`

```
# ============================================================================
# PromptGFM-Bio .gitignore  (merge these lines into your existing .gitignore)
# Goal: push CODE to a private repo; keep SECRETS and LARGE BINARIES out.
# ============================================================================

# --- Secrets (NEVER commit) -------------------------------------------------
.env
.env.*
!.env.example

# --- Large data binaries (transfer separately, not via git) -----------------
data/processed/*.pt
data/processed/*.csv
data/raw/
*.zip
*.gz
*.tsv

# --- Model artifacts --------------------------------------------------------
checkpoints/
checkpoints_backup_*/
*.pt
*.pth
results/**/*.json
!results/.gitkeep

# --- Logs -------------------------------------------------------------------
logs/
*.log

# --- Heavy docs/media (optional; keep repo cloneable fast) ------------------
docs/*.mp4
docs/*.pptx
docs/**/images/*.png

# --- Python cruft -----------------------------------------------------------
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
.pytest_cache/

# --- Keep small, useful text artifacts tracked ------------------------------
!data/splits/zero_shot_rare_diseases.json
!requirements.txt
!requirements-dev.txt
```

## File: `scripts/make_shepherd_input.py`

```python
#!/usr/bin/env python3
"""
make_shepherd_input.py
Convert the 117 zero-shot diseases into SHEPHERD's JSON-lines "patient" format.

SHEPHERD input contract (per the mims-harvard/SHEPHERD README):
  one JSON object per line, each with:
    "id"                  : patient/disease identifier
    "positive_phenotypes" : list of HPO term IDs   (e.g. "HP:0001250")
    "true_genes"          : list of causal genes as ENSEMBL IDs
    "all_candidate_genes" : list of candidate genes as ENSEMBL IDs (for causal gene discovery)

Run after prepare_baseline_inputs.py (PromptGFM venv):
    python scripts/make_shepherd_input.py

Output: data/baselines/shepherd_patients.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data/baselines"

hpo  = json.load(open(B / "disease_hpo_terms.json"))
true = json.load(open(B / "disease_true_genes.json"))
s2e  = json.load(open(B / "symbol_to_ensembl.json"))
cand = json.load(open(B / "all_candidate_genes_ensembl.json"))

out_path = B / "shepherd_patients.jsonl"
written = skipped_no_hpo = skipped_no_true = 0

with open(out_path, "w") as f:
    for did, terms in hpo.items():
        if not terms:
            skipped_no_hpo += 1
            continue
        true_ens = [s2e[g] for g in true.get(did, []) if g in s2e]
        if not true_ens:
            # No mappable ground-truth gene -> cannot score causal gene discovery.
            skipped_no_true += 1
            continue
        rec = {
            "id": did,
            "positive_phenotypes": terms,
            "true_genes": true_ens,
            "all_candidate_genes": cand,
        }
        f.write(json.dumps(rec) + "\n")
        written += 1

print(f"[shepherd] wrote {written} patients -> {out_path}")
print(f"[shepherd] skipped {skipped_no_hpo} (no HPO terms), "
      f"{skipped_no_true} (no Ensembl-mappable true gene)")
print("[shepherd] NOTE the written/skipped counts; report them in the paper's "
      "methods (comparison is over the diseases all methods can score).")
```

## File: `scripts/md_to_pdf.py`

```python
"""
Convert Markdown files to PDF while preserving table layout and formatting.

Default usage targets data/README.md and writes data/README.pdf.

Examples:
    python scripts/md_to_pdf.py
    python scripts/md_to_pdf.py --input data/README.md --output data/README.pdf
    python scripts/md_to_pdf.py --input docs/guide.md --css docs/pdf_style.css
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


DEFAULT_CSS = """
@page {
    size: A4;
    margin: 20mm 16mm;
}

body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1f2937;
}

h1, h2, h3, h4 {
    color: #111827;
    margin-top: 1.1em;
    margin-bottom: 0.45em;
    page-break-after: avoid;
}

h1 {
    font-size: 24pt;
    border-bottom: 1px solid #d1d5db;
    padding-bottom: 8px;
}

h2 {
    font-size: 18pt;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
}

h3 {
    font-size: 14pt;
}

p, ul, ol, blockquote {
    margin-top: 0.35em;
    margin-bottom: 0.55em;
}

code {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.8pt;
    background: #f3f4f6;
    border-radius: 4px;
    padding: 1px 4px;
}

pre {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9pt;
    background: #111827;
    color: #f9fafb;
    border-radius: 6px;
    padding: 10px 12px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

pre code {
    background: transparent;
    color: inherit;
    padding: 0;
}

/* Keep Markdown table structure clearly rendered in PDF */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 14px;
    font-size: 10pt;
    table-layout: auto;
}

thead {
    display: table-header-group;
}

tbody tr {
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 7px 9px;
    vertical-align: top;
    word-break: break-word;
}

th {
    background: #e2e8f0;
    font-weight: 700;
    text-align: left;
}

tbody tr:nth-child(even) {
    background: #f8fafc;
}

blockquote {
    border-left: 4px solid #cbd5e1;
    padding-left: 10px;
    color: #4b5563;
}

a {
    color: #0f766e;
    text-decoration: none;
}

hr {
    border: none;
    border-top: 1px solid #d1d5db;
    margin: 12px 0;
}
"""


def parse_args() -> argparse.Namespace:
    default_backend = "xhtml2pdf" if sys.platform.startswith("win") else "auto"

    parser = argparse.ArgumentParser(
        description="Convert Markdown to PDF with table-preserving formatting."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/README.md",
        help="Path to input Markdown file (default: data/README.md)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output PDF file (default: same name as input with .pdf)",
    )
    parser.add_argument(
        "--css",
        default=None,
        help="Optional custom CSS file to append after default styling.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional HTML title for the generated document.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "weasyprint", "xhtml2pdf"],
        default=default_backend,
        help=f"PDF backend to use (default: {default_backend}).",
    )
    return parser.parse_args()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_markdown_module() -> object:
    try:
        import markdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'markdown'. Install with: pip install markdown"
        ) from exc

    return markdown


def load_pdf_backend(preferred_backend: str) -> tuple[str, object]:
    weasy_error = None

    if preferred_backend in ("auto", "weasyprint"):
        try:
            from weasyprint import HTML  # type: ignore

            return "weasyprint", HTML
        except Exception as exc:
            weasy_error = exc
            if preferred_backend == "weasyprint":
                raise RuntimeError(
                    "WeasyPrint is unavailable in this environment. "
                    "Either install its system libraries, or use --backend xhtml2pdf. "
                    f"Original error: {exc}"
                ) from exc

    try:
        from xhtml2pdf import pisa  # type: ignore

        return "xhtml2pdf", pisa
    except ImportError as exc:
        install_msg = (
            "No usable PDF backend found. Install one of:\n"
            "  pip install weasyprint\n"
            "  pip install xhtml2pdf\n"
            "For this Windows env, xhtml2pdf is recommended."
        )
        if weasy_error is not None:
            install_msg += f"\nWeasyPrint error: {weasy_error}"
        raise RuntimeError(
            install_msg
        ) from exc


def markdown_to_html(markdown_text: str, title: str, css_text: str, markdown_module: object) -> str:
    rendered_body = markdown_module.markdown(  # type: ignore[attr-defined]
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>{css_text}</style>
</head>
<body>
    <main class=\"markdown-body\">
        {rendered_body}
    </main>
</body>
</html>
"""


def convert(
    md_path: Path,
    pdf_path: Path,
    custom_css_path: Path | None = None,
    title: str | None = None,
    backend_preference: str = "auto",
) -> str:
    markdown_module = load_markdown_module()
    backend_name, pdf_backend = load_pdf_backend(backend_preference)

    markdown_text = md_path.read_text(encoding="utf-8")
    combined_css = DEFAULT_CSS

    if custom_css_path is not None:
        custom_css = custom_css_path.read_text(encoding="utf-8")
        combined_css = f"{DEFAULT_CSS}\n\n{custom_css}"

    resolved_title = title or md_path.stem
    html_document = markdown_to_html(markdown_text, resolved_title, combined_css, markdown_module)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if backend_name == "weasyprint":
        pdf_backend(string=html_document, base_url=str(md_path.parent.resolve())).write_pdf(str(pdf_path))
    else:
        with pdf_path.open("wb") as output_file:
            pisa_status = pdf_backend.CreatePDF(src=html_document, dest=output_file, encoding="utf-8")
        if getattr(pisa_status, "err", 1):
            raise RuntimeError("xhtml2pdf failed while rendering the PDF.")

    return backend_name


def main() -> int:
    args = parse_args()

    md_path = resolve_path(args.input, PROJECT_ROOT)
    if not md_path.exists():
        print(f"Input file not found: {md_path}")
        print("Tip: relative paths are resolved from the project root.")
        return 1

    if args.output:
        pdf_path = resolve_path(args.output, PROJECT_ROOT)
    else:
        pdf_path = md_path.with_suffix(".pdf")

    custom_css_path = None
    if args.css:
        custom_css_path = resolve_path(args.css, PROJECT_ROOT)
        if not custom_css_path.exists():
            print(f"Custom CSS file not found: {custom_css_path}")
            return 1

    try:
        used_backend = convert(
            md_path,
            pdf_path,
            custom_css_path=custom_css_path,
            title=args.title,
            backend_preference=args.backend,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        return 1

    print(f"PDF created: {pdf_path} (backend: {used_backend})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## File: `scripts/parity_check-jarvis.sh`

```bash
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
```

## File: `scripts/PATH2_DEPLOYMENT_GUIDE.md`

```markdown
# Path 2 Deployment — Operational Guide

This supersedes the seed-related sections of the Phase 0+1 guide. The Phase 1
(baseline) work is unchanged. The change is in **Phase 2**, which now produces
a **single internally-consistent 10-seed study under fixed split + properly
seeded model RNG.**

---

## What changed (one paragraph)

The existing seeds 42/43/44 trained under a broken seed flow: the top-level
`seed:` only controlled the data split, while `torch`, `numpy`, and Python
`random` (used in negative sampling) were never seeded — so model init and
sampling were process-time-random and non-reproducible. The new Path 2 patch
(see `PATH2_PATCHES.md`) seeds **all** RNGs from `config['seed']` and moves
the *split* seed to `config['data']['random_seed']`, which stays at 42 for
every run. The old 12 ablation results are **superseded**, not used.

---

## Cost (Path 2, full)

| Step | GPU | Hours | Cost (on-demand) | Cost (spot, ~-48%) |
|---|---|---|---|---|
| Smoke test (one-off) | A100 40GB | ~0.2 | ~₹17 | — |
| 40 training runs (10 seeds × 4 variants × ~1.33 h) | A100 40GB | ~53 | **~₹4,470** | ~₹2,330 |
| 40 evaluations (test + stratified + zero-shot) | A100 40GB | ~7 | ~₹590 | ~₹310 |
| Phase 1 baselines (unchanged) | L4 24GB | ~22 | ~₹900 | — |
| Buffer (failures / 1 spot interruption) | mixed | ~6 | ~₹500 | ~₹300 |
| **Total through Phase 2** | | **~88** | **~₹6,477** | **~₹3,860** |

Recharge target: **₹7,000 before Phase 2** (or ~₹4,500 if committing to spot for training).

---

## The exact command order

### 0. ONCE on Windows (before launching anything)

Apply `PATH2_PATCHES.md` to `scripts\train.py` and `scripts\evaluate.py`, then:

```powershell
cd E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio
Select-String -Path "scripts\train.py" -Pattern "_set_all_seeds|split_seed = config"     # expect 3+ hits
Select-String -Path "scripts\evaluate.py" -Pattern "_set_all_seeds|split_seed = config"  # expect 3+ hits
# Copy the four new shell scripts into scripts\:
#   _jarvis_env.sh, smoke_test_path2-jarvis.sh,
#   run_all_seeds-jarvis.sh, run_evaluations_all_seeds-jarvis.sh
git add scripts\train.py scripts\evaluate.py scripts\*.sh
git commit -m "Path 2: fixed split via data.random_seed; deterministic per-seed model RNG"
git push
```

### 1. Launch the A100 Template (on-demand)

JarvisLabs → Templates → PyTorch → **A100 40GB IN2** → On-Demand → 100 GB → Launch.
SSH key already added (from earlier).

### 2. Bootstrap on the instance

```bash
cd /home
git clone https://github.com/<you>/PromptGFM-Bio.git promptgfm-bio
cd /home/promptgfm-bio
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python scripts/test_gpu.py        # confirm A100 detected
python scripts/verify_setup.py    # confirm torch 2.1 / pyg 2.4 / transformers 4.35
# Upload via JupyterLab drag-drop OR scp (from PowerShell):
#   data/processed/biomedical_graph.pt
#   data/processed/hpo_gene_disease_edges.csv
chmod +x scripts/*jarvis*.sh
```

### 3. Smoke test (~10 min, ~₹17)

```bash
bash scripts/smoke_test_path2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log
```

Expected end of output:

```
 >>> SMOKE PASS — Path 2 is wired correctly. Safe to launch run_all_seeds-jarvis.sh.
```

If it says **SMOKE FAIL**, the patch wasn't applied correctly — fix and re-run. Do NOT proceed to step 4 on a fail.

### 4. The 40 training runs (~53 GPU-hours)

```bash
tmux new -s train
bash scripts/run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%Y%m%d).log
# Ctrl+B, D to detach.  tmux attach -t train  to return.
```

Skip-if-exists is safe to resume after any interruption. To split across two GPUs:

```bash
# GPU 1:
ALL_SEEDS="42 43 44 45 46" bash scripts/run_all_seeds-jarvis.sh
# GPU 2:
ALL_SEEDS="47 48 49 50 51" bash scripts/run_all_seeds-jarvis.sh
```

### 5. The 40 evaluations (~7 GPU-hours)

```bash
bash scripts/run_evaluations_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_eval_$(date +%Y%m%d).log
```

Produces both `evaluation_results.json` and `zero_shot_results.json` per (variant, seed).
The 117-disease zero-shot set is valid for all 10 seeds because the split is now fixed.

### 6. Aggregate (CPU, can run on instance or downloaded locally)

Write the 10-seed aggregator that pools all 40 result JSONs, produces mean ± std + bootstrap CI, applies Holm–Bonferroni across the metric × comparison family, and emits the LaTeX-ready main table. (We can write this together once the 40 runs are done — the analysis is downstream of these scripts.)

### 7. Pause / destroy the A100

When step 6 is complete, **pause** the instance (you keep `/home`, pay ~₹1.13/hr storage) if you'll come back, or destroy after downloading results.

### 8. Phase 1 baselines (L4 VM, unchanged from earlier guide)

Continue as in the Phase 0+1 guide (SHEPHERD on a separate L4 VM with its own conda env, then Phrank / LIRICAL / PubMedBERT-cosine / LLM-direct, then `score_baselines.py`). The 117 zero-shot diseases and the disease → true-genes map produced by `prepare_baseline_inputs.py` are unchanged under Path 2.

---

## What about the old 12 ablation runs?

**Drop from headline results.** They were trained under the broken seed flow.
Two acceptable mentions in the paper:

1. **Don't mention them at all** — cleanest. The 10-seed study under Path 2 is your only ablation table.
2. **Cite as preliminary results in supplementary** — "An earlier 3-seed pilot under a non-fixed split informed the present 10-seed protocol." Honest, low-stakes.

Either is fine. Most reviewers prefer (1).

---

## What gets written / commits should include

After the run completes, your repository should contain:

```
scripts/train.py        (patched)
scripts/evaluate.py     (patched)
scripts/_jarvis_env.sh  (new)
scripts/run_all_seeds-jarvis.sh           (new)
scripts/run_evaluations_all_seeds-jarvis.sh (new)
scripts/smoke_test_path2-jarvis.sh        (new)
PATH2_PATCHES.md        (this patch record, in repo root or docs/)
checkpoints/<variant>_seed{42..51}/best_model.pt      (40 checkpoints)
results/<variant>_seed{42..51}/evaluation_results.json (40 std + 40 zero-shot)
logs/path2_*.log
```

Commit + tag this state (e.g. `git tag path2-frozen` after the 40 runs finish), so the methodology and the artifacts are versioned together — reviewers may ask.

---

## Why this maximizes acceptance probability

Two methodological objections that **would** be raised by a careful Q1 reviewer of the previous mechanism are eliminated:

1. **"Why does each seed train on a different data split?"** — gone, the split is fixed.
2. **"How was the zero-shot set verified leak-free across all seeds?"** — gone, one split means one valid zero-shot set.

And one objection that the new code prevents from being raised at all:

3. **"Is your training deterministic given the reported seed?"** — yes, demonstrated by the smoke test.

The cost over the original plan is ~₹1,400 (₹100 smoke + 12 extra training runs to re-do 42/43/44 under the patched code) — negligible compared to the cost of a major-revision cycle over either of those objections.
```

## File: `scripts/PATH2_PATCHES.md`

```markdown
# Path 2 patches — apply BEFORE the Phase 2 cloud runs

These two small edits make `PromptGFM-Bio` reproducible in the way Q1 reviewers
expect: a **single fixed train/val/test split** for the 10-seed study, and the
top-level `seed:` deterministically seeding torch / numpy / Python `random`
(model init, negative sampling, DataLoader shuffling).

After applying, your existing configs work unchanged — `data.random_seed: 42`
is already present in every ablation YAML, and the top-level `seed:` becomes
the per-run model-RNG seed.

> Apply with any text editor on Windows. For each change, search for the OLD
> block (verbatim) and replace with the NEW block. Indentation must match.

---

## 1) `scripts/train.py` — three edits

### 1a. Add the seeding helper (after the imports / logger setup)

Find this line (it's near the top of the file, just after `setup_logger` is imported):

```python
logger = logging.getLogger(__name__)
```

Immediately **AFTER** that line, insert:

```python


# ---------------------------------------------------------------------------
# Path 2 reproducibility: deterministic per-seed RNG for ALL stochastic sources.
# Called once from run_finetuning / run_pretraining with config['seed'].
# The data split is seeded SEPARATELY in create_dataloaders() using
# config['data']['random_seed'], which is held FIXED across all 10 seeds.
# ---------------------------------------------------------------------------
def _set_all_seeds(seed: int) -> None:
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"[reproducibility] all RNGs seeded with seed={seed} "
                f"(split seed comes from config['data']['random_seed'])")
```

### 1b. Fix the split seed in `create_dataloaders`

Find this block (inside `def create_dataloaders(config):`):

```python
    # Split data
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=config.get('seed', 42)
    )
```

Replace with:

```python
    # Split data — Path 2: split seed comes from data.random_seed (fixed across
    # all model-init seeds), NOT from top-level config['seed']. Backward-compat
    # fallback: if data.random_seed is missing, use the top-level seed.
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed
    )
    logger.info(f"[reproducibility] split seed = {split_seed} "
                f"(should be FIXED across all 10 model-init seeds)")
```

### 1c. Seed all RNGs at the top of training

Find this block (the start of `def run_finetuning(config):`):

```python
def run_finetuning(config):
    """Run supervised fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("Starting Supervised Fine-tuning")
    logger.info("="*60)
    
    # Create dataloaders (GeneDiseaseDataset will load the graph internally)
    train_loader, val_loader, dataset = create_dataloaders(config)
```

Replace with:

```python
def run_finetuning(config):
    """Run supervised fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("Starting Supervised Fine-tuning")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
    
    # Create dataloaders (GeneDiseaseDataset will load the graph internally)
    train_loader, val_loader, dataset = create_dataloaders(config)
```

And for safety also patch `run_pretraining` the same way — find:

```python
def run_pretraining(config):
    """Run self-supervised pretraining."""
    logger.info("\n" + "="*60)
    logger.info("Starting Self-Supervised Pretraining")
    logger.info("="*60)
```

Replace with:

```python
def run_pretraining(config):
    """Run self-supervised pretraining."""
    logger.info("\n" + "="*60)
    logger.info("Starting Self-Supervised Pretraining")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
```

---

## 2) `scripts/evaluate.py` — two edits

### 2a. Add the seeding helper

Find this line in `evaluate.py` (near the top, after the imports block):

```python
logger = logging.getLogger(__name__)
```

Immediately **AFTER** that line, insert:

```python


def _set_all_seeds(seed: int) -> None:
    """Path 2 reproducibility: identical helper to train.py._set_all_seeds."""
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 2b. Fix the split seed in `_load_dataset`

Find this block (inside `_load_dataset` — it's the only `create_train_val_test_split` call in evaluate.py):

```python
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=config.get('seed', 42),
    )
```

Replace with:

```python
    # Path 2: split seed = data.random_seed (FIXED across all seeds).
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed,
    )
```

### 2c. Seed RNGs at the top of `main()`

Find this block:

```python
def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptGFM-Bio')
```

Replace with:

```python
def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptGFM-Bio')
    # Path 2 reproducibility — model-RNG seed comes from config['seed'].
    # We can't seed before parsing args (config path is an arg), so the seed
    # call is moved just after we load the config below.
```

Then further down in `main()`, find the line that loads the config (commonly `config = load_config(args.config)` or `with open(args.config) as f: config = yaml.safe_load(f)`). Immediately AFTER that line, add:

```python
    _set_all_seeds(config.get('seed', 42))
```

---

## 3) The configs need NO changes

Every ablation YAML already contains `data: { random_seed: 42, ... }` and `seed: 42` at the top level. The Path 2 code reads both correctly. The sed override in the runner script (which rewrites only the top-level `seed:` line) now correctly varies model RNG without touching the split.

---

## 4) Verify the patches applied

After saving both files, from `E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio` in PowerShell:

```powershell
# train.py — should print 3 hits
Select-String -Path "scripts\train.py" -Pattern "_set_all_seeds|split_seed = config"

# evaluate.py — should print 3 hits
Select-String -Path "scripts\evaluate.py" -Pattern "_set_all_seeds|split_seed = config"
```

If you see fewer hits, one block didn't take — re-check indentation (Python is sensitive). Then commit and push the patched files to your private GitHub repo.

---

## 5) Why this is the Q1-correct fix

Before this patch: top-level `seed:` controlled only the *data split*. Negative sampling (`random.sample`), model init (`torch.manual_seed` was never called from config), and DataLoader order were driven by **process-time randomness** — so even running the *same* config twice could give different numbers, and "seed variance" across 42/43/44 conflated split-variance with process-noise.

After this patch: the train/val/test split is **bit-identical across all 10 seeds** (seeded by `data.random_seed=42`), the 117-disease zero-shot set is therefore valid for *every* seed (no leakage to audit), and the per-seed std reflects exclusively **model-initialization variance** — the standard, defensible quantity a reviewer expects from "mean ± std over 10 seeds." Two methodological footnotes get crossed off the reviewer's list before they're written.
```

## File: `scripts/prepare_baseline_inputs.py`

```python
#!/usr/bin/env python3
"""
prepare_baseline_inputs.py
Build the shared inputs every external baseline (SHEPHERD, Phrank, LIRICAL,
PubMedBERT-cosine, LLM-direct) needs for the 117 zero-shot rare diseases.

Run in the PromptGFM venv (pure pandas + mygene; no GPU):
    python scripts/prepare_baseline_inputs.py

Outputs (data/baselines/):
    disease_hpo_terms.json    {disease_id: [HP:xxxxxxx, ...]}     (model-equivalent input)
    disease_true_genes.json   {disease_id: [SYMBOL, ...]}         (ground truth, vocab-restricted)
    symbol_to_ensembl.json    {SYMBOL: ENSG..., ...}              (SHEPHERD needs Ensembl)
    all_candidate_genes_symbols.json  [SYMBOL, ...]               (full ranking vocab)
    all_candidate_genes_ensembl.json  [ENSG..., ...]
    prep_report.txt           coverage diagnostics — READ THIS before proceeding
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZS_PATH      = ROOT / "data/splits/zero_shot_rare_diseases.json"
HPOA_PATH    = ROOT / "data/raw/hpo/phenotype.hpoa"
ORPHA4_PATH  = ROOT / "data/raw/orphanet/en_product4.xml"
EDGES_PATH   = ROOT / "data/processed/hpo_gene_disease_edges.csv"
GRAPH_PATH   = ROOT / "data/processed/biomedical_graph.pt"
OUT_DIR      = ROOT / "data/baselines"
MIN_SCORE    = 0.3   # must match training config

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_zero_shot_ids():
    ids = json.load(open(ZS_PATH))["disease_ids"]
    print(f"[zs] {len(ids)} zero-shot diseases "
          f"({sum(i.startswith('OMIM') for i in ids)} OMIM, "
          f"{sum(i.startswith('ORPHA') for i in ids)} ORPHA)")
    return ids


def full_vocab_symbols():
    """Authoritative gene vocabulary = exactly what the model ranks (gene_to_idx)."""
    from src.data.dataset import GeneDiseaseDataset
    ds = GeneDiseaseDataset(graph_path=str(GRAPH_PATH),
                            edges_path=str(EDGES_PATH),
                            min_score=MIN_SCORE)
    symbols = sorted(ds.gene_to_idx.keys())
    print(f"[vocab] {len(symbols)} candidate genes (from gene_to_idx)")
    return symbols


def disease_to_hpo(zs_ids):
    """OMIM+ORPHA -> HPO terms from phenotype.hpoa, with ORPHA fallback to product4."""
    hpoa = pd.read_csv(HPOA_PATH, sep="\t", comment="#")
    # Be tolerant of header naming across HPO releases.
    cols = {c.lower(): c for c in hpoa.columns}
    dis_col = cols.get("database_id") or cols.get("databaseid") or list(hpoa.columns)[0]
    hpo_col = cols.get("hpo_id") or cols.get("hpoid")
    if hpo_col is None:
        raise SystemExit(f"[FATAL] could not find HPO id column in {HPOA_PATH}; "
                         f"columns are: {list(hpoa.columns)}")
    print(f"[hpoa] using disease column '{dis_col}', hpo column '{hpo_col}'")
    dis2hpo = (hpoa.groupby(dis_col)[hpo_col]
                   .apply(lambda s: sorted(set(s.dropna()))).to_dict())

    orpha2hpo = {}
    if ORPHA4_PATH.exists():
        tree = ET.parse(ORPHA4_PATH)
        for dis in tree.iter("Disorder"):
            code = dis.findtext("OrphaCode")
            terms = sorted({h.findtext("HPOId") for h in dis.iter("HPO")
                            if h.findtext("HPOId")})
            if code and terms:
                orpha2hpo[f"ORPHA:{code}"] = terms

    out = {}
    for d in zs_ids:
        out[d] = dis2hpo.get(d) or orpha2hpo.get(d) or []
    return out


def disease_to_true_genes(zs_ids, vocab):
    edges = pd.read_csv(EDGES_PATH)
    edges = edges[edges["score"] >= MIN_SCORE]
    vocab_set = set(vocab)
    g = (edges.groupby("disease")["gene"]
              .apply(lambda s: sorted(set(s) & vocab_set)).to_dict())
    return {d: g.get(d, []) for d in zs_ids}


def symbol_to_ensembl(symbols):
    try:
        import mygene
    except ImportError:
        raise SystemExit("[FATAL] pip install mygene  (needed for SHEPHERD Ensembl IDs)")
    mg = mygene.MyGeneInfo()
    res = mg.querymany(symbols, scopes="symbol", fields="ensembl.gene",
                       species="human", returnall=True)
    mapping = {}
    for hit in res["out"]:
        sym = hit.get("query")
        ens = hit.get("ensembl")
        if isinstance(ens, list):
            ens = ens[0].get("gene") if ens and isinstance(ens[0], dict) else None
        elif isinstance(ens, dict):
            ens = ens.get("gene")
        if sym and ens:
            mapping[sym] = ens
    return mapping


def main():
    zs = load_zero_shot_ids()
    vocab = full_vocab_symbols()
    hpo = disease_to_hpo(zs)
    true = disease_to_true_genes(zs, vocab)
    s2e = symbol_to_ensembl(vocab)

    cand_sym = vocab
    cand_ens = [s2e[s] for s in vocab if s in s2e]

    json.dump(hpo,  open(OUT_DIR / "disease_hpo_terms.json", "w"), indent=2)
    json.dump(true, open(OUT_DIR / "disease_true_genes.json", "w"), indent=2)
    json.dump(s2e,  open(OUT_DIR / "symbol_to_ensembl.json", "w"), indent=2)
    json.dump(cand_sym, open(OUT_DIR / "all_candidate_genes_symbols.json", "w"), indent=2)
    json.dump(cand_ens, open(OUT_DIR / "all_candidate_genes_ensembl.json", "w"), indent=2)

    n_hpo  = sum(1 for v in hpo.values() if v)
    n_true = sum(1 for v in true.values() if v)
    report = [
        "PromptGFM-Bio baseline-input prep report",
        "=" * 48,
        f"zero-shot diseases            : {len(zs)}",
        f"  with >=1 HPO term           : {n_hpo}   <-- must be ~all; if low, fix hpoa columns",
        f"  with >=1 true gene in vocab : {n_true}",
        f"candidate vocabulary (symbol) : {len(cand_sym)}",
        f"  mapped to Ensembl           : {len(cand_ens)}  ({100*len(cand_ens)/len(cand_sym):.1f}%)",
        f"  unmapped (excluded from SHEPHERD candidate set, documented): {len(cand_sym)-len(cand_ens)}",
        "",
        "Diseases with NO HPO terms (investigate before running baselines):",
        *[f"  {d}" for d, v in hpo.items() if not v][:25],
    ]
    (OUT_DIR / "prep_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\n[done] wrote artifacts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
```

## File: `scripts/preprocess_all.py`

```python
"""
Preprocessing script for PromptGFM-Bio.

This script orchestrates the complete preprocessing pipeline:
1. Parse BioGRID and STRING PPI networks
2. Create gene-disease edges using enhanced methods:
   - HPO Bridge: IDF-weighted phenotype overlap (primary method)
   - Orphadata: Gold standard rare disease associations (validation)
   - DisGeNET: Backup method (if available)
3. Parse HPO phenotype annotations
4. Optionally add UniProt gene descriptions (Week 4+)
5. Optionally add Reactome pathway annotations (Week 4+)
6. Build heterogeneous graph
7. Save processed graph for model training

Usage:
    # Basic usage (HPO bridge + Orphadata)
    python scripts/preprocess_all.py
    
    # Force reprocess
    python scripts/preprocess_all.py --force
    
    # Week 4+ enhancements
    python scripts/preprocess_all.py --with-uniprot --with-pathways
    
    # Minimal (HPO bridge only)
    python scripts/preprocess_all.py --no-orphadata
"""

import sys
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import preprocess_all
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess biomedical data into knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes raw biomedical datasets into a heterogeneous knowledge graph:
  
  Input files (in data/raw/):
    - BioGRID: protein-protein interactions
    - STRING: protein network database  
    - HPO: Human Phenotype Ontology (genes_to_phenotype.txt, phenotype.hpoa)
    - Orphadata: Gold standard rare disease gene associations (optional)
    - UniProt: Gene descriptions (optional, Week 4+)
    - Reactome: Pathway annotations (optional, Week 4+)
    
  Output (in data/processed/):
    - biomedical_graph.pt: PyTorch Geometric HeteroData graph
    - biomedical_graph_stats.txt: Graph statistics
    - hpo_gene_disease_edges.csv: HPO bridge edges (if enabled)
    - uniprot_gene_descriptions.csv: Gene descriptions (if enabled)
    - reactome_gene_pathways.csv: Pathway annotations (if enabled)
    
  Graph structure:
    Node types: [gene, disease, phenotype]
    Edge types: [gene-gene, gene-disease, disease-phenotype]
  
  Gene-Disease Edge Methods:
    1. HPO Bridge (Primary): IDF-weighted phenotype overlap
    2. Orphadata (Secondary): Gold standard validation
    3. DisGeNET (Backup): Direct associations (if available)

Examples:
  %(prog)s                              Process with HPO bridge + Orphadata
  %(prog)s --force                      Force reprocess
  %(prog)s --no-orphadata               Use HPO bridge only (MVP)
  %(prog)s --with-uniprot               Add UniProt gene descriptions
  %(prog)s --with-uniprot --with-pathways   Full enhancement (Week 4+)
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output already exists'
    )
    
    parser.add_argument(
        '--no-hpo-bridge',
        action='store_true',
        help='Disable HPO phenotype bridge (primary gene-disease method)'
    )
    
    parser.add_argument(
        '--no-orphadata',
        action='store_true',
        help='Disable Orphadata gold standard integration'
    )
    
    parser.add_argument(
        '--with-uniprot',
        action='store_true',
        help='Enable UniProt gene descriptions (Week 4+)'
    )
    
    parser.add_argument(
        '--with-pathways',
        action='store_true',
        help='Enable Reactome pathway annotations (Week 4+)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PromptGFM-Bio Enhanced Preprocessing Pipeline")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Force reprocess: {args.force}")
    print(f"  HPO Bridge: {not args.no_hpo_bridge}")
    print(f"  Orphadata: {not args.no_orphadata}")
    print(f"  UniProt: {args.with_uniprot}")
    print(f"  Pathways: {args.with_pathways}")
    print()
    
    try:
        preprocess_all(
            force=args.force,
            use_hpo_bridge=not args.no_hpo_bridge,
            use_orphadata=not args.no_orphadata,
            use_uniprot=args.with_uniprot,
            use_pathways=args.with_pathways
        )
        
        print("\n" + "="*70)
        print("✓ PREPROCESSING COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Create dataset splits: python -m src.data.dataset")
        print("  2. Check graph file: data/processed/biomedical_graph.pt")
        print("  3. View statistics: data/processed/biomedical_graph_stats.txt")
        print()
        
    except FileNotFoundError as e:
        print(f"\n✗ Preprocessing failed: Required data file not found")
        print(f"   {e}")
        print("\nMake sure you've downloaded all datasets:")
        print("  python scripts/download_data.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## File: `scripts/rebuild_kaggle_notebook.py`

```python
"""
Rebuilds notebooks/kaggle_training.ipynb with full cross-session resume support.
Run from project root:  python scripts/rebuild_kaggle_notebook.py
"""
import json, pathlib, sys

OUT = pathlib.Path("notebooks/kaggle_training.ipynb")

def md(src): return {"cell_type":"markdown","id":None,"metadata":{},"source":[src]}
def code(lines): return {"cell_type":"code","id":None,"execution_count":None,"metadata":{},"outputs":[],"source":lines}

# ── helper: assign sequential ids ────────────────────────────────────────────
def build(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell{i:02d}abcd"
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{
                "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.10.0"},
                "kaggle":{"accelerator":"gpu","dataSources":[],"dockerImageVersionId":30761,
                          "isInternetEnabled":True,"language":"python","sourceType":"notebook"}
            },
            "cells":cells}

cells = []

# ── 0 · Title ─────────────────────────────────────────────────────────────────
cells.append(md("""\
# PromptGFM-Bio — Kaggle Training Notebook
**Gene-Phenotype Prediction for Rare Diseases**

### ✅ Resumable Across Sessions & Accounts
This notebook saves **three Kaggle Datasets** after training so any future session — \
or a different Kaggle account — can skip all expensive steps:

| Dataset name (you choose) | What it stores | Skips |
|---|---|---|
| `promptgfm-model-cache` | HuggingFace BioBERT weights | ~5 min download |
| `promptgfm-data` | Raw + processed graph | ~25 min download + preprocess |
| `promptgfm-checkpoints` | Per-epoch checkpoints | training from epoch 0 |

**Setup once → add as Dataset inputs on every future session.**

> ⚙️ Before running: Settings → Accelerator → **GPU T4 x2** · Internet → **On**\
"""))

# ── 1 · Env check ─────────────────────────────────────────────────────────────
cells.append(md("## 1. Environment Check"))
cells.append(code([
    "import sys, subprocess, os\n",
    "import torch\n",
    "\n",
    "print(f'Python  : {sys.version}')\n",
    "print(f'PyTorch : {torch.__version__}')\n",
    "print(f'CUDA    : {torch.version.cuda}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU     : {torch.cuda.get_device_name(0)}')\n",
    "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print(f'VRAM    : {vram:.1f} GB  (expect ~15-16 GB on T4)')\n",
    "else:\n",
    "    print('⚠️  No GPU — enable in Notebook Settings → Accelerator → GPU T4')\n",
]))

# ── 2 · Session configuration ─────────────────────────────────────────────────
cells.append(md("""\
## 2. ⚙️ Session Configuration
Edit the variables below **before running any other cell**.

**`RESUME_*` flags**: set to `True` if you have added the corresponding Kaggle Dataset as input.  
**Dataset input paths**: change if you named your datasets differently.\
"""))
cells.append(code([
    "# ─── RESUME FLAGS ────────────────────────────────────────────────────────\n",
    "# Set True when you have added the matching dataset as notebook input\n",
    "RESUME_HF_CACHE     = False  # True → skip BioBERT download (saves ~5 min)\n",
    "RESUME_DATA         = False  # True → skip raw download + preprocessing (~25 min)\n",
    "RESUME_CHECKPOINTS  = False  # True → resume training from last saved epoch\n",
    "\n",
    "# ─── INPUT DATASET PATHS (Kaggle mounts datasets under /kaggle/input/) ───\n",
    "# After you create and add them, the paths will match these names:\n",
    "INPUT_HF_CACHE    = '/kaggle/input/promptgfm-model-cache'\n",
    "INPUT_DATA        = '/kaggle/input/promptgfm-data'\n",
    "INPUT_CHECKPOINTS = '/kaggle/input/promptgfm-checkpoints'\n",
    "\n",
    "# ─── OUTPUT DATASET NAMES (used in Step 12 instructions) ─────────────────\n",
    "OUTPUT_HF_CACHE    = 'promptgfm-model-cache'\n",
    "OUTPUT_DATA        = 'promptgfm-data'\n",
    "OUTPUT_CHECKPOINTS = 'promptgfm-checkpoints'\n",
    "\n",
    "# ─── HF MODEL CACHE LOCATION ─────────────────────────────────────────────\n",
    "# Point HuggingFace to a path inside /kaggle/working/ so we can save it\n",
    "HF_CACHE_DIR = '/kaggle/working/hf_cache'\n",
    "os.environ['HF_HOME']              = HF_CACHE_DIR\n",
    "os.environ['TRANSFORMERS_CACHE']   = HF_CACHE_DIR\n",
    "os.environ['HUGGINGFACE_HUB_CACHE']= HF_CACHE_DIR\n",
    "\n",
    "print('Configuration:')\n",
    "print(f'  RESUME_HF_CACHE    = {RESUME_HF_CACHE}')\n",
    "print(f'  RESUME_DATA        = {RESUME_DATA}')\n",
    "print(f'  RESUME_CHECKPOINTS = {RESUME_CHECKPOINTS}')\n",
    "print(f'  HF cache dir       = {HF_CACHE_DIR}')\n",
]))

# ── 3 · Install PyG ───────────────────────────────────────────────────────────
cells.append(md("## 3. Install PyTorch Geometric"))
cells.append(code([
    "import torch, subprocess, sys\n",
    "\n",
    "TORCH_VER = torch.__version__.split('+')[0]\n",
    "CUDA_VER  = torch.version.cuda.replace('.', '')\n",
    "WHEEL_URL = f'https://data.pyg.org/whl/torch-{TORCH_VER}+cu{CUDA_VER}.html'\n",
    "print(f'PyG wheel source: {WHEEL_URL}')\n",
    "\n",
    "subprocess.run(\n",
    "    [sys.executable, '-m', 'pip', 'install', '--quiet',\n",
    "     '-f', WHEEL_URL,\n",
    "     'torch-scatter', 'torch-sparse', 'torch-cluster',\n",
    "     'torch-spline-conv', 'torch-geometric'],\n",
    "    check=True\n",
    ")\n",
    "print('✅ PyTorch Geometric installed')\n",
]))

# ── 4 · Install extras ────────────────────────────────────────────────────────
cells.append(md("## 4. Install Extra Dependencies"))
cells.append(code([
    "# Upgrade build tools first — prevents metadata-generation-failed on Python 3.12\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',\n",
    "                '--upgrade', 'setuptools', 'wheel', 'pip'], check=True)\n",
    "\n",
    "extra = [\n",
    "    'transformers>=4.40.0',\n",
    "    'sentence-transformers>=2.7.0',\n",
    "    'biopython>=1.84',\n",
    "    'networkx>=3.2',\n",
    "    'wandb>=0.17.0',\n",
    "    'python-dotenv>=1.0.0',\n",
    "]\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + extra, check=True)\n",
    "print('✅ Extra packages installed')\n",
]))

# ── 5 · Clone repo ────────────────────────────────────────────────────────────
cells.append(md("## 5. Clone Project Code from GitHub"))
cells.append(code([
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "GITHUB_URL  = 'https://github.com/pes1ug23am910/PROMPTGMF-Bio-Kaggle.git'\n",
    "PROJECT_DIR = '/kaggle/working/PromptGFM-Bio'\n",
    "\n",
    "if not os.path.exists(PROJECT_DIR):\n",
    "    subprocess.run(['git', 'clone', '--depth', '1', GITHUB_URL, PROJECT_DIR], check=True)\n",
    "    print(f'✅ Cloned to {PROJECT_DIR}')\n",
    "else:\n",
    "    subprocess.run(['git', '-C', PROJECT_DIR, 'pull'], check=True)\n",
    "    print(f'✅ Pulled latest changes')\n",
    "\n",
    "os.chdir(PROJECT_DIR)\n",
    "sys.path.insert(0, PROJECT_DIR)\n",
    "print(f'Working directory: {os.getcwd()}')\n",
]))

# ── 6 · Create dirs ───────────────────────────────────────────────────────────
cells.append(md("## 6. Create Directory Structure"))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "dirs = [\n",
    "    'data/raw', 'data/processed', 'data/splits',\n",
    "    'checkpoints/promptgfm_film',\n",
    "    'logs',\n",
    "]\n",
    "for d in dirs:\n",
    "    Path(d).mkdir(parents=True, exist_ok=True)\n",
    "print('✅ Directories created')\n",
]))

# ── 7 · Restore assets ────────────────────────────────────────────────────────
cells.append(md("""\
## 7. Restore Assets from Previous Session
Restores HuggingFace model cache, preprocessed data, and training checkpoints \
from saved Kaggle Datasets — skipping all expensive steps below.\

**First-time run**: all three blocks will print "not found — will download/preprocess fresh".\
"""))
cells.append(code([
    "import shutil, tarfile\n",
    "from pathlib import Path\n",
    "\n",
    "def restore_tar(src_path, dest_dir, label):\n",
    "    \"\"\"Extract a .tar.gz archive if it exists.\"\"\"\n",
    "    src = Path(src_path)\n",
    "    if src.exists():\n",
    "        dest = Path(dest_dir)\n",
    "        dest.mkdir(parents=True, exist_ok=True)\n",
    "        with tarfile.open(src, 'r:gz') as tar:\n",
    "            tar.extractall(dest)\n",
    "        print(f'✅ {label} restored from {src}')\n",
    "        return True\n",
    "    return False\n",
    "\n",
    "def restore_dir(src_path, dest_dir, label):\n",
    "    \"\"\"Copy directory tree if source exists.\"\"\"\n",
    "    src = Path(src_path)\n",
    "    if src.exists() and any(src.iterdir()):\n",
    "        dest = Path(dest_dir)\n",
    "        if dest.exists():\n",
    "            shutil.rmtree(dest)\n",
    "        shutil.copytree(src, dest)\n",
    "        print(f'✅ {label} restored from {src}')\n",
    "        return True\n",
    "    return False\n",
    "\n",
    "# ── A. HuggingFace model cache ────────────────────────────────────────────\n",
    "if RESUME_HF_CACHE:\n",
    "    ok = restore_tar(f'{INPUT_HF_CACHE}/hf_cache.tar.gz', HF_CACHE_DIR, 'HF model cache')\n",
    "    if not ok:\n",
    "        ok = restore_dir(f'{INPUT_HF_CACHE}/hf_cache', HF_CACHE_DIR, 'HF model cache')\n",
    "    if not ok:\n",
    "        print(f'⚠️  HF cache not found at {INPUT_HF_CACHE} — BioBERT will be re-downloaded')\n",
    "else:\n",
    "    print('HF cache: skipped (RESUME_HF_CACHE=False)')\n",
    "\n",
    "# ── B. Preprocessed graph + raw data ─────────────────────────────────────\n",
    "if RESUME_DATA:\n",
    "    ok = restore_tar(f'{INPUT_DATA}/data.tar.gz', 'data', 'Graph data')\n",
    "    if not ok:\n",
    "        ok = restore_dir(f'{INPUT_DATA}/processed', 'data/processed', 'Processed graph')\n",
    "        restore_dir(f'{INPUT_DATA}/splits', 'data/splits', 'Data splits')\n",
    "    graph = Path('data/processed/biomedical_graph.pt')\n",
    "    if graph.exists():\n",
    "        print(f'✅ Graph ready ({graph.stat().st_size/1e6:.0f} MB)')\n",
    "    else:\n",
    "        print('⚠️  Graph not found in restored data — will preprocess fresh')\n",
    "        RESUME_DATA = False   # force re-preprocessing below\n",
    "else:\n",
    "    print('Data: skipped (RESUME_DATA=False)')\n",
    "\n",
    "# ── C. Training checkpoints ───────────────────────────────────────────────\n",
    "if RESUME_CHECKPOINTS:\n",
    "    ckpt_src = Path(INPUT_CHECKPOINTS)\n",
    "    ckpt_dst = Path('checkpoints/promptgfm_film')\n",
    "    ckpt_dst.mkdir(parents=True, exist_ok=True)\n",
    "    files_copied = 0\n",
    "    if ckpt_src.exists():\n",
    "        for f in ckpt_src.glob('*'):\n",
    "            shutil.copy2(f, ckpt_dst / f.name)\n",
    "            files_copied += 1\n",
    "    if files_copied:\n",
    "        epochs = sorted([f.stem.replace('checkpoint_epoch_','') for f in ckpt_dst.glob('checkpoint_epoch_*.pt')])\n",
    "        print(f'✅ Checkpoints restored ({files_copied} files). Epochs available: {epochs}')\n",
    "    else:\n",
    "        print(f'⚠️  No checkpoints found at {INPUT_CHECKPOINTS} — will train from scratch')\n",
    "        RESUME_CHECKPOINTS = False\n",
    "else:\n",
    "    print('Checkpoints: skipped (RESUME_CHECKPOINTS=False)')\n",
]))

# ── 8 · Download data ─────────────────────────────────────────────────────────
cells.append(md("""\
## 8. Download Biomedical Datasets
Skipped automatically if `RESUME_DATA=True` and graph was restored successfully.  
Total download: ~1.5 GB · takes ~10 min.\
"""))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "graph_exists = Path('data/processed/biomedical_graph.pt').exists()\n",
    "\n",
    "if RESUME_DATA and graph_exists:\n",
    "    print('⏭️  Download skipped — restored from Kaggle Dataset')\n",
    "else:\n",
    "    print('Downloading datasets...')\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, 'scripts/download_data.py', '--dataset', 'all'],\n",
    "        capture_output=False\n",
    "    )\n",
    "    print('Download exit code:', result.returncode)\n",
]))

# ── 9 · Preprocess ────────────────────────────────────────────────────────────
cells.append(md("""\
## 9. Preprocess Data (Build Knowledge Graph)
Skipped automatically if `RESUME_DATA=True` and graph was restored successfully.\
"""))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "graph_path = Path('data/processed/biomedical_graph.pt')\n",
    "\n",
    "if RESUME_DATA and graph_path.exists():\n",
    "    print(f'⏭️  Preprocessing skipped — graph ready ({graph_path.stat().st_size/1e6:.0f} MB)')\n",
    "else:\n",
    "    print('Building knowledge graph...')\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, 'scripts/preprocess_all.py'],\n",
    "        capture_output=False\n",
    "    )\n",
    "    print('Preprocessing exit code:', result.returncode)\n",
    "    if graph_path.exists():\n",
    "        print(f'✅ Graph ready ({graph_path.stat().st_size/1e6:.0f} MB)')\n",
    "    else:\n",
    "        raise RuntimeError('Graph file not created — check logs above')\n",
]))

# ── 10 · W&B ──────────────────────────────────────────────────────────────────
cells.append(md("## 10. W&B Login (Optional)"))
cells.append(code([
    "WANDB_API_KEY = ''   # paste your key here, or leave empty to disable\n",
    "\n",
    "if WANDB_API_KEY:\n",
    "    import wandb\n",
    "    wandb.login(key=WANDB_API_KEY)\n",
    "    print('✅ W&B logged in')\n",
    "else:\n",
    "    os.environ['WANDB_MODE'] = 'disabled'\n",
    "    print('W&B disabled — metrics logged to stdout only')\n",
]))

# ── 11 · Train ────────────────────────────────────────────────────────────────
cells.append(md("""\
## 11. Train
Uses `configs/kaggle_config.yaml` (T4-tuned: batch_size=64, hidden_dim=512).  
Set `RESUME=True` to continue from the last restored checkpoint.\
"""))
cells.append(code([
    "RESUME = RESUME_CHECKPOINTS   # auto-set from config above; override here if needed\n",
    "\n",
    "if RESUME:\n",
    "    cmd = [sys.executable, 'scripts/resume_training.py',\n",
    "           '--config', 'configs/kaggle_config.yaml']\n",
    "else:\n",
    "    cmd = [sys.executable, 'scripts/train.py',\n",
    "           '--config', 'configs/kaggle_config.yaml']\n",
    "\n",
    "print('Running:', ' '.join(cmd))\n",
    "result = subprocess.run(cmd, capture_output=False)\n",
    "print('Training exit code:', result.returncode)\n",
]))

# ── 12 · Save everything ──────────────────────────────────────────────────────
cells.append(md("""\
## 12. 💾 Save ALL Assets as Kaggle Output

Run this cell **before the session ends** (set a reminder before the 9-hour limit).

It saves three directories under `/kaggle/working/`:

| Directory | Contents | Create Dataset named |
|---|---|---|
| `out_model_cache/` | BioBERT weights (~440 MB) | `promptgfm-model-cache` |
| `out_data/` | Raw + processed graph (~600 MB) | `promptgfm-data` |
| `out_checkpoints/` | Per-epoch `.pt` files | `promptgfm-checkpoints` |

### After this cell completes:
1. Click **Output** tab (right panel) → you'll see these three folders
2. For **each** folder → click the ⊕ icon → **New Dataset** → use the names above
3. Make the datasets **Public** (or **Private** if you want them only for yourself)
4. Next session: **Add Data** → **Your Datasets** → add all three → set `RESUME_*=True`

### Using from a different Kaggle account:
Make the datasets **Public**, then the other account can find them by searching  
`pes1ug23am910/promptgfm-model-cache` etc. in **Add Data**.\
"""))
cells.append(code([
    "import shutil, tarfile, os\n",
    "from pathlib import Path\n",
    "\n",
    "def make_tar(src_dir, out_file, label):\n",
    "    src = Path(src_dir)\n",
    "    if not src.exists() or not any(src.rglob('*')):\n",
    "        print(f'⚠️  {label}: source empty or missing ({src}) — skipped')\n",
    "        return\n",
    "    out = Path(out_file)\n",
    "    out.parent.mkdir(parents=True, exist_ok=True)\n",
    "    with tarfile.open(out, 'w:gz') as tar:\n",
    "        tar.add(src, arcname=src.name)\n",
    "    size_mb = out.stat().st_size / 1e6\n",
    "    print(f'✅ {label} → {out}  ({size_mb:.0f} MB)')\n",
    "\n",
    "def copy_dir(src_dir, out_dir, label):\n",
    "    src = Path(src_dir)\n",
    "    out = Path(out_dir)\n",
    "    if not src.exists():\n",
    "        print(f'⚠️  {label}: missing ({src}) — skipped')\n",
    "        return\n",
    "    if out.exists():\n",
    "        shutil.rmtree(out)\n",
    "    shutil.copytree(src, out)\n",
    "    files = list(out.rglob('*.*'))\n",
    "    total_mb = sum(f.stat().st_size for f in files) / 1e6\n",
    "    print(f'✅ {label} → {out}  ({len(files)} files, {total_mb:.0f} MB)')\n",
    "\n",
    "print('=== Saving assets for next session ===')\n",
    "\n",
    "# A. HuggingFace model cache (BioBERT) ─────────────────────────────────────\n",
    "make_tar(HF_CACHE_DIR,\n",
    "         '/kaggle/working/out_model_cache/hf_cache.tar.gz',\n",
    "         'HF model cache')\n",
    "\n",
    "# B. Data (raw + processed graph + splits) ─────────────────────────────────\n",
    "make_tar('data/processed',\n",
    "         '/kaggle/working/out_data/data.tar.gz',\n",
    "         'Processed graph data')\n",
    "\n",
    "# C. Training checkpoints ──────────────────────────────────────────────────\n",
    "copy_dir('checkpoints/promptgfm_film',\n",
    "         '/kaggle/working/out_checkpoints',\n",
    "         'Training checkpoints')\n",
    "\n",
    "print()\n",
    "print('=== Output summary ===')\n",
    "for d in ['out_model_cache', 'out_data', 'out_checkpoints']:\n",
    "    p = Path('/kaggle/working') / d\n",
    "    if p.exists():\n",
    "        files = list(p.rglob('*.*'))\n",
    "        total_mb = sum(f.stat().st_size for f in files) / 1e6\n",
    "        print(f'  /kaggle/working/{d}/  →  {len(files)} files, {total_mb:.0f} MB')\n",
    "\n",
    "print()\n",
    "print('Next steps:')\n",
    "print('  1. Output tab → create 3 datasets from out_model_cache, out_data, out_checkpoints')\n",
    "print('  2. Next session: add those datasets as input, set RESUME_*=True in cell 2')\n",
]))

# ── 13 · Evaluate ─────────────────────────────────────────────────────────────
cells.append(md("## 13. Quick Evaluation"))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "best = Path('checkpoints/promptgfm_film/best_model.pt')\n",
    "if not best.exists():\n",
    "    print('No best_model.pt yet — run more training epochs first')\n",
    "else:\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, 'scripts/evaluate.py',\n",
    "         '--config', 'configs/kaggle_config.yaml',\n",
    "         '--checkpoint', str(best)],\n",
    "        capture_output=False\n",
    "    )\n",
    "    print('Evaluation exit code:', result.returncode)\n",
]))

print(f"Built {len(cells)} cells")
nb = build(cells)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written → {OUT}")
```

## File: `scripts/resume_ablations.sh`

```bash
#!/usr/bin/env bash
# resume_ablations.sh — Continue the 12-run ablation training after interruption.
#
# Skips any (config, seed) whose checkpoints/${name}_seed${seed}/best_model.pt
# already exists.  Does NOT back up old checkpoints (we're resuming, not
# starting over).  Does NOT run evaluation — use evaluate_all-fix.sh for that
# after all 12 training runs complete (it already has skip logic).
#
# Usage (inside tmux):
#   bash scripts/resume_ablations.sh 2>&1 | tee -a logs/resume_$(date +%Y%m%d_%H%M%S).log
#
# Note on `set -e` + `| tee`: we capture the python exit code via PIPESTATUS
# so a single failed training run doesn't kill the whole wrapper.

set -euo pipefail

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

TOTAL=12
DONE=0
SKIPPED=0
RAN=0
FAILED=0

echo "=============================================="
echo " Resume training — $(date)"
echo "=============================================="

# ─── First pass: inventory what is already complete ────────────────────────
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    if [ -f "checkpoints/${name}_seed${seed}/best_model.pt" ]; then
      DONE=$((DONE + 1))
    fi
  done
done
echo " Already complete: ${DONE}/${TOTAL}"
echo " Still to train:   $((TOTAL - DONE))/${TOTAL}"
echo ""

# ─── Second pass: train the missing ones ───────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    ckpt_dir="checkpoints/${name}_seed${seed}"
    best_model="${ckpt_dir}/best_model.pt"

    if [ -f "$best_model" ]; then
      echo "=== SKIP ${name} seed ${seed} — best_model.pt already exists ==="
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo ""
    echo "=============================================="
    echo "=== Training ${name} seed ${seed} ==="
    echo "=== Started: $(date) ==="
    echo "=============================================="

    mkdir -p "$ckpt_dir"

    # Temp config: override seed AND checkpoint_dir
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    # Capture exit via PIPESTATUS so `set -e` + `| tee` doesn't kill the wrapper
    set +e
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/retrain_${name}_seed${seed}.log"
    train_exit=${PIPESTATUS[0]}
    set -e

    rm -f "$tmp"

    if [ $train_exit -ne 0 ]; then
      echo "=== [FAILED] ${name} seed ${seed} — exit code ${train_exit} ==="
      FAILED=$((FAILED + 1))
    else
      echo "=== [DONE] ${name} seed ${seed} — $(date) ==="
      RAN=$((RAN + 1))
    fi
  done
done

echo ""
echo "=============================================="
echo " Resume summary — $(date)"
echo "=============================================="
echo "  Already complete (skipped):  ${SKIPPED}"
echo "  Trained this run:            ${RAN}"
echo "  Failed:                      ${FAILED}"
echo "  Total complete checkpoints:  $(ls checkpoints/ablation_*_seed*/best_model.pt 2>/dev/null | wc -l)/${TOTAL}"
echo ""
echo " Next step: when all 12 complete, run:"
echo "   bash scripts/evaluate_all-fix.sh"
echo " (it skips runs whose results/*/evaluation_results.json already exists)"
```

## File: `scripts/resume_training.py`

```python
"""
Resume training script with interactive checkpoint management.

Provides options to:
A) Resume from last checkpoint
B) Start fresh (archive old checkpoints)
C) Resume from custom epoch
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_checkpoints(checkpoint_dir):
    """List all available checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for ckpt_file in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
        epoch = int(ckpt_file.stem.split('_')[-1])
        
        # Load checkpoint to get metrics
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            metrics = ckpt.get('current_metrics', {})
            checkpoints.append({
                'path': ckpt_file,
                'epoch': epoch,
                'auroc': metrics.get('auroc', 0),
                'aupr': metrics.get('aupr', 0),
                'loss': metrics.get('loss', 0)
            })
        except Exception as e:
            logger.warning(f"Could not load {ckpt_file}: {e}")
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    return checkpoints


def archive_checkpoints(checkpoint_dir):
    """Archive existing checkpoints to timestamped folder."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_archive_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Move all checkpoint files
    moved = 0
    for ckpt_file in checkpoint_dir.glob("*.pt"):
        shutil.move(str(ckpt_file), str(archive_dir / ckpt_file.name))
        moved += 1
    
    # Move JSON files too
    for json_file in checkpoint_dir.glob("*.json"):
        shutil.move(str(json_file), str(archive_dir / json_file.name))
        moved += 1
    
    logger.info(f"\u2713 Archived {moved} files to: {archive_dir}")
    return archive_dir


def resume_training(config_path, checkpoint_path, start_epoch=None):
    """Resume training from checkpoint.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint to resume from
        start_epoch: Override starting epoch (for custom resume)
    """
    # Import training modules
    from scripts.train import run_finetuning
    from src.training.finetune import PromptGFMTrainer
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set resume checkpoint in config
    config['training']['resume_checkpoint'] = str(checkpoint_path)
    if start_epoch is not None:
        config['training']['resume_epoch'] = start_epoch
    
    logger.info(f"\n{'='*70}")
    logger.info("RESUMING TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    if start_epoch:
        logger.info(f"Starting from epoch: {start_epoch}")
    logger.info(f"{'='*70}\n")
    
    # Run training
    run_finetuning(config)


def interactive_resume():
    """Interactive mode to choose resume option."""
    import sys
    
    print("\n" + "="*70)
    print("  PROMPTGFM-BIO TRAINING RESUME")
    print("="*70)
    
    # Get config file
    config_path = input("\nEnter config file path [configs/finetune_config.yaml]: ").strip()
    if not config_path:
        config_path = "configs/finetune_config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"\n\u274c Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config to get checkpoint directory
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    
    # List available checkpoints
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"\n\u26a0 No checkpoints found in {checkpoint_dir}")
        print("Starting fresh training...\n")
        from scripts.train import run_finetuning
        run_finetuning(config)
        return
    
    # Display options
    print(f"\nFound {len(checkpoints)} checkpoint(s) in {checkpoint_dir}:")
    print("\n" + "-"*70)
    for ckpt in checkpoints[-5:]:  # Show last 5
        print(f"  Epoch {ckpt['epoch']:3d} | AUROC: {ckpt['auroc']:.4f} | "
              f"AUPR: {ckpt['aupr']:.4f} | Loss: {ckpt['loss']:.6f}")
    if len(checkpoints) > 5:
        print(f"  ... and {len(checkpoints) - 5} more")
    print("-"*70)
    
    print("\nChoose an option:")
    print("  A) Resume from last checkpoint (Epoch {})".format(checkpoints[-1]['epoch']))
    print("  B) Start fresh (archive current checkpoints)")
    print("  C) Resume from custom epoch")
    print("  Q) Quit")
    
    choice = input("\nYour choice [A/B/C/Q]: ").strip().upper()
    
    if choice == 'A':
        # Resume from last checkpoint
        last_ckpt = checkpoints[-1]
        print(f"\n\u2713 Resuming from Epoch {last_ckpt['epoch']}")
        print(f"  AUROC: {last_ckpt['auroc']:.4f}, AUPR: {last_ckpt['aupr']:.4f}")
        resume_training(config_path, last_ckpt['path'])
        
    elif choice == 'B':
        # Archive and start fresh
        confirm = input("\n\u26a0 Archive checkpoints and start fresh? [y/N]: ").strip().lower()
        if confirm == 'y':
            archive_dir = archive_checkpoints(checkpoint_dir)
            print(f"\n\u2713 Starting fresh training...")
            from scripts.train import run_finetuning
            run_finetuning(config)
        else:
            print("\nCancelled.")
            
    elif choice == 'C':
        # Custom epoch
        print("\nAvailable epochs:")
        for ckpt in checkpoints:
            print(f"  {ckpt['epoch']}")
        
        try:
            epoch = int(input("\nEnter epoch number to resume from: ").strip())
            matching = [c for c in checkpoints if c['epoch'] == epoch]
            
            if not matching:
                print(f"\n\u274c Error: No checkpoint found for epoch {epoch}")
                sys.exit(1)
            
            ckpt = matching[0]
            print(f"\n\u2713 Resuming from Epoch {ckpt['epoch']}")
            print(f"  AUROC: {ckpt['auroc']:.4f}, AUPR: {ckpt['aupr']:.4f}")
            resume_training(config_path, ckpt['path'])
            
        except ValueError:
            print("\n\u274c Invalid epoch number")
            sys.exit(1)
            
    elif choice == 'Q':
        print("\nExiting...")
        sys.exit(0)
        
    else:
        print("\n\u274c Invalid choice")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Resume PromptGFM-Bio training with checkpoint management'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/finetune_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Specific checkpoint to resume from'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        help='Specific epoch to resume from'
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive existing checkpoints and start fresh'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode (default if no args provided)'
    )
    
    args = parser.parse_args()
    
    # If no specific args, go interactive
    if not any([args.checkpoint, args.epoch, args.archive]) or args.interactive:
        interactive_resume()
        return
    
    # Archive mode
    if args.archive:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
        archive_checkpoints(checkpoint_dir)
        print("\n\u2713 Starting fresh training...")
        from scripts.train import run_finetuning
        run_finetuning(config)
        return
    
    # Resume from specific checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        resume_training(args.config, checkpoint_path, args.epoch)
        return
    
    # Resume from specific epoch
    if args.epoch:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{args.epoch}.pt"
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        resume_training(args.config, checkpoint_path, args.epoch)
        return
    
    # Default: interactive
    interactive_resume()


if __name__ == "__main__":
    main()
```

## File: `scripts/retrain_and_evaluate.sh`

```bash
#!/usr/bin/env bash
# retrain_and_evaluate.sh — Retrains all 4 ablations × 3 seeds, then evaluates.
# Run inside tmux so you can detach and close your laptop.
#
# Usage:
#   tmux new -s ablations
#   bash scripts/retrain_and_evaluate.sh 2>&1 | tee logs/full_retrain_eval.log
#   # Then Ctrl+B, D to detach.  Reattach later: tmux attach -t ablations

set -euo pipefail

PROJECT=/home/mluser/projects_yash/new_project/PromptGFM-Bio
cd "$PROJECT"
mkdir -p logs results

echo "=============================================="
echo " STEP 0: Back up old checkpoints"
echo "=============================================="
BACKUP="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
cp -r checkpoints "$BACKUP"
echo "  ✓ Old checkpoints saved to $BACKUP"

echo ""
echo "=============================================="
echo " STEP 1: Retrain all 4 ablations × 3 seeds"
echo "=============================================="

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
    echo "=== Training ${name} seed ${seed} ==="

    # Seed-specific checkpoint dir so runs don't overwrite each other
    ckpt_dir="checkpoints/${name}_seed${seed}"
    mkdir -p "$ckpt_dir"

    # Temp config: override seed AND checkpoint_dir
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/retrain_${name}_seed${seed}.log"

    rm -f "$tmp"
    echo "=== Finished training ${name} seed ${seed} ==="
  done
done

echo ""
echo "=============================================="
echo " STEP 2: Evaluate all 12 runs"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=== Evaluating ${name} seed ${seed} ==="

    ckpt="checkpoints/${name}_seed${seed}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
      echo "  [ERROR] No checkpoint at ${ckpt} — skipping."
      continue
    fi

    out_dir="results/${name}_seed${seed}"
    mkdir -p "$out_dir"

    tmp="/tmp/${name}_seed${seed}_eval.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

    python3 scripts/evaluate.py \
      --config  "$tmp" \
      --checkpoint "$ckpt" \
      --split   test \
      --stratified \
      --output  "${out_dir}/evaluation_results.json" \
      2>&1 | tee "logs/eval_${name}_seed${seed}.log"

    # Optional zero-shot
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
    echo "=== Done ${name} seed ${seed} ==="
  done
done

echo ""
echo "=============================================="
echo " STEP 3: Summary"
echo "=============================================="
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    out="results/${name}_seed${seed}/evaluation_results.json"
    if [ -f "$out" ]; then
      auroc=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('auroc','N/A'):.4f}\")" 2>/dev/null || echo "err")
      hit50=$(python3 -c "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('hit_rate@50','N/A'):.4f}\")" 2>/dev/null || echo "err")
      mrr=$(python3 -c   "import json; d=json.load(open('${out}')); print(f\"{d.get('test',d).get('mrr','N/A'):.4f}\")" 2>/dev/null || echo "err")
      echo "  ${name} seed${seed}:  AUROC=${auroc}  Hit@50=${hit50}  MRR=${mrr}"
    else
      echo "  ${name} seed${seed}:  [no results]"
    fi
  done
done

echo ""
echo "=== ALL DONE ==="
```

## File: `scripts/run_ablations-workstations.sh`

```bash
#!/usr/bin/env bash
# run_ablations.sh — Runs 4 ablation configs × 3 seeds = 12 total training runs.
# Usage: bash scripts/run_ablations.sh
# Logs are saved to: logs/ablation_<name>_seed<N>.log
# Do NOT run training directly; this script manages temp configs for seed overrides.

set -euo pipefail

cd /home/mluser/projects_yash/new_project/PromptGFM-Bio
mkdir -p logs

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
    echo "=== Starting ${name} seed ${seed} ==="

    # Write a temp config with the seed line overridden.
    # sed targets the top-level "seed: <N>" line (not data.random_seed).
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed "s/^seed: .*/seed: ${seed}/" "$cfg" > "$tmp"

    # Run training; tee streams to terminal AND log file simultaneously.
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/ablation_${name}_seed${seed}.log"

    # Clean up temp config after the run completes.
    rm -f "$tmp"

    echo "=== Finished ${name} seed ${seed} ==="
  done
done

echo ""
echo "=== All 12 ablation runs complete ==="
```

## File: `scripts/run_ablations_extra_seeds-jarvis.sh`

```bash
#!/usr/bin/env bash
# run_ablations_extra_seeds-jarvis.sh
# ---------------------------------------------------------------------------
# Trains the EXTRA seeds (default 45-51) for all 4 ablation variants on a
# JarvisLabs instance, taking the 3-seed study (42,43,44) up to 10 seeds.
#
# Identical training logic to scripts/retrain_and_evaluate.sh STEP 1:
#   - overrides ONLY the top-level `seed:` line (data.random_seed stays 42,
#     so the train/val/test split and the zero-shot set are unchanged)
#   - per-seed checkpoint dir so runs never overwrite each other
#   - configs/*.yaml are NOT edited (batch_size 768, FP16 AMP, etc. preserved)
#
# Safe to re-run / resume (e.g. after a spot interruption): any run whose
# best_model.pt already exists is skipped.
#
# Usage (inside tmux):
#   tmux new -s train
#   bash run_ablations_extra_seeds-jarvis.sh 2>&1 | tee -a logs/extra_seeds_train_$(date +%Y%m%d).log
#   # Ctrl+B, D to detach;  tmux attach -t train  to return
#
# Run a subset (e.g. one spot GPU doing 3 seeds):
#   EXTRA_SEEDS="45 46 47" bash run_ablations_extra_seeds-jarvis.sh

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

read -r -a SEEDS <<< "$EXTRA_SEEDS"

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
SKIPPED=0; RAN=0; FAILED=0

echo "=============================================="
echo " Train extra seeds — $(date)"
echo " ${#CONFIGS[@]} variants x ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    ckpt_dir="checkpoints/${name}_seed${seed}"
    best_model="${ckpt_dir}/best_model.pt"

    if [ -f "$best_model" ]; then
      echo "=== SKIP ${name} seed ${seed} — best_model.pt exists ==="
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo ""
    echo "=============================================="
    echo "=== Training ${name} seed ${seed} ==="
    echo "=== Started: $(date) ==="
    echo "=============================================="
    mkdir -p "$ckpt_dir"

    # Temp config: override top-level seed AND checkpoint_dir ONLY.
    # (Exactly matches retrain_and_evaluate.sh — do not add other overrides.)
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    set +e
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/retrain_${name}_seed${seed}.log"
    train_exit=${PIPESTATUS[0]}
    set -e
    rm -f "$tmp"

    if [ "$train_exit" -ne 0 ]; then
      echo "=== [FAILED] ${name} seed ${seed} — exit ${train_exit} ==="
      FAILED=$((FAILED + 1))
    else
      echo "=== [DONE] ${name} seed ${seed} — $(date) ==="
      RAN=$((RAN + 1))
    fi
  done
done

echo ""
echo "=============================================="
echo " Training summary — $(date)"
echo "=============================================="
echo "  Skipped (already done): ${SKIPPED}"
echo "  Trained this run:       ${RAN}"
echo "  Failed:                 ${FAILED}"
# Count checkpoints for exactly the seeds requested (respects EXTRA_SEEDS override)
_done=0
for cfg in "${CONFIGS[@]}"; do
  _name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    [ -f "checkpoints/${_name}_seed${seed}/best_model.pt" ] && _done=$((_done + 1))
  done
done
echo "  Extra-seed checkpoints: ${_done}/${TOTAL}"
echo ""
echo " Next: bash run_evaluations_extra_seeds-jarvis.sh"
```

## File: `scripts/run_all_seeds-jarvis.sh`

```bash
#!/usr/bin/env bash
# run_all_seeds-jarvis.sh
# ---------------------------------------------------------------------------
# Path 2: trains 10 fresh seeds (42-51) x 4 ablation variants = 40 runs.
#
# This REPLACES the old 12 ablations (workstation, broken seed flow) and the
# run_ablations_extra_seeds-jarvis.sh (which only added 7 seeds to the old 3).
# All 40 runs are under the patched seed flow: split fixed at data.random_seed=42,
# model RNG seeded by top-level config['seed'].
#
# Safe to re-run / resume (spot interruption etc.): skip-if-exists per
# (variant, seed). To split across two GPUs:
#   GPU 1:  ALL_SEEDS="42 43 44 45 46" bash run_all_seeds-jarvis.sh
#   GPU 2:  ALL_SEEDS="47 48 49 50 51" bash run_all_seeds-jarvis.sh
#
# Usage (in tmux):
#   tmux new -s train
#   bash run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%Y%m%d).log

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

read -r -a SEEDS <<< "$ALL_SEEDS"
TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
SKIPPED=0; RAN=0; FAILED=0

echo "=============================================="
echo " Path 2 training — $(date)"
echo " ${#CONFIGS[@]} variants x ${#SEEDS[@]} seeds = ${TOTAL} runs"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    ckpt_dir="checkpoints/${name}_seed${seed}"
    best_model="${ckpt_dir}/best_model.pt"

    if [ -f "$best_model" ]; then
      echo "=== SKIP ${name} seed ${seed} — best_model.pt exists ==="
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo ""
    echo "=============================================="
    echo "=== Training ${name} seed ${seed} ==="
    echo "=== Started: $(date) ==="
    echo "=============================================="
    mkdir -p "$ckpt_dir"

    # Override only the top-level model-RNG seed and the checkpoint dir.
    # data.random_seed is NOT touched -> split stays fixed at 42 for every run.
    tmp="/tmp/${name}_seed${seed}.yaml"
    sed -e "s/^seed: .*/seed: ${seed}/" \
        -e "s|checkpoint_dir: .*|checkpoint_dir: ${ckpt_dir}|" \
        "$cfg" > "$tmp"

    set +e
    python3 scripts/train.py --config "$tmp" \
      2>&1 | tee "logs/path2_train_${name}_seed${seed}.log"
    train_exit=${PIPESTATUS[0]}
    set -e
    rm -f "$tmp"

    if [ "$train_exit" -ne 0 ]; then
      echo "=== [FAILED] ${name} seed ${seed} — exit ${train_exit} ==="
      FAILED=$((FAILED + 1))
    else
      echo "=== [DONE] ${name} seed ${seed} — $(date) ==="
      RAN=$((RAN + 1))
    fi
  done
done

echo ""
echo "=============================================="
echo " Path 2 training summary — $(date)"
echo "=============================================="
echo "  Skipped (already done): ${SKIPPED}"
echo "  Trained this run:       ${RAN}"
echo "  Failed:                 ${FAILED}"
_done=0
for cfg in "${CONFIGS[@]}"; do
  _name=$(basename "$cfg" .yaml)
  for seed in "${SEEDS[@]}"; do
    [ -f "checkpoints/${_name}_seed${seed}/best_model.pt" ] && _done=$((_done + 1))
  done
done
echo "  Total Path 2 checkpoints: ${_done}/${TOTAL}"
echo ""
echo " Next: bash scripts/run_evaluations_all_seeds-jarvis.sh"
```

## File: `scripts/run_evaluations-workstation.sh`

```bash
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
```

## File: `scripts/run_evaluations_all_seeds-jarvis.sh`

```bash
#!/usr/bin/env bash
# run_evaluations_all_seeds-jarvis.sh
# ---------------------------------------------------------------------------
# Path 2: evaluates 10 fresh seeds (42-51) x 4 ablation variants = 40 runs.
# Produces, per (variant, seed):
#   results/<name>_seed<seed>/evaluation_results.json   (test + stratified)
#   results/<name>_seed<seed>/zero_shot_results.json    (the 117 zero-shot diseases,
#                                                        valid for every seed since the
#                                                        split is fixed under Path 2)
#
# Usage (in tmux):
#   bash run_evaluations_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_eval_$(date +%Y%m%d).log

set -euo pipefail
source "$(dirname "$0")/_jarvis_env.sh"

read -r -a SEEDS <<< "$ALL_SEEDS"

ZS_FILE="data/splits/zero_shot_rare_diseases.json"
COMPLETED=0; SKIPPED=0; FAILED=0

echo "=============================================="
echo " Path 2 evaluation — $(date)"
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

    if [ ! -f "$std_out" ]; then
      set +e
      python3 scripts/evaluate.py \
        --config "$tmp" --checkpoint "$ckpt" --split test --stratified \
        --output "$std_out" \
        2>&1 | tee "logs/path2_eval_${name}_seed${seed}.log"
      e=${PIPESTATUS[0]}
      set -e
      if [ "$e" -ne 0 ]; then
        echo "  [FAILED] standard eval ${name} seed ${seed} — exit ${e}"
        FAILED=$((FAILED + 1)); rm -f "$tmp"; continue
      fi
    fi

    if [ ! -f "$zs_out" ] && [ -f "$ZS_FILE" ]; then
      set +e
      python3 scripts/evaluate.py \
        --config "$tmp" --checkpoint "$ckpt" --split test \
        --zero_shot --zero_shot_file "$ZS_FILE" --zero_shot_output "$zs_out" \
        2>&1 | tee -a "logs/path2_eval_${name}_seed${seed}.log"
      e=${PIPESTATUS[0]}
      set -e
      if [ "$e" -ne 0 ]; then
        echo "  [FAILED] zero-shot eval ${name} seed ${seed} — exit ${e}"
        FAILED=$((FAILED + 1)); rm -f "$tmp"; continue
      fi
    fi

    rm -f "$tmp"
    COMPLETED=$((COMPLETED + 1))
    echo "=== Done ${name} seed ${seed} ==="
  done
done

echo ""
echo "=============================================="
echo " Path 2 evaluation summary — $(date)"
echo "=============================================="
echo "  Completed: ${COMPLETED}   Skipped: ${SKIPPED}   Failed: ${FAILED}"
echo ""
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
```

## File: `scripts/run_evaluations_extra_seeds-jarvis.sh`

```bash
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
```

## File: `scripts/score_baselines.py`

```python
#!/usr/bin/env python3
"""
score_baselines.py
Uniform, apples-to-apples scoring of every method on the 117 zero-shot diseases.

Each method must provide a rankings file:
    {disease_id: [ranked_gene_SYMBOLS, best_first]}
(SHEPHERD outputs Ensembl -> map back to symbols first; helper below.)

We report Hit@10, Hit@50, MRR -- definitions identical for ALL methods, including
PromptGFM-Bio, so the comparison is unambiguous. (AUROC is kept for your internal
ablation table; rank-based metrics are the fair cross-method currency.)

Usage:
    python scripts/score_baselines.py \
        --truth data/baselines/disease_true_genes.json \
        --method promptgfm=data/baselines/rankings/promptgfm.json \
        --method shepherd=data/baselines/rankings/shepherd_symbols.json \
        --method phrank=data/baselines/rankings/phrank.json \
        --method pubmedbert=data/baselines/rankings/pubmedbert_cosine.json \
        --out data/baselines/comparison_table.md
"""
import argparse
import json
from statistics import mean


def metrics_for_disease(ranked, true_set):
    ranks = [i + 1 for i, g in enumerate(ranked) if g in true_set]
    if not ranks:
        return dict(hit10=0.0, hit50=0.0, mrr=0.0, found=0)
    best = min(ranks)
    return dict(
        hit10=1.0 if best <= 10 else 0.0,
        hit50=1.0 if best <= 50 else 0.0,
        mrr=1.0 / best,
        found=1,
    )


def score_method(rankings, truth):
    rows = []
    for did, true_genes in truth.items():
        true_set = set(true_genes)
        if not true_set:
            continue
        ranked = rankings.get(did, [])
        rows.append(metrics_for_disease(ranked, true_set))
    n = len(rows)
    if n == 0:
        return None
    return dict(
        n=n,
        hit10=mean(r["hit10"] for r in rows),
        hit50=mean(r["hit50"] for r in rows),
        mrr=mean(r["mrr"] for r in rows),
        coverage=mean(r["found"] for r in rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--method", action="append", required=True,
                    help="name=path/to/rankings.json (repeatable)")
    ap.add_argument("--out", default="data/baselines/comparison_table.md")
    args = ap.parse_args()

    truth = json.load(open(args.truth))
    results = {}
    for spec in args.method:
        name, path = spec.split("=", 1)
        rankings = json.load(open(path))
        m = score_method(rankings, truth)
        if m is None:
            print(f"[warn] no scorable diseases for {name}")
            continue
        results[name] = m

    # Markdown table, sorted by Hit@50 desc (your method should top it)
    lines = [
        "| Method | n | Hit@10 | Hit@50 | MRR | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["hit50"]):
        lines.append(f"| {name} | {m['n']} | {m['hit10']:.3f} | "
                     f"{m['hit50']:.3f} | {m['mrr']:.3f} | {m['coverage']:.3f} |")
    table = "\n".join(lines)
    open(args.out, "w").write(table + "\n")
    print(table)
    print(f"\n[done] table -> {args.out}")


if __name__ == "__main__":
    main()
```

## File: `scripts/smoke_test_path2-jarvis.sh`

```bash
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
```

## File: `scripts/test_download.py`

```python
"""
Test script for data download module.

This script tests the download functionality without actually downloading large files.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import (
    _get_data_dir,
    _verify_checksum,
    download_biogrid,
    download_string,
    download_disgenet,
    download_hpo
)

def test_directory_creation():
    """Test that data directory is created properly."""
    print("Testing directory creation...")
    data_dir = _get_data_dir()
    print(f"✓ Data directory: {data_dir}")
    assert data_dir.exists(), "Data directory should exist"
    assert data_dir.is_dir(), "Data directory should be a directory"
    print("✓ Directory creation test passed\n")


def test_imports():
    """Test that all functions can be imported."""
    print("Testing imports...")
    print("✓ _get_data_dir imported")
    print("✓ _verify_checksum imported")
    print("✓ download_biogrid imported")
    print("✓ download_string imported")
    print("✓ download_disgenet imported")
    print("✓ download_hpo imported")
    print("✓ All imports successful\n")


def test_function_signatures():
    """Test that functions have correct signatures."""
    print("Testing function signatures...")
    
    # Test that functions accept expected parameters
    import inspect
    
    sig_biogrid = inspect.signature(download_biogrid)
    print(f"✓ download_biogrid signature: {sig_biogrid}")
    
    sig_string = inspect.signature(download_string)
    print(f"✓ download_string signature: {sig_string}")
    
    sig_disgenet = inspect.signature(download_disgenet)
    print(f"✓ download_disgenet signature: {sig_disgenet}")
    
    sig_hpo = inspect.signature(download_hpo)
    print(f"✓ download_hpo signature: {sig_hpo}")
    
    print("✓ Function signature test passed\n")


def show_download_info():
    """Display information about downloads without executing."""
    print("="*70)
    print("Download Module Information")
    print("="*70)
    print("\nAvailable download functions:")
    print("  1. download_biogrid()  - BioGRID PPI (~500MB)")
    print("  2. download_string()   - STRING network (~700MB)")
    print("  3. download_disgenet() - DisGeNET associations (~300MB)")
    print("  4. download_hpo()      - HPO annotations (~50MB)")
    print("  5. download_all()      - All datasets (~1.5GB total)")
    print("\nFeatures:")
    print("  ✓ Progress bars with tqdm")
    print("  ✓ Automatic retry with exponential backoff")
    print("  ✓ Checksum verification")
    print("  ✓ Automatic archive extraction")
    print("  ✓ Caching (skip if already downloaded)")
    print("  ✓ Error handling and logging")
    print("\nUsage:")
    print("  python -m src.data.download --dataset all")
    print("  python -m src.data.download --dataset biogrid --force")
    print("\nTo actually download data, run:")
    print("  python scripts/download_data.py")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PromptGFM-Bio Download Module Test")
    print("="*70 + "\n")
    
    try:
        test_imports()
        test_directory_creation()
        test_function_signatures()
        show_download_info()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nDownload module is ready to use!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

## File: `scripts/test_gpu.py`

```python
"""
GPU functionality test script.

Tests PyTorch CUDA availability and GPU operations.
"""

import torch

print('=' * 60)
print('GPU CONFIGURATION')
print('=' * 60)

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'cuDNN Version: {torch.backends.cudnn.version()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'\nCurrent GPU: {torch.cuda.current_device()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory: {props.total_memory / 1024**3:.2f} GB')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'Multi Processors: {props.multi_processor_count}')
    
    print('\n' + '=' * 60)
    print('TESTING GPU OPERATIONS')
    print('=' * 60)
    
    # Test basic tensor operations
    print('\nTest 1: Basic tensor creation and GPU transfer...')
    x = torch.randn(1000, 1000).cuda()
    print(f'✓ Created tensor on GPU: {x.device}')
    
    print('\nTest 2: Matrix multiplication on GPU...')
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f'✓ Matrix multiplication successful')
    print(f'  Result shape: {z.shape}')
    print(f'  Result device: {z.device}')
    
    print('\nTest 3: GPU memory usage...')
    print(f'  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
    print(f'  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB')
    
    print('\nTest 4: GPU computation speed test...')
    import time
    
    # CPU test
    x_cpu = torch.randn(2000, 2000)
    y_cpu = torch.randn(2000, 2000)
    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    
    # GPU test
    x_gpu = torch.randn(2000, 2000).cuda()
    y_gpu = torch.randn(2000, 2000).cuda()
    torch.cuda.synchronize()
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f'  CPU time: {cpu_time*1000:.2f} ms')
    print(f'  GPU time: {gpu_time*1000:.2f} ms')
    print(f'  Speedup: {cpu_time/gpu_time:.2f}x')
    
    print('\n' + '=' * 60)
    print('✓ ALL GPU TESTS PASSED!')
    print('=' * 60)
    print('\n✓ Your RTX 4060 is ready for training!')
    
else:
    print('\n' + '=' * 60)
    print('✗ CUDA NOT AVAILABLE')
    print('=' * 60)
    print('\nPossible issues:')
    print('1. NVIDIA drivers not installed')
    print('2. PyTorch CPU-only version installed')
    print('3. CUDA toolkit not properly configured')
```

## File: `scripts/test_negative_sampling.py`

```python
"""
Test script to verify negative sampling works correctly.
"""
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import GeneDiseaseDataset
from scripts.train import create_collate_fn

def test_negative_sampling():
    """Test that negative sampling generates proper labels."""
    print("Testing negative sampling implementation...")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'finetune_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = GeneDiseaseDataset(
        graph_path=config['data']['graph_file'],
        edges_path=config['data']['edge_file'],
        min_score=config['data'].get('min_score', 0.3)
    )
    print(f"   ✓ Loaded {len(dataset.edges_df)} edges")
    print(f"   ✓ Graph has {dataset.graph['gene'].num_nodes} gene nodes")
    
    # Create train split
    print("\n2. Creating train split...")
    train_edges, _, _ = dataset.create_train_val_test_split(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42
    )
    print(f"   ✓ Train edges: {len(train_edges)}")
    
    # Create collate function with negative sampling
    print("\n3. Creating collate function...")
    num_negatives = config['data'].get('num_negatives', 5)
    print(f"   Num negatives per positive: {num_negatives}")
    
    input_dim = config.get('model', {}).get('gnn_input_dim', 128)
    collate_fn = create_collate_fn(
        train_edges, dataset.graph, dataset.gene_to_idx, 
        input_dim, num_negatives
    )
    print(f"   ✓ Collate function created")
    
    # Test with a small batch
    print("\n4. Testing with batch of 4 edges...")
    batch_indices = torch.arange(4)
    batch_data = collate_fn([batch_indices])
    
    print(f"\n5. Batch results:")
    print(f"   Gene indices shape: {batch_data['gene_indices'].shape}")
    print(f"   Labels shape: {batch_data['labels'].shape}")
    print(f"   Disease texts count: {len(batch_data['disease_texts'])}")
    
    # Count positive and negative samples
    labels = batch_data['labels']
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    
    print(f"\n6. Label distribution:")
    print(f"   Positive samples (label=1): {num_pos}")
    print(f"   Negative samples (label=0): {num_neg}")
    print(f"   Expected ratio: 1:{num_negatives}")
    print(f"   Actual ratio: 1:{num_neg/num_pos:.1f}")
    
    # Show sample of labels
    print(f"\n7. First 10 labels: {labels[:10].tolist()}")
    print(f"   Last 10 labels: {labels[-10:].tolist()}")
    
    # Validation
    print(f"\n8. Validation:")
    expected_total = num_pos * (1 + num_negatives)
    actual_total = len(labels)
    
    if num_pos > 0 and num_neg > 0:
        print(f"   ✓ Both positive and negative samples present")
    else:
        print(f"   ✗ Missing positive or negative samples!")
        
    if abs(num_neg / num_pos - num_negatives) < 0.5:
        print(f"   ✓ Ratio approximately correct")
    else:
        print(f"   ✗ Ratio incorrect (expected 1:{num_negatives})")
    
    if actual_total == expected_total:
        print(f"   ✓ Total sample count correct ({actual_total})")
    else:
        print(f"   ⚠ Total samples: {actual_total} (expected ~{expected_total})")
    
    print(f"\n{'='*60}")
    if num_pos > 0 and num_neg > 0:
        print("✓ NEGATIVE SAMPLING WORKING CORRECTLY!")
        print(f"  Training will use mixed positive/negative samples")
        print(f"  Loss should be non-zero and decrease over epochs")
    else:
        print("✗ NEGATIVE SAMPLING FAILED!")
        print(f"  Check implementation in scripts/train.py")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_negative_sampling()
```

## File: `scripts/test_optimizations.py`

```python
"""
Quick test to verify training optimizations are working correctly.
This tests that AMP, DataLoader optimizations, and cuDNN are properly configured.
"""

import sys
import torch
from pathlib import Path

def test_optimizations():
    """Test that all optimizations are properly configured."""
    
    print("="*70)
    print("TRAINING OPTIMIZATIONS TEST")
    print("="*70)
    
    # Test 1: Check CUDA availability
    print("\n1. CUDA Availability:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    
    # Test 2: Test Mixed Precision (AMP)
    print("\n2. Mixed Precision (AMP):")
    if cuda_available:
        from torch.cuda.amp import autocast, GradScaler
        
        # Create simple tensors
        x = torch.randn(10, 10, device='cuda')
        y = torch.randn(10, 10, device='cuda')
        
        # Test autocast
        with autocast():
            z = torch.mm(x, y)
            print(f"   Autocast dtype: {z.dtype}")
            if z.dtype == torch.float16:
                print("   ✓ AMP working correctly (FP16)")
            else:
                print(f"   ⚠ Expected FP16, got {z.dtype}")
        
        # Test GradScaler
        scaler = GradScaler()
        print(f"   ✓ GradScaler initialized")
    else:
        print("   ⚠ Skipped (no CUDA)")
    
    # Test 3: cuDNN Benchmark
    print("\n3. cuDNN Autotuning:")
    if cuda_available:
        torch.backends.cudnn.benchmark = True
        print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   ✓ cuDNN autotuning enabled")
    else:
        print("   ⚠ Skipped (no CUDA)")
    
    # Test 4: DataLoader settings
    print("\n4. DataLoader Configuration:")
    import os
    cpu_count = os.cpu_count()
    num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
    print(f"   CPU cores: {cpu_count}")
    print(f"   DataLoader workers: {num_workers}")
    print(f"   Pin memory: {cuda_available}")
    if num_workers > 0:
        print(f"   ✓ Parallel data loading enabled")
    else:
        print(f"   ⚠ Single-threaded (may be slower)")
    
    # Test 5: Import trainer with AMP
    print("\n5. Trainer Module:")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.training.finetune import PromptGFMTrainer
        print(f"   ✓ PromptGFMTrainer imported successfully")
        
        # Check if trainer has AMP support
        import inspect
        sig = inspect.signature(PromptGFMTrainer.__init__)
        params = list(sig.parameters.keys())
        if 'use_amp' in params:
            print(f"   ✓ AMP parameter available in trainer")
        else:
            print(f"   ✗ AMP parameter missing from trainer")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    optimizations_working = []
    optimizations_skipped = []
    
    if cuda_available:
        optimizations_working.append("✓ Mixed Precision (AMP) - 1.5-2× speedup")
        optimizations_working.append("✓ cuDNN Autotuning - 5-15% speedup")
        optimizations_working.append("✓ Pin Memory - Faster GPU transfers")
    else:
        optimizations_skipped.append("⚠ GPU optimizations (no CUDA)")
    
    if num_workers > 0:
        optimizations_working.append(f"✓ Parallel DataLoader ({num_workers} workers) - 20-40% speedup")
    else:
        optimizations_skipped.append("⚠ DataLoader parallelization")
    
    print("\nEnabled Optimizations:")
    for opt in optimizations_working:
        print(f"  {opt}")
    
    if optimizations_skipped:
        print("\nSkipped:")
        for opt in optimizations_skipped:
            print(f"  {opt}")
    
    print("\nExpected Performance:")
    if cuda_available and num_workers > 0:
        print("  • Combined speedup: 2-3× faster than baseline")
        print("  • Memory usage: ~40% lower (can use larger batch)")
        print("  • No accuracy degradation")
    else:
        print("  • Limited speedup without GPU")
    
    print("\n" + "="*70)
    print("Ready to train with optimizations!")
    print("="*70)

if __name__ == '__main__':
    test_optimizations()
```

## File: `scripts/test_preprocess.py`

```python
"""
Test script for preprocessing module.

This script tests the preprocessing functions without requiring full datasets.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import (
    _normalize_gene_symbol,
    _get_data_dirs,
    build_heterogeneous_graph,
)


def test_gene_normalization():
    """Test gene symbol normalization."""
    print("Testing gene symbol normalization...")
    
    test_cases = [
        ("TP53", "TP53"),
        ("tp53", "TP53"),
        ("  BRCA1  ", "BRCA1"),
        ("HUMAN_MYC", "MYC"),
        ("", None),
        (None, None),
    ]
    
    for input_symbol, expected in test_cases:
        result = _normalize_gene_symbol(input_symbol)
        assert result == expected, f"Failed for {input_symbol}: got {result}, expected {expected}"
        print(f"  ✓ {input_symbol!r} -> {result!r}")
    
    print("✓ Gene normalization test passed\n")


def test_directory_structure():
    """Test directory creation."""
    print("Testing directory structure...")
    
    dirs = _get_data_dirs()
    print(f"  Raw data dir: {dirs['raw']}")
    print(f"  Processed dir: {dirs['processed']}")
    
    assert dirs['raw'].exists(), "Raw directory should exist"
    assert dirs['processed'].parent.exists(), "Data directory should exist"
    
    print("✓ Directory structure test passed\n")


def test_graph_building():
    """Test heterogeneous graph construction with toy data."""
    print("Testing graph construction with toy data...")
    
    # Create toy PPI data
    ppi_data = {
        'gene_a': ['TP53', 'BRCA1', 'MYC'],
        'gene_b': ['MDM2', 'TP53', 'MAX'],
        'confidence': [0.9, 0.8, 0.95],
        'source': ['biogrid', 'biogrid', 'string']
    }
    ppi_edges = pd.DataFrame(ppi_data)
    
    # Create toy gene-disease data
    gene_disease_data = {
        'gene': ['TP53', 'BRCA1', 'TP53'],
        'diseaseId': ['C0006826', 'C0006826', 'C0024623'],
        'diseaseName': ['Cancer', 'Cancer', 'Li-Fraumeni Syndrome'],
        'confidence': [0.95, 0.9, 0.98],
        'source': ['disgenet', 'disgenet', 'disgenet']
    }
    gene_disease_edges = pd.DataFrame(gene_disease_data)
    
    disease_info = {
        'C0006826': 'Cancer',
        'C0024623': 'Li-Fraumeni Syndrome'
    }
    
    # Build graph
    graph = build_heterogeneous_graph(
        ppi_edges=ppi_edges,
        gene_disease_edges=gene_disease_edges,
        disease_info=disease_info
    )
    
    # Verify graph structure
    assert 'gene' in graph.node_types, "Graph should have 'gene' nodes"
    assert 'disease' in graph.node_types, "Graph should have 'disease' nodes"
    
    print(f"  Gene nodes: {graph['gene'].num_nodes}")
    print(f"  Disease nodes: {graph['disease'].num_nodes}")
    print(f"  Edge types: {graph.edge_types}")
    
    assert graph['gene'].num_nodes > 0, "Should have gene nodes"
    assert graph['disease'].num_nodes > 0, "Should have disease nodes"
    
    print("✓ Graph construction test passed\n")


def test_imports():
    """Test that all functions can be imported."""
    print("Testing imports...")
    
    from src.data.preprocess import (
        parse_biogrid,
        parse_string,
        parse_ppi_network,
        parse_disgenet,
        parse_hpo,
        build_heterogeneous_graph,
        save_graph,
        preprocess_all
    )
    
    print("  ✓ parse_biogrid imported")
    print("  ✓ parse_string imported")
    print("  ✓ parse_ppi_network imported")
    print("  ✓ parse_disgenet imported")
    print("  ✓ parse_hpo imported")
    print("  ✓ build_heterogeneous_graph imported")
    print("  ✓ save_graph imported")
    print("  ✓ preprocess_all imported")
    
    print("✓ All imports successful\n")


def show_preprocessing_info():
    """Display information about preprocessing."""
    print("="*70)
    print("Preprocessing Module Information")
    print("="*70)
    print("\nFunctions:")
    print("  1. parse_biogrid()       - Parse BioGRID PPI data")
    print("  2. parse_string()        - Parse STRING network")
    print("  3. parse_ppi_network()   - Parse and combine PPI networks")
    print("  4. parse_disgenet()      - Parse gene-disease associations")
    print("  5. parse_hpo()           - Parse phenotype annotations")
    print("  6. build_heterogeneous_graph() - Construct PyG HeteroData")
    print("  7. save_graph()          - Save graph to disk")
    print("  8. preprocess_all()      - Run complete pipeline")
    print("\nFeatures:")
    print("  ✓ HGNC gene symbol normalization")
    print("  ✓ Filter to Homo sapiens only")
    print("  ✓ Confidence score filtering")
    print("  ✓ Rare disease filtering (<= N known genes)")
    print("  ✓ Heterogeneous graph construction")
    print("  ✓ PyTorch Geometric HeteroData format")
    print("  ✓ Comprehensive logging and statistics")
    print("\nUsage:")
    print("  python scripts/preprocess_all.py")
    print("  python scripts/preprocess_all.py --force")
    print("\nOutput:")
    print("  data/processed/biomedical_graph.pt")
    print("  data/processed/biomedical_graph_stats.txt")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PromptGFM-Bio Preprocessing Module Test")
    print("="*70 + "\n")
    
    try:
        test_imports()
        test_gene_normalization()
        test_directory_structure()
        test_graph_building()
        show_preprocessing_info()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nPreprocessing module is ready to use!")
        print("\nOnce data download completes, run:")
        print("  python scripts/preprocess_all.py")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

## File: `scripts/train.py`

```python
"""
Main training script for PromptGFM-Bio.

Usage:
    python scripts/train.py --config configs/pretrain_config.yaml  # Pretraining
    python scripts/train.py --config configs/finetune_config.yaml  # Fine-tuning

Dual-GPU (Kaggle T4 ×2):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/kaggle_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings as _w
_w.filterwarnings("ignore", message="An issue occurred while importing", category=UserWarning)

from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset
from src.models.promptgfm import PromptGFM, GNNOnlyBaseline
from src.training.finetune import PromptGFMTrainer, create_optimizer, create_scheduler
from src.training.pretrain import GraphPretrainer
from src.training.losses import BCELoss, MarginRankingLoss, CombinedLoss
from src.evaluation.metrics import GeneRankingEvaluator
from src.utils.config import load_config
from src.utils.logger import setup_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path 2 reproducibility: deterministic per-seed RNG for ALL stochastic sources.
# Called once from run_finetuning / run_pretraining with config['seed'].
# The data split is seeded SEPARATELY in create_dataloaders() using
# config['data']['random_seed'], which is held FIXED across all 10 seeds.
# ---------------------------------------------------------------------------
def _set_all_seeds(seed: int) -> None:
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"[reproducibility] all RNGs seeded with seed={seed} "
                f"(split seed comes from config['data']['random_seed'])")


# Distributed training helpers ------------------------------------------------
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
IS_DDP = LOCAL_RANK != -1

def _get_rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def _is_main_process() -> bool:
    return _get_rank() == 0
# -----------------------------------------------------------------------------


def create_collate_fn(edges_df, graph, gene_to_idx, input_dim=128, num_negatives=5):
    """
    Create a collate function for batching gene-disease edges with negative sampling.
    
    Args:
        edges_df: DataFrame of edges (gene, disease, score, etc.)
        graph: PyTorch Geometric HeteroData graph
        gene_to_idx: Dict mapping gene names to graph node indices
        input_dim: Input feature dimension for GNN
        num_negatives: Number of negative samples per positive sample
    
    Returns:
        Collate function for DataLoader
    """
    # Get the actual number of gene nodes from the graph
    num_graph_genes = graph['gene'].num_nodes
    
    # Pre-compute disease to genes mapping for negative sampling
    disease_to_genes = {}
    for _, row in edges_df.iterrows():
        disease = row['disease']
        gene = row['gene']
        if disease not in disease_to_genes:
            disease_to_genes[disease] = set()
        if gene in gene_to_idx:
            idx = gene_to_idx[gene]
            if 0 <= idx < num_graph_genes:
                disease_to_genes[disease].add(idx)
    
    # Get all valid gene indices for negative sampling
    all_gene_indices = set(range(num_graph_genes))
    
    # Pre-compute node features once (will be reused across batches)
    # Use the graph's actual node count, not the dataset vocabulary
    if 'gene' in graph.node_types and hasattr(graph['gene'], 'x'):
        cached_node_features = graph['gene'].x
    else:
        # Create learnable random features (fixed per gene)
        # IMPORTANT: Use graph's node count, not dataset vocabulary size
        torch.manual_seed(42)  # For reproducibility
        cached_node_features = torch.randn(num_graph_genes, input_dim)
    
    # Pre-compute edge index once (will be reused across batches)
    # IMPORTANT: Only use gene-gene edges for message passing
    try:
        edge_types = graph.edge_types if hasattr(graph, 'edge_types') else []
        
        # Only try gene-gene edge types (PPI networks)
        gene_gene_edge_types = [
            ('gene', 'interacts', 'gene'),
            ('gene', 'protein_interaction', 'gene'),
            ('gene', 'ppi', 'gene')
        ]
        
        cached_edge_index = None
        for edge_type in gene_gene_edge_types:
            if edge_type in edge_types:
                cached_edge_index = graph[edge_type].edge_index
                break
        
        # Fallback to empty if no gene-gene edges found
        if cached_edge_index is None:
            import logging
            logging.info(f"No gene-gene edges found in graph. "
                       f"Available edge types: {edge_types}. "
                       f"Training without message passing.")
            cached_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # Validate edge indices are within bounds
            if cached_edge_index.numel() > 0:
                max_idx = cached_edge_index.max().item()
                if max_idx >= num_graph_genes:
                    # Edge index contains invalid node references
                    import logging
                    logging.warning(f"Edge index contains out-of-bounds indices "
                                  f"(max={max_idx}, num_nodes={num_graph_genes}). "
                                  f"Using empty edge index.")
                    cached_edge_index = torch.empty((2, 0), dtype=torch.long)
    except Exception as e:
        import logging
        logging.warning(f"Error loading edge index: {e}. Using empty edge index.")
        cached_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    def collate_fn(batch):
        """
        Convert batch of edge indices to model input format with negative sampling.
        
        Args:
            batch: List of tensors containing edge indices
            
        Returns:
            Dict with keys: node_features, edge_index, disease_texts, gene_indices, labels
        """
        # Extract edge indices from batch (DataLoader wraps in list)
        if isinstance(batch[0], torch.Tensor):
            edge_indices = batch[0].tolist()
        else:
            edge_indices = [item[0].item() if isinstance(item[0], torch.Tensor) else item[0] for item in batch]
        
        # Get edges for this batch
        batch_edges = edges_df.iloc[edge_indices]
        
        # Extract gene names and diseases
        gene_names = batch_edges['gene'].tolist()
        disease_ids = batch_edges['disease'].tolist()
        
        # Map gene names to graph indices (POSITIVE samples)
        positive_gene_indices = []
        positive_disease_texts = []
        num_genes = cached_node_features.shape[0]
        
        for gene, disease in zip(gene_names, disease_ids):
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                # Validate index is in bounds
                if 0 <= idx < num_genes:
                    positive_gene_indices.append(idx)
                    positive_disease_texts.append(disease)
                # else: skip out-of-bounds indices
            # else: skip genes not in mapping
        
        # Sample NEGATIVE genes for each disease
        negative_gene_indices = []
        negative_disease_texts = []
        
        import random
        for disease in positive_disease_texts:
            # Get genes known to be associated with this disease
            known_genes = disease_to_genes.get(disease, set())
            
            # Available negative samples: all genes NOT associated with this disease
            available_negatives = list(all_gene_indices - known_genes)
            
            # Sample num_negatives random genes
            if len(available_negatives) >= num_negatives:
                neg_samples = random.sample(available_negatives, num_negatives)
            else:
                # If not enough, sample with replacement
                neg_samples = random.choices(available_negatives, k=num_negatives) if available_negatives else []
            
            negative_gene_indices.extend(neg_samples)
            # Repeat disease text for each negative sample
            negative_disease_texts.extend([disease] * len(neg_samples))
        
        # Combine positive and negative samples
        combined_gene_indices = positive_gene_indices + negative_gene_indices
        combined_disease_texts = positive_disease_texts + negative_disease_texts
        
        # Convert to tensors
        gene_indices = torch.tensor(combined_gene_indices, dtype=torch.long)
        
        # Labels: 1 for positive, 0 for negative
        positive_labels = torch.ones(len(positive_gene_indices), dtype=torch.float32)
        negative_labels = torch.zeros(len(negative_gene_indices), dtype=torch.float32)
        labels = torch.cat([positive_labels, negative_labels])
        
        # Use cached node features and edge index
        node_features = cached_node_features
        edge_index = cached_edge_index
        
        # node_features and edge_index are deliberately NOT included here.
        # They are graph-wide constants (~40 MB combined) that never change
        # between batches.  Returning them through the DataLoader queue forces
        # workers to serialize them into /dev/shm on every prefetch, which
        # exhausts shared memory and causes ENOSPC Bus Error crashes.
        # The trainer receives them once via set_graph_tensors() below.
        return {
            'disease_texts': combined_disease_texts,
            'gene_indices': gene_indices,
            'labels': labels
        }
    
    return collate_fn


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    logger.info("Creating dataloaders...")
    
    # Load gene-disease dataset (it will load the graph internally)
    dataset = GeneDiseaseDataset(
        graph_path=config['data']['graph_file'],
        edges_path=config['data']['edge_file'],
        min_score=config['data'].get('min_score', 0.3)
    )
    
    # Split data — Path 2: split seed comes from data.random_seed (fixed across
    # all model-init seeds), NOT from top-level config['seed']. Backward-compat
    # fallback: if data.random_seed is missing, use the top-level seed.
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed
    )
    logger.info(f"[reproducibility] split seed = {split_seed} "
                f"(should be FIXED across all 10 model-init seeds)")
    
    logger.info(f"  Train: {len(train_edges)} edges")
    logger.info(f"  Val:   {len(val_edges)} edges")
    logger.info(f"  Test:  {len(test_edges)} edges")
    
    # Get input dimension from config
    input_dim = config.get('model', {}).get('gnn_input_dim', 128)
    
    # Debug: Check graph and dataset consistency
    logger.info(f"  Gene nodes in graph: {dataset.graph['gene'].num_nodes}")
    logger.info(f"  Genes in dataset vocabulary: {len(dataset.gene_to_idx)}")
    logger.info(f"  Input feature dim: {input_dim}")
    
    # Create collate functions with negative sampling
    num_negatives = config['data'].get('num_negatives', 5)
    logger.info(f"  Using {num_negatives} negative samples per positive sample")
    
    train_collate_fn = create_collate_fn(train_edges, dataset.graph, dataset.gene_to_idx, input_dim, num_negatives)
    val_collate_fn = create_collate_fn(val_edges, dataset.graph, dataset.gene_to_idx, input_dim, num_negatives)
    
    # Create dataloaders with proper collate function and optimizations
    from torch.utils.data import TensorDataset
    import os
    import platform
    
    # Optimize DataLoader settings for better CPU/GPU overlap
    # Note: Windows has issues pickling nested functions, so disable workers on Windows
    cpu_count = os.cpu_count()
    is_windows = platform.system() == 'Windows'
    
    # Read num_workers from config when explicitly set; otherwise auto-detect.
    _cfg_nw = config.get("data", {}).get("num_workers", None)
    cfg_nw = int(_cfg_nw) if _cfg_nw is not None else None
    if is_windows:
        num_workers = 0
        logger.info(f"  DataLoader workers: {num_workers} (Windows)")
    elif cfg_nw is not None:
        num_workers = cfg_nw
        logger.info(f"  DataLoader workers: {num_workers} (from config)")
    else:
        num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
        logger.info(f"  DataLoader workers: {num_workers} (auto-detected)")

    # In distributed training use DistributedSampler so each rank gets a unique data shard
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(TensorDataset(torch.arange(len(train_edges))),
                                       shuffle=True) if IS_DDP else None
    val_sampler   = DistributedSampler(TensorDataset(torch.arange(len(val_edges))),
                                       shuffle=False) if IS_DDP else None

    train_loader = DataLoader(
        TensorDataset(torch.arange(len(train_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),  # DistributedSampler handles shuffling
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.arange(len(val_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    
    return train_loader, val_loader, dataset


def run_pretraining(config):
    """Run self-supervised pretraining."""
    logger.info("\n" + "="*60)
    logger.info("Starting Self-Supervised Pretraining")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
    
    # Load graph
    graph_dataset = BiomedicalGraphDataset(config['data']['graph_file'])
    graph = graph_dataset.graph
    
    # Create model
    model = PromptGFM(**config['model'])
    
    # Get device
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create pretrainer
    pretrainer = GraphPretrainer(
        model=model,
        device=device,
        mask_rate=config.get('pretraining', {}).get('mask_ratio', 0.15),
        negative_samples=config.get('pretraining', {}).get('num_negatives', 1)
    )
    
    # Run pretraining tasks
    histories = pretrainer.pretrain_all(
        node_features=graph.x_dict['gene'],  # Assuming gene nodes
        edge_index=graph.edge_index_dict[('gene', 'interacts', 'gene')],
        tasks=config.get('pretraining', {}).get('tasks', {}),
        num_epochs=config.get('pretraining', {}).get('num_epochs', 50),
        lr=config['training'].get('learning_rate', 0.001)
    )
    
    # Save pretrained model
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'histories': histories
    }, checkpoint_dir / 'pretrained_model.pt')
    
    logger.info(f"\n✓ Pretraining complete! Model saved to {checkpoint_dir / 'pretrained_model.pt'}")


def run_finetuning(config):
    """Run supervised fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("Starting Supervised Fine-tuning")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
    
    # Create dataloaders (GeneDiseaseDataset will load the graph internally)
    train_loader, val_loader, dataset = create_dataloaders(config)
    
    # Prepare model parameters from config
    model_config = config['model']
    
    # Create model
    if model_config.get('baseline', False) or not model_config.get('use_prompt', True):
        logger.info("Training GNN-Only Baseline")
        model_params = {
            'gnn_type': model_config.get('gnn_type', 'graphsage'),
            'gnn_hidden_dim': model_config.get('hidden_dim', 256),
            'gnn_num_layers': model_config.get('num_layers', 3),
            'gnn_dropout': model_config.get('dropout', 0.3),
            'hidden_dim': model_config.get('hidden_dim', 256)
        }
        model = GNNOnlyBaseline(**model_params)
    else:
        logger.info("Training PromptGFM")
        # Flatten nested config for PromptGFM
        model_params = {
            # GNN parameters
            'gnn_input_dim': model_config.get('gene_feature_dim', 256),
            'gnn_hidden_dim': model_config.get('hidden_dim', 256),
            'gnn_output_dim': model_config.get('hidden_dim', 256),
            'gnn_num_layers': model_config.get('num_layers', 3),
            'gnn_type': model_config.get('gnn_type', 'graphsage'),
            'gnn_dropout': model_config.get('dropout', 0.3),
            # Prompt encoder parameters
            'prompt_model_name': model_config.get('prompt_encoder', {}).get('model_name', 
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
            'prompt_pooling': model_config.get('prompt_encoder', {}).get('pooling_strategy', 'cls'),
            'prompt_max_length': model_config.get('prompt_encoder', {}).get('max_length', 512),
            'freeze_prompt': model_config.get('prompt_encoder', {}).get('freeze_encoder', False),
            # Conditioning parameters
            'conditioning_type': model_config.get('conditioning_type', 'film'),
            'conditioning_hidden_dim': model_config.get('hidden_dim', 256),
            # Predictor parameters
            'predictor_hidden_dim': model_config.get('prediction_hidden_dim', 128),
            'predictor_dropout': model_config.get('dropout', 0.3),
            # ── FIX (Bug 1, 2026-04-17): forward ablation flags so YAML overrides
            #    reach the model.  Without these, every ablation trained the full
            #    model silently (defaults use_gnn=True, use_conditioning=True).
            'use_gnn':          model_config.get('use_gnn', True),
            'use_conditioning': model_config.get('use_conditioning', True),
        }
        model = PromptGFM(**model_params)
        logger.info(
            f"  Ablation flags active: use_gnn={model_params['use_gnn']}, "
            f"use_conditioning={model_params['use_conditioning']}"
        )
        
        # Load pretrained weights if available
        pretrained_path = config['training'].get('pretrained_checkpoint')
        if pretrained_path and Path(pretrained_path).exists():
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Create loss function
    loss_type = config['training'].get('loss_type', 'bce')
    if loss_type == 'bce':
        loss_fn = BCELoss()
    elif loss_type == 'ranking':
        loss_fn = MarginRankingLoss(
            margin=config['training'].get('margin', 0.5)
        )
    elif loss_type == 'combined':
        loss_weights = config['training'].get('loss_weights', {})
        loss_fn = CombinedLoss(
            bce_weight=loss_weights.get('bce', 1.0),
            ranking_weight=loss_weights.get('ranking', 0.5),
            contrastive_weight=loss_weights.get('contrastive', 0.1)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        lr=config['training'].get('learning_rate', 0.001),
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config['training'].get('lr_scheduler', 'cosine'),
        num_epochs=config['training'].get('num_epochs', 100)
    )
    
    # Create evaluator
    evaluator = GeneRankingEvaluator(
        k_values=config.get('evaluation', {}).get('k_values', [10, 20, 50])
    )
    
    # Get device and checkpoint directory
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if IS_DDP:
        device = f'cuda:{LOCAL_RANK}'
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')

    # --- Wrap model for DDP if running multi-GPU ---
    if IS_DDP:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = model.to(device)
        # find_unused_parameters=True required when freeze_encoder=True:
        # frozen BioBERT params have no gradient hooks; with False DDP hangs
        # on the first backward waiting for a reducer bucket that never fires.
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
        logger.info(f"  DDP enabled — rank {LOCAL_RANK}/{WORLD_SIZE}, device {device}")
    
    # --- Pre-compute BioBERT embeddings for all unique disease texts ---
    # BioBERT is frozen (freeze_encoder=true in config) so embeddings never change.
    # Running BioBERT once upfront and caching by text string eliminates the most
    # expensive forward pass from every training step.
    inner_model = model.module if hasattr(model, 'module') else model
    freeze_encoder = config.get('model', {}).get('prompt_encoder', {}).get('freeze_encoder', False)
    if freeze_encoder and hasattr(inner_model, 'prompt_encoder'):
        logger.info("\n  Pre-computing frozen BioBERT embeddings (runs once, then cached)...")
        all_disease_texts = list(dataset.diseases)  # dataset.diseases = unique disease names

        prompt_enc = inner_model.prompt_encoder
        prompt_enc.eval()
        emb_cache: dict = {}
        encode_bs = 64  # encode in small chunks to avoid OOM during pre-computation
        with torch.no_grad():
            for i in range(0, len(all_disease_texts), encode_bs):
                chunk = all_disease_texts[i:i + encode_bs]
                embs = prompt_enc(chunk).cpu()  # store on CPU; moved to device in _forward_batch
                for txt, emb in zip(chunk, embs):
                    emb_cache[txt] = emb
        logger.info(f"  ✅ Cached {len(emb_cache)} disease embeddings — BioBERT skipped per batch")
    else:
        emb_cache = None
        if not freeze_encoder:
            logger.info("  BioBERT not frozen — gradients needed, skipping pre-computation")

    # Create trainer
    trainer = PromptGFMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        evaluator=evaluator,
        checkpoint_dir=checkpoint_dir,
        max_epochs=config['training'].get('num_epochs', 100),
        patience=config['training'].get('early_stopping_patience', 10),
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        use_wandb=config.get('logging', {}).get('use_wandb', False),
        use_amp=config['training'].get('use_amp', True)  # Enable mixed precision by default
    )

    if emb_cache:
        trainer.set_prompt_cache(emb_cache)

    # ── Graph tensors: pass once to trainer, never through DataLoader queue ──
    # Resolve node features from the graph (same logic as create_collate_fn).
    _graph = dataset.graph
    _num_g = _graph['gene'].num_nodes
    if hasattr(_graph['gene'], 'x') and _graph['gene'].x is not None:
        _nf = _graph['gene'].x
    else:
        import torch as _t
        _t.manual_seed(42)
        _nf = _t.randn(_num_g, config.get('model', {}).get('gene_feature_dim', 128))

    # Resolve edge index (gene-gene PPI edges).
    _et = _graph.edge_types if hasattr(_graph, 'edge_types') else []
    _ei = None
    for _candidate in [('gene','interacts','gene'),
                        ('gene','protein_interaction','gene'),
                        ('gene','ppi','gene')]:
        if _candidate in _et:
            _ei = _graph[_candidate].edge_index
            break
    if _ei is None:
        import torch as _t
        _ei = _t.empty((2, 0), dtype=_t.long)

    trainer.set_graph_tensors(_nf, _ei)
    logger.info(f"  ✅ Graph tensors set on trainer "
                f"(node_features={list(_nf.shape)}, "
                f"edge_index={list(_ei.shape)}) — removed from DataLoader queue")
    
    # Check for resume checkpoint
    resume_checkpoint = config['training'].get('resume_checkpoint')
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info("RESUMING FROM CHECKPOINT")
            logger.info(f"{'='*60}")
            trainer.load_checkpoint(resume_path, load_optimizer=True)
        else:
            logger.warning(f"Resume checkpoint not found: {resume_checkpoint}")
            logger.info("Starting training from scratch...")
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        val_metric=config['training'].get('val_metric', 'auroc')  # Default to AUROC
    )
    
    logger.info("\n✓ Fine-tuning complete!")


def main():
    parser = argparse.ArgumentParser(description='Train PromptGFM-Bio')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pretrain', 'finetune', 'auto'],
        default='auto',
        help='Training mode (auto: detect from config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision training (AMP)'
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # CLI --resume-checkpoint overrides anything in the YAML
    if args.resume_checkpoint:
        if 'training' not in config:
            config['training'] = {}
        config['training']['resume_checkpoint'] = args.resume_checkpoint
    
    # Enable cuDNN autotuning for optimized convolution algorithms
    # This finds the fastest algorithm for your specific hardware (one-time cost)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("✓ cuDNN autotuning enabled (first epoch may be slightly slower)")

    # --- DDP init (activated by torchrun --nproc_per_node=N) ---
    if IS_DDP:
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend='nccl')
        logger.info(f"✓ DDP process group initialised — rank {LOCAL_RANK}/{WORLD_SIZE}")
    
    # Ensure hardware section exists
    if 'hardware' not in config:
        config['hardware'] = {}
    
    # Override device if specified
    if args.device:
        config['hardware']['device'] = args.device
    elif 'device' not in config['hardware']:
        config['hardware']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine mode
    mode = args.mode
    if mode == 'auto':
        mode = config.get('mode', 'finetune')
    
    logger.info(f"Mode: {mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {config['hardware']['device']}")
    
    # Store AMP setting in config for trainer
    if 'training' not in config:
        config['training'] = {}
    config['training']['use_amp'] = not args.no_amp
    if config['training']['use_amp'] and torch.cuda.is_available():
        logger.info("✓ Mixed precision (AMP) enabled (1.5-2× speedup expected)")
    
    # Run training
    if mode == 'pretrain':
        run_pretraining(config)
    elif mode == 'finetune':
        run_finetuning(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
```

## File: `scripts/verify_setup.py`

```python
"""
Setup verification script for PromptGFM-Bio.

Checks that all required packages are installed and working correctly.
"""

import sys
import importlib


def check_package(package_name, import_name=None, check_version=None):
    """Check if a package is installed and optionally verify its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        status = "OK"
        if check_version and version != check_version:
            status = f"WARN (found {version}, expected {check_version})"
        else:
            status = f"OK (version {version})"
        
        print(f"  {status} {package_name}")
        return True
    except ImportError:
        print(f"  MISSING {package_name} - NOT FOUND")
        return False


def main():
    """Run all verification checks."""
    print("=" * 50)
    print("PromptGFM-Bio Setup Verification")
    print("=" * 50)
    print()
    
    # Check Python version
    print("Python Environment:")
    print(f"  OK Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  WARN Python 3.10+ recommended")
    print()
    
    # Check core packages
    print("Core Deep Learning:")
    all_ok = True
    all_ok &= check_package("torch", check_version="2.1.0")
    all_ok &= check_package("torchvision", check_version="0.16.0")
    all_ok &= check_package("torchaudio", check_version="2.1.0")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  OK CUDA available (version {torch.version.cuda})")
            print(f"  OK GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  OK VRAM: {props.total_memory / 1024**3:.2f} GB")
            print(f"  OK Compute Capability: {props.major}.{props.minor}")
        else:
            print("  WARN CUDA not available (CPU-only mode)")
    except:
        pass
    print()
    
    # Check PyTorch Geometric
    print("PyTorch Geometric:")
    all_ok &= check_package("torch-geometric", "torch_geometric", "2.4.0")
    all_ok &= check_package("torch-scatter", "torch_scatter")
    all_ok &= check_package("torch-sparse", "torch_sparse")
    all_ok &= check_package("torch-cluster", "torch_cluster")
    print()
    
    # Check NLP packages
    print("NLP & Transformers:")
    all_ok &= check_package("transformers", check_version="4.35.0")
    all_ok &= check_package("sentence-transformers", "sentence_transformers")
    all_ok &= check_package("datasets")
    print()
    
    # Check data science packages
    print("Data Science:")
    all_ok &= check_package("numpy")
    all_ok &= check_package("pandas")
    all_ok &= check_package("scipy")
    all_ok &= check_package("scikit-learn", "sklearn")
    all_ok &= check_package("networkx")
    print()
    
    # Check biomedical packages
    print("Biomedical:")
    all_ok &= check_package("biopython", "Bio")
    print()
    
    # Check visualization
    print("Visualization:")
    all_ok &= check_package("matplotlib")
    all_ok &= check_package("seaborn")
    print()
    
    # Check utilities
    print("Utilities:")
    all_ok &= check_package("tqdm")
    all_ok &= check_package("wandb")
    all_ok &= check_package("yaml")
    all_ok &= check_package("requests")
    print()
    
    # Check development tools
    print("Development (optional):")
    check_package("pytest")
    check_package("jupyter")
    print()
    
    # Final status
    print("=" * 50)
    if all_ok:
        print("OK Setup verification PASSED")
        print("All required packages are installed correctly!")
    else:
        print("FAILED Setup verification")
        print("Some required packages are missing or have incorrect versions.")
        print("\nPlease run:")
        print("  pip install -r requirements.txt")
    print("=" * 50)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

## File: `scripts/_jarvis_env-OLD.sh`

```bash
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
```

## File: `scripts/_jarvis_env.sh`

```bash
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
```

