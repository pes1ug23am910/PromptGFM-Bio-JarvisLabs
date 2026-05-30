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


def _dump_per_disease(rankings, disease_ids, out_path, k_values):
    """Path 2 (additive, guarded): write per-disease ranking records so the
    aggregator can bootstrap-resample the 117 zero-shot diseases for Hit@K and
    MRR. Mirrors metrics.py exactly: 1-indexed rank of the FIRST positive;
    Hit@K == (best_rank <= K). Pooled AUROC is NOT a per-disease mean, so a
    *macro* per-disease AUROC is stored separately (clearly different from the
    headline pooled AUROC). Wrapped so it can never alter metrics or crash the
    eval — on any error it logs a warning and returns."""
    try:
        import numpy as _np
        import json as _json
        import os as _os
        from sklearn.metrics import roc_auc_score as _auroc
        records = {}
        for did, (labels, scores) in zip(disease_ids, rankings):
            labels = _np.asarray(labels)
            scores = _np.asarray(scores)
            order = _np.argsort(scores)[::-1]          # descending == metrics.py
            sorted_labels = labels[order]
            pos = _np.where(sorted_labels == 1)[0]
            best_rank = int(pos[0] + 1) if pos.size > 0 else None   # 1-indexed
            rec = {
                "num_true": int(labels.sum()),
                "best_rank": best_rank,
                "rr": (1.0 / best_rank) if best_rank is not None else 0.0,
            }
            for k in k_values:
                rec[f"hit@{k}"] = 1 if (best_rank is not None and best_rank <= k) else 0
            try:
                rec["auroc"] = (float(_auroc(labels, scores))
                                if 0 < labels.sum() < labels.size else None)
            except Exception:
                rec["auroc"] = None
            records[str(did)] = rec
        _os.makedirs(_os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as _f:
            _json.dump(records, _f, indent=2)
        logger.info(f"  [per-disease] wrote {len(records)} records -> {out_path}")
    except Exception as _e:
        logger.warning(f"  [per-disease] dump skipped ({_e!r}); metrics unaffected.")


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

def evaluate_split(model, dataset, edges_df, config, device='cuda', train_edges_df=None,
                   per_disease_out=None):
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
    ranked_disease_ids = []  # Path 2: kept in lock-step with `rankings`

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
            ranked_disease_ids.append(disease)  # Path 2: alignment with rankings

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

    if per_disease_out:
        _dump_per_disease(rankings, ranked_disease_ids, per_disease_out, k_values)

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
                       train_edges_df=None, zero_shot_json='data/splits/zero_shot_rare_diseases.json',
                       per_disease_out=None):
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
        per_disease_out=per_disease_out,
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
        zs_per_disease = str(Path(args.zero_shot_output).with_suffix("")) + "_per_disease.json"
        zs_metrics = evaluate_zero_shot(
            model,
            dataset,
            dataset.edges,          # full edge set — ground truth for all diseases
            config,
            device,
            train_edges_df=train_edges,
            zero_shot_json=args.zero_shot_file,
            per_disease_out=zs_per_disease,
        )
        if zs_metrics:
            save_results(zs_metrics, args.zero_shot_output)

    # ── Save ───────────────────────────────────────────────────────────────
    save_results(all_results, args.output)
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
