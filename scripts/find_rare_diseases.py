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
