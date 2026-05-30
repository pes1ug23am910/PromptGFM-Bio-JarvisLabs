"""
Patch an existing biomedical graph with STRING gene-gene edges.

This utility is intended for cases where the graph was built without STRING
edges (for example due to missing .txt/.txt.gz path handling during preprocessing).

Usage:
    python add_string_ppi_edges.py
    python add_string_ppi_edges.py --min-score 700 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "biomedical_graph.pt"
DEFAULT_STRING_DIR = PROJECT_ROOT / "data" / "raw" / "string"
EDGE_TYPE = ("gene", "interacts", "gene")


def _resolve_first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_graph_cpu(path: Path):
    """Load a PyG graph on CPU across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_gene_symbol(symbol: str) -> Optional[str]:
    if pd.isna(symbol) or not symbol:
        return None
    symbol = str(symbol).strip().upper()
    return symbol if symbol else None


def _load_protein_to_gene_map(info_path: Path) -> Dict[str, str]:
    info_df = pd.read_csv(info_path, sep="\t", compression="infer", low_memory=False)

    normalized_cols = {col.strip().lstrip("#").lower(): col for col in info_df.columns}

    protein_col = None
    for candidate in ("protein_external_id", "string_protein_id", "protein_id"):
        if candidate in normalized_cols:
            protein_col = normalized_cols[candidate]
            break

    gene_col = None
    for candidate in ("preferred_name", "gene_name", "gene"):
        if candidate in normalized_cols:
            gene_col = normalized_cols[candidate]
            break

    if protein_col is None or gene_col is None:
        raise ValueError(
            "Could not identify protein->gene mapping columns in STRING info file. "
            f"Found columns: {list(info_df.columns)}"
        )

    protein_to_gene = {
        str(protein_id): _normalize_gene_symbol(gene_name)
        for protein_id, gene_name in zip(info_df[protein_col], info_df[gene_col])
        if pd.notna(protein_id)
    }
    protein_to_gene = {
        protein_id: gene_name
        for protein_id, gene_name in protein_to_gene.items()
        if gene_name
    }

    return protein_to_gene


def _parse_string_edges(
    links_path: Path,
    protein_to_gene: Dict[str, str],
    min_score: int,
) -> pd.DataFrame:
    links_df = pd.read_csv(links_path, sep=r"\s+", engine="python", compression="infer")

    required_cols = {"protein1", "protein2", "combined_score"}
    missing_cols = required_cols - set(links_df.columns)
    if missing_cols:
        raise ValueError(f"STRING links file missing required columns: {sorted(missing_cols)}")

    links_df = links_df[links_df["combined_score"] >= min_score].copy()

    links_df["gene_a"] = links_df["protein1"].map(protein_to_gene)
    links_df["gene_b"] = links_df["protein2"].map(protein_to_gene)
    links_df = links_df.dropna(subset=["gene_a", "gene_b"])
    links_df = links_df[links_df["gene_a"] != links_df["gene_b"]]

    links_df["confidence"] = links_df["combined_score"] / 1000.0

    return links_df[["gene_a", "gene_b", "confidence"]]


def _build_existing_edge_conf_map(graph) -> Dict[Tuple[int, int], float]:
    edge_conf: Dict[Tuple[int, int], float] = {}

    if EDGE_TYPE not in graph.edge_types:
        return edge_conf

    edge_index = graph[EDGE_TYPE].edge_index
    if edge_index.numel() == 0:
        return edge_conf

    edge_attr = None
    if hasattr(graph[EDGE_TYPE], "edge_attr") and graph[EDGE_TYPE].edge_attr is not None:
        edge_attr = graph[EDGE_TYPE].edge_attr.view(-1).tolist()

    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    for idx, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
        confidence = float(edge_attr[idx]) if edge_attr and idx < len(edge_attr) else 1.0
        key = (int(src), int(dst))
        edge_conf[key] = max(edge_conf.get(key, 0.0), confidence)

    return edge_conf


def _validate_edge_indices(edge_conf: Dict[Tuple[int, int], float], num_nodes: int) -> None:
    for src, dst in edge_conf.keys():
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            raise ValueError(
                "Edge index out of bounds: "
                f"({src}, {dst}) with num_nodes={num_nodes}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add STRING PPI edges into biomedical_graph.pt")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--string-links", type=Path, default=None)
    parser.add_argument("--string-info", type=Path, default=None)
    parser.add_argument("--min-score", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--directed", action="store_true", help="Do not add reverse edges")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup before writing")
    args = parser.parse_args()

    graph_path = args.graph.resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    string_links_path = args.string_links or _resolve_first_existing_path(
        [
            DEFAULT_STRING_DIR / "9606.protein.links.v12.0.txt",
            DEFAULT_STRING_DIR / "9606.protein.links.v12.0.txt.gz",
        ]
    )
    string_info_path = args.string_info or _resolve_first_existing_path(
        [
            DEFAULT_STRING_DIR / "9606.protein.info.v12.0.txt",
            DEFAULT_STRING_DIR / "9606.protein.info.v12.0.txt.gz",
        ]
    )

    if string_links_path is None:
        raise FileNotFoundError(
            "Could not find STRING links file. Expected one of: "
            "data/raw/string/9606.protein.links.v12.0.txt(.gz)"
        )
    if string_info_path is None:
        raise FileNotFoundError(
            "Could not find STRING protein info file. Expected one of: "
            "data/raw/string/9606.protein.info.v12.0.txt(.gz)"
        )

    print("=" * 70)
    print("Patch STRING PPI Edges")
    print("=" * 70)
    print(f"Graph file       : {graph_path}")
    print(f"STRING links     : {string_links_path}")
    print(f"STRING info      : {string_info_path}")
    print(f"Min score        : {args.min_score}")
    print(f"Undirected edges : {not args.directed}")
    print()

    print("Loading graph...")
    graph = _load_graph_cpu(graph_path)

    if "gene" not in graph.node_types or not hasattr(graph["gene"], "node_id"):
        raise ValueError("Graph does not contain gene node_id metadata")

    graph_genes = list(graph["gene"].node_id)
    gene_to_idx = {gene: idx for idx, gene in enumerate(graph_genes)}
    num_gene_nodes = graph["gene"].num_nodes

    print(f"Gene nodes in graph: {num_gene_nodes:,}")

    print("Loading STRING protein->gene mapping...")
    protein_to_gene = _load_protein_to_gene_map(string_info_path)
    print(f"Protein->gene mappings: {len(protein_to_gene):,}")

    print("Loading and filtering STRING links...")
    string_edges = _parse_string_edges(string_links_path, protein_to_gene, min_score=args.min_score)
    print(f"STRING edges after score/mapping filter: {len(string_edges):,}")

    ppi_genes = set(string_edges["gene_a"]) | set(string_edges["gene_b"])

    # Keep only edges where both endpoints already exist in graph gene nodes.
    in_graph_mask = (
        string_edges["gene_a"].isin(gene_to_idx)
        & string_edges["gene_b"].isin(gene_to_idx)
    )
    mapped_edges = string_edges[in_graph_mask].copy()

    print(f"STRING edges with both genes in graph: {len(mapped_edges):,}")

    existing_edge_conf = _build_existing_edge_conf_map(graph)
    existing_count = len(existing_edge_conf)

    for gene_a, gene_b, confidence in zip(
        mapped_edges["gene_a"], mapped_edges["gene_b"], mapped_edges["confidence"]
    ):
        src = gene_to_idx[str(gene_a)]
        dst = gene_to_idx[str(gene_b)]
        conf = float(confidence) if pd.notna(confidence) else 0.0

        existing_edge_conf[(src, dst)] = max(existing_edge_conf.get((src, dst), 0.0), conf)
        if not args.directed:
            existing_edge_conf[(dst, src)] = max(existing_edge_conf.get((dst, src), 0.0), conf)

    _validate_edge_indices(existing_edge_conf, num_gene_nodes)

    final_edges = sorted(existing_edge_conf.items())
    src_nodes = [src for (src, _), _ in final_edges]
    dst_nodes = [dst for (_, dst), _ in final_edges]
    confidences = [conf for _, conf in final_edges]

    graph[EDGE_TYPE].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    graph[EDGE_TYPE].edge_attr = torch.tensor(confidences, dtype=torch.float).unsqueeze(1)

    graph_genes_with_ppi = set()
    for src, dst in zip(src_nodes, dst_nodes):
        graph_genes_with_ppi.add(graph_genes[src])
        graph_genes_with_ppi.add(graph_genes[dst])

    unmapped_ppi_genes = sorted(ppi_genes - set(graph_genes))
    graph_genes_without_ppi = sorted(set(graph_genes) - graph_genes_with_ppi)

    print()
    print("Diagnostics")
    print("-" * 70)
    print(f"Existing gene-gene edges before patch: {existing_count:,}")
    print(f"Final gene-gene edges after patch    : {len(final_edges):,}")
    print(f"Unique STRING genes after filtering  : {len(ppi_genes):,}")
    print(f"STRING genes present in graph        : {len(ppi_genes & set(graph_genes)):,}")
    print(f"Graph genes with >=1 PPI edge        : {len(graph_genes_with_ppi):,} / {num_gene_nodes:,} "
          f"({(len(graph_genes_with_ppi) / max(num_gene_nodes, 1)):.1%})")

    if unmapped_ppi_genes:
        print(f"Sample STRING genes not in graph ({min(len(unmapped_ppi_genes), 20)} shown):")
        print("  " + ", ".join(unmapped_ppi_genes[:20]))

    if graph_genes_without_ppi:
        print(f"Sample graph genes without PPI edges ({min(len(graph_genes_without_ppi), 20)} shown):")
        print("  " + ", ".join(graph_genes_without_ppi[:20]))

    if args.dry_run:
        print("\nDry run enabled; graph file was not modified.")
        return

    if not args.no_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = graph_path.with_suffix(graph_path.suffix + f".bak_string_patch_{timestamp}")
        shutil.copy2(graph_path, backup_path)
        print(f"Backup written: {backup_path}")

    torch.save(graph, graph_path)
    print(f"\nUpdated graph saved: {graph_path}")


if __name__ == "__main__":
    main()
