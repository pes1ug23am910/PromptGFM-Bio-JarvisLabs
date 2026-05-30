"""
Graph preprocessing module for PromptGFM-Bio.

This module constructs heterogeneous biomedical knowledge graphs from:
- BioGRID/STRING PPI networks
- DisGeNET gene-disease associations
- HPO phenotype annotations

Output: PyTorch Geometric HeteroData graph with:
- Node types: [gene, disease, phenotype]
- Edge types: [gene-gene (PPI), gene-disease, disease-phenotype]
"""

import logging
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import gzip
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_data_dirs() -> Dict[str, Path]:
    """Get data directory paths."""
    project_root = Path(__file__).parent.parent.parent
    return {
        'raw': project_root / "data" / "raw",
        'processed': project_root / "data" / "processed"
    }


def _resolve_first_existing_path(candidates: List[Path]) -> Optional[Path]:
    """Return the first path that exists from a candidate list."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_gene_symbol(symbol: str) -> str:
    """
    Normalize gene symbol to HGNC standard format.
    
    Args:
        symbol: Gene symbol to normalize
        
    Returns:
        Normalized gene symbol (uppercase, stripped)
    """
    if pd.isna(symbol) or not symbol:
        return None
    # Convert to uppercase and strip whitespace
    symbol = str(symbol).strip().upper()
    # Remove common prefixes from other species
    symbol = re.sub(r'^(HUMAN_|Hs_|ENSP\d+)', '', symbol)
    return symbol if symbol else None


def parse_biogrid(filepath: Path, organism: str = "Homo sapiens", 
                  min_score: float = 0.0) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Parse BioGRID protein-protein interaction data.
    
    Args:
        filepath: Path to BioGRID tab3 file
        organism: Organism name to filter (default: Homo sapiens)
        min_score: Minimum interaction score (not used in BioGRID, kept for API consistency)
        
    Returns:
        Tuple of (edge dataframe, set of unique genes)
    """
    logger.info(f"Parsing BioGRID from {filepath}...")
    
    # BioGRID tab3 format has many columns, we need:
    # Columns: Official Symbol Interactor A, Official Symbol Interactor B, Organism Interactor A/B
    relevant_cols = [
        'Official Symbol Interactor A',
        'Official Symbol Interactor B', 
        'Organism Interactor A',
        'Organism Interactor B',
        'Experimental System',
        'Experimental System Type'
    ]
    
    try:
        # Read the file
        df = pd.read_csv(filepath, sep='\t', usecols=relevant_cols, low_memory=False)
        logger.info(f"Loaded {len(df)} BioGRID interactions")
        
        # Filter for human-human interactions
        human_mask = (
            (df['Organism Interactor A'] == organism) & 
            (df['Organism Interactor B'] == organism)
        )
        df = df[human_mask]
        logger.info(f"Filtered to {len(df)} human interactions")
        
        # Normalize gene symbols
        df['gene_a'] = df['Official Symbol Interactor A'].apply(_normalize_gene_symbol)
        df['gene_b'] = df['Official Symbol Interactor B'].apply(_normalize_gene_symbol)
        
        # Remove rows with missing gene symbols
        df = df.dropna(subset=['gene_a', 'gene_b'])
        
        # Remove self-loops
        df = df[df['gene_a'] != df['gene_b']]
        
        # Get unique genes
        genes = set(df['gene_a'].unique()) | set(df['gene_b'].unique())
        logger.info(f"Found {len(genes)} unique genes in BioGRID")
        
        # Create edge dataframe
        edges = df[['gene_a', 'gene_b']].copy()
        edges['source'] = 'biogrid'
        edges['confidence'] = 1.0  # BioGRID doesn't have confidence scores
        
        return edges, genes
        
    except Exception as e:
        logger.error(f"Failed to parse BioGRID: {e}")
        return pd.DataFrame(), set()


def parse_string(filepath: Path, info_filepath: Optional[Path] = None,
                 min_score: float = 400) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Parse STRING protein-protein interaction network.
    
    Args:
        filepath: Path to STRING protein.links file
        info_filepath: Path to STRING protein.info file for gene name mapping
        min_score: Minimum combined score (0-1000, default 400 = medium confidence)
        
    Returns:
        Tuple of (edge dataframe, set of unique genes)
    """
    logger.info(f"Parsing STRING from {filepath}...")
    
    try:
        # Read STRING links (space-separated, supports .txt/.txt.gz)
        df = pd.read_csv(filepath, sep=r'\s+', engine='python', compression='infer')
        required_cols = {'protein1', 'protein2', 'combined_score'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"STRING links file missing required columns: {sorted(missing_cols)}")

        logger.info(f"Loaded {len(df)} STRING interactions")
        
        # Filter by confidence score
        df = df[df['combined_score'] >= min_score]
        logger.info(f"Filtered to {len(df)} interactions with score >= {min_score}")
        
        # Load gene name mapping if available
        protein_to_gene = {}
        if info_filepath and info_filepath.exists():
            logger.info(f"Loading gene name mapping from {info_filepath}...")
            info_df = pd.read_csv(info_filepath, sep='\t', compression='infer', low_memory=False)

            # Support both v11/v12 headers (e.g., protein_external_id or #string_protein_id).
            normalized_cols = {
                col.strip().lstrip('#').lower(): col for col in info_df.columns
            }
            id_col = None
            for candidate in ('protein_external_id', 'string_protein_id', 'protein_id'):
                if candidate in normalized_cols:
                    id_col = normalized_cols[candidate]
                    break

            name_col = None
            for candidate in ('preferred_name', 'gene_name', 'gene'):
                if candidate in normalized_cols:
                    name_col = normalized_cols[candidate]
                    break

            if id_col and name_col:
                protein_to_gene = {
                    str(protein_id): _normalize_gene_symbol(gene_name)
                    for protein_id, gene_name in zip(info_df[id_col], info_df[name_col])
                    if pd.notna(protein_id)
                }
                protein_to_gene = {
                    protein_id: gene_name
                    for protein_id, gene_name in protein_to_gene.items()
                    if gene_name
                }
                logger.info(f"Loaded {len(protein_to_gene)} protein-gene mappings")
            else:
                logger.warning(
                    "STRING info file is missing expected mapping columns. "
                    f"Found columns: {list(info_df.columns)}"
                )
        
        # Map protein IDs to gene symbols
        if protein_to_gene:
            df['gene_a'] = df['protein1'].map(protein_to_gene)
            df['gene_b'] = df['protein2'].map(protein_to_gene)
            mapped_mask = df['gene_a'].notna() & df['gene_b'].notna()
            mapping_coverage = float(mapped_mask.mean()) if len(df) > 0 else 0.0
            logger.info(f"STRING mapping coverage after filtering: {mapping_coverage:.1%}")
            df = df[mapped_mask].copy()
        
        # If no mapping available, use protein IDs directly.
        # This keeps edges but may not overlap HGNC symbols from gene-disease sources.
        if not protein_to_gene:
            logger.warning("No STRING protein->gene mapping available, using protein IDs")
            df['gene_a'] = df['protein1'].str.replace(r'9606\.', '', regex=True)
            df['gene_b'] = df['protein2'].str.replace(r'9606\.', '', regex=True)
        
        # Remove rows with missing gene symbols
        df = df.dropna(subset=['gene_a', 'gene_b'])
        
        # Remove self-loops
        df = df[df['gene_a'] != df['gene_b']]
        
        # Get unique genes
        genes = set(df['gene_a'].unique()) | set(df['gene_b'].unique())
        logger.info(f"Found {len(genes)} unique genes in STRING")
        
        # Create edge dataframe with normalized confidence scores
        edges = df[['gene_a', 'gene_b', 'combined_score']].copy()
        edges['source'] = 'string'
        edges['confidence'] = edges['combined_score'] / 1000.0  # Normalize to [0, 1]
        edges = edges.drop('combined_score', axis=1)
        
        return edges, genes
        
    except Exception as e:
        logger.error(f"Failed to parse STRING: {e}")
        return pd.DataFrame(), set()


def parse_ppi_network(biogrid_path: Optional[Path] = None, 
                      string_path: Optional[Path] = None,
                      string_info_path: Optional[Path] = None,
                      min_confidence: float = 0.4) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Parse and combine PPI networks from BioGRID and STRING.
    
    Args:
        biogrid_path: Path to BioGRID file
        string_path: Path to STRING links file
        string_info_path: Path to STRING info file
        min_confidence: Minimum confidence threshold for STRING
        
    Returns:
        Tuple of (combined edge dataframe, set of unique genes)
    """
    all_edges = []
    all_genes = set()
    
    # Parse BioGRID if available
    if biogrid_path and biogrid_path.exists():
        biogrid_edges, biogrid_genes = parse_biogrid(biogrid_path)
        if not biogrid_edges.empty:
            all_edges.append(biogrid_edges)
            all_genes.update(biogrid_genes)
    
    # Parse STRING if available
    if string_path and string_path.exists():
        string_score = int(min_confidence * 1000)
        string_edges, string_genes = parse_string(
            string_path, string_info_path, min_score=string_score
        )
        if not string_edges.empty:
            all_edges.append(string_edges)
            all_genes.update(string_genes)
    
    # Combine all edges
    if all_edges:
        combined_edges = pd.concat(all_edges, ignore_index=True)
        
        # Remove duplicate edges (keep highest confidence)
        combined_edges = combined_edges.sort_values('confidence', ascending=False)
        combined_edges = combined_edges.drop_duplicates(
            subset=['gene_a', 'gene_b'], keep='first'
        )
        
        logger.info(f"Combined PPI network: {len(combined_edges)} edges, {len(all_genes)} genes")
        return combined_edges, all_genes
    
    logger.warning("No PPI data available")
    return pd.DataFrame(), set()


def parse_disgenet(filepath: Path, rare_only: bool = True, 
                   max_known_genes: int = 100) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse DisGeNET gene-disease associations.
    
    Args:
        filepath: Path to DisGeNET file
        rare_only: If True, filter to rare/orphan diseases
        max_known_genes: Maximum known genes per disease (for rare disease filtering)
        
    Returns:
        Tuple of (edge dataframe, disease info dict {disease_id: description})
    """
    logger.info(f"Parsing DisGeNET from {filepath}...")
    
    try:
        # Read DisGeNET file
        df = pd.read_csv(filepath, sep='\t')
        logger.info(f"Loaded {len(df)} gene-disease associations")
        
        # Expected columns: geneId, geneSymbol, diseaseId, diseaseName, score, source
        # Normalize gene symbols
        df['gene'] = df['geneSymbol'].apply(_normalize_gene_symbol)
        df = df.dropna(subset=['gene'])
        
        # Get disease info
        disease_info = dict(zip(df['diseaseId'], df['diseaseName']))
        
        # Filter to rare diseases if requested
        if rare_only:
            # Count genes per disease
            genes_per_disease = df.groupby('diseaseId')['gene'].nunique()
            rare_diseases = genes_per_disease[genes_per_disease <= max_known_genes].index
            df = df[df['diseaseId'].isin(rare_diseases)]
            logger.info(f"Filtered to {len(rare_diseases)} rare diseases with <={max_known_genes} genes")
            logger.info(f"Remaining associations: {len(df)}")
        
        # Create edge dataframe
        edges = df[['gene', 'diseaseId', 'diseaseName']].copy()
        edges['source'] = 'disgenet'
        
        # Add confidence score if available
        if 'score' in df.columns:
            edges['confidence'] = df['score']
        else:
            edges['confidence'] = 1.0
        
        unique_diseases = df['diseaseId'].nunique()
        unique_genes = df['gene'].nunique()
        logger.info(f"Found {unique_genes} genes associated with {unique_diseases} diseases")
        
        return edges, disease_info
        
    except Exception as e:
        logger.error(f"Failed to parse DisGeNET: {e}")
        return pd.DataFrame(), {}


def parse_hpo(phenotype_to_genes_path: Path, 
              genes_to_phenotype_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Parse HPO (Human Phenotype Ontology) annotations.
    
    Args:
        phenotype_to_genes_path: Path to phenotype_to_genes.txt
        genes_to_phenotype_path: Path to genes_to_phenotype.txt (optional)
        
    Returns:
        Tuple of (disease-phenotype edges, phenotype info dict {HPO_id: description})
    """
    logger.info(f"Parsing HPO from {phenotype_to_genes_path}...")
    
    try:
        # Read phenotype to genes mapping
        # Format: HPO-id, HPO-name, Gene-symbol, Gene-ID
        df = pd.read_csv(phenotype_to_genes_path, sep='\t', comment='#')
        logger.info(f"Loaded {len(df)} phenotype-gene annotations")
        
        # Expected columns vary, try to identify them
        # Common formats: HPO_ID, HPO_Name, Gene_Symbol, Gene_ID
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'hpo' in col_lower and 'id' in col_lower:
                col_mapping['hpo_id'] = col
            elif 'hpo' in col_lower and 'name' in col_lower:
                col_mapping['hpo_name'] = col
            elif 'gene' in col_lower and 'symbol' in col_lower:
                col_mapping['gene_symbol'] = col
        
        # Rename columns
        df = df.rename(columns={v: k for k, v in col_mapping.items()})
        
        # Normalize gene symbols
        if 'gene_symbol' in df.columns:
            df['gene'] = df['gene_symbol'].apply(_normalize_gene_symbol)
            df = df.dropna(subset=['gene'])
        
        # Get phenotype descriptions
        if 'hpo_id' in df.columns and 'hpo_name' in df.columns:
            phenotype_info = dict(zip(df['hpo_id'], df['hpo_name']))
        else:
            phenotype_info = {}
        
        # Create edges (for now, connect genes to phenotypes)
        # In full implementation, we'll need disease-phenotype links
        edges = df[['gene', 'hpo_id']].copy() if 'hpo_id' in df.columns else pd.DataFrame()
        
        if not edges.empty:
            logger.info(f"Found {len(phenotype_info)} unique phenotypes")
            logger.info(f"Created {len(edges)} gene-phenotype associations")
        
        return edges, phenotype_info
        
    except Exception as e:
        logger.error(f"Failed to parse HPO: {e}")
        return pd.DataFrame(), {}


def build_heterogeneous_graph(ppi_edges: pd.DataFrame,
                              gene_disease_edges: pd.DataFrame,
                              disease_info: Dict[str, str],
                              disease_pheno_edges: Optional[pd.DataFrame] = None,
                              phenotype_info: Optional[Dict[str, str]] = None) -> HeteroData:
    """
    Build PyTorch Geometric HeteroData graph from parsed data.
    
    Args:
        ppi_edges: DataFrame with columns [gene_a, gene_b, confidence]
        gene_disease_edges: DataFrame with columns [gene, diseaseId, confidence]
        disease_info: Dict mapping disease IDs to names
        disease_pheno_edges: Optional DataFrame with disease-phenotype edges
        phenotype_info: Optional dict with phenotype descriptions
        
    Returns:
        HeteroData graph with node types [gene, disease, phenotype] and edges
    """
    logger.info("Building heterogeneous graph...")
    
    data = HeteroData()
    
    # Create gene node mapping
    all_genes_in_ppi = set(ppi_edges['gene_a'].unique()) | set(ppi_edges['gene_b'].unique()) if not ppi_edges.empty else set()
    
    # Handle both old ('geneSymbol') and new ('gene') column names
    if not gene_disease_edges.empty:
        if 'gene' in gene_disease_edges.columns:
            gene_col = 'gene'
            disease_col = 'disease'
        elif 'geneSymbol' in gene_disease_edges.columns:
            gene_col = 'geneSymbol'
            disease_col = 'diseaseId'
        else:
            logger.error(f"Gene-disease edges missing expected columns. Found: {gene_disease_edges.columns.tolist()}")
            gene_col = gene_disease_edges.columns[0]  # Fallback
            disease_col = gene_disease_edges.columns[1] if len(gene_disease_edges.columns) > 1 else gene_disease_edges.columns[0]
        
        all_genes_in_disease = set(gene_disease_edges[gene_col].unique())
    else:
        all_genes_in_disease = set()
    
    all_genes = sorted(all_genes_in_ppi | all_genes_in_disease)
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    logger.info(f"Created {len(all_genes)} gene nodes")
    
    # Create disease node mapping
    if not gene_disease_edges.empty:
        all_diseases = sorted(gene_disease_edges[disease_col].unique())
    else:
        all_diseases = []
    disease_to_idx = {disease: idx for idx, disease in enumerate(all_diseases)}
    logger.info(f"Created {len(all_diseases)} disease nodes")
    
    # Store node info
    data['gene'].num_nodes = len(all_genes)
    data['gene'].node_id = all_genes  # Store gene symbols
    
    data['disease'].num_nodes = len(all_diseases)
    data['disease'].node_id = all_diseases  # Store disease IDs
    data['disease'].description = [disease_info.get(d, '') for d in all_diseases]
    
    # Build gene-gene (PPI) edges
    if not ppi_edges.empty:
        gene_gene_src = [gene_to_idx[g] for g in ppi_edges['gene_a'] if g in gene_to_idx]
        gene_gene_dst = [gene_to_idx[g] for g in ppi_edges['gene_b'] if g in gene_to_idx]
        
        data['gene', 'interacts', 'gene'].edge_index = torch.tensor(
            [gene_gene_src, gene_gene_dst], dtype=torch.long
        )
        data['gene', 'interacts', 'gene'].edge_attr = torch.tensor(
            ppi_edges['confidence'].values[:len(gene_gene_src)], dtype=torch.float
        ).unsqueeze(1)
        
        logger.info(f"Added {len(gene_gene_src)} gene-gene edges")
    
    # Build gene-disease edges
    if not gene_disease_edges.empty:
        gene_disease_gene_idx = [gene_to_idx[g] for g in gene_disease_edges[gene_col] if g in gene_to_idx]
        gene_disease_disease_idx = [disease_to_idx[d] for d in gene_disease_edges[disease_col] if d in disease_to_idx]
        
        # Ensure same length
        min_len = min(len(gene_disease_gene_idx), len(gene_disease_disease_idx))
        gene_disease_gene_idx = gene_disease_gene_idx[:min_len]
        gene_disease_disease_idx = gene_disease_disease_idx[:min_len]
        
        data['gene', 'associated_with', 'disease'].edge_index = torch.tensor(
            [gene_disease_gene_idx, gene_disease_disease_idx], dtype=torch.long
        )
        
        # Reverse edge
        data['disease', 'rev_associated_with', 'gene'].edge_index = torch.tensor(
            [gene_disease_disease_idx, gene_disease_gene_idx], dtype=torch.long
        )
        
        logger.info(f"Added {len(gene_disease_gene_idx)} gene-disease edges")
    
    # Add phenotype nodes and edges if available
    if disease_pheno_edges is not None and not disease_pheno_edges.empty and phenotype_info:
        all_phenotypes = sorted(phenotype_info.keys())
        phenotype_to_idx = {pheno: idx for idx, pheno in enumerate(all_phenotypes)}
        
        data['phenotype'].num_nodes = len(all_phenotypes)
        data['phenotype'].node_id = all_phenotypes
        data['phenotype'].description = [phenotype_info[p] for p in all_phenotypes]
        
        logger.info(f"Added {len(all_phenotypes)} phenotype nodes")
    
    # Print graph statistics
    logger.info("\n" + "="*70)
    logger.info("Graph Statistics:")
    logger.info(f"  Gene nodes: {data['gene'].num_nodes}")
    logger.info(f"  Disease nodes: {data['disease'].num_nodes}")
    if 'phenotype' in data.node_types:
        logger.info(f"  Phenotype nodes: {data['phenotype'].num_nodes}")
    logger.info(f"  Edge types: {data.edge_types}")
    for edge_type in data.edge_types:
        logger.info(f"    {edge_type}: {data[edge_type].edge_index.shape[1]} edges")
    logger.info("="*70)
    
    return data


def save_graph(graph: HeteroData, output_path: Path):
    """
    Save processed graph to disk.
    
    Args:
        graph: HeteroData graph to save
        output_path: Path to save the graph
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_path)
    logger.info(f"Saved graph to {output_path}")
    
    # Save human-readable statistics
    stats_path = output_path.parent / f"{output_path.stem}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("Biomedical Knowledge Graph Statistics\n")
        f.write("="*70 + "\n\n")
        f.write(f"Node types: {graph.node_types}\n")
        for node_type in graph.node_types:
            f.write(f"  {node_type}: {graph[node_type].num_nodes} nodes\n")
        f.write(f"\nEdge types: {graph.edge_types}\n")
        for edge_type in graph.edge_types:
            f.write(f"  {edge_type}: {graph[edge_type].edge_index.shape[1]} edges\n")
    
    logger.info(f"Saved statistics to {stats_path}")


def create_gene_disease_edges_enhanced(dirs: Dict[str, Path],
                                       use_hpo_bridge: bool = True,
                                       use_orphadata: bool = True,
                                       use_disgenet: bool = False,
                                       min_hpo_score: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create gene-disease edges using enhanced methods.
    
    Strategy (in priority order):
    1. HPO Bridge - IDF-weighted phenotype overlap
    2. Orphadata - Gold standard rare disease associations
    3. DisGeNET - Backup if available
    
    Args:
        dirs: Data directories dict
        use_hpo_bridge: Use HPO phenotype bridge
        use_orphadata: Use Orphadata gold standard
        use_disgenet: Use DisGeNET (backup)
        min_hpo_score: Minimum score for HPO edges
        
    Returns:
        Tuple of (gene-disease edges DataFrame, disease info dict)
    """
    all_edges = []
    disease_info = {}
    
    # Method 1: HPO Bridge (Primary method)
    if use_hpo_bridge:
        try:
            logger.info("\n[Method 1] Creating gene-disease edges from HPO bridge...")
            from src.data.hpo_bridge import create_gene_disease_from_hpo
            
            gene_pheno_path = dirs['raw'] / "hpo" / "genes_to_phenotype.txt"
            disease_pheno_path = dirs['raw'] / "hpo" / "phenotype.hpoa"
            hpo_output_path = dirs['processed'] / "hpo_gene_disease_edges.csv"
            
            if gene_pheno_path.exists() and disease_pheno_path.exists():
                hpo_edges = create_gene_disease_from_hpo(
                    gene_pheno_path=gene_pheno_path,
                    disease_pheno_path=disease_pheno_path,
                    output_path=hpo_output_path,
                    min_score=min_hpo_score,
                    scoring_method='weighted_overlap'
                )
                
                if not hpo_edges.empty:
                    all_edges.append(hpo_edges)
                    # Extract disease info from HPO edges
                    for _, row in hpo_edges.iterrows():
                        if row['disease'] not in disease_info:
                            disease_info[row['disease']] = f"Disease {row['disease']}"
                    
                    logger.info(f"✓ HPO bridge created {len(hpo_edges)} edges")
                else:
                    logger.warning("HPO bridge returned no edges")
            else:
                logger.warning(f"HPO files not found: {gene_pheno_path}, {disease_pheno_path}")
                
        except Exception as e:
            logger.error(f"HPO bridge failed: {e}")
    
    # Method 2: Orphadata (Gold standard)
    if use_orphadata:
        try:
            logger.info("\n[Method 2] Downloading and parsing Orphadata...")
            from src.data.orphadata import get_orphadata_gene_disease_edges, merge_orphadata_with_hpo
            
            orphadata_edges = get_orphadata_gene_disease_edges(
                download_dir=dirs['raw'] / "orphanet",
                force_download=False
            )
            
            if not orphadata_edges.empty:
                # If we have HPO edges, merge them
                if all_edges:
                    hpo_edges_df = all_edges[0]  # HPO edges from Method 1
                    merged_edges = merge_orphadata_with_hpo(orphadata_edges, hpo_edges_df)
                    all_edges = [merged_edges]  # Replace with merged version
                else:
                    # No HPO edges, just use Orphadata
                    # Convert to standard format
                    orpha_formatted = orphadata_edges[['gene_symbol', 'orpha_code', 'disease_name']].copy()
                    orpha_formatted.rename(columns={
                        'gene_symbol': 'gene',
                        'orpha_code': 'disease',
                        'disease_name': 'diseaseName'
                    }, inplace=True)
                    orpha_formatted['score'] = 1.0
                    orpha_formatted['provenance'] = 'Orphadata'
                    orpha_formatted['supporting_phenotypes'] = ''
                    orpha_formatted['num_shared_phenotypes'] = 0
                    all_edges.append(orpha_formatted)
                
                # Update disease info
                for _, row in orphadata_edges.iterrows():
                    disease_info[row['orpha_code']] = row['disease_name']
                
                logger.info(f"✓ Orphadata added {len(orphadata_edges)} gold-standard edges")
            else:
                logger.warning("Orphadata returned no edges")
                
        except Exception as e:
            logger.error(f"Orphadata integration failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Method 3: DisGeNET (Backup)
    if use_disgenet:
        try:
            logger.info("\n[Method 3] Parsing DisGeNET (backup method)...")
            disgenet_path = dirs['raw'] / "disgenet" / "curated_gene_disease_associations.tsv"
            if not disgenet_path.exists():
                disgenet_path = dirs['raw'] / "disgenet" / "curated_gene_disease_associations.tsv.gz"
            
            if disgenet_path.exists():
                disgenet_edges, disgenet_info = parse_disgenet(
                    disgenet_path,
                    rare_only=True,
                    max_known_genes=100
                )
                
                if not disgenet_edges.empty:
                    all_edges.append(disgenet_edges)
                    disease_info.update(disgenet_info)
                    logger.info(f"✓ DisGeNET added {len(disgenet_edges)} edges")
            else:
                logger.warning("DisGeNET file not found, skipping")
                
        except Exception as e:
            logger.error(f"DisGeNET parsing failed: {e}")
    
    # Combine all edges
    if all_edges:
        combined_edges = pd.concat(all_edges, ignore_index=True) if len(all_edges) > 1 else all_edges[0]
        
        # Ensure required columns exist
        required_cols = ['gene', 'disease', 'score']
        for col in required_cols:
            if col not in combined_edges.columns:
                if col == 'score':
                    combined_edges['score'] = 1.0
        
        logger.info(f"\n✓ Total gene-disease edges: {len(combined_edges)}")
        logger.info(f"✓ Unique genes: {combined_edges['gene'].nunique()}")
        logger.info(f"✓ Unique diseases: {combined_edges['disease'].nunique()}")
        
        return combined_edges, disease_info
    else:
        logger.error("No gene-disease edges created")
        return pd.DataFrame(), {}


def preprocess_all(force: bool = False,
                  use_hpo_bridge: bool = True,
                  use_orphadata: bool = True,
                  use_uniprot: bool = False,
                  use_pathways: bool = False):
    """
    Run complete preprocessing pipeline with enhanced data sources.
    
    Args:
        force: If True, reprocess even if output exists
        use_hpo_bridge: Use HPO phenotype bridge for gene-disease edges
        use_orphadata: Use Orphadata gold standard
        use_uniprot: Add UniProt gene descriptions
        use_pathways: Add Reactome pathway annotations
    """
    dirs = _get_data_dirs()
    output_path = dirs['processed'] / "biomedical_graph.pt"
    
    if output_path.exists() and not force:
        logger.info(f"Processed graph already exists at {output_path}")
        logger.info("Use force=True to reprocess")
        return
    
    logger.info("="*70)
    logger.info("Starting enhanced preprocessing pipeline...")
    logger.info("="*70)
    logger.info(f"Options:")
    logger.info(f"  HPO Bridge: {use_hpo_bridge}")
    logger.info(f"  Orphadata: {use_orphadata}")
    logger.info(f"  UniProt: {use_uniprot}")
    logger.info(f"  Pathways: {use_pathways}")
    logger.info("="*70)
    
    # Parse PPI networks
    logger.info("\n[Step 1] Parsing PPI networks...")
    biogrid_path = dirs['raw'] / "biogrid" / "BIOGRID-ALL-4.4.224.tab3.txt"
    string_path = _resolve_first_existing_path([
        dirs['raw'] / "string" / "9606.protein.links.v12.0.txt",
        dirs['raw'] / "string" / "9606.protein.links.v12.0.txt.gz",
    ])
    string_info_path = _resolve_first_existing_path([
        dirs['raw'] / "string" / "9606.protein.info.v12.0.txt",
        dirs['raw'] / "string" / "9606.protein.info.v12.0.txt.gz",
    ])

    if string_path:
        logger.info(f"Using STRING links file: {string_path}")
    else:
        logger.warning("STRING links file not found (.txt/.txt.gz)")

    if string_info_path:
        logger.info(f"Using STRING protein info file: {string_info_path}")
    else:
        logger.warning("STRING protein info file not found (.txt/.txt.gz)")
    
    ppi_edges, ppi_genes = parse_ppi_network(
        biogrid_path=biogrid_path if biogrid_path.exists() else None,
        string_path=string_path,
        string_info_path=string_info_path,
        min_confidence=0.4
    )
    
    # Create gene-disease edges using enhanced methods
    logger.info("\n[Step 2] Creating gene-disease edges...")
    gene_disease_edges, disease_info = create_gene_disease_edges_enhanced(
        dirs=dirs,
        use_hpo_bridge=use_hpo_bridge,
        use_orphadata=use_orphadata,
        use_disgenet=False,  # DisGeNET has issues, use as last resort
        min_hpo_score=0.1
    )
    
    # Parse HPO phenotypes
    logger.info("\n[Step 3] Parsing HPO phenotypes...")
    hpo_path = dirs['raw'] / "hpo" / "phenotype_to_genes.txt"
    disease_pheno_edges, phenotype_info = parse_hpo(
        hpo_path if hpo_path.exists() else None
    ) if hpo_path and hpo_path.exists() else (pd.DataFrame(), {})
    
    # Optional: Add UniProt descriptions
    if use_uniprot:
        try:
            logger.info("\n[Step 4] Adding UniProt gene descriptions...")
            from src.data.uniprot_pathways import get_uniprot_gene_descriptions
            
            uniprot_df = get_uniprot_gene_descriptions(
                download_dir=dirs['raw'] / "uniprot",
                force_download=False
            )
            
            if not uniprot_df.empty:
                # Save for later use in model
                uniprot_output = dirs['processed'] / "uniprot_gene_descriptions.csv"
                uniprot_df.to_csv(uniprot_output, index=False)
                logger.info(f"✓ Saved UniProt descriptions to {uniprot_output}")
            
        except Exception as e:
            logger.error(f"UniProt integration failed: {e}")
    
    # Optional: Add Reactome pathways
    if use_pathways:
        try:
            logger.info("\n[Step 5] Adding Reactome pathway annotations...")
            from src.data.uniprot_pathways import get_reactome_gene_pathways
            
            pathways_df = get_reactome_gene_pathways(
                download_dir=dirs['raw'] / "reactome",
                force_download=False
            )
            
            if not pathways_df.empty:
                # Save for later validation
                pathways_output = dirs['processed'] / "reactome_gene_pathways.csv"
                pathways_df.to_csv(pathways_output, index=False)
                logger.info(f"✓ Saved pathway annotations to {pathways_output}")
            
        except Exception as e:
            logger.error(f"Pathway integration failed: {e}")
    
    # Build graph
    logger.info(f"\n[Step {4 + use_uniprot + use_pathways}] Building heterogeneous graph...")
    graph = build_heterogeneous_graph(
        ppi_edges=ppi_edges,
        gene_disease_edges=gene_disease_edges,
        disease_info=disease_info,
        disease_pheno_edges=disease_pheno_edges,
        phenotype_info=phenotype_info
    )
    
    # Save graph
    save_graph(graph, output_path)
    
    logger.info("\n" + "="*70)
    logger.info("✓ Enhanced preprocessing complete!")
    logger.info(f"Graph saved to: {output_path}")
    logger.info("="*70)


if __name__ == "__main__":
    """Command-line interface for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess biomedical data into graph")
    parser.add_argument('--force', action='store_true', help='Reprocess even if output exists')
    
    args = parser.parse_args()
    
    preprocess_all(force=args.force)
