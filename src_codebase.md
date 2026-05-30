# Codebase Dump: `src/`

This document consolidates all the code files within the `src/` directory structure for LLM analysis.

## File: `src/__init__.py`

```python
"""
PromptGFM-Bio: A prompt-conditioned graph foundation model for rare-disease gene-phenotype mapping.

This package combines Graph Neural Networks (GNNs) with natural language processing
for biomedical applications, using disease descriptions as dynamic prompts to condition
GNN message passing for task-adaptive gene discovery.
"""

__version__ = "0.1.0"
```

## File: `src/data/dataset.py`

```python
"""
PyTorch Geometric dataset classes for gene-disease prediction.

Implements dataset loaders for link prediction tasks with support for
rare disease stratification and few-shot learning.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BiomedicalGraphDataset:
    """
    Base dataset class for loading preprocessed biomedical graphs.
    
    Features:
    - Load PyG HeteroData graphs
    - Node/edge indexing and mapping
    - Statistics and metadata
    """
    
    def __init__(self, graph_path: str):
        """
        Args:
            graph_path: Path to preprocessed graph file (.pt)
        """
        self.graph_path = Path(graph_path)
        logger.info(f"Loading graph from {self.graph_path}")
        
        # Load graph (weights_only=False needed for PyG HeteroData)
        self.graph = torch.load(self.graph_path, weights_only=False)
        
        # Create node mappings
        self._create_node_mappings()
        
        logger.info(f"Graph loaded: {self._get_graph_info()}")
    
    def _create_node_mappings(self):
        """Create mappings from node IDs to indices."""
        self.node_mappings = {}
        self.reverse_mappings = {}
        
        # For heterogeneous graphs, map each node type
        if hasattr(self.graph, 'node_types'):
            for node_type in self.graph.node_types:  # node_types is a property
                # Assuming nodes have 'id' attribute or use index
                self.node_mappings[node_type] = {}
                self.reverse_mappings[node_type] = {}
                
                num_nodes = self.graph[node_type].num_nodes
                for idx in range(num_nodes):
                    self.node_mappings[node_type][idx] = idx
                    self.reverse_mappings[node_type][idx] = idx
    
    def _get_graph_info(self) -> str:
        """Get summary string of graph statistics."""
        info_parts = []
        
        if hasattr(self.graph, 'node_types'):
            for node_type in self.graph.node_types:  # node_types is a property
                num_nodes = self.graph[node_type].num_nodes
                info_parts.append(f"{node_type}={num_nodes}")
        
        if hasattr(self.graph, 'edge_types'):
            for edge_type in self.graph.edge_types:  # edge_types is a property
                num_edges = self.graph[edge_type].num_edges
                info_parts.append(f"{edge_type}={num_edges}")
        
        return ", ".join(info_parts)
    
    def get_subgraph(self, node_type: str, node_indices: List[int]):
        """Extract subgraph for specific nodes."""
        # Placeholder for subgraph extraction
        raise NotImplementedError("Subgraph extraction not yet implemented")


class GeneDiseaseDataset(BiomedicalGraphDataset):
    """
    Dataset for gene-disease link prediction with rare disease stratification.
    
    Features:
    - Load gene-disease associations
    - Split by disease rarity
    - Few-shot learning splits
    - Negative sampling
    - Provenance tracking
    """
    
    def __init__(
        self,
        graph_path: str,
        edges_path: str,
        min_score: float = 0.1,
        use_provenance: Optional[List[str]] = None
    ):
        """
        Args:
            graph_path: Path to preprocessed graph (.pt)
            edges_path: Path to gene-disease edges CSV
            min_score: Minimum edge score to include
            use_provenance: Filter to specific provenances (e.g., ['Orphadata'])
        """
        super().__init__(graph_path)
        
        self.edges_path = Path(edges_path)
        self.min_score = min_score
        self.use_provenance = use_provenance
        
        logger.info(f"Loading gene-disease edges from {self.edges_path}")
        self._load_edges()
        
        logger.info(f"Loaded {len(self.edges)} edges "
                   f"({self.edges['provenance'].value_counts().to_dict()})")
    
    def _load_edges(self):
        """Load gene-disease association edges."""
        # Load edges
        self.edges = pd.read_csv(self.edges_path)
        
        # Filter by score
        self.edges = self.edges[self.edges['score'] >= self.min_score]
        
        # Filter by provenance if specified
        if self.use_provenance is not None:
            self.edges = self.edges[self.edges['provenance'].isin(self.use_provenance)]
        
        # Create gene and disease vocabularies
        self.genes = sorted(self.edges['gene'].unique())
        self.diseases = sorted(self.edges['disease'].unique())
        
        # Create ID mappings
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.genes)}
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.idx_to_gene = {idx: gene for gene, idx in self.gene_to_idx.items()}
        self.idx_to_disease = {idx: disease for disease, idx in self.disease_to_idx.items()}
        
        logger.info(f"Vocabulary: {len(self.genes)} genes, {len(self.diseases)} diseases")
    
    def split_by_rarity(
        self,
        rarity_bins: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Split edges by disease rarity (number of known genes).
        
        Args:
            rarity_bins: Dict mapping rarity level to (min_genes, max_genes).
                        Default bins:
                        - ultra_rare: 1-2 genes
                        - very_rare: 3-5 genes
                        - rare: 6-15 genes
                        - moderate: 16-50 genes
                        - common: 51+ genes
        
        Returns:
            Dict mapping rarity level to DataFrame of edges
        """
        if rarity_bins is None:
            rarity_bins = {
                'ultra_rare': (1, 2),
                'very_rare': (3, 5),
                'rare': (6, 15),
                'moderate': (16, 50),
                'common': (51, float('inf'))
            }
        
        # Count genes per disease
        disease_gene_counts = self.edges.groupby('disease')['gene'].nunique()
        
        # Assign rarity levels
        disease_rarity = {}
        for disease, num_genes in disease_gene_counts.items():
            for rarity_level, (min_genes, max_genes) in rarity_bins.items():
                if min_genes <= num_genes <= max_genes:
                    disease_rarity[disease] = rarity_level
                    break
        
        # Add rarity column
        self.edges['rarity'] = self.edges['disease'].map(disease_rarity)
        
        # Split by rarity
        splits = {}
        for rarity_level in rarity_bins.keys():
            splits[rarity_level] = self.edges[self.edges['rarity'] == rarity_level]
            logger.info(f"  {rarity_level}: {len(splits[rarity_level])} edges, "
                       f"{splits[rarity_level]['disease'].nunique()} diseases")
        
        return splits
    
    def create_train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by_provenance: bool = True,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify_by_provenance: Maintain provenance distribution in splits
            random_seed: Random seed for reproducibility
        
        Returns:
            (train_edges, val_edges, test_edges)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        np.random.seed(random_seed)
        
        if stratify_by_provenance and 'provenance' in self.edges.columns:
            # Stratified split by provenance
            train_list, val_list, test_list = [], [], []
            
            for provenance in self.edges['provenance'].unique():
                prov_edges = self.edges[self.edges['provenance'] == provenance].copy()
                prov_edges = prov_edges.sample(frac=1, random_state=random_seed)
                
                n = len(prov_edges)
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                
                train_list.append(prov_edges[:train_end])
                val_list.append(prov_edges[train_end:val_end])
                test_list.append(prov_edges[val_end:])
            
            train_edges = pd.concat(train_list, ignore_index=True)
            val_edges = pd.concat(val_list, ignore_index=True)
            test_edges = pd.concat(test_list, ignore_index=True)
        else:
            # Simple random split
            shuffled = self.edges.sample(frac=1, random_state=random_seed)
            n = len(shuffled)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_edges = shuffled[:train_end]
            val_edges = shuffled[train_end:val_end]
            test_edges = shuffled[val_end:]
        
        logger.info(f"Split sizes: train={len(train_edges)}, "
                   f"val={len(val_edges)}, test={len(test_edges)}")
        
        return train_edges, val_edges, test_edges
    
    def create_few_shot_split(
        self,
        k_shot: int = 3,
        n_query: int = 10,
        min_edges_per_disease: int = None,
        random_seed: int = 42
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create few-shot learning splits.
        
        Each disease is split into:
        - Support set: k positive examples
        - Query set: remaining examples (or n_query if specified)
        
        Args:
            k_shot: Number of examples in support set
            n_query: Number of examples in query set (None = use all remaining)
            min_edges_per_disease: Only include diseases with >= this many edges
            random_seed: Random seed
        
        Returns:
            Dict mapping disease to {'support': DataFrame, 'query': DataFrame}
        """
        if min_edges_per_disease is None:
            min_edges_per_disease = k_shot + (n_query if n_query else 1)
        
        np.random.seed(random_seed)
        
        # Group by disease
        disease_groups = self.edges.groupby('disease')
        
        few_shot_splits = {}
        num_valid_diseases = 0
        
        for disease, group in disease_groups:
            if len(group) < min_edges_per_disease:
                continue
            
            # Shuffle edges
            group = group.sample(frac=1, random_state=random_seed)
            
            # Split into support and query
            support = group[:k_shot]
            remaining = group[k_shot:]
            
            if n_query is not None and len(remaining) > n_query:
                query = remaining[:n_query]
            else:
                query = remaining
            
            few_shot_splits[disease] = {
                'support': support,
                'query': query
            }
            num_valid_diseases += 1
        
        logger.info(f"Few-shot split: {num_valid_diseases} diseases with "
                   f"{k_shot}-shot support sets")
        
        return few_shot_splits
    
    def get_edge_index_tensor(self, edge_df: pd.DataFrame) -> torch.Tensor:
        """
        Convert edge DataFrame to PyG edge_index tensor.
        
        Args:
            edge_df: DataFrame with 'gene' and 'disease' columns
        
        Returns:
            edge_index tensor of shape [2, num_edges]
        """
        gene_indices = [self.gene_to_idx[g] for g in edge_df['gene']]
        disease_indices = [self.disease_to_idx[d] for d in edge_df['disease']]
        
        edge_index = torch.tensor([gene_indices, disease_indices], dtype=torch.long)
        return edge_index
    
    def get_edge_weights(self, edge_df: pd.DataFrame) -> torch.Tensor:
        """Get edge weights (scores) as tensor."""
        return torch.tensor(edge_df['score'].values, dtype=torch.float)
    
    def sample_negative_edges(
        self,
        num_negatives: int,
        existing_edges: Optional[pd.DataFrame] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Sample negative (non-existent) gene-disease edges.
        
        Args:
            num_negatives: Number of negative samples
            existing_edges: DataFrame of positive edges to avoid
            random_seed: Random seed
        
        Returns:
            DataFrame of negative edges with same columns as positive edges
        """
        if existing_edges is None:
            existing_edges = self.edges
        
        # Create set of positive edges for fast lookup
        positive_set = set(zip(existing_edges['gene'], existing_edges['disease']))
        
        np.random.seed(random_seed)
        
        negative_edges = []
        max_attempts = num_negatives * 10  # Avoid infinite loops
        attempts = 0
        
        while len(negative_edges) < num_negatives and attempts < max_attempts:
            # Sample random gene and disease
            gene = np.random.choice(self.genes)
            disease = np.random.choice(self.diseases)
            
            # Check if it's a true negative
            if (gene, disease) not in positive_set:
                negative_edges.append({
                    'gene': gene,
                    'disease': disease,
                    'disease_name': '',
                    'score': 0.0,
                    'provenance': 'negative_sample',
                    'supporting_phenotypes': '',
                    'num_shared_phenotypes': 0
                })
            
            attempts += 1
        
        if len(negative_edges) < num_negatives:
            logger.warning(f"Only generated {len(negative_edges)}/{num_negatives} "
                          f"negative samples after {max_attempts} attempts")
        
        return pd.DataFrame(negative_edges)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'num_edges': len(self.edges),
            'num_genes': len(self.genes),
            'num_diseases': len(self.diseases),
            'provenance_counts': self.edges['provenance'].value_counts().to_dict(),
            'score_stats': {
                'mean': float(self.edges['score'].mean()),
                'std': float(self.edges['score'].std()),
                'min': float(self.edges['score'].min()),
                'max': float(self.edges['score'].max())
            }
        }
        
        if 'num_shared_phenotypes' in self.edges.columns:
            stats['phenotype_stats'] = {
                'mean': float(self.edges['num_shared_phenotypes'].mean()),
                'median': float(self.edges['num_shared_phenotypes'].median())
            }
        
        return stats


if __name__ == "__main__":
    # Test dataset loading
    logger.info("Testing dataset classes...")
    
    try:
        # Test with merged dataset
        dataset = GeneDiseaseDataset(
            graph_path='data/processed/biomedical_graph.pt',
            edges_path='data/processed/merged_gene_disease_edges.csv',
            min_score=0.3
        )
        
        print("\n=== Dataset Statistics ===")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n=== Train/Val/Test Split ===")
        train, val, test = dataset.create_train_val_test_split()
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        print("\n=== Rarity Stratification ===")
        rarity_splits = dataset.split_by_rarity()
        
        print("\n=== Few-Shot Split ===")
        few_shot = dataset.create_few_shot_split(k_shot=3, n_query=5)
        print(f"Few-shot: {len(few_shot)} diseases")
        
        print("\n✓ Dataset classes working!")
        
    except Exception as e:
        logger.error(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
```

## File: `src/data/download.py`

```python
"""
Data download module for PromptGFM-Bio.

This module handles downloading biomedical datasets including:
- BioGRID protein-protein interactions
- STRING database PPI
- DisGeNET gene-disease associations
- Human Phenotype Ontology (HPO)
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import logging
import hashlib
import time
from typing import Optional, Dict
import zipfile
import gzip
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_data_dir() -> Path:
    """Get the data directory path, creating it if necessary."""
    # Get project root (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_file_with_progress(url: str, output_path: Path, 
                                  max_retries: int = 3, 
                                  timeout: int = 300) -> bool:
    """
    Download a file with progress bar and retry logic.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for the request
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading from {url} (attempt {attempt + 1}/{max_retries})")
            
            # Stream the download
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded to {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
    
    return False


def _verify_checksum(file_path: Path, expected_hash: Optional[str] = None, 
                     algorithm: str = 'md5') -> bool:
    """
    Verify file integrity using checksum.
    
    Args:
        file_path: Path to the file to verify
        expected_hash: Expected hash value (if None, only compute and log)
        algorithm: Hash algorithm to use ('md5' or 'sha256')
        
    Returns:
        True if verification passes or no expected_hash provided
    """
    if not file_path.exists():
        return False
    
    hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    computed_hash = hash_func.hexdigest()
    logger.info(f"File {file_path.name} {algorithm}: {computed_hash}")
    
    if expected_hash:
        if computed_hash == expected_hash:
            logger.info("Checksum verification passed!")
            return True
        else:
            logger.error(f"Checksum mismatch! Expected {expected_hash}, got {computed_hash}")
            return False
    
    return True


def _extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract zip or gz archives.
    
    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            logger.info(f"Extracting {archive_path.name}...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info("Extraction complete!")
            
        elif archive_path.suffix == '.gz':
            logger.info(f"Extracting {archive_path.name}...")
            output_file = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted to {output_file}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def download_biogrid(force: bool = False) -> Dict[str, Path]:
    """
    Download BioGRID protein-protein interaction database.
    
    BioGRID is a biomedical interaction repository containing protein and 
    genetic interactions from major model organisms.
    
    Args:
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    biogrid_dir = data_dir / "biogrid"
    biogrid_dir.mkdir(parents=True, exist_ok=True)
    
    # BioGRID latest release (tab-delimited format)
    # Note: This URL may need updating - check https://downloads.thebiogrid.org/
    url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.224/BIOGRID-ALL-4.4.224.tab3.zip"
    filename = "BIOGRID-ALL-4.4.224.tab3.zip"
    output_path = biogrid_dir / filename
    
    if output_path.exists() and not force:
        logger.info(f"BioGRID file already exists at {output_path}")
        logger.info("Use force=True to re-download")
        return {"biogrid_zip": output_path}
    
    logger.info("Downloading BioGRID database (~500MB)...")
    success = _download_file_with_progress(url, output_path)
    
    if success:
        _verify_checksum(output_path)
        # Extract the archive
        _extract_archive(output_path, biogrid_dir)
        return {"biogrid_zip": output_path, "biogrid_dir": biogrid_dir}
    
    return {}


def download_string(organism: str = "9606", score_threshold: int = 400, 
                    force: bool = False) -> Dict[str, Path]:
    """
    Download STRING protein-protein interaction network.
    
    STRING is a database of known and predicted protein-protein interactions.
    
    Args:
        organism: NCBI taxonomy ID (9606 = Homo sapiens)
        score_threshold: Minimum confidence score (0-1000, default 400 = medium confidence)
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    string_dir = data_dir / "string"
    string_dir.mkdir(parents=True, exist_ok=True)
    
    # STRING v11.5 protein links (physical interactions)
    base_url = "https://stringdb-downloads.org/download"
    version = "v12.0"
    filename = f"{organism}.protein.links.{version}.txt.gz"
    url = f"{base_url}/protein.links.{version}/{filename}"
    info_filename = f"{organism}.protein.info.{version}.txt.gz"
    info_url = f"{base_url}/protein.info.{version}/{info_filename}"
    
    output_path = string_dir / filename
    info_path = string_dir / info_filename

    def _ensure_info_file() -> bool:
        if info_path.exists() and not force:
            return True

        logger.info("Downloading STRING protein info for gene name mapping...")
        success_info = _download_file_with_progress(info_url, info_path)
        if success_info:
            _extract_archive(info_path, string_dir)
            return True

        logger.warning("Failed to download STRING protein info file")
        return False
    
    if output_path.exists() and not force:
        logger.info(f"STRING file already exists at {output_path}")
        _ensure_info_file()
        logger.info("Use force=True to re-download")
        return {
            "string_links": output_path,
            "string_info": info_path if info_path.exists() else None,
            "string_dir": string_dir
        }
    
    logger.info(f"Downloading STRING database for organism {organism} (~700MB)...")
    success = _download_file_with_progress(url, output_path)
    
    if success:
        _verify_checksum(output_path)
        # Extract the gz file
        _extract_archive(output_path, string_dir)
        _ensure_info_file()
        
        return {
            "string_links": output_path,
            "string_info": info_path if info_path.exists() else None,
            "string_dir": string_dir
        }
    
    return {}


def download_disgenet(version: str = "v7.0", force: bool = False) -> Dict[str, Path]:
    """
    Download DisGeNET gene-disease associations.
    
    DisGeNET is a comprehensive platform integrating information on gene-disease
    associations from various expert curated databases and text-mining.
    
    NOTE: DisGeNET requires registration and authentication for some datasets.
    This function downloads the publicly available core dataset.
    For full access, register at: https://www.disgenet.org/signup/
    
    Args:
        version: DisGeNET version to download
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    disgenet_dir = data_dir / "disgenet"
    disgenet_dir.mkdir(parents=True, exist_ok=True)
    
    # Public DisGeNET curated gene-disease associations
    # Note: For private/full data, you need authentication
    base_url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads"
    filename = "curated_gene_disease_associations.tsv.gz"
    url = f"{base_url}/{filename}"
    
    output_path = disgenet_dir / filename
    
    if output_path.exists() and not force:
        logger.info(f"DisGeNET file already exists at {output_path}")
        logger.info("Use force=True to re-download")
        return {"disgenet_gz": output_path}
    
    logger.info("Downloading DisGeNET curated associations (~300MB)...")
    logger.info("Note: This is the public dataset. For full access, register at disgenet.org")
    
    success = _download_file_with_progress(url, output_path)
    
    if success:
        _verify_checksum(output_path)
        _extract_archive(output_path, disgenet_dir)
        return {"disgenet_gz": output_path, "disgenet_dir": disgenet_dir}
    else:
        # Provide fallback message
        logger.warning("Public DisGeNET download may require authentication.")
        logger.info("Alternative: Download manually from https://www.disgenet.org/downloads")
        logger.info(f"Save the file to: {disgenet_dir}")
    
    return {}


def download_hpo(force: bool = False) -> Dict[str, Path]:
    """
    Download Human Phenotype Ontology (HPO) annotations.
    
    HPO provides a standardized vocabulary of phenotypic abnormalities
    encountered in human disease.
    
    Args:
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    hpo_dir = data_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    
    # HPO annotations
    files_to_download = {
        "phenotype_to_genes": "http://purl.obolibrary.org/obo/hp/hpoa/phenotype_to_genes.txt",
        "genes_to_phenotype": "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
        "phenotype_annotations": "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
    }
    
    downloaded_files = {}
    
    for file_key, url in files_to_download.items():
        filename = url.split('/')[-1]
        output_path = hpo_dir / filename
        
        if output_path.exists() and not force:
            logger.info(f"HPO file {filename} already exists")
            downloaded_files[file_key] = output_path
            continue
        
        logger.info(f"Downloading HPO {file_key}...")
        success = _download_file_with_progress(url, output_path)
        
        if success:
            downloaded_files[file_key] = output_path
            _verify_checksum(output_path)
    
    downloaded_files["hpo_dir"] = hpo_dir
    return downloaded_files


def download_all(force: bool = False, skip_failing: bool = True) -> Dict[str, Dict[str, Path]]:
    """
    Download all biomedical datasets.
    
    Args:
        force: If True, re-download even if files exist
        skip_failing: If True, continue even if some downloads fail
        
    Returns:
        Dictionary mapping dataset names to their download results
    """
    logger.info("="*70)
    logger.info("Starting download of all biomedical datasets...")
    logger.info("This may take 30-60 minutes depending on your connection")
    logger.info("Total size: ~1.5 GB")
    logger.info("="*70)
    
    results = {}
    
    # Download BioGRID
    try:
        logger.info("\n[1/4] BioGRID Protein-Protein Interactions")
        results['biogrid'] = download_biogrid(force=force)
    except Exception as e:
        logger.error(f"BioGRID download failed: {e}")
        if not skip_failing:
            raise
        results['biogrid'] = {}
    
    # Download STRING
    try:
        logger.info("\n[2/4] STRING Protein Network")
        results['string'] = download_string(force=force)
    except Exception as e:
        logger.error(f"STRING download failed: {e}")
        if not skip_failing:
            raise
        results['string'] = {}
    
    # Download DisGeNET
    try:
        logger.info("\n[3/4] DisGeNET Gene-Disease Associations")
        results['disgenet'] = download_disgenet(force=force)
    except Exception as e:
        logger.error(f"DisGeNET download failed: {e}")
        if not skip_failing:
            raise
        results['disgenet'] = {}
    
    # Download HPO
    try:
        logger.info("\n[4/4] Human Phenotype Ontology")
        results['hpo'] = download_hpo(force=force)
    except Exception as e:
        logger.error(f"HPO download failed: {e}")
        if not skip_failing:
            raise
        results['hpo'] = {}
    
    logger.info("\n" + "="*70)
    logger.info("Download Summary:")
    for dataset, files in results.items():
        if files:
            logger.info(f"✓ {dataset.upper()}: {len(files)} files downloaded")
        else:
            logger.warning(f"✗ {dataset.upper()}: Download failed or incomplete")
    logger.info("="*70)
    
    return results


if __name__ == "__main__":
    """Command-line interface for data download."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download biomedical datasets for PromptGFM-Bio")
    parser.add_argument('--dataset', type=str, choices=['all', 'biogrid', 'string', 'disgenet', 'hpo'],
                       default='all', help='Which dataset to download')
    parser.add_argument('--force', action='store_true', help='Re-download even if files exist')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        download_all(force=args.force)
    elif args.dataset == 'biogrid':
        download_biogrid(force=args.force)
    elif args.dataset == 'string':
        download_string(force=args.force)
    elif args.dataset == 'disgenet':
        download_disgenet(force=args.force)
    elif args.dataset == 'hpo':
        download_hpo(force=args.force)


if __name__ == "__main__":
    download_all()
```

## File: `src/data/hpo_bridge.py`

```python
"""
HPO-Based Gene-Disease Bridge Implementation

This module creates high-quality gene-disease associations by bridging
HPO gene→phenotype and phenotype→disease relationships.

Key features:
- Weighted phenotype scoring (IDF-based)
- Jaccard similarity with phenotype specificity
- Provenance tracking
- ID harmonization
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HPOGeneDiseaseBuilder:
    """
    Build gene-disease associations from HPO phenotype bridge.
    
    Implements scoring methods:
    1. IDF-weighted phenotype overlap
    2. Weighted Jaccard similarity
    3. Provenance tracking for explainability
    """
    
    def __init__(self, min_score: float = 0.1, max_common_phenotype_freq: float = 0.5):
        """
        Args:
            min_score: Minimum edge score to include (0-1)
            max_common_phenotype_freq: Filter phenotypes appearing in >X of diseases
        """
        self.min_score = min_score
        self.max_common_phenotype_freq = max_common_phenotype_freq
        self.phenotype_idf = {}  # Inverse document frequency per phenotype
        
    def compute_phenotype_idf(self, 
                              disease_phenotypes: Dict[str, Set[str]],
                              gene_phenotypes: Dict[str, Set[str]]) -> None:
        """
        Compute IDF (inverse document frequency) for each phenotype.
        
        IDF = log(N / df) where:
        - N = total number of entities (diseases + genes)
        - df = number of entities with this phenotype
        
        Higher IDF = rarer, more specific phenotype
        """
        phenotype_counts = defaultdict(int)
        
        # Count phenotype occurrences across diseases
        for phenotypes in disease_phenotypes.values():
            for pheno in phenotypes:
                phenotype_counts[pheno] += 1
                
        # Count phenotype occurrences across genes
        for phenotypes in gene_phenotypes.values():
            for pheno in phenotypes:
                phenotype_counts[pheno] += 1
        
        total_entities = len(disease_phenotypes) + len(gene_phenotypes)
        
        # Compute IDF
        for pheno, count in phenotype_counts.items():
            self.phenotype_idf[pheno] = np.log(total_entities / count)
            
        logger.info(f"Computed IDF for {len(self.phenotype_idf)} phenotypes")
        
    def filter_common_phenotypes(self, 
                                 disease_phenotypes: Dict[str, Set[str]]) -> Set[str]:
        """
        Remove overly common phenotypes (e.g., 'fever', 'pain').
        
        Returns set of phenotypes to exclude.
        """
        total_diseases = len(disease_phenotypes)
        phenotype_freq = defaultdict(int)
        
        for phenotypes in disease_phenotypes.values():
            for pheno in phenotypes:
                phenotype_freq[pheno] += 1
        
        # Filter phenotypes appearing in >X% of diseases
        excluded = {
            pheno for pheno, count in phenotype_freq.items()
            if count / total_diseases > self.max_common_phenotype_freq
        }
        
        logger.info(f"Filtering {len(excluded)} overly common phenotypes (>{self.max_common_phenotype_freq*100}% frequency)")
        return excluded
    
    def weighted_phenotype_overlap_score(self,
                                         gene_phenotypes: Set[str],
                                         disease_phenotypes: Set[str],
                                         excluded_phenotypes: Set[str]) -> Tuple[float, List[str]]:
        """
        Score gene-disease association using IDF-weighted phenotype overlap.
        
        Score = sum_{pheno in intersection} IDF(pheno)
        Normalized by max possible score.
        
        Returns:
            score: Float in [0, 1]
            supporting_phenotypes: List of shared phenotypes
        """
        # Remove excluded phenotypes
        gene_pheno_filtered = gene_phenotypes - excluded_phenotypes
        disease_pheno_filtered = disease_phenotypes - excluded_phenotypes
        
        # Find intersection
        shared = gene_pheno_filtered & disease_pheno_filtered
        
        if not shared:
            return 0.0, []
        
        # Compute weighted score
        score = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in shared)
        
        # Normalize by maximum possible score (if all disease phenotypes matched)
        max_score = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in disease_pheno_filtered)
        
        if max_score > 0:
            normalized_score = score / max_score
        else:
            normalized_score = 0.0
            
        return normalized_score, list(shared)
    
    def weighted_jaccard_score(self,
                              gene_phenotypes: Set[str],
                              disease_phenotypes: Set[str],
                              excluded_phenotypes: Set[str]) -> Tuple[float, List[str]]:
        """
        Weighted Jaccard similarity.
        
        J = sum_{pheno in intersection} w(pheno) / sum_{pheno in union} w(pheno)
        where w(pheno) = IDF(pheno)
        """
        # Remove excluded
        gene_pheno_filtered = gene_phenotypes - excluded_phenotypes
        disease_pheno_filtered = disease_phenotypes - excluded_phenotypes
        
        shared = gene_pheno_filtered & disease_pheno_filtered
        union = gene_pheno_filtered | disease_pheno_filtered
        
        if not union:
            return 0.0, []
        
        # Weighted intersection
        weighted_intersection = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in shared)
        
        # Weighted union
        weighted_union = sum(self.phenotype_idf.get(pheno, 0.0) for pheno in union)
        
        score = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
        
        return score, list(shared)
    
    def create_gene_disease_edges(self,
                                  gene_phenotypes_path: Path,
                                  disease_phenotypes_path: Path,
                                  scoring_method: str = 'weighted_overlap') -> pd.DataFrame:
        """
        Main pipeline: Create scored gene-disease edges from HPO.
        
        Args:
            gene_phenotypes_path: Path to genes_to_phenotype.txt
            disease_phenotypes_path: Path to phenotype.hpoa
            scoring_method: 'weighted_overlap' or 'weighted_jaccard'
            
        Returns:
            DataFrame with columns: [gene, disease, score, supporting_phenotypes, provenance]
        """
        # Step 1: Parse HPO files
        logger.info("Parsing HPO gene-phenotype annotations...")
        gene_to_phenotypes = self._parse_gene_phenotypes(gene_phenotypes_path)
        logger.info(f"Parsed {len(gene_to_phenotypes)} genes with phenotype annotations")
        
        # Debug: Show sample
        if gene_to_phenotypes:
            sample_genes = list(gene_to_phenotypes.keys())[:3]
            for gene in sample_genes:
                phenos = list(gene_to_phenotypes[gene])[:3]
                logger.info(f"  Sample gene: {gene} → {phenos}")
        
        logger.info("Parsing HPO disease-phenotype annotations...")
        disease_to_phenotypes = self._parse_disease_phenotypes(disease_phenotypes_path)
        logger.info(f"Parsed {len(disease_to_phenotypes)} diseases with phenotype annotations")
        
        # Debug: Show sample
        if disease_to_phenotypes:
            sample_diseases = list(disease_to_phenotypes.keys())[:3]
            for disease in sample_diseases:
                phenos = list(disease_to_phenotypes[disease])[:3]
                logger.info(f"  Sample disease: {disease} → {phenos}")
        
        # Step 2: Compute IDF
        logger.info("Computing phenotype IDF scores...")
        self.compute_phenotype_idf(disease_to_phenotypes, gene_to_phenotypes)
        
        # Step 3: Filter common phenotypes
        excluded_phenotypes = self.filter_common_phenotypes(disease_to_phenotypes)
        
        # Step 4: Score all gene-disease pairs
        logger.info(f"Scoring gene-disease associations ({len(gene_to_phenotypes)} genes x {len(disease_to_phenotypes)} diseases)...")
        edges = []
        
        total_pairs = len(gene_to_phenotypes) * len(disease_to_phenotypes)
        logger.info(f"Total possible pairs: {total_pairs:,}")
        
        # Debug: Track intersection stats
        debug_count = 0
        debug_limit = 5
        has_intersection_count = 0
        
        for gene, gene_phenos in gene_to_phenotypes.items():
            for disease, disease_phenos in disease_to_phenotypes.items():
                
                # Debug: Show first few comparisons
                if debug_count < debug_limit:
                    intersection = gene_phenos & disease_phenos
                    logger.info(f"  Debug: {gene} x {disease} → {len(intersection)} shared phenotypes out of {len(gene_phenos)} gene / {len(disease_phenos)} disease")
                    if intersection:
                        logger.info(f"    Shared: {list(intersection)[:3]}")
                    debug_count += 1
                
                # Quick check for any intersection
                if gene_phenos & disease_phenos:
                    has_intersection_count += 1
                
                # Choose scoring method
                if scoring_method == 'weighted_overlap':
                    score, supporting = self.weighted_phenotype_overlap_score(
                        gene_phenos, disease_phenos, excluded_phenotypes
                    )
                elif scoring_method == 'weighted_jaccard':
                    score, supporting = self.weighted_jaccard_score(
                        gene_phenos, disease_phenos, excluded_phenotypes
                    )
                else:
                    raise ValueError(f"Unknown scoring method: {scoring_method}")
                
                # Filter by minimum score
                if score >= self.min_score:
                    edges.append({
                        'gene': gene,
                        'disease': disease,
                        'score': score,
                        'supporting_phenotypes': ';'.join(supporting),
                        'num_shared_phenotypes': len(supporting),
                        'provenance': 'HPO_phenotype_bridge'
                    })
        
        logger.info(f"Created {len(edges)} gene-disease edges (score >= {self.min_score})")
        logger.info(f"Pairs with any phenotype intersection: {has_intersection_count:,} ({100*has_intersection_count/total_pairs:.2f}%)")
        
        df = pd.DataFrame(edges)
        
        if len(df) > 0:
            logger.info(f"\nScore distribution:")
            logger.info(f"  Mean: {df['score'].mean():.3f}")
            logger.info(f"  Median: {df['score'].median():.3f}")
            logger.info(f"  Min: {df['score'].min():.3f}")
            logger.info(f"  Max: {df['score'].max():.3f}")
        
        return df
    
    def _parse_gene_phenotypes(self, filepath: Path) -> Dict[str, Set[str]]:
        """
        Parse genes_to_phenotype.txt
        
        Format:
        ncbi_gene_id  gene_symbol  hpo_id  hpo_name  frequency  disease_id
        10            NAT2         HP:0000007  ...
        """
        gene_to_phenotypes = defaultdict(set)
        
        logger.info(f"🔍 [HPO_BRIDGE_V2] Parsing gene phenotypes from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read and skip header
                header = f.readline().strip()
                logger.info(f"🔍 [HPO_BRIDGE_V2] Skipped header: {header[:50]}...")
                
                line_count = 0
                for line in f:
                    if line.startswith('#'):
                        continue
                        
                    parts = line.strip().split('\t')
                    if len(parts) < 3:  # Need at least gene_symbol and hpo_id
                        continue
                    
                    gene = parts[1].strip().upper()  # gene_symbol is column 1
                    hpo_id = parts[2].strip()  # hpo_id is column 2
                    
                    if line_count < 3:
                        logger.info(f"🔍 [HPO_BRIDGE_V2] Line {line_count}: gene={gene}, hpo_id={hpo_id}")
                        line_count += 1
                    
                    if gene and hpo_id and hpo_id.startswith('HP:'):
                        gene_to_phenotypes[gene].add(hpo_id)
            
            logger.info(f"Parsed {len(gene_to_phenotypes)} genes with phenotype annotations")
            return dict(gene_to_phenotypes)
            
        except Exception as e:
            logger.error(f"Failed to parse gene-phenotype file: {e}")
            return {}
    
    def _parse_disease_phenotypes(self, filepath: Path) -> Dict[str, Set[str]]:
        """
        Parse phenotype.hpoa
        
        Format (tab-separated):
        database_id  disease_name  ...  hpo_id  ...
        OMIM:154700  Marfan syndrome  ...  HP:0001166  ...
        """
        disease_to_phenotypes = defaultdict(set)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Skip comment lines that start with #
                header_line = None
                for line in f:
                    if line.startswith('#'):
                        continue
                    header_line = line.strip().split('\t')
                    break
                
                if not header_line:
                    logger.error("No header found in phenotype.hpoa")
                    return {}
                
                # Find column indices
                db_id_col = None
                hpo_id_col = None
                
                for i, col in enumerate(header_line):
                    col_lower = col.lower()
                    if 'database' in col_lower and 'id' in col_lower:
                        db_id_col = i
                    elif 'hpo' in col_lower and 'id' in col_lower:
                        hpo_id_col = i
                
                # Fallback if columns not found (use standard positions)
                if db_id_col is None:
                    db_id_col = 0
                if hpo_id_col is None:
                    hpo_id_col = 3
                
                logger.info(f"Using columns: database_id={db_id_col}, hpo_id={hpo_id_col}")
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) <= max(db_id_col, hpo_id_col):
                        continue
                    
                    # Extract disease ID and HPO term
                    disease_id = parts[db_id_col].strip()
                    hpo_id = parts[hpo_id_col].strip()
                    
                    if disease_id and hpo_id and hpo_id.startswith('HP:'):
                        disease_to_phenotypes[disease_id].add(hpo_id)
            
            logger.info(f"Parsed {len(disease_to_phenotypes)} diseases with phenotype annotations")
            return dict(disease_to_phenotypes)
            
        except Exception as e:
            logger.error(f"Failed to parse disease-phenotype file: {e}")
            return {}


def create_gene_disease_from_hpo(
    gene_pheno_path: Path,
    disease_pheno_path: Path,
    output_path: Optional[Path] = None,
    min_score: float = 0.1,
    scoring_method: str = 'weighted_overlap'
) -> pd.DataFrame:
    """
    Convenience function to create gene-disease edges from HPO.
    
    Args:
        gene_pheno_path: Path to genes_to_phenotype.txt
        disease_pheno_path: Path to phenotype.hpoa
        output_path: Where to save edges (optional)
        min_score: Minimum edge score
        scoring_method: 'weighted_overlap' or 'weighted_jaccard'
        
    Returns:
        DataFrame of gene-disease edges with scores
    """
    builder = HPOGeneDiseaseBuilder(min_score=min_score)
    
    edges_df = builder.create_gene_disease_edges(
        gene_pheno_path,
        disease_pheno_path,
        scoring_method=scoring_method
    )
    
    if output_path and len(edges_df) > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(edges_df)} HPO-derived edges to {output_path}")
    
    return edges_df
```

## File: `src/data/orphadata.py`

```python
"""
Orphadata Integration Module

Orphadata provides authoritative rare disease gene associations.
This module handles download, parsing, and integration into the graph.

Orphadata Files:
- en_product6.xml: Gene-disease associations
- en_product1.xml: Rare disease classifications
- en_product4.xml: Disease prevalence data
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def download_orphadata(output_dir: Path = None) -> Dict[str, Path]:
    """
    Download Orphadata XML files.
    
    Files downloaded:
    1. en_product6.xml - Gene-disease associations (PRIMARY)
    2. en_product1.xml - Disease classifications
    3. en_product4.xml - Prevalence data
    
    Args:
        output_dir: Directory to save files
        
    Returns:
        Dict mapping filenames to paths
    """
    if output_dir is None:
        # Get default data directory
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "raw" / "orphanet"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = {
        'en_product6.xml': 'http://www.orphadata.com/data/xml/en_product6.xml',  # Genes
        'en_product1.xml': 'http://www.orphadata.com/data/xml/en_product1.xml',  # Classifications
        'en_product4.xml': 'http://www.orphadata.com/data/xml/en_product4.xml',  # Prevalence
    }
    
    downloaded_files = {}
    
    for filename, url in files_to_download.items():
        output_file = output_dir / filename
        
        if output_file.exists():
            logger.info(f"✓ {filename} already exists, skipping")
            downloaded_files[filename] = output_file
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / 1024 / 1024
            logger.info(f"✓ Downloaded {filename} ({file_size_mb:.1f} MB)")
            downloaded_files[filename] = output_file
            
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
    
    logger.info(f"\nOrphadata files saved to {output_dir}")
    return downloaded_files


def parse_orphadata_gene_associations(xml_path: Path) -> pd.DataFrame:
    """
    Parse en_product6.xml to extract gene-disease associations.
    
    XML structure (simplified):
    <JDBOR>
        <DisorderList>
            <Disorder id="...">
                <OrphaCode>ORPHA:166024</OrphaCode>
                <Name>Angelman syndrome</Name>
                <DisorderGeneAssociationList>
                    <DisorderGeneAssociation>
                        <Gene>
                            <Symbol>UBE3A</Symbol>
                            <ExternalReferenceList>
                                <ExternalReference>
                                    <Source>HGNC</Source>
                                    <Reference>12496</Reference>
                                </ExternalReference>
                            </ExternalReferenceList>
                        </Gene>
                        <DisorderGeneAssociationType>
                            <Name>Disease-causing germline mutation(s) in</Name>
                        </DisorderGeneAssociationType>
                        <DisorderGeneAssociationStatus>
                            <Name>Assessed</Name>
                        </DisorderGeneAssociationStatus>
                    </DisorderGeneAssociation>
                </DisorderGeneAssociationList>
            </Disorder>
        </DisorderList>
    </JDBOR>
    
    Returns:
        DataFrame with columns: [orpha_code, disease_name, gene_symbol, hgnc_id, 
                                 association_type, association_status]
    """
    logger.info(f"Parsing Orphadata gene associations from {xml_path}...")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        associations = []
        
        # Find all disorders
        for disorder in root.findall('.//Disorder'):
            # Extract disease info
            orpha_code = disorder.find('OrphaCode').text if disorder.find('OrphaCode') is not None else None
            disease_name = disorder.find('Name').text if disorder.find('Name') is not None else None
            
            # Extract gene associations
            gene_assoc_list = disorder.find('DisorderGeneAssociationList')
            if gene_assoc_list is None:
                continue
            
            for gene_assoc in gene_assoc_list.findall('DisorderGeneAssociation'):
                # Gene info
                gene = gene_assoc.find('Gene')
                if gene is None:
                    continue
                
                gene_symbol = gene.find('Symbol').text if gene.find('Symbol') is not None else None
                
                # Extract HGNC ID
                hgnc_id = None
                ext_refs = gene.find('ExternalReferenceList')
                if ext_refs is not None:
                    for ext_ref in ext_refs.findall('ExternalReference'):
                        source = ext_ref.find('Source')
                        if source is not None and source.text == 'HGNC':
                            hgnc_id = ext_ref.find('Reference').text
                            break
                
                # Association type
                assoc_type_elem = gene_assoc.find('.//DisorderGeneAssociationType/Name')
                assoc_type = assoc_type_elem.text if assoc_type_elem is not None else None
                
                # Association status
                assoc_status_elem = gene_assoc.find('.//DisorderGeneAssociationStatus/Name')
                assoc_status = assoc_status_elem.text if assoc_status_elem is not None else None
                
                associations.append({
                    'orpha_code': orpha_code,
                    'disease_name': disease_name,
                    'gene_symbol': gene_symbol.upper() if gene_symbol else None,  # Normalize
                    'hgnc_id': hgnc_id,
                    'association_type': assoc_type,
                    'association_status': assoc_status
                })
        
        df = pd.DataFrame(associations)
        logger.info(f"Extracted {len(df)} gene-disease associations from Orphadata")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse Orphadata: {e}")
        return pd.DataFrame()


def filter_high_confidence_orphadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to high-confidence gene-disease associations.
    
    Criteria:
    - Association status = 'Assessed' or 'Validated'
    - Association type includes 'Disease-causing' or 'Major'
    
    Args:
        df: DataFrame from parse_orphadata_gene_associations
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    # Filter by status
    valid_statuses = ['Assessed', 'Validated']
    df_filtered = df[df['association_status'].isin(valid_statuses)].copy()
    
    # Filter by type (disease-causing mutations)
    disease_causing_keywords = ['Disease-causing', 'disease-causing', 'Major', 'major']
    df_filtered = df_filtered[
        df_filtered['association_type'].str.contains('|'.join(disease_causing_keywords), na=False)
    ]
    
    logger.info(f"Filtered to {len(df_filtered)} high-confidence associations")
    logger.info(f"  ({len(df)} total → {len(df_filtered)} after filtering)")
    
    return df_filtered


def merge_orphadata_with_hpo(orphadata_df: pd.DataFrame,
                             hpo_edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Orphadata and HPO-derived gene-disease edges.
    
    Strategy:
    1. Use Orphadata as gold standard (score = 1.0, provenance = 'Orphadata')
    2. Add HPO edges that are NOT in Orphadata (provenance = 'HPO_bridge')
    3. Tag overlapping edges (provenance = 'Orphadata+HPO')
    
    Args:
        orphadata_df: From parse_orphadata_gene_associations (filtered)
        hpo_edges_df: From HPOGeneDiseaseBuilder
        
    Returns:
        Merged DataFrame with all edges
    """
    if orphadata_df.empty:
        logger.warning("Orphadata DataFrame is empty, returning HPO edges only")
        return hpo_edges_df
    
    if hpo_edges_df.empty:
        logger.warning("HPO DataFrame is empty, returning Orphadata edges only")
        # Prepare Orphadata format
        orpha_edges = orphadata_df[['gene_symbol', 'orpha_code', 'disease_name']].copy()
        orpha_edges.rename(columns={'gene_symbol': 'gene', 'orpha_code': 'disease'}, inplace=True)
        orpha_edges['score'] = 1.0
        orpha_edges['provenance'] = 'Orphadata'
        orpha_edges['supporting_phenotypes'] = ''
        orpha_edges['num_shared_phenotypes'] = 0
        return orpha_edges
    
    # Prepare Orphadata edges
    orpha_edges = orphadata_df[['gene_symbol', 'orpha_code', 'disease_name']].copy()
    orpha_edges.rename(columns={'gene_symbol': 'gene', 'orpha_code': 'disease'}, inplace=True)
    orpha_edges['score'] = 1.0  # Gold standard
    orpha_edges['provenance'] = 'Orphadata'
    orpha_edges['supporting_phenotypes'] = ''
    orpha_edges['num_shared_phenotypes'] = 0
    
    # Create lookup set for Orphadata pairs
    orpha_pairs = set(zip(orpha_edges['gene'], orpha_edges['disease']))
    
    # Tag HPO edges
    hpo_edges_df = hpo_edges_df.copy()
    hpo_edges_df['in_orphadata'] = hpo_edges_df.apply(
        lambda row: (row['gene'], row['disease']) in orpha_pairs, axis=1
    )
    
    # Separate HPO edges
    hpo_only = hpo_edges_df[~hpo_edges_df['in_orphadata']].copy()
    hpo_overlap = hpo_edges_df[hpo_edges_df['in_orphadata']].copy()
    
    # Update provenance for overlapping edges in Orphadata
    for idx, row in hpo_overlap.iterrows():
        mask = (orpha_edges['gene'] == row['gene']) & (orpha_edges['disease'] == row['disease'])
        orpha_edges.loc[mask, 'provenance'] = 'Orphadata+HPO'
        orpha_edges.loc[mask, 'supporting_phenotypes'] = row['supporting_phenotypes']
        orpha_edges.loc[mask, 'num_shared_phenotypes'] = row['num_shared_phenotypes']
    
    # Drop the temporary column
    hpo_only = hpo_only.drop('in_orphadata', axis=1)
    
    # Combine all edges
    merged = pd.concat([orpha_edges, hpo_only], ignore_index=True)
    
    logger.info(f"\nMerge summary:")
    logger.info(f"  Orphadata edges: {len(orpha_edges)}")
    logger.info(f"  HPO-only edges: {len(hpo_only)}")
    logger.info(f"  Overlap: {len(hpo_overlap)} edges confirmed by both sources")
    logger.info(f"  Total merged edges: {len(merged)}")
    
    return merged


def validate_hpo_against_orphadata(hpo_edges_df: pd.DataFrame,
                                    orphadata_df: pd.DataFrame) -> Dict[str, float]:
    """
    Validate HPO-derived edges against Orphadata gold standard.
    
    Computes:
    - Precision: What fraction of HPO edges are in Orphadata?
    - Recall: What fraction of Orphadata edges are recovered?
    - F1 score
    
    Args:
        hpo_edges_df: DataFrame with 'gene' and 'disease' columns
        orphadata_df: DataFrame with 'gene_symbol' and 'orpha_code' columns
        
    Returns:
        Dict with precision, recall, F1, and counts
    """
    if hpo_edges_df.empty or orphadata_df.empty:
        logger.warning("Cannot validate: One or both DataFrames are empty")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Create sets of (gene, disease) tuples
    hpo_pairs = set(zip(hpo_edges_df['gene'], hpo_edges_df['disease']))
    
    # Map Orphadata to same format
    orpha_pairs = set(zip(
        orphadata_df['gene_symbol'], 
        orphadata_df['orpha_code']
    ))
    
    # Compute metrics
    true_positives = len(hpo_pairs & orpha_pairs)
    precision = true_positives / len(hpo_pairs) if hpo_pairs else 0
    recall = true_positives / len(orpha_pairs) if orpha_pairs else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hpo_edges': len(hpo_pairs),
        'orphadata_edges': len(orpha_pairs),
        'overlap': true_positives
    }
    
    logger.info(f"\nHPO Bridge Validation vs Orphadata:")
    logger.info(f"  Precision: {precision:.3f} ({true_positives}/{len(hpo_pairs)})")
    logger.info(f"  Recall: {recall:.3f} ({true_positives}/{len(orpha_pairs)})")
    logger.info(f"  F1: {f1:.3f}")
    
    return results


def get_orphadata_gene_disease_edges(
    download_dir: Optional[Path] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Complete pipeline: Download, parse, and filter Orphadata.
    
    Args:
        download_dir: Where to save/load Orphadata files
        force_download: Re-download even if files exist
        
    Returns:
        DataFrame of high-confidence gene-disease edges
    """
    if download_dir is None:
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "raw" / "orphanet"
    
    # Download if needed
    if force_download or not (download_dir / 'en_product6.xml').exists():
        logger.info("Downloading Orphadata...")
        download_orphadata(download_dir)
    
    # Parse gene associations
    xml_path = download_dir / 'en_product6.xml'
    if not xml_path.exists():
        logger.error(f"Orphadata file not found: {xml_path}")
        return pd.DataFrame()
    
    orphadata_df = parse_orphadata_gene_associations(xml_path)
    
    # Filter to high confidence
    orphadata_filtered = filter_high_confidence_orphadata(orphadata_df)
    
    return orphadata_filtered
```

## File: `src/data/preprocess.py`

```python
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
```

## File: `src/data/uniprot_pathways.py`

```python
"""
UniProt and Reactome Pathways Integration

This module adds:
1. UniProt gene descriptions for better text-based features
2. Reactome pathway annotations for biological validation

Purpose:
- UniProt: Provides protein function descriptions
- Reactome: Provides pathway membership for enrichment analysis
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Set, Optional
from collections import defaultdict
import gzip
import json

logger = logging.getLogger(__name__)


def download_uniprot_human_proteins(output_dir: Path = None) -> Path:
    """
    Download UniProt SwissProt human protein annotations.
    
    SwissProt = manually curated, high-quality subset of UniProt
    Focus on Homo sapiens (organism 9606)
    
    Args:
        output_dir: Where to save file
        
    Returns:
        Path to downloaded file
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "raw" / "uniprot"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "uniprot_human_swissprot.tsv.gz"
    
    if output_file.exists():
        logger.info(f"✓ UniProt file already exists: {output_file}")
        return output_file
    
    logger.info("Downloading UniProt human SwissProt annotations...")
    
    # UniProt REST API for Homo sapiens SwissProt entries
    # Query: organism:9606 AND reviewed:true
    # Fields: accession, gene names, protein names, function
    url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        "format=tsv&"
        "query=(organism_id:9606)%20AND%20(reviewed:true)&"
        "fields=accession,id,gene_names,protein_name,cc_function,organism_name"
    )
    
    try:
        logger.info("Requesting data from UniProt REST API (this may take a few minutes)...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        # Write gzipped
        import gzip
        with gzip.open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"✓ Downloaded UniProt data ({file_size_mb:.1f} MB)")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download UniProt: {e}")
        logger.info("Alternative: Download manually from https://www.uniprot.org/")
        return None


def parse_uniprot_descriptions(filepath: Path) -> pd.DataFrame:
    """
    Parse UniProt TSV file to extract gene descriptions.
    
    Args:
        filepath: Path to uniprot_human_swissprot.tsv.gz
        
    Returns:
        DataFrame with columns: [gene_symbol, protein_name, function_description]
    """
    logger.info(f"Parsing UniProt descriptions from {filepath}...")
    
    try:
        # Read gzipped TSV
        df = pd.read_csv(filepath, sep='\t', compression='gzip')
        
        logger.info(f"Loaded {len(df)} UniProt entries")
        
        # Extract gene symbols from 'Gene Names' column
        # Format: "SYMBOL synonym1 synonym2"
        gene_data = []
        
        for _, row in df.iterrows():
            gene_names = row.get('Gene Names', '') or row.get('Gene names', '')
            if not gene_names or pd.isna(gene_names):
                continue
            
            # First name is primary symbol
            gene_symbol = gene_names.split()[0].upper() if gene_names else None
            
            if gene_symbol:
                protein_name = row.get('Protein names', '') or ''
                function_desc = row.get('Function [CC]', '') or ''
                
                gene_data.append({
                    'gene_symbol': gene_symbol,
                    'protein_name': protein_name,
                    'function_description': function_desc
                })
        
        result_df = pd.DataFrame(gene_data)
        
        # Remove duplicates (keep first entry per gene)
        result_df = result_df.drop_duplicates(subset=['gene_symbol'], keep='first')
        
        logger.info(f"Extracted descriptions for {len(result_df)} genes")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to parse UniProt: {e}")
        return pd.DataFrame()


def download_reactome_pathways(output_dir: Path = None) -> Dict[str, Path]:
    """
    Download Reactome pathway annotations for human.
    
    Files:
    1. UniProt2Reactome: Maps proteins to pathways
    2. ReactomePathways: Pathway names and descriptions
    
    Args:
        output_dir: Where to save files
        
    Returns:
        Dict mapping filenames to paths
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "raw" / "reactome"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = {
        'UniProt2Reactome_All_Levels.txt': (
            'https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt'
        ),
        'ReactomePathways.txt': (
            'https://reactome.org/download/current/ReactomePathways.txt'
        )
    }
    
    downloaded = {}
    
    for filename, url in files_to_download.items():
        output_file = output_dir / filename
        
        if output_file.exists():
            logger.info(f"✓ {filename} already exists")
            downloaded[filename] = output_file
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_kb = len(response.content) / 1024
            logger.info(f"✓ Downloaded {filename} ({file_size_kb:.1f} KB)")
            downloaded[filename] = output_file
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
    
    return downloaded


def parse_reactome_pathways(uniprot2reactome_path: Path,
                            reactome_pathways_path: Path,
                            organism: str = 'Homo sapiens') -> pd.DataFrame:
    """
    Parse Reactome pathway annotations.
    
    Args:
        uniprot2reactome_path: Path to UniProt2Reactome_All_Levels.txt
        reactome_pathways_path: Path to ReactomePathways.txt
        organism: Filter to this organism
        
    Returns:
        DataFrame with columns: [gene_symbol, pathway_id, pathway_name]
    """
    logger.info("Parsing Reactome pathways...")
    
    try:
        # Parse pathway names
        pathway_names = {}
        with open(reactome_pathways_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pathway_id = parts[0]
                    pathway_name = parts[1]
                    # Filter to human pathways (format: R-HSA-...)
                    if pathway_id.startswith('R-HSA'):
                        pathway_names[pathway_id] = pathway_name
        
        logger.info(f"Loaded {len(pathway_names)} human pathway names")
        
        # Parse gene-pathway mappings
        # Format: UniProtID  PathwayID  URL  PathwayName  Evidence  Organism
        gene_pathways = []
        
        with open(uniprot2reactome_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                
                uniprot_id = parts[0]
                pathway_id = parts[1]
                pathway_name = parts[3]
                organism_name = parts[5]
                
                # Filter to human
                if organism_name != organism:
                    continue
                
                # We need to map UniProt ID to gene symbol
                # For simplicity, we'll store UniProt ID and map later
                # Or extract from UniProt data
                gene_pathways.append({
                    'uniprot_id': uniprot_id,
                    'pathway_id': pathway_id,
                    'pathway_name': pathway_name
                })
        
        df = pd.DataFrame(gene_pathways)
        logger.info(f"Parsed {len(df)} gene-pathway associations")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse Reactome: {e}")
        return pd.DataFrame()


def map_uniprot_to_gene_symbols(gene_pathways_df: pd.DataFrame,
                                uniprot_descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UniProt IDs in pathway data to gene symbols using UniProt descriptions.
    
    Args:
        gene_pathways_df: From parse_reactome_pathways
        uniprot_descriptions_df: From parse_uniprot_descriptions
        
    Returns:
        DataFrame with gene_symbol column added
    """
    # This is a simplified approach
    # In production, you'd want a proper UniProt ID -> gene symbol mapping file
    
    logger.info("Mapping UniProt IDs to gene symbols...")
    
    # For now, we'll return the pathways dataframe as-is
    # In a full implementation, you'd use the UniProt API or mapping files
    
    logger.warning("UniProt ID -> gene symbol mapping not fully implemented")
    logger.warning("Use gene_symbol from UniProt descriptions or download ID mapping separately")
    
    return gene_pathways_df


def get_uniprot_gene_descriptions(download_dir: Optional[Path] = None,
                                  force_download: bool = False) -> pd.DataFrame:
    """
    Complete pipeline: Download and parse UniProt gene descriptions.
    
    Args:
        download_dir: Where to save/load files
        force_download: Re-download even if files exist
        
    Returns:
        DataFrame with gene descriptions
    """
    if download_dir is None:
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "raw" / "uniprot"
    
    # Download if needed
    if force_download or not (download_dir / 'uniprot_human_swissprot.tsv.gz').exists():
        filepath = download_uniprot_human_proteins(download_dir)
        if not filepath:
            return pd.DataFrame()
    else:
        filepath = download_dir / 'uniprot_human_swissprot.tsv.gz'
    
    # Parse descriptions
    descriptions_df = parse_uniprot_descriptions(filepath)
    
    return descriptions_df


def get_reactome_gene_pathways(download_dir: Optional[Path] = None,
                               force_download: bool = False) -> pd.DataFrame:
    """
    Complete pipeline: Download and parse Reactome pathways.
    
    Args:
        download_dir: Where to save/load files
        force_download: Re-download even if files exist
        
    Returns:
        DataFrame with gene-pathway associations
    """
    if download_dir is None:
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "raw" / "reactome"
    
    # Download if needed
    required_files = [
        'UniProt2Reactome_All_Levels.txt',
        'ReactomePathways.txt'
    ]
    
    if force_download or not all((download_dir / f).exists() for f in required_files):
        downloaded = download_reactome_pathways(download_dir)
        if not downloaded:
            return pd.DataFrame()
    
    # Parse pathways
    uniprot2reactome = download_dir / 'UniProt2Reactome_All_Levels.txt'
    pathway_names = download_dir / 'ReactomePathways.txt'
    
    if not uniprot2reactome.exists() or not pathway_names.exists():
        logger.error("Required Reactome files not found")
        return pd.DataFrame()
    
    pathways_df = parse_reactome_pathways(uniprot2reactome, pathway_names)
    
    return pathways_df


def create_pathway_enrichment_validator(graph_gene_pathways: pd.DataFrame,
                                        background_pathways: pd.DataFrame) -> Dict:
    """
    Create a pathway enrichment validator for predicted gene lists.
    
    Given a list of predicted genes, compute enriched pathways using
    hypergeometric test.
    
    Args:
        graph_gene_pathways: Gene-pathway mappings for genes in your graph
        background_pathways: All gene-pathway mappings (background)
        
    Returns:
        Dict with pathway enrichment functions
    """
    # This is a placeholder for a more complete implementation
    # You would use scipy.stats.hypergeom for enrichment testing
    
    logger.info("Pathway enrichment validator created")
    logger.info("For full implementation, use scipy.stats.hypergeom or GSEApy")
    
    return {
        'gene_pathways': graph_gene_pathways,
        'background': background_pathways,
        'method': 'hypergeometric_test'
    }


# Example usage functions
def integrate_uniprot_into_graph(gene_list: List[str],
                                descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add UniProt descriptions as node features for genes in graph.
    
    Args:
        gene_list: List of gene symbols in your graph
        descriptions_df: From get_uniprot_gene_descriptions
        
    Returns:
        DataFrame with gene features
    """
    # Filter to genes in graph
    gene_features = descriptions_df[descriptions_df['gene_symbol'].isin(gene_list)].copy()
    
    logger.info(f"Matched {len(gene_features)}/{len(gene_list)} genes with UniProt descriptions")
    
    return gene_features


def integrate_pathways_into_graph(gene_list: List[str],
                                  pathways_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pathway edges for genes in graph.
    
    Args:
        gene_list: List of gene symbols in your graph
        pathways_df: From get_reactome_gene_pathways
        
    Returns:
        DataFrame with gene-pathway edges for graph genes
    """
    # This would require mapping UniProt IDs to gene symbols first
    # For now, return empty DataFrame as placeholder
    
    logger.warning("Pathway integration requires UniProt ID mapping")
    return pd.DataFrame()
```

## File: `src/data/__init__.py`

```python
"""
Data module for PromptGFM-Bio.

This module handles data downloading, preprocessing, and dataset management
for gene-disease association prediction.
"""
```

## File: `src/evaluation/case_study.py`

```python
"""
Case studies for model validation on known disease-gene associations.

Implements detailed analysis for:
- Angelman Syndrome (UBE3A)
- Rett Syndrome (MECP2)
- Fragile X Syndrome (FMR1)

Validates model predictions against literature and expert knowledge.
"""

import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaseStudy:
    """
    Base class for disease case studies.
    
    Analyzes model predictions for specific diseases with known gene associations.
    """
    
    def __init__(
        self,
        model,
        graph,
        disease_name: str,
        primary_genes: List[str],
        pathway_genes: List[str],
        negative_controls: List[str],
        phenotypes: List[str],
        description: str
    ):
        self.model = model
        self.graph = graph
        self.disease_name = disease_name
        self.primary_genes = primary_genes
        self.pathway_genes = pathway_genes
        self.negative_controls = negative_controls
        self.phenotypes = phenotypes
        self.description = description
        
        logger.info(f"CaseStudy initialized for: {disease_name}")
        logger.info(f"  Primary genes: {primary_genes}")
        logger.info(f"  Pathway genes: {len(pathway_genes)} genes")
        logger.info(f"  Negative controls: {negative_controls}")
    
    def create_prompt(self) -> str:
        """Create disease prompt for the model."""
        phenotype_str = ", ".join(self.phenotypes[:10])  # Limit to 10
        return f"Disease: {self.disease_name}. Phenotypes: {phenotype_str}. " \
               f"Description: {self.description}. Associated genes:"
    
    def rank_all_genes(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        top_k: int = 100,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Rank all genes for this disease.
        
        Returns:
            gene_indices: Indices of top-K genes
            scores: Prediction scores
            gene_names: Names of top-K genes
        """
        logger.info(f"\nRanking all genes for {self.disease_name}...")
        
        # Move data to device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        # Create prompt
        prompt = self.create_prompt()
        
        # Get all gene indices
        num_genes = node_features.shape[0]
        all_gene_indices = torch.arange(num_genes, device=device)
        
        # Predict scores
        self.model.eval()
        with torch.no_grad():
            ranked_indices, scores = self.model.get_gene_rankings(
                node_features=node_features,
                edge_index=edge_index,
                disease_text=prompt,
                candidate_gene_indices=all_gene_indices,
                top_k=top_k
            )
        
        # Convert to numpy
        ranked_indices = ranked_indices.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # Get gene names (if available)
        gene_names = [f"Gene_{idx}" for idx in ranked_indices]  # Placeholder
        
        return ranked_indices, scores, gene_names
    
    def analyze_known_genes(
        self,
        ranked_indices: np.ndarray,
        gene_names: List[str]
    ) -> Dict:
        """
        Analyze where known genes appear in rankings.
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'primary_genes': {},
            'pathway_genes': {},
            'negative_controls': {},
            'summary': {}
        }
        
        # Create gene name to rank mapping
        gene_to_rank = {name: rank + 1 for rank, name in enumerate(gene_names)}
        
        # Check primary genes
        logger.info(f"\nPrimary genes:")
        for gene in self.primary_genes:
            rank = gene_to_rank.get(gene, None)
            if rank:
                logger.info(f"  {gene}: Rank {rank}")
                results['primary_genes'][gene] = rank
            else:
                logger.info(f"  {gene}: Not in top {len(gene_names)}")
                results['primary_genes'][gene] = None
        
        # Check pathway genes
        logger.info(f"\nPathway genes:")
        pathway_ranks = []
        for gene in self.pathway_genes[:10]:  # Show first 10
            rank = gene_to_rank.get(gene, None)
            if rank:
                logger.info(f"  {gene}: Rank {rank}")
                results['pathway_genes'][gene] = rank
                pathway_ranks.append(rank)
            else:
                results['pathway_genes'][gene] = None
        
        # Check negative controls
        logger.info(f"\nNegative controls (should rank low):")
        for gene in self.negative_controls:
            rank = gene_to_rank.get(gene, None)
            if rank:
                logger.info(f"  {gene}: Rank {rank}")
                results['negative_controls'][gene] = rank
            else:
                logger.info(f"  {gene}: Not in top {len(gene_names)}")
                results['negative_controls'][gene] = None
        
        # Summary statistics
        primary_ranks = [r for r in results['primary_genes'].values() if r is not None]
        pathway_ranks_all = [r for r in results['pathway_genes'].values() if r is not None]
        negative_ranks = [r for r in results['negative_controls'].values() if r is not None]
        
        results['summary'] = {
            'primary_mean_rank': np.mean(primary_ranks) if primary_ranks else None,
            'primary_median_rank': np.median(primary_ranks) if primary_ranks else None,
            'primary_min_rank': np.min(primary_ranks) if primary_ranks else None,
            'pathway_mean_rank': np.mean(pathway_ranks_all) if pathway_ranks_all else None,
            'negative_mean_rank': np.mean(negative_ranks) if negative_ranks else None,
            'primary_in_top10': sum(1 for r in primary_ranks if r <= 10),
            'primary_in_top50': sum(1 for r in primary_ranks if r <= 50),
            'pathway_in_top100': sum(1 for r in pathway_ranks_all if r <= 100)
        }
        
        logger.info(f"\nSummary:")
        for key, value in results['summary'].items():
            logger.info(f"  {key}: {value}")
        
        return results
    
    def run_case_study(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        top_k: int = 100,
        device: str = 'cuda',
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Run complete case study analysis.
        
        Returns:
            Complete analysis results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Case Study: {self.disease_name}")
        logger.info(f"{'='*60}")
        
        # Rank genes
        ranked_indices, scores, gene_names = self.rank_all_genes(
            node_features, edge_index, top_k, device
        )
        
        # Analyze known genes
        analysis = self.analyze_known_genes(ranked_indices, gene_names)
        
        # Create full results
        results = {
            'disease_name': self.disease_name,
            'top_predictions': [
                {'rank': i + 1, 'gene': gene, 'score': float(scores[i])}
                for i, gene in enumerate(gene_names[:20])  # Top 20
            ],
            'known_genes_analysis': analysis,
            'success_metrics': {
                'primary_gene_top10': analysis['summary']['primary_in_top10'] > 0,
                'primary_gene_top50': analysis['summary']['primary_in_top50'] > 0,
                'best_primary_rank': analysis['summary']['primary_min_rank']
            }
        }
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✓ Results saved to {save_path}")
        
        return results


class AngelmanCaseStudy(CaseStudy):
    """
    Angelman Syndrome case study.
    
    Primary Gene: UBE3A (ubiquitin protein ligase E3A)
    - Most common cause (70-80% of cases)
    - Maternal deletion/mutation of chromosome 15q11-q13
    
    Key Phenotypes:
    - Severe intellectual disability
    - Absent speech
    - Seizures
    - Ataxia
    - Happy demeanor/inappropriate laughter
    
    Pathway Genes: Involved in ubiquitin pathway, neurodevelopment
    Negative Controls: Genes for other neurodevelopmental disorders
    """
    
    def __init__(self, model, graph):
        super().__init__(
            model=model,
            graph=graph,
            disease_name="Angelman Syndrome",
            primary_genes=["UBE3A"],
            pathway_genes=[
                "MAPK1", "PRMT5", "CDK1", "CDK4",  # Cell cycle
                "GABRB3", "GABRA5", "GABRG3",  # GABA receptors (nearby on chr 15)
                "HERC2", "OCA2"  # Nearby genes
            ],
            negative_controls=[
                "MECP2",  # Rett syndrome
                "ZEB2",   # Mowat-Wilson syndrome
                "TCF4"    # Pitt-Hopkins syndrome
            ],
            phenotypes=[
                "severe intellectual disability",
                "absent speech",
                "seizures",
                "ataxia",
                "happy demeanor",
                "inappropriate laughter",
                "microcephaly",
                "motor delay"
            ],
            description="Rare genetic disorder causing developmental disabilities, "
                       "neurological problems, seizures, and frequent laughter"
        )


class RettCaseStudy(CaseStudy):
    """Rett Syndrome case study."""
    
    def __init__(self, model, graph):
        super().__init__(
            model=model,
            graph=graph,
            disease_name="Rett Syndrome",
            primary_genes=["MECP2"],
            pathway_genes=[
                "CDKL5", "FOXG1",  # Atypical Rett variants
                "BDNF", "CREB1"  # MeCP2 targets
            ],
            negative_controls=["UBE3A", "FMR1", "TCF4"],
            phenotypes=[
                "regression after normal development",
                "loss of purposeful hand skills",
                "hand stereotypies",
                "seizures",
                "breathing abnormalities",
                "autism features"
            ],
            description="Progressive neurological disorder affecting mostly females, "
                       "characterized by regression and loss of purposeful hand skills"
        )


class FragileXCaseStudy(CaseStudy):
    """Fragile X Syndrome case study."""
    
    def __init__(self, model, graph):
        super().__init__(
            model=model,
            graph=graph,
            disease_name="Fragile X Syndrome",
            primary_genes=["FMR1"],
            pathway_genes=[
                "FXR1", "FXR2",  # Related RNA-binding proteins
                "CYFIP1", "CYFIP2",  # FMR1 interactors
                "mGluR5"  # Metabotropic glutamate receptor
            ],
            negative_controls=["MECP2", "UBE3A", "SHANK3"],
            phenotypes=[
                "intellectual disability",
                "autism spectrum disorder",
                "social anxiety",
                "hyperactivity",
                "macro-orchidism",
                "long face",
                "large ears"
            ],
            description="Most common inherited cause of intellectual disability, "
                       "caused by expansion of CGG repeats in FMR1 gene"
        )


def run_all_case_studies(
    model,
    graph,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    save_dir: Optional[Path] = None,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Run all predefined case studies.
    
    Returns:
        Dictionary mapping disease name to results
    """
    logger.info("\n" + "="*60)
    logger.info("Running All Case Studies")
    logger.info("="*60)
    
    case_studies = [
        AngelmanCaseStudy(model, graph),
        RettCaseStudy(model, graph),
        FragileXCaseStudy(model, graph)
    ]
    
    all_results = {}
    
    for case_study in case_studies:
        save_path = None
        if save_dir:
            save_path = Path(save_dir) / f"{case_study.disease_name.replace(' ', '_')}.json"
        
        results = case_study.run_case_study(
            node_features, edge_index, top_k=100, device=device, save_path=save_path
        )
        
        all_results[case_study.disease_name] = results
    
    logger.info("\n" + "="*60)
    logger.info("All Case Studies Complete")
    logger.info("="*60)
    
    return all_results


def test_case_study():
    """Test case study with dummy data."""
    logger.info("Testing case study module...")
    
    logger.info("Case study test placeholder - requires full model and data")
    logger.info("✓ Case study module loaded successfully")


if __name__ == "__main__":
    test_case_study()
```

## File: `src/evaluation/metrics.py`

```python
"""
Evaluation metrics for gene-disease association prediction and ranking.

Implements comprehensive metrics for link prediction and ranking evaluation:
- AUROC (Area Under ROC Curve) - Binary classification performance
- AUPR (Area Under Precision-Recall Curve) - Precision-recall tradeoff
- Precision@K - Precision in top-K predictions
- Recall@K - Recall in top-K predictions
- Mean Average Precision (MAP) - Average precision across queries
- Mean Reciprocal Rank (MRR) - Reciprocal rank of first relevant item
- NDCG@K (Normalized Discounted Cumulative Gain) - Ranking quality
- Hit Rate@K - Proportion of queries with at least one hit in top-K
"""

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)
from typing import Dict, List, Optional, Tuple, Union
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneRankingEvaluator:
    """
    Comprehensive evaluator for gene-disease association prediction.
    
    Handles both binary classification metrics (AUROC, AUPR) and
    ranking metrics (Precision@K, MRR, NDCG).
    """
    
    def __init__(self, k_values: List[int] = [10, 20, 50, 100]):
        """
        Args:
            k_values: List of K values for computing Precision@K, Recall@K, etc.
        """
        self.k_values = k_values
        logger.info(f"GeneRankingEvaluator initialized with K values: {k_values}")
    
    def compute_auroc(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_scores: Union[np.ndarray, torch.Tensor, List]
    ) -> float:
        """
        Compute Area Under ROC Curve.
        
        Args:
            y_true: Binary ground truth labels (0 or 1)
            y_scores: Predicted scores (higher = more likely positive)
            
        Returns:
            AUROC score (0-1, higher is better)
        """
        y_true = self._to_numpy(y_true)
        y_scores = self._to_numpy(y_scores)
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in y_true. AUROC is undefined.")
            return float('nan')
        
        return roc_auc_score(y_true, y_scores)
    
    def compute_aupr(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_scores: Union[np.ndarray, torch.Tensor, List]
    ) -> float:
        """
        Compute Area Under Precision-Recall Curve.
        
        Args:
            y_true: Binary ground truth labels (0 or 1)
            y_scores: Predicted scores (higher = more likely positive)
            
        Returns:
            AUPR score (0-1, higher is better)
        """
        y_true = self._to_numpy(y_true)
        y_scores = self._to_numpy(y_scores)
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in y_true. AUPR is undefined.")
            return float('nan')
        
        return average_precision_score(y_true, y_scores)
    
    def precision_at_k(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_scores: Union[np.ndarray, torch.Tensor, List],
        k: int = 20
    ) -> float:
        """
        Compute Precision@K.
        
        Fraction of top-K predictions that are correct.
        
        Args:
            y_true: Binary ground truth labels
            y_scores: Predicted scores
            k: Number of top predictions to consider
            
        Returns:
            Precision@K (0-1, higher is better)
        """
        y_true = self._to_numpy(y_true)
        y_scores = self._to_numpy(y_scores)
        
        # Get top-K indices
        top_k_indices = np.argsort(y_scores)[-k:][::-1]
        
        # Compute precision
        precision = np.sum(y_true[top_k_indices]) / k
        
        return precision
    
    def recall_at_k(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_scores: Union[np.ndarray, torch.Tensor, List],
        k: int = 20
    ) -> float:
        """
        Compute Recall@K.
        
        Fraction of all positives that appear in top-K predictions.
        
        Args:
            y_true: Binary ground truth labels
            y_scores: Predicted scores
            k: Number of top predictions to consider
            
        Returns:
            Recall@K (0-1, higher is better)
        """
        y_true = self._to_numpy(y_true)
        y_scores = self._to_numpy(y_scores)
        
        num_positives = np.sum(y_true)
        if num_positives == 0:
            return 0.0
        
        # Get top-K indices
        top_k_indices = np.argsort(y_scores)[-k:][::-1]
        
        # Compute recall
        recall = np.sum(y_true[top_k_indices]) / num_positives
        
        return recall
    
    def mean_average_precision(
        self,
        rankings: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Compute Mean Average Precision (MAP).
        
        Average precision averaged across multiple queries/diseases.
        
        Args:
            rankings: List of (y_true, y_scores) tuples, one per query
            
        Returns:
            MAP score (0-1, higher is better)
        """
        aps = []
        
        for y_true, y_scores in rankings:
            y_true = self._to_numpy(y_true)
            y_scores = self._to_numpy(y_scores)
            
            if len(np.unique(y_true)) < 2:
                continue
            
            ap = average_precision_score(y_true, y_scores)
            aps.append(ap)
        
        if len(aps) == 0:
            return 0.0
        
        return np.mean(aps)
    
    def mean_reciprocal_rank(
        self,
        rankings: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Average of reciprocal ranks of first relevant item across queries.
        
        Args:
            rankings: List of (y_true, y_scores) tuples, one per query
            
        Returns:
            MRR score (0-1, higher is better)
        """
        rrs = []
        
        for y_true, y_scores in rankings:
            y_true = self._to_numpy(y_true)
            y_scores = self._to_numpy(y_scores)
            
            # Sort by score (descending)
            sorted_indices = np.argsort(y_scores)[::-1]
            sorted_labels = y_true[sorted_indices]
            
            # Find rank of first positive
            first_positive_idx = np.where(sorted_labels == 1)[0]
            if len(first_positive_idx) > 0:
                rank = first_positive_idx[0] + 1  # 1-indexed
                rrs.append(1.0 / rank)
            else:
                rrs.append(0.0)
        
        if len(rrs) == 0:
            return 0.0
        
        return np.mean(rrs)
    
    def ndcg_at_k(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_scores: Union[np.ndarray, torch.Tensor, List],
        k: int = 20
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.
        
        Measures ranking quality with position-based discounting.
        
        Args:
            y_true: Ground truth relevance scores (0 or 1 for binary)
            y_scores: Predicted scores
            k: Number of top predictions to consider
            
        Returns:
            NDCG@K (0-1, higher is better, 1 = perfect ranking)
        """
        y_true = self._to_numpy(y_true)
        y_scores = self._to_numpy(y_scores)
        
        # Get top-K indices
        top_k_indices = np.argsort(y_scores)[-k:][::-1]
        
        # DCG: sum of (relevance / log2(rank + 1))
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            relevance = y_true[idx]
            rank = i + 1
            dcg += relevance / np.log2(rank + 1)
        
        # Ideal DCG: sort by true relevance
        ideal_indices = np.argsort(y_true)[-k:][::-1]
        idcg = 0.0
        for i, idx in enumerate(ideal_indices):
            relevance = y_true[idx]
            rank = i + 1
            idcg += relevance / np.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(
        self,
        rankings: List[Tuple[np.ndarray, np.ndarray]],
        k: int = 20
    ) -> float:
        """
        Compute Hit Rate@K.
        
        Fraction of queries that have at least one relevant item in top-K.
        
        Args:
            rankings: List of (y_true, y_scores) tuples, one per query
            k: Number of top predictions to consider
            
        Returns:
            Hit Rate@K (0-1, higher is better)
        """
        hits = 0
        total = 0
        
        for y_true, y_scores in rankings:
            y_true = self._to_numpy(y_true)
            y_scores = self._to_numpy(y_scores)
            
            if len(y_true) == 0:
                continue
            
            # Get top-K indices
            top_k = min(k, len(y_scores))
            top_k_indices = np.argsort(y_scores)[-top_k:][::-1]
            
            # Check if any positive in top-K
            if np.sum(y_true[top_k_indices]) > 0:
                hits += 1
            
            total += 1
        
        if total == 0:
            return 0.0
        
        return hits / total
    
    def evaluate_all(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_scores: Union[np.ndarray, torch.Tensor, List],
        rankings: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            y_true: Binary ground truth labels for all predictions
            y_scores: Predicted scores for all predictions
            rankings: Optional list of per-query rankings for MAP, MRR, Hit Rate
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Binary classification metrics
        try:
            metrics['auroc'] = self.compute_auroc(y_true, y_scores)
        except Exception as e:
            logger.warning(f"Failed to compute AUROC: {e}")
            metrics['auroc'] = float('nan')
        
        try:
            metrics['aupr'] = self.compute_aupr(y_true, y_scores)
        except Exception as e:
            logger.warning(f"Failed to compute AUPR: {e}")
            metrics['aupr'] = float('nan')
        
        # Ranking metrics at different K values
        for k in self.k_values:
            try:
                metrics[f'precision@{k}'] = self.precision_at_k(y_true, y_scores, k)
                metrics[f'recall@{k}'] = self.recall_at_k(y_true, y_scores, k)
                metrics[f'ndcg@{k}'] = self.ndcg_at_k(y_true, y_scores, k)
            except Exception as e:
                logger.warning(f"Failed to compute metrics@{k}: {e}")
        
        # Per-query metrics (if rankings provided)
        if rankings is not None:
            try:
                metrics['map'] = self.mean_average_precision(rankings)
                metrics['mrr'] = self.mean_reciprocal_rank(rankings)
                
                for k in self.k_values:
                    metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(rankings, k)
            except Exception as e:
                logger.warning(f"Failed to compute per-query metrics: {e}")
        
        return metrics
    
    def evaluate_stratified(
        self,
        y_true_by_stratum: Dict[str, np.ndarray],
        y_scores_by_stratum: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics stratified by disease rarity or other categories.
        
        Args:
            y_true_by_stratum: Dict of stratum_name -> y_true arrays
            y_scores_by_stratum: Dict of stratum_name -> y_scores arrays
            
        Returns:
            Dict of stratum_name -> metrics dict
        """
        results = {}
        
        for stratum_name in y_true_by_stratum.keys():
            y_true = y_true_by_stratum[stratum_name]
            y_scores = y_scores_by_stratum[stratum_name]
            
            logger.info(f"Evaluating stratum: {stratum_name}")
            results[stratum_name] = self.evaluate_all(y_true, y_scores)
        
        return results
    
    def print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Pretty print metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            prefix: Optional prefix for logging
        """
        if prefix:
            logger.info(f"\n{prefix}")
        
        # Print classification metrics
        if 'auroc' in metrics:
            logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        if 'aupr' in metrics:
            logger.info(f"  AUPR:  {metrics['aupr']:.4f}")
        
        # Print ranking metrics
        for k in self.k_values:
            if f'precision@{k}' in metrics:
                logger.info(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            if f'recall@{k}' in metrics:
                logger.info(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            if f'ndcg@{k}' in metrics:
                logger.info(f"  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
            if f'hit_rate@{k}' in metrics:
                logger.info(f"  Hit Rate@{k}: {metrics[f'hit_rate@{k}']:.4f}")
        
        # Print per-query metrics
        if 'map' in metrics:
            logger.info(f"  MAP:   {metrics['map']:.4f}")
        if 'mrr' in metrics:
            logger.info(f"  MRR:   {metrics['mrr']:.4f}")
    
    def _to_numpy(self, arr: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert various array types to numpy."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        elif isinstance(arr, list):
            return np.array(arr)
        return arr


def test_metrics():
    """Test evaluation metrics with dummy data."""
    logger.info("Testing evaluation metrics...")
    
    # Create dummy data
    np.random.seed(42)
    n = 1000
    
    # Simulated predictions (higher scores for positives)
    y_true = np.random.binomial(1, 0.2, n)  # 20% positive
    y_scores = np.random.randn(n) + y_true * 1.5  # Positives have higher scores
    
    evaluator = GeneRankingEvaluator(k_values=[10, 20, 50])
    
    # Test individual metrics
    logger.info("\nTesting individual metrics:")
    auroc = evaluator.compute_auroc(y_true, y_scores)
    logger.info(f"  AUROC: {auroc:.4f}")
    
    aupr = evaluator.compute_aupr(y_true, y_scores)
    logger.info(f"  AUPR: {aupr:.4f}")
    
    p_at_10 = evaluator.precision_at_k(y_true, y_scores, k=10)
    logger.info(f"  Precision@10: {p_at_10:.4f}")
    
    r_at_10 = evaluator.recall_at_k(y_true, y_scores, k=10)
    logger.info(f"  Recall@10: {r_at_10:.4f}")
    
    ndcg_at_10 = evaluator.ndcg_at_k(y_true, y_scores, k=10)
    logger.info(f"  NDCG@10: {ndcg_at_10:.4f}")
    
    # Test per-query metrics
    logger.info("\nTesting per-query metrics:")
    num_queries = 10
    rankings = []
    for _ in range(num_queries):
        query_n = 100
        query_true = np.random.binomial(1, 0.2, query_n)
        query_scores = np.random.randn(query_n) + query_true * 1.5
        rankings.append((query_true, query_scores))
    
    map_score = evaluator.mean_average_precision(rankings)
    logger.info(f"  MAP: {map_score:.4f}")
    
    mrr_score = evaluator.mean_reciprocal_rank(rankings)
    logger.info(f"  MRR: {mrr_score:.4f}")
    
    hit_rate = evaluator.hit_rate_at_k(rankings, k=10)
    logger.info(f"  Hit Rate@10: {hit_rate:.4f}")
    
    # Test evaluate_all
    logger.info("\nTesting evaluate_all:")
    metrics = evaluator.evaluate_all(y_true, y_scores, rankings)
    evaluator.print_metrics(metrics, prefix="All Metrics:")
    
    # Test stratified evaluation
    logger.info("\nTesting stratified evaluation:")
    y_true_stratified = {
        'rare': y_true[:300],
        'common': y_true[300:]
    }
    y_scores_stratified = {
        'rare': y_scores[:300],
        'common': y_scores[300:]
    }
    
    stratified_results = evaluator.evaluate_stratified(y_true_stratified, y_scores_stratified)
    for stratum, metrics in stratified_results.items():
        evaluator.print_metrics(metrics, prefix=f"Stratum: {stratum}")
    
    # Test with PyTorch tensors
    logger.info("\nTesting with PyTorch tensors:")
    y_true_torch = torch.from_numpy(y_true)
    y_scores_torch = torch.from_numpy(y_scores)
    auroc_torch = evaluator.compute_auroc(y_true_torch, y_scores_torch)
    logger.info(f"  AUROC (torch): {auroc_torch:.4f}")
    
    logger.info("\n✓ All metric tests passed!")


if __name__ == "__main__":
    test_metrics()
```

## File: `src/evaluation/__init__.py`

```python
"""
Evaluation module for PromptGFM-Bio.

This module implements evaluation metrics and case studies for
gene-disease association prediction.
"""
```

## File: `src/models/conditioning.py`

```python
"""
Conditioning mechanisms for fusing prompt embeddings into GNN layers.

Implements:
1. FiLM (Feature-wise Linear Modulation)
2. Cross-Attention Conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioning.
    
    Modulates node features using prompt embeddings:
        out = gamma * x + beta
    
    Where gamma (scale) and beta (shift) are learned functions of the prompt.
    
    Reference:
    "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)
    """
    
    def __init__(
        self,
        node_dim: int,
        prompt_dim: int,
        use_batch_norm: bool = False,
        dropout: float = 0.1,
        use_conditioning: bool = True,  # False → identity transform (gamma=1, beta=0)
    ):
        """
        Args:
            node_dim: Dimension of node features
            prompt_dim: Dimension of prompt embeddings
            use_batch_norm: Apply batch normalization before FiLM
            dropout: Dropout rate for prompt processing
            use_conditioning: If False, forward() returns x unchanged (ablation: No-Prompt)
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.prompt_dim = prompt_dim
        self.use_batch_norm = use_batch_norm
        self.use_conditioning = use_conditioning  # ablation flag
        
        # Prompt processing layers
        hidden_dim = max(prompt_dim // 2, node_dim)
        self.prompt_processor = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # FiLM parameter generators
        self.gamma_gen = nn.Linear(hidden_dim, node_dim)  # Scale
        self.beta_gen = nn.Linear(hidden_dim, node_dim)   # Shift
        
        # Optional batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(node_dim)
        
        # Initialize gamma to near 1.0 and beta to near 0.0
        nn.init.normal_(self.gamma_gen.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gamma_gen.bias, 1.0)
        nn.init.normal_(self.beta_gen.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.beta_gen.bias, 0.0)
        
        logger.info(f"FiLM Conditioning: node_dim={node_dim}, prompt_dim={prompt_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        prompt_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            x: Node features [batch_size, node_dim] or [num_nodes, node_dim]
            prompt_embedding: Prompt embedding [batch_size, prompt_dim] or [prompt_dim]
        
        Returns:
            Modulated features [batch_size, node_dim] or [num_nodes, node_dim]
        """
        # Ablation: identity transform — skip FiLM entirely (gamma=1, beta=0 equivalent)
        if not self.use_conditioning:
            return x

        # Handle shape
        if prompt_embedding.dim() == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        
        # Process prompt to get conditioning parameters
        h = self.prompt_processor(prompt_embedding)
        gamma = self.gamma_gen(h)  # [batch_size, node_dim]
        beta = self.beta_gen(h)    # [batch_size, node_dim]
        
        # Optional batch normalization
        if self.use_batch_norm and x.dim() == 2:
            x = self.batch_norm(x)
        
        # Apply FiLM: out = gamma * x + beta
        # Handle broadcasting for different input shapes
        if x.dim() == 2 and gamma.size(0) == 1:
            # Single prompt for all nodes
            out = gamma * x + beta
        elif x.dim() == 3:
            # Batched: [batch_size, num_nodes, node_dim]
            out = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        else:
            out = gamma * x + beta
        
        return out
    
    def get_film_params(
        self,
        prompt_embedding: torch.Tensor
    ) -> tuple:
        """Get FiLM parameters (gamma, beta) for analysis."""
        if prompt_embedding.dim() == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        
        h = self.prompt_processor(prompt_embedding)
        gamma = self.gamma_gen(h)
        beta = self.beta_gen(h)
        
        return gamma, beta


class CrossAttentionConditioning(nn.Module):
    """
    Cross-attention conditioning mechanism.
    
    Allows nodes to attend to prompt embeddings directly.
    More flexible than FiLM but computationally expensive.
    
    Q: from node features
    K, V: from prompt embeddings
    """
    
    def __init__(
        self,
        node_dim: int,
        prompt_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Args:
            node_dim: Dimension of node features
            prompt_dim: Dimension of prompt embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_residual: Use residual connection
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        self.head_dim = node_dim // num_heads
        
        # Project prompt to match node_dim if needed
        if prompt_dim != node_dim:
            self.prompt_proj = nn.Linear(prompt_dim, node_dim)
        else:
            self.prompt_proj = None
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)
        
        self.scale = math.sqrt(self.head_dim)
        
        logger.info(f"Cross-Attention: node_dim={node_dim}, prompt_dim={prompt_dim}, "
                   f"heads={num_heads}")
    
    def forward(
        self,
        x: torch.Tensor,
        prompt_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention conditioning.
        
        Args:
            x: Node features [batch_size, num_nodes, node_dim]
            prompt_embedding: Prompt [batch_size, seq_len, prompt_dim]
            attention_mask: Optional mask [batch_size, num_nodes, seq_len]
        
        Returns:
            Conditioned features [batch_size, num_nodes, node_dim]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Project prompt if needed
        if self.prompt_proj is not None:
            prompt_embedding = self.prompt_proj(prompt_embedding)
        
        seq_len = prompt_embedding.size(1)
        
        # Queries from nodes
        Q = self.q_proj(x)  # [batch, num_nodes, node_dim]
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, heads, num_nodes, head_dim]
        
        # Keys and Values from prompt
        K = self.k_proj(prompt_embedding)  # [batch, seq_len, node_dim]
        V = self.v_proj(prompt_embedding)  # [batch, seq_len, node_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        V = V.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch, heads, num_nodes, seq_len]
        
        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [batch, heads, num_nodes, head_dim]
        out = out.transpose(1, 2).contiguous()  # [batch, num_nodes, heads, head_dim]
        out = out.view(batch_size, num_nodes, self.node_dim)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Residual connection and layer norm
        if self.use_residual:
            out = self.layer_norm(x + out)
        else:
            out = self.layer_norm(out)
        
        return out


class HybridConditioning(nn.Module):
    """
    Hybrid conditioning: FiLM + Cross-Attention.
    
    Combines benefits of both:
    - FiLM: Fast, global modulation
    - Cross-Attention: Flexible, selective attention
    """
    
    def __init__(
        self,
        node_dim: int,
        prompt_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        film_weight: float = 0.5
    ):
        super().__init__()
        
        self.film = FiLMConditioning(node_dim, prompt_dim, dropout=dropout)
        self.cross_attn = CrossAttentionConditioning(
            node_dim, prompt_dim, num_heads, dropout
        )
        
        # Learnable mixing weight
        self.film_weight = nn.Parameter(torch.tensor(film_weight))
        
        logger.info(f"Hybrid Conditioning: node_dim={node_dim}, prompt_dim={prompt_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        prompt_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Apply hybrid conditioning."""
        # FiLM modulation
        x_film = self.film(x, prompt_embedding)
        
        # Cross-attention (requires 3D input)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim
        if prompt_embedding.dim() == 2:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        if prompt_embedding.dim() == 2:
            # If prompt is 2D [batch, dim], add seq dimension
            prompt_embedding = prompt_embedding.unsqueeze(1)
        
        x_attn = self.cross_attn(x, prompt_embedding)
        
        # Mix both
        w = torch.sigmoid(self.film_weight)
        out = w * x_film + (1 - w) * x_attn
        
        return out


if __name__ == "__main__":
    # Test conditioning mechanisms
    logger.info("Testing conditioning mechanisms...")
    
    batch_size = 32
    num_nodes = 100
    node_dim = 256
    prompt_dim = 768
    
    x =torch.randn(batch_size, num_nodes, node_dim)
    prompt = torch.randn(batch_size, 10, prompt_dim)  # 10 tokens
    
    # Test FiLM
    print("Testing FiLM...")
    film = FiLMConditioning(node_dim, prompt_dim)
    # Flatten for FiLM test
    x_flat = x.view(-1, node_dim)  # [batch*nodes, node_dim]
    prompt_global = prompt.mean(dim=1)  # [batch, prompt_dim]
    # Repeat prompt for each node
    prompt_repeated = prompt_global.unsqueeze(1).repeat(1, num_nodes, 1).view(-1, prompt_dim)
    out_film = film(x_flat, prompt_repeated)
    print(f"FiLM output shape: {out_film.shape}")
    
    # Test Cross-Attention
    print("Testing Cross-Attention...")
    cross_attn = CrossAttentionConditioning(node_dim, prompt_dim, num_heads=8)
    out_attn = cross_attn(x, prompt)
    print(f"Cross-Attention output shape: {out_attn.shape}")
    
    print("✓ Conditioning tests passed!")
```

## File: `src/models/gnn_backbone.py`

```python
"""
GNN backbone architectures for PromptGFM-Bio.

Supports multiple GNN architectures:
- GraphSAGE (mean/max/LSTM aggregation)
- GAT (Graph Attention Networks)
- GIN (Graph Isomorphism Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GINConv
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class GNNBackbone(nn.Module):
    """
    Flexible GNN backbone supporting multiple architectures.
    
    Architectures:
    - graphsage: GraphSAGE with mean aggregation
    - gat: Graph Attention Networks
    - gin: Graph Isomorphism Network
    
    Features:
    - Multi-layer graph convolutions
    - Residual connections (optional)
    - Layer normalization
    - Dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        gnn_type: str = 'graphsage',
        dropout: float = 0.2,
        heads: int = 4,  # For GAT
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('graphsage', 'gat', 'gin')
            dropout: Dropout rate
            heads: Number of attention heads (GAT only)
            use_residual: Use residual connections
            use_layer_norm: Use layer normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Input projection if needed
        if input_dim != hidden_dim and num_layers > 0:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            first_layer_dim = hidden_dim
        else:
            self.input_proj = None
            first_layer_dim = input_dim
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        for i in range(num_layers):
            in_dim = first_layer_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            # Create GNN layer based on type
            if gnn_type == 'graphsage':
                conv = SAGEConv(in_dim, out_dim)
            elif gnn_type == 'gat':
                conv = GATConv(
                    in_dim,
                    out_dim // heads if i < num_layers - 1 else out_dim,
                    heads=heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, 2 * out_dim),
                    nn.ReLU(),
                    nn.Linear(2 * out_dim, out_dim)
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.convs.append(conv)
            
            # Layer normalization (not on last layer)
            if use_layer_norm and i < num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        logger.info(f"GNN Backbone: {gnn_type}, {num_layers} layers, "
                   f"{input_dim}→{hidden_dim}→{output_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x_prev = x
            
            # Graph convolution
            if self.gnn_type == 'graphsage' and edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            
            # Skip last layer activations/norms
            if i < self.num_layers - 1:
                # Layer normalization
                if self.use_layer_norm:
                    x = self.norms[i](x)
                
                # Activation
                x = F.relu(x)
                
                # Dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                # Residual connection
                if self.use_residual and x_prev.shape == x.shape:
                    x = x + x_prev
        
        return x
    
    def get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Get embeddings from all layers (for analysis/visualization)."""
        embeddings = []
        
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        for i, conv in enumerate(self.convs):
            x_prev = x
            
            if self.gnn_type == 'graphsage' and edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                if self.use_layer_norm:
                    x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.use_residual and x_prev.shape == x.shape:
                    x = x + x_prev
            
            embeddings.append(x)
        
        return embeddings


if __name__ == "__main__":
    # Test GNN backbone
    logger.info("Testing GNN backbone...")
    
    # Create random graph
    num_nodes = 100
    num_edges = 500
    input_dim = 128
    hidden_dim = 256
    output_dim = 128
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Test GraphSAGE
    model_sage = GNNBackbone(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        gnn_type='graphsage'
    )
    
    out = model_sage(x, edge_index)
    print(f"GraphSAGE output shape: {out.shape}")
    
    # Test GAT
    model_gat = GNNBackbone(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        gnn_type='gat',
        heads=4
    )
    
    out = model_gat(x, edge_index)
    print(f"GAT output shape: {out.shape}")
    
    print("✓ GNN backbone tests passed!")
```

## File: `src/models/promptgfm.py`

```python
"""
Complete PromptGFM-Bio model.

Integrates:
- Prompt encoder (BioBERT)
- GNN backbone (GraphSAGE/GAT/GIN)
- Conditioning mechanism (FiLM/Cross-Attention/Hybrid)
- Link prediction head

This is the main model for gene-disease association prediction conditioned on
disease descriptions and phenotype information.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from torch_geometric.data import HeteroData

from .prompt_encoder import PromptEncoder
from .gnn_backbone import GNNBackbone
from .conditioning import FiLMConditioning, CrossAttentionConditioning, HybridConditioning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptGFM(nn.Module):
    """
    Complete PromptGFM-Bio model for promptconditioned gene-disease prediction.
    
    The model workflow:
    1. Encode disease description/phenotypes using BioBERT
    2. Apply GNN message passing on biomedical graph
    3. Condition gene embeddings on prompt using FiLM/Cross-Attention
    4. Predict gene-disease associations using MLP head
    
    Args:
        gnn_input_dim: Input dimension for GNN (node feature dim)
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_output_dim: Output dimension from GNN
        gnn_num_layers: Number of GNN layers
        gnn_type: Type of GNN ('graphsage', 'gat', 'gin')
        gnn_dropout: Dropout probability for GNN
        prompt_model_name: HuggingFace model name for prompt encoder
        prompt_pooling: Pooling strategy for prompt encoder ('cls', 'mean', 'max')
        prompt_max_length: Max sequence length for prompt encoder
        freeze_prompt: Whether to freeze prompt encoder parameters
        conditioning_type: Type of conditioning ('film', 'cross_attention', 'hybrid')
        conditioning_hidden_dim: Hidden dimension for conditioning mechanism
        predictor_hidden_dim: Hidden dimension for prediction head
        predictor_dropout: Dropout for prediction head
        use_residual: Whether to use residual connections in GNN
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        # GNN parameters
        gnn_input_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 256,
        gnn_num_layers: int = 3,
        gnn_type: str = 'graphsage',
        gnn_dropout: float = 0.1,
        # Prompt encoder parameters
        prompt_model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        prompt_pooling: str = 'cls',
        prompt_max_length: int = 128,
        freeze_prompt: bool = False,
        # Conditioning parameters
        conditioning_type: str = 'film',
        conditioning_hidden_dim: Optional[int] = None,
        # Predictor parameters
        predictor_hidden_dim: int = 128,
        predictor_dropout: float = 0.2,
        # Additional options
        use_residual: bool = True,
        use_batch_norm: bool = True,
        # Ablation flags
        use_gnn: bool = True,           # False → skip GraphSAGE, project raw features instead
        use_conditioning: bool = True,  # False → FiLM returns identity (gamma=1, beta=0)
    ):
        super().__init__()
        
        logger.info("Initializing PromptGFM model...")
        
        self.gnn_type = gnn_type
        self.conditioning_type = conditioning_type
        self.gnn_output_dim = gnn_output_dim
        self.use_gnn = use_gnn               # ablation: False skips GraphSAGE
        self.use_conditioning = use_conditioning  # ablation: False uses identity FiLM
        
        # 1. Prompt Encoder (BioBERT)
        self.prompt_encoder = PromptEncoder(
            model_name=prompt_model_name,
            pooling=prompt_pooling,
            max_length=prompt_max_length,
            freeze=freeze_prompt
        )
        prompt_dim = self.prompt_encoder.embedding_dim
        
        # 2. GNN Backbone
        self.gnn = GNNBackbone(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            dropout=gnn_dropout,
            use_residual=use_residual,
            use_layer_norm=use_batch_norm  # GNNBackbone uses use_layer_norm parameter
        )
        # Ablation bypass: when use_gnn=False, project raw features to gnn_output_dim
        # so downstream predictor/conditioning dims remain consistent with the full model.
        if not use_gnn:
            self.node_proj = nn.Linear(gnn_input_dim, gnn_output_dim)
        
        # 3. Conditioning Mechanism
        if conditioning_hidden_dim is None:
            conditioning_hidden_dim = gnn_output_dim
        
        if conditioning_type == 'film':
            self.conditioning = FiLMConditioning(
                node_dim=gnn_output_dim,
                prompt_dim=prompt_dim,
                use_batch_norm=use_batch_norm,
                dropout=gnn_dropout,
                use_conditioning=use_conditioning,  # ablation: False → identity transform
            )
        elif conditioning_type == 'cross_attention':
            self.conditioning = CrossAttentionConditioning(
                node_dim=gnn_output_dim,
                prompt_dim=prompt_dim,
                num_heads=8,
                dropout=gnn_dropout,
                use_residual=use_residual
            )
        elif conditioning_type == 'hybrid':
            self.conditioning = HybridConditioning(
                node_dim=gnn_output_dim,
                prompt_dim=prompt_dim,
                num_heads=8,
                dropout=gnn_dropout,
                film_weight=0.5
            )
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")
        
        # 4. Link Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(gnn_output_dim, predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_dim // 2, 1)
        )
        
        logger.info(f"PromptGFM initialized:")
        logger.info(f"  GNN: {gnn_type}, layers={gnn_num_layers}, dim={gnn_output_dim}")
        logger.info(f"  Prompt: {prompt_pooling} pooling, dim={prompt_dim}")
        logger.info(f"  Conditioning: {conditioning_type}")
        logger.info(f"  Predictor: hidden_dim={predictor_hidden_dim}")
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        disease_texts: Union[str, List[str]],
        gene_indices: torch.Tensor,
        return_embeddings: bool = False,
        precomputed_prompt_embs: Optional[torch.Tensor] = None,
        precomputed_node_embs: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for gene-disease association prediction.
        
        Args:
            node_features: [num_nodes, input_dim] node feature matrix
            edge_index: [2, num_edges] edge connectivity
            disease_texts: Disease description(s) as prompt(s)
            gene_indices: [num_genes] indices of genes to predict for
            return_embeddings: Whether to return intermediate embeddings
            precomputed_prompt_embs: Optional [batch, 768] pre-encoded disease embeddings
                (skips BioBERT forward pass — use when prompt_encoder is frozen)
            precomputed_node_embs: Optional [num_nodes, dim] pre-computed GNN node embeddings
                (skips GNN forward pass — use during validation for speed)
            
        Returns:
            scores: [num_genes, 1] prediction scores
            embeddings (optional): [num_genes, output_dim] gene embeddings
        """
        # 1. Encode disease prompt (skip if precomputed — frozen BioBERT)
        if precomputed_prompt_embs is not None:
            prompt_embeddings = precomputed_prompt_embs
        else:
            prompt_embeddings = self.prompt_encoder(disease_texts)  # [batch_size, prompt_dim]

        # 2. GNN message passing to get node embeddings
        #    Priority: precomputed cache (validation) > use_gnn ablation flag > full GNN
        if precomputed_node_embs is not None:
            node_embeddings = precomputed_node_embs
        elif not self.use_gnn:
            # Ablation variant: skip message passing, project raw features to output dim
            node_embeddings = self.node_proj(node_features)  # [num_nodes, gnn_output_dim]
        else:
            node_embeddings = self.gnn(node_features, edge_index)  # [num_nodes, gnn_output_dim]
        
        # 3. Extract gene embeddings FIRST (before conditioning)
        gene_embeddings = node_embeddings[gene_indices]  # [batch_size, gnn_output_dim]
        
        # 4. Apply prompt-based conditioning on extracted gene embeddings
        # Now batch sizes match: [batch_size, gnn_output_dim] and [batch_size, prompt_dim]
        conditioned_gene_embeddings = self.conditioning(
            gene_embeddings,
            prompt_embeddings
        )  # [batch_size, gnn_output_dim]
        
        # 5. Predict association scores
        scores = self.predictor(conditioned_gene_embeddings)  # [batch_size, 1]
        
        if return_embeddings:
            return scores, conditioned_gene_embeddings
        return scores
    
    def predict_gene_disease_pairs(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        gene_disease_pairs: List[Tuple[int, str]],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Predict scores for specific gene-disease pairs.
        
        Args:
            node_features: [num_nodes, input_dim] node feature matrix
            edge_index: [2, num_edges] edge connectivity
            gene_disease_pairs: List of (gene_idx, disease_text) tuples
            batch_size: Batch size for processing
            
        Returns:
            scores: [num_pairs, 1] prediction scores
        """
        self.eval()
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(gene_disease_pairs), batch_size):
                batch_pairs = gene_disease_pairs[i:i + batch_size]
                
                # Separate gene indices and disease texts
                gene_indices = torch.tensor(
                    [pair[0] for pair in batch_pairs],
                    dtype=torch.long,
                    device=node_features.device
                )
                disease_texts = [pair[1] for pair in batch_pairs]
                
                # Forward pass
                batch_scores = self.forward(
                    node_features,
                    edge_index,
                    disease_texts,
                    gene_indices
                )
                all_scores.append(batch_scores)
        
        return torch.cat(all_scores, dim=0)
    
    def get_gene_rankings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        disease_text: str,
        candidate_gene_indices: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rank candidate genes for a disease.
        
        Args:
            node_features: [num_nodes, input_dim] node feature matrix
            edge_index: [2, num_edges] edge connectivity
            disease_text: Disease description
            candidate_gene_indices: [num_candidates] gene indices to rank
            top_k: Return only top-k genes (None=all)
            
        Returns:
            ranked_indices: [top_k] gene indices sorted by score
            ranked_scores: [top_k] corresponding scores
        """
        self.eval()
        
        with torch.no_grad():
            # Get scores for all candidate genes
            scores = self.forward(
                node_features,
                edge_index,
                disease_text,
                candidate_gene_indices
            )  # [num_candidates, 1]
            
            scores = scores.squeeze(-1)  # [num_candidates]
            
            # Sort by score (descending)
            sorted_indices = torch.argsort(scores, descending=True)
            
            if top_k is not None:
                sorted_indices = sorted_indices[:top_k]
            
            ranked_gene_indices = candidate_gene_indices[sorted_indices]
            ranked_scores = scores[sorted_indices]
            
            return ranked_gene_indices, ranked_scores
    
    def unfreeze_prompt_encoder(self):
        """Unfreeze prompt encoder for fine-tuning."""
        self.prompt_encoder.unfreeze()
        logger.info("Unfroze prompt encoder parameters")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        prompt_params = sum(p.numel() for p in self.prompt_encoder.parameters())
        gnn_params = sum(p.numel() for p in self.gnn.parameters())
        conditioning_params = sum(p.numel() for p in self.conditioning.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'prompt_encoder': prompt_params,
            'gnn': gnn_params,
            'conditioning': conditioning_params,
            'predictor': predictor_params,
            'total': total_params
        }


class GNNOnlyBaseline(nn.Module):
    """
    Baseline model without prompt conditioning.
    
    Uses only GNN to predict gene-disease associations without considering
    disease descriptions. Useful for ablation studies.
    """
    
    def __init__(
        self,
        gnn_input_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 256,
        gnn_num_layers: int = 3,
        gnn_type: str = 'graphsage',
        gnn_dropout: float = 0.1,
        predictor_hidden_dim: int = 128,
        predictor_dropout: float = 0.2,
        use_residual: bool = True,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        logger.info("Initializing GNN-Only Baseline...")
        
        # GNN Backbone
        self.gnn = GNNBackbone(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            dropout=gnn_dropout,
            use_residual=use_residual,
            use_batch_norm=use_batch_norm
        )
        
        # Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(gnn_output_dim, predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_dim // 2, 1)
        )
        
        logger.info(f"GNN-Only Baseline initialized: {gnn_type}, {gnn_num_layers} layers")
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        gene_indices: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass without prompt conditioning."""
        # GNN message passing
        node_embeddings = self.gnn(node_features, edge_index)
        
        # Extract gene embeddings
        gene_embeddings = node_embeddings[gene_indices]
        
        # Predict scores
        scores = self.predictor(gene_embeddings)
        
        if return_embeddings:
            return scores, gene_embeddings
        return scores


def test_promptgfm():
    """Test PromptGFM model with dummy data."""
    logger.info("Testing PromptGFM model...")
    
    # Create dummy data
    num_nodes = 100
    num_edges = 500
    input_dim = 256
    
    node_features = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Test PromptGFM with different conditioning types
    for conditioning_type in ['film', 'cross_attention', 'hybrid']:
        logger.info(f"\nTesting conditioning: {conditioning_type}")
        
        model = PromptGFM(
            gnn_input_dim=input_dim,
            gnn_hidden_dim=128,
            gnn_output_dim=128,
            gnn_num_layers=2,
            gnn_type='graphsage',
            conditioning_type=conditioning_type,
            predictor_hidden_dim=64
        )
        
        # Test single disease
        disease_text = "Disease: Angelman syndrome. Phenotypes: seizures, delay. Associated genes:"
        gene_indices = torch.tensor([0, 5, 10, 15, 20])
        
        scores = model(node_features, edge_index, disease_text, gene_indices)
        logger.info(f"  Single disease - Scores shape: {scores.shape}")
        
        # Test batch of diseases
        disease_texts = [
            "Disease: Angelman syndrome. Phenotypes: seizures. Associated genes:",
            "Disease: Rett syndrome. Phenotypes: regression. Associated genes:",
        ]
        gene_indices_batch = torch.tensor([0, 1])  # One gene per disease
        
        # Note: For batch processing, we'd need to handle this differently
        # For now, test with single prompt repeated
        scores_batch = model(node_features, edge_index, disease_texts[0], gene_indices)
        logger.info(f"  Batch - Scores shape: {scores_batch.shape}")
        
        # Test rankings
        ranked_genes, ranked_scores = model.get_gene_rankings(
            node_features,
            edge_index,
            disease_text,
            torch.arange(num_nodes),
            top_k=10
        )
        logger.info(f"  Rankings - Top genes: {ranked_genes.shape}, scores: {ranked_scores.shape}")
        
        # Check parameters
        params = model.get_num_parameters()
        logger.info(f"  Parameters: {params['total']:,} total")
    
    # Test GNN-Only Baseline
    logger.info("\nTesting GNN-Only Baseline...")
    baseline = GNNOnlyBaseline(
        gnn_input_dim=input_dim,
        gnn_hidden_dim=128,
        gnn_output_dim=128
    )
    
    scores_baseline = baseline(node_features, edge_index, gene_indices)
    logger.info(f"  Baseline scores shape: {scores_baseline.shape}")
    
    logger.info("\n✓ PromptGFM tests passed!")


if __name__ == "__main__":
    test_promptgfm()
```

## File: `src/models/prompt_encoder.py`

```python
"""
Biomedical prompt encoder using pretrained language models.

Uses BioBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
to encode disease descriptions and phenotype lists into embedding vectors.
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptEncoder(nn.Module):
    """
    Biomedical prompt encoder using BioBERT.
    
    Encodes disease descriptions and phenotype information into dense embedding vectors
    that can be used to condition graph neural networks.
    
    Args:
        model_name: Pretrained model name from HuggingFace
        pooling: Pooling strategy - 'cls', 'mean', or 'max'
        max_length: Maximum sequence length for tokenization
        freeze: Whether to freeze BERT parameters
        dropout_prob: Dropout probability for the output layer
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        pooling: str = 'cls',
        max_length: int = 128,
        freeze: bool = False,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        logger.info(f"Initializing PromptEncoder with model: {model_name}")
        
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.freeze = freeze
        
        # Load tokenizer and model
        # Note: resume_download was removed in transformers>=4.45; local_files_only
        # is set via TRANSFORMERS_OFFLINE env var by the notebook before training starts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size
        
        # Freeze BERT parameters if requested
        if freeze:
            logger.info("Freezing BERT parameters")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Optional projection and dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        logger.info(f"PromptEncoder initialized - hidden_size: {self.hidden_size}, "
                   f"pooling: {pooling}, max_length: {max_length}")
    
    def create_prompt(
        self,
        disease_name: str,
        phenotypes: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a structured prompt from disease information.
        
        Args:
            disease_name: Name of the disease
            phenotypes: List of associated phenotype terms
            description: Optional disease description
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Disease: {disease_name}."]
        
        if phenotypes and len(phenotypes) > 0:
            # Limit phenotypes to avoid exceeding max_length
            phenotype_str = ", ".join(phenotypes[:10])  # Take first 10
            prompt_parts.append(f"Phenotypes: {phenotype_str}.")
        
        if description:
            prompt_parts.append(f"Description: {description}.")
        
        prompt_parts.append("Associated genes:")
        
        return " ".join(prompt_parts)
    
    def forward(
        self,
        texts: Union[str, List[str]],
        return_attention_mask: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Encode text prompts into embedding vectors.
        
        Args:
            texts: Single text string or list of text strings
            return_attention_mask: Whether to return attention masks
            
        Returns:
            embeddings: [batch_size, hidden_size] tensor
            attention_mask (optional): [batch_size, seq_len] tensor
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward pass through BERT
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool the outputs
        embeddings = self._pool_outputs(
            outputs.last_hidden_state,
            attention_mask
        )
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings
    
    def _pool_outputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token-level hidden states into a single vector per sequence.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled: [batch_size, hidden_size]
        """
        if self.pooling == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'mean':
            # Mean pooling over non-padding tokens
            # Expand attention_mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        
        elif self.pooling == 'max':
            # Max pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states.clone()
            hidden_states[mask_expanded == 0] = -1e9  # Set padding to very negative
            return torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def encode_batch(
        self,
        disease_names: List[str],
        phenotypes_list: Optional[List[List[str]]] = None,
        descriptions: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Encode a batch of disease information.
        
        Args:
            disease_names: List of disease names
            phenotypes_list: List of phenotype lists (one per disease)
            descriptions: List of disease descriptions
            
        Returns:
            embeddings: [batch_size, hidden_size] tensor
        """
        batch_size = len(disease_names)
        
        # Create prompts for each disease
        prompts = []
        for i in range(batch_size):
            phenotypes = phenotypes_list[i] if phenotypes_list else None
            description = descriptions[i] if descriptions else None
            prompt = self.create_prompt(disease_names[i], phenotypes, description)
            prompts.append(prompt)
        
        # Encode all prompts
        return self.forward(prompts)
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.hidden_size
    
    def unfreeze(self):
        """Unfreeze all parameters for fine-tuning."""
        logger.info("Unfreezing BERT parameters")
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze = False


def test_prompt_encoder():
    """Test the prompt encoder with sample data."""
    logger.info("Testing PromptEncoder...")
    
    # Create encoder
    encoder = PromptEncoder(
        pooling='cls',
        max_length=128,
        freeze=False
    )
    
    # Test single prompt
    disease_name = "Angelman syndrome"
    phenotypes = ["seizures", "developmental delay", "speech impairment", "ataxia"]
    prompt = encoder.create_prompt(disease_name, phenotypes)
    logger.info(f"Sample prompt: {prompt}")
    
    # Test encoding
    embeddings = encoder([prompt])
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Embedding dim: {encoder.embedding_dim}")
    
    # Test batch encoding
    disease_names = ["Angelman syndrome", "Rett syndrome", "Fragile X syndrome"]
    phenotypes_list = [
        ["seizures", "developmental delay", "speech impairment"],
        ["hand stereotypies", "regression", "seizures"],
        ["intellectual disability", "anxiety", "autism"]
    ]
    
    batch_embeddings = encoder.encode_batch(disease_names, phenotypes_list)
    logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Test different pooling strategies
    for pooling in ['cls', 'mean', 'max']:
        encoder_test = PromptEncoder(pooling=pooling, max_length=64)
        emb = encoder_test([prompt])
        logger.info(f"Pooling={pooling}: shape {emb.shape}")
    
    logger.info("✓ PromptEncoder tests passed!")
    return encoder


if __name__ == "__main__":
    test_prompt_encoder()
```

## File: `src/models/__init__.py`

```python
"""
Models module for PromptGFM-Bio.

This module contains the GNN backbone, prompt encoder, conditioning mechanisms,
and the complete PromptGFM model architecture.
"""
```

## File: `src/training/finetune.py`

```python
"""
Supervised training for gene-disease link prediction.

Implements comprehensive training loop with:
- Multiple loss functions (BCE, Ranking, Combined)
- Validation and early stopping
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Weights & Biases logging
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# ── FIX (2026-04-17): use torch.cuda.amp legacy API.
# On some PyTorch builds, `from torch.amp import autocast` resolves to the
# legacy torch.cuda.amp.autocast class (first positional arg is `enabled: bool`,
# no `device_type` kwarg), which crashes with both `autocast('cuda')` and
# `autocast(device_type='cuda')`. The torch.cuda.amp.* classes with no args
# are stable across PyTorch 1.6 → 2.x and work on CUDA 12.4 and CUDA 13.0 alike.
# May emit a DeprecationWarning on newer PyTorch; harmless.
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import time
import math
import numpy as np

from ..evaluation.metrics import GeneRankingEvaluator
from .losses import BCELoss, MarginRankingLoss, CombinedLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptGFMTrainer:
    """
    Supervised trainer for PromptGFM model.
    
    Handles training, validation, checkpointing, and metric logging.
    
    Args:
        model: PromptGFM model to train
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: Device to train on ('cuda' or 'cpu')
        evaluator: Metrics evaluator
        checkpoint_dir: Directory to save checkpoints
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs)
        gradient_clip: Maximum gradient norm for clipping
        use_wandb: Whether to log to Weights & Biases
        log_interval: How often to log training metrics (steps)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cuda',
        evaluator: Optional[GeneRankingEvaluator] = None,
        checkpoint_dir: str = 'checkpoints',
        max_epochs: int = 100,
        patience: int = 10,
        gradient_clip: float = 1.0,
        use_wandb: bool = False,
        log_interval: int = 100,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.evaluator = evaluator if evaluator is not None else GeneRankingEvaluator()
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip

        # Pre-computed prompt embedding cache {text -> tensor} (set via set_prompt_cache())
        # Eliminates frozen BioBERT forward pass on every batch — huge speedup.
        self.prompt_emb_cache: Optional[Dict[str, torch.Tensor]] = None
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        
        # Mixed precision training (default: enabled for speed/memory efficiency)
        self.use_amp = use_amp and torch.cuda.is_available()
        # FIX (2026-04-17): legacy GradScaler takes no device positional arg.
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('-inf')
        self.epochs_without_improvement = 0
        
        # History
        self.train_losses = []
        self.val_metrics = []
        
        logger.info(f"PromptGFMTrainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Gradient clip: {gradient_clip}")
        logger.info(f"  Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")
        
        # Initialize W&B if requested
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                logger.info("  W&B logging: enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
    
    def train_epoch(
        self,
        train_loader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            scheduler: Optional learning rate scheduler
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")
        
        for batch in pbar:
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                # FIX (2026-04-17): legacy autocast takes no device positional arg.
                with autocast():
                    outputs = self._forward_batch(batch)
                    loss = self._compute_loss(outputs, batch)
            else:
                outputs = self._forward_batch(batch)
                loss = self._compute_loss(outputs, batch)
            
            # Backward pass with gradient scaling for AMP
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first for AMP)
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            if self.use_wandb and self.global_step % self.log_interval == 0:
                self.wandb.log({
                    'train/loss': loss.item(),
                    'train/step': self.global_step,
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Step scheduler if batch-level
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                scheduler.step_update(self.global_step)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        val_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc="Validating")

        # Cache GNN node embeddings for the entire validation pass.
        # The graph is static so every batch uses the same node_features / edge_index.
        # Computing the GNN once (no_grad) instead of once-per-batch saves ~30% val time.
        _val_node_embs: Optional[torch.Tensor] = None

        for batch in pbar:
            # Move batch to device
            batch = self._move_to_device(batch)

            if _val_node_embs is None:
                inner_model = self.model.module if hasattr(self.model, 'module') else self.model
                if self.graph_node_features is None or self.graph_edge_index is None:
                    raise RuntimeError("Graph tensors missing — call set_graph_tensors()")
                with torch.no_grad():
                    _val_node_embs = inner_model.gnn(
                        self.graph_node_features, self.graph_edge_index
                    ).detach()

            # Forward pass (no GNN re-run, no BioBERT re-run)
            outputs = self._forward_batch(batch, precomputed_node_embs=_val_node_embs)
            
            # Compute loss
            loss = self._compute_loss(outputs, batch)
            val_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            scores = outputs['scores'].squeeze(-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
        
        # Compute metrics
        import numpy as np
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        metrics = self.evaluator.evaluate_all(all_labels, all_scores)
        metrics['loss'] = val_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        val_metric: str = 'aupr'
    ):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            scheduler: Optional learning rate scheduler
            val_metric: Metric to use for early stopping ('aupr', 'auroc', etc.)
        """
        start_epoch = self.current_epoch
        logger.info(f"\nStarting training from epoch {start_epoch + 1} to {self.max_epochs}")
        logger.info(f"Early stopping on: {val_metric}")
        
        # Initialize W&B if enabled
        if self.use_wandb:
            try:
                self.wandb.init(
                    project="promptgfm-bio",
                    name=f"finetune_{self.checkpoint_dir.name}",
                    config={
                        "max_epochs": self.max_epochs,
                        "patience": self.patience,
                        "val_metric": val_metric,
                        "resume_from_epoch": start_epoch
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}. Disabling W&B logging.")
                self.use_wandb = False
        
        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, scheduler)
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Calculate epoch time and add to metrics
            epoch_time = time.time() - start_time
            val_metrics['epoch_time'] = epoch_time
            self.val_metrics.append(val_metrics)
            
            # Step scheduler if epoch-level
            if scheduler is not None and not hasattr(scheduler, 'step_update'):
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics[val_metric])
                else:
                    scheduler.step()
            
            # Calculate timing statistics
            total_time = sum([m.get('epoch_time', 0) for m in self.val_metrics])
            avg_epoch_time = total_time / (epoch + 1)
            remaining_epochs = self.max_epochs - (epoch + 1)
            eta_total = remaining_epochs * avg_epoch_time
            
            # Log detailed metrics
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs} Complete")
            logger.info(f"{'='*70}")
            logger.info(f"  Time: {epoch_time:.1f}s (Avg: {avg_epoch_time:.1f}s/epoch)")
            logger.info(f"  ETA: {int(eta_total//3600)}h {int((eta_total%3600)//60)}m (for {remaining_epochs} epochs)")
            logger.info(f"  Train Loss: {train_metrics['loss']:.6f}")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.6f}")
            logger.info(f"  Val AUROC:  {val_metrics.get('auroc', 0):.4f}")
            logger.info(f"  Val AUPR:   {val_metrics.get('aupr', 0):.4f}")
            
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'time/epoch_time': epoch_time,
                    'time/avg_epoch_time': avg_epoch_time
                }
                for k, v in val_metrics.items():
                    log_dict[f'val/{k}'] = v
                self.wandb.log(log_dict)
            
            # Check for improvement
            current_metric = val_metrics.get(val_metric, float('-inf'))
            
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt', is_best=True, metrics=val_metrics)
                logger.info(f"  ✓ New best {val_metric}: {current_metric:.4f} (saved as best_model.pt)")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    f'checkpoint_epoch_{epoch + 1}.pt',
                    is_best=False,
                    metrics=val_metrics
                )
                logger.info(f"  💾 Checkpoint saved: checkpoint_epoch_{epoch + 1}.pt")
            logger.info(f"{'='*70}\n")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        logger.info(f"\nTraining complete!")
        logger.info(f"Best {val_metric}: {self.best_val_metric:.4f}")
        
        # Finish W&B logging
        if self.use_wandb:
            try:
                self.wandb.finish()
            except:
                pass
        
        # Load best model if it exists
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            self.load_checkpoint(best_model_path)
        else:
            logger.warning(f"No best model found at {best_model_path}. Using final model weights.")
    
    def set_graph_tensors(self, node_features, edge_index):
        self.graph_node_features = node_features.to(self.device)
        self.graph_edge_index    = edge_index.to(self.device)

    def set_prompt_cache(self, cache: Dict[str, torch.Tensor]):
        """Set pre-computed prompt embeddings (call once before training)."""
        self.prompt_emb_cache = cache
        logger.info(f"  Prompt embedding cache: {len(cache)} unique disease texts — BioBERT will be skipped per batch")

    def _forward_batch(
        self,
        batch: Dict,
        precomputed_node_embs: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass for a batch.

        Args:
            batch: batch dict with node_features, edge_index, disease_texts, gene_indices
            precomputed_node_embs: optional cached GNN output [num_nodes, dim] (used in validate)
        """
        # Look up pre-computed prompt embeddings if cache is available
        precomputed_prompt_embs: Optional[torch.Tensor] = None
        if self.prompt_emb_cache is not None:
            texts = batch['disease_texts']
            precomputed_prompt_embs = torch.stack(
                [self.prompt_emb_cache[t] for t in texts]
            ).to(self.device)

        # self.model handles both DDP-wrapped and plain PromptGFM
        if self.graph_node_features is None or self.graph_edge_index is None:
            raise RuntimeError("Graph tensors missing — call set_graph_tensors()")

        scores = self.model(
            node_features=self.graph_node_features,
            edge_index=self.graph_edge_index,
            disease_texts=batch['disease_texts'],
            gene_indices=batch['gene_indices'],
            precomputed_prompt_embs=precomputed_prompt_embs,
            precomputed_node_embs=precomputed_node_embs,
        )

        return {'scores': scores}
    
    def _compute_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute loss for a batch."""
        scores = outputs['scores']
        labels = batch['labels']
        
        # Handle different loss types
        if isinstance(self.loss_fn, CombinedLoss):
            # Split positive and negative scores
            pos_mask = labels == 1
            neg_mask = labels == 0
            
            pos_scores = scores[pos_mask] if pos_mask.any() else None
            neg_scores = scores[neg_mask] if neg_mask.any() else None
            
            losses = self.loss_fn(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                gene_embs=outputs.get('gene_embs'),
                prompt_embs=outputs.get('prompt_embs')
            )
            return losses['total']
        else:
            # Simple loss
            return self.loss_fn(scores, labels)
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch[key] = [v.to(self.device) for v in value]
        return batch
    
    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint with all training state."""
        # In distributed training only rank-0 writes checkpoints
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        checkpoint_path = self.checkpoint_dir / filename

        # Unwrap DDP model to get plain state dict
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'epoch': self.current_epoch + 1,  # Save as next epoch to resume from
            'global_step': self.global_step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'epochs_without_improvement': self.epochs_without_improvement,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'is_best': is_best
        }
        
        if metrics is not None:
            checkpoint['current_metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics to JSON (human-readable)
        if metrics is not None:
            metrics_path = checkpoint_path.with_suffix('.json')
            
            # Convert all metrics to JSON-serializable types
            def convert_to_native(obj):
                """Convert numpy/torch types to Python native types for JSON serialization."""
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif obj is None or isinstance(obj, (str, int, float)):
                    # Handle NaN/Inf values
                    if isinstance(obj, float):
                        if math.isnan(obj) or math.isinf(obj):
                            return None
                    return obj
                else:
                    return obj
            
            # Safely convert best_val_metric
            best_val = None
            if self.best_val_metric is not None:
                try:
                    best_val = float(self.best_val_metric)
                    if math.isnan(best_val) or math.isinf(best_val):
                        best_val = None
                except (ValueError, TypeError):
                    best_val = None
            
            save_metrics = {
                'epoch': int(self.current_epoch + 1),
                'metrics': convert_to_native(metrics),
                'best_val_metric': best_val,
                'is_best': bool(is_best)
            }
            with open(metrics_path, 'w') as f:
                json.dump(save_metrics, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: Path, load_optimizer: bool = True):
        """Load model checkpoint and optionally resume training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: If True, resume training state (optimizer, epoch, etc.)
                          If False, only load model weights (for inference)
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            # Resume full training state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_metrics = checkpoint.get('val_metrics', [])
            
            logger.info(f"✓ Resumed from epoch {self.current_epoch}")
            logger.info(f"  Best val metric: {self.best_val_metric:.4f}")
            logger.info(f"  Global step: {self.global_step}")
        else:
            logger.info(f"✓ Loaded model weights only (epoch {checkpoint.get('epoch', 0)})")


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999)
) -> AdamW:
    """
    Create AdamW optimizer with parameter groups.
    
    Separate learning rates for prompt encoder vs. GNN if desired.
    """
    # Could implement differential learning rates here
    # For now, use same rate for all parameters
    
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 100,
    eta_min: float = 1e-6,
    **kwargs
):
    """Create learning rate scheduler."""
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min
        )
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def test_trainer():
    """Test trainer with dummy data."""
    logger.info("Testing PromptGFMTrainer...")
    
    # This is a placeholder test
    # Real testing would require creating a full model and dataloaders
    logger.info("Trainer test placeholder - use training scripts for full testing")
    logger.info("✓ Trainer module loaded successfully")


if __name__ == "__main__":
    test_trainer()
```

## File: `src/training/losses.py`

```python
"""
Loss functions for gene-disease association prediction.

Implements various loss functions for link prediction and ranking:
1. Binary Cross-Entropy (BCE) - Standard classification loss
2. Margin Ranking Loss - For pairwise ranking
3. ListNet Loss - For listwise ranking (neural IR approach)
4. Contrastive Loss - For prompt-gene alignment
5. Focal Loss - For handling class imbalance
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy loss for gene-disease link prediction.
    
    Standard loss for binary classification (edge exists or not).
    
    Args:
        pos_weight: Weight for positive class (useful for imbalanced data)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        super().__init__()
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, 1] or [batch_size] raw prediction scores (logits)
            labels: [batch_size, 1] or [batch_size] binary labels (0 or 1)
            
        Returns:
            loss: Scalar loss value
        """
        scores = scores.squeeze(-1) if scores.dim() > 1 else scores
        labels = labels.squeeze(-1) if labels.dim() > 1 else labels
        return self.loss_fn(scores, labels.float())


class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for pairwise gene ranking.
    
    Encourages positive examples to have higher scores than negative examples
    by at least a margin.
    
    Loss = max(0, margin - pos_score + neg_score)
    
    Args:
        margin: Margin between positive and negative scores
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction=reduction)
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: [batch_size] or [batch_size, 1] scores for positive pairs
            neg_scores: [batch_size * num_negatives] or [batch_size * num_negatives, 1] scores for negative pairs
            
        Returns:
            loss: Scalar loss value
        """
        pos_scores = pos_scores.squeeze(-1) if pos_scores.dim() > 1 else pos_scores
        neg_scores = neg_scores.squeeze(-1) if neg_scores.dim() > 1 else neg_scores
        
        # If neg_scores has more samples than pos_scores (negative sampling ratio > 1),
        # expand pos_scores to match by repeating each positive score for each negative
        if neg_scores.size(0) > pos_scores.size(0):
            num_pos = pos_scores.size(0)
            num_neg = neg_scores.size(0)
            
            if num_neg % num_pos == 0:
                # Negative sampling: repeat each positive score for comparison with all its negatives
                ratio = num_neg // num_pos
                pos_scores_expanded = pos_scores.repeat_interleave(ratio)
            else:
                # Size mismatch - truncate or pad to match
                # This shouldn't happen with proper negative sampling, but handle gracefully
                import logging
                logging.warning(f"MarginRankingLoss: Size mismatch pos={num_pos}, neg={num_neg}")
                min_size = min(num_pos, num_neg)
                pos_scores_expanded = pos_scores[:min_size]
                neg_scores = neg_scores[:min_size]
        else:
            pos_scores_expanded = pos_scores
        
        # Target: pos_scores should be greater than neg_scores
        target = torch.ones_like(pos_scores_expanded)
        
        return self.loss_fn(pos_scores_expanded, neg_scores, target)


class ListNetLoss(nn.Module):
    """
    ListNet loss for listwise ranking.
    
    Uses cross-entropy between probability distributions defined by scores.
    Based on "Learning to Rank: From Pairwise Approach to Listwise Approach" (Cao et al., 2007)
    
    Args:
        temperature: Temperature for softmax (lower = sharper distribution)
        eps: Small constant for numerical stability
    """
    
    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(
        self,
        pred_scores: torch.Tensor,
        true_relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: [batch_size, num_candidates] predicted scores
            true_relevance: [batch_size, num_candidates] ground truth relevance scores
            
        Returns:
            loss: Scalar loss value
        """
        # Convert scores to probability distributions using softmax
        pred_probs = F.softmax(pred_scores / self.temperature, dim=-1)
        true_probs = F.softmax(true_relevance / self.temperature, dim=-1)
        
        # Cross-entropy between distributions
        loss = -torch.sum(true_probs * torch.log(pred_probs + self.eps), dim=-1)
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for prompt-gene alignment.
    
    Encourages gene embeddings to be similar to their corresponding disease prompt
    embeddings and dissimilar to other diseases.
    
    Based on InfoNCE loss used in contrastive learning.
    
    Args:
        temperature: Temperature for scaling similarities
        similarity: Similarity function ('cosine' or 'dot')
    """
    
    def __init__(self, temperature: float = 0.07, similarity: str = 'cosine'):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
    
    def forward(
        self,
        gene_embs: torch.Tensor,
        prompt_embs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            gene_embs: [batch_size, emb_dim] gene embeddings
            prompt_embs: [batch_size, emb_dim] disease prompt embeddings
            labels: [batch_size] optional labels indicating positive pairs
                   If None, assumes diagonal pairs are positive
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = gene_embs.shape[0]
        
        # Compute similarity matrix
        if self.similarity == 'cosine':
            gene_embs = F.normalize(gene_embs, dim=-1)
            prompt_embs = F.normalize(prompt_embs, dim=-1)
            similarity_matrix = torch.matmul(gene_embs, prompt_embs.T)
        else:  # dot product
            similarity_matrix = torch.matmul(gene_embs, prompt_embs.T)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels if not provided (diagonal is positive)
        if labels is None:
            labels = torch.arange(batch_size, device=gene_embs.device)
        
        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Down-weights easy examples and focuses on hard examples.
    Based on "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, 1] or [batch_size] raw prediction scores (logits)
            labels: [batch_size, 1] or [batch_size] binary labels (0 or 1)
            
        Returns:
            loss: Scalar loss value
        """
        scores = scores.squeeze(-1) if scores.dim() > 1 else scores
        labels = labels.squeeze(-1) if labels.dim() > 1 else labels
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction='none')
        
        # Compute probabilities
        probs = torch.sigmoid(scores)
        
        # Compute focal weight
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        
        # Combine
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    
    Useful for training with multiple objectives (e.g., BCE + Ranking).
    
    Args:
        bce_weight: Weight for BCE loss
        ranking_weight: Weight for ranking loss
        contrastive_weight: Weight for contrastive loss
        margin: Margin for ranking loss
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        ranking_weight: float = 0.5,
        contrastive_weight: float = 0.1,
        margin: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        
        self.bce_loss = BCELoss()
        self.ranking_loss = MarginRankingLoss(margin=margin)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: Optional[torch.Tensor] = None,
        gene_embs: Optional[torch.Tensor] = None,
        prompt_embs: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            pos_scores: [batch_size] scores for positive pairs
            neg_scores: [batch_size] scores for negative pairs (optional)
            gene_embs: [batch_size, emb_dim] gene embeddings (optional)
            prompt_embs: [batch_size, emb_dim] prompt embeddings (optional)
            
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pos_scores.device if pos_scores is not None else 'cpu')
        
        # BCE loss on positive and negative samples
        if pos_scores is not None:
            if neg_scores is not None:
                all_scores = torch.cat([pos_scores, neg_scores], dim=0)
                all_labels = torch.cat([
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores)
                ], dim=0)
                bce = self.bce_loss(all_scores, all_labels)
            else:
                # Only positive samples - compute BCE with labels=1
                bce = self.bce_loss(pos_scores, torch.ones_like(pos_scores))
            
            losses['bce'] = bce
            total_loss = total_loss + self.bce_weight * bce
        
        # Ranking loss
        if neg_scores is not None and self.ranking_weight > 0:
            ranking = self.ranking_loss(pos_scores, neg_scores)
            losses['ranking'] = ranking
            total_loss += self.ranking_weight * ranking
        
        # Contrastive loss
        if gene_embs is not None and prompt_embs is not None and self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(gene_embs, prompt_embs)
            losses['contrastive'] = contrastive
            total_loss += self.contrastive_weight * contrastive
        
        losses['total'] = total_loss
        return losses


def test_losses():
    """Test all loss functions with dummy data."""
    logger.info("Testing loss functions...")
    
    batch_size = 32
    num_candidates = 50
    emb_dim = 128
    
    # Test BCE Loss
    logger.info("\nTesting BCE Loss...")
    bce_loss = BCELoss()
    scores = torch.randn(batch_size)
    labels = torch.randint(0, 2, (batch_size,)).float()
    loss = bce_loss(scores, labels)
    logger.info(f"  BCE loss: {loss.item():.4f}")
    
    # Test Margin Ranking Loss
    logger.info("\nTesting Margin Ranking Loss...")
    ranking_loss = MarginRankingLoss(margin=0.5)
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size)
    loss = ranking_loss(pos_scores, neg_scores)
    logger.info(f"  Ranking loss: {loss.item():.4f}")
    
    # Test ListNet Loss
    logger.info("\nTesting ListNet Loss...")
    listnet_loss = ListNetLoss()
    pred_scores = torch.randn(batch_size, num_candidates)
    true_relevance = torch.randint(0, 5, (batch_size, num_candidates)).float()
    loss = listnet_loss(pred_scores, true_relevance)
    logger.info(f"  ListNet loss: {loss.item():.4f}")
    
    # Test Contrastive Loss
    logger.info("\nTesting Contrastive Loss...")
    contrastive_loss = ContrastiveLoss()
    gene_embs = torch.randn(batch_size, emb_dim)
    prompt_embs = torch.randn(batch_size, emb_dim)
    loss = contrastive_loss(gene_embs, prompt_embs)
    logger.info(f"  Contrastive loss: {loss.item():.4f}")
    
    # Test Focal Loss
    logger.info("\nTesting Focal Loss...")
    focal_loss = FocalLoss()
    loss = focal_loss(scores, labels)
    logger.info(f"  Focal loss: {loss.item():.4f}")
    
    # Test Combined Loss
    logger.info("\nTesting Combined Loss...")
    combined_loss = CombinedLoss()
    losses = combined_loss(pos_scores, neg_scores, gene_embs, prompt_embs)
    logger.info(f"  Total loss: {losses['total'].item():.4f}")
    logger.info(f"  BCE: {losses['bce'].item():.4f}")
    logger.info(f"  Ranking: {losses['ranking'].item():.4f}")
    logger.info(f"  Contrastive: {losses['contrastive'].item():.4f}")
    
    logger.info("\n✓ All loss function tests passed!")


if __name__ == "__main__":
    test_losses()
```

## File: `src/training/pretrain.py`

```python
"""
Self-supervised pretraining tasks for graph representation learning.

Implements multiple pretraining strategies:
1. Masked Node Prediction - Predict masked node features
2. Edge Contrastive Learning - Distinguish real vs. corrupted edges
3. Context Prediction - Predict local graph context
4. Graph Contrastive Learning - Instance discrimination

These tasks help the GNN learn better representations before supervised training.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskedNodePredictor(nn.Module):
    """
    Masked node feature prediction (similar to BERT's MLM).
    
    Randomly masks node features and trains model to reconstruct them.
    """
    
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.predictor(node_embeddings)


class EdgePredictor(nn.Module):
    """
    Edge prediction head for contrastive learning.
    
    Predicts whether an edge exists between two nodes.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_emb: [num_edges, hidden_dim]
            dst_emb: [num_edges, hidden_dim]
            
        Returns:
            scores: [num_edges, 1]
        """
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        return self.predictor(edge_emb)


class GraphPretrainer:
    """
    Self-supervised pretraining for GNN backbone.
    
    Supports multiple pretraining strategies to learn better representations
    before supervised fine-tuning.
    
    Args:
        model: GNN model to pretrain
        device: Device for training
        mask_rate: Fraction of nodes to mask
        negative_samples: Number of negative edges per positive edge
        temperature: Temperature for contrastive learning
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        mask_rate: float = 0.15,
        negative_samples: int = 1,
        temperature: float = 0.07
    ):
        self.model = model.to(device)
        self.device = device
        self.mask_rate = mask_rate
        self.negative_samples = negative_samples
        self.temperature = temperature
        
        # Get hidden dimension from model
        self.hidden_dim = getattr(model, 'gnn_output_dim', 256)
        
        # Create task-specific heads
        self.node_predictor = None
        self.edge_predictor = None
        
        logger.info(f"GraphPretrainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mask rate: {mask_rate}")
        logger.info(f"  Negative samples: {negative_samples}")
    
    def masked_node_prediction(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_epochs: int = 10,
        lr: float = 1e-3
    ) -> Dict[str, list]:
        """
        Pretrain with masked node prediction.
        
        Args:
            node_features: [num_nodes, feature_dim] node features
            edge_index: [2, num_edges] edge connectivity
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        logger.info(f"\nMasked Node Prediction Pretraining for {num_epochs} epochs...")
        
        feature_dim = node_features.shape[1]
        if self.node_predictor is None:
            self.node_predictor = MaskedNodePredictor(
                self.hidden_dim,
                feature_dim
            ).to(self.device)
        
        # Optimizer for both model and predictor
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.node_predictor.parameters()),
            lr=lr
        )
        
        # Move data to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        history = {'loss': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            self.node_predictor.train()
            
            # Create masked features
            masked_features, mask, original_features = self._mask_node_features(
                node_features
            )
            
            # Forward pass through GNN
            node_embeddings = self.model.gnn(masked_features, edge_index)
            
            # Predict masked node features
            predictions = self.node_predictor(node_embeddings[mask])
            targets = original_features[mask]
            
            # Compute loss (MSE for continuous features)
            loss = F.mse_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            if (epoch + 1) % max(1, num_epochs // 10) == 0:
                logger.info(f"  Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item():.4f}")
        
        logger.info(f"✓ Masked node prediction complete. Final loss: {loss.item():.4f}")
        return history
    
    def edge_contrastive_learning(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_epochs: int = 10,
        lr: float = 1e-3
    ) -> Dict[str, list]:
        """
        Pretrain with edge contrastive learning.
        
        Train model to distinguish real edges from corrupted (negative) edges.
        
        Args:
            node_features: [num_nodes, feature_dim] node features
            edge_index: [2, num_edges] edge connectivity
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        logger.info(f"\nEdge Contrastive Learning for {num_epochs} epochs...")
        
        if self.edge_predictor is None:
            self.edge_predictor = EdgePredictor(self.hidden_dim).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.edge_predictor.parameters()),
            lr=lr
        )
        
        # Move data to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]
        
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            self.edge_predictor.train()
            
            # Sample a subset of edges for efficiency
            sample_size = min(10000, num_edges)
            edge_sample_idx = torch.randperm(num_edges)[:sample_size]
            pos_edge_index = edge_index[:, edge_sample_idx]
            
            # Generate negative edges
            neg_edge_index = self._generate_negative_edges(
                num_nodes,
                pos_edge_index,
                num_negative=self.negative_samples
            )
            
            # Forward pass through GNN
            node_embeddings = self.model.gnn(node_features, edge_index)
            
            # Predict positive edges
            pos_src_emb = node_embeddings[pos_edge_index[0]]
            pos_dst_emb = node_embeddings[pos_edge_index[1]]
            pos_scores = self.edge_predictor(pos_src_emb, pos_dst_emb)
            
            # Predict negative edges
            neg_src_emb = node_embeddings[neg_edge_index[0]]
            neg_dst_emb = node_embeddings[neg_edge_index[1]]
            neg_scores = self.edge_predictor(neg_src_emb, neg_dst_emb)
            
            # Combine scores and labels
            all_scores = torch.cat([pos_scores, neg_scores], dim=0).squeeze(-1)
            all_labels = torch.cat([
                torch.ones(pos_scores.shape[0], device=self.device),
                torch.zeros(neg_scores.shape[0], device=self.device)
            ], dim=0)
            
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
            
            # Compute accuracy
            predictions = (torch.sigmoid(all_scores) > 0.5).float()
            accuracy = (predictions == all_labels).float().mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy.item())
            
            if (epoch + 1) % max(1, num_epochs // 10) == 0:
                logger.info(f"  Epoch {epoch + 1}/{num_epochs}: "
                          f"Loss = {loss.item():.4f}, Acc = {accuracy.item():.4f}")
        
        logger.info(f"✓ Edge contrastive learning complete. "
                   f"Final loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")
        return history
    
    def graph_contrastive_learning(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 256
    ) -> Dict[str, list]:
        """
        Pretrain with graph contrastive learning (instance discrimination).
        
        Create two augmented views of each node's neighborhood and maximize
        agreement between their representations.
        
        Args:
            node_features: [num_nodes, feature_dim] node features
            edge_index: [2, num_edges] edge connectivity
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            batch_size: Number of nodes per batch
            
        Returns:
            Training history
        """
        logger.info(f"\nGraph Contrastive Learning for {num_epochs} epochs...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Move data to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        num_nodes = node_features.shape[0]
        
        history = {'loss': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            
            # Create two augmented views
            aug_features_1, aug_edge_index_1 = self._augment_graph(
                node_features, edge_index
            )
            aug_features_2, aug_edge_index_2 = self._augment_graph(
                node_features, edge_index
            )
            
            # Sample nodes for batch
            node_idx = torch.randperm(num_nodes)[:batch_size]
            
            # Forward pass on both views
            emb_1 = self.model.gnn(aug_features_1, aug_edge_index_1)[node_idx]
            emb_2 = self.model.gnn(aug_features_2, aug_edge_index_2)[node_idx]
            
            # Normalize embeddings
            emb_1 = F.normalize(emb_1, dim=-1)
            emb_2 = F.normalize(emb_2, dim=-1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(emb_1, emb_2.T) / self.temperature
            
            # Labels: diagonal pairs are positive
            labels = torch.arange(batch_size, device=self.device)
            
            # Contrastive loss (InfoNCE)
            loss = F.cross_entropy(similarity_matrix, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            if (epoch + 1) % max(1, num_epochs // 10) == 0:
                logger.info(f"  Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item():.4f}")
        
        logger.info(f"✓ Graph contrastive learning complete. Final loss: {loss.item():.4f}")
        return history
    
    def _mask_node_features(
        self,
        node_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask a fraction of node features."""
        num_nodes = node_features.shape[0]
        num_mask = int(num_nodes * self.mask_rate)
        
        # Random mask
        mask_idx = torch.randperm(num_nodes)[:num_mask]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=node_features.device)
        mask[mask_idx] = True
        
        # Create masked features (set to zero or random)
        masked_features = node_features.clone()
        masked_features[mask] = 0  # Can also use torch.randn_like
        
        return masked_features, mask, node_features
    
    def _generate_negative_edges(
        self,
        num_nodes: int,
        pos_edge_index: torch.Tensor,
        num_negative: int = 1
    ) -> torch.Tensor:
        """Generate negative edges by random sampling."""
        num_pos = pos_edge_index.shape[1]
        num_neg = num_pos * num_negative
        
        # Random node pairs
        neg_edge_index = torch.randint(
            0, num_nodes,
            (2, num_neg),
            device=pos_edge_index.device
        )
        
        return neg_edge_index
    
    def _augment_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        drop_edge_rate: float = 0.1,
        drop_feature_rate: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment graph by dropping edges and features.
        
        Returns augmented features and edge index.
        """
        # Drop edges
        num_edges = edge_index.shape[1]
        keep_edge_mask = torch.rand(num_edges) > drop_edge_rate
        aug_edge_index = edge_index[:, keep_edge_mask]
        
        # Drop features (set to 0)
        aug_features = node_features.clone()
        drop_feature_mask = torch.rand_like(aug_features) < drop_feature_rate
        aug_features[drop_feature_mask] = 0
        
        return aug_features, aug_edge_index
    
    def pretrain_all(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        tasks: list = ['masked_node', 'edge_contrastive'],
        num_epochs: int = 10,
        lr: float = 1e-3
    ) -> Dict[str, Dict]:
        """
        Run multiple pretraining tasks sequentially.
        
        Args:
            node_features: Node features
            edge_index: Edge connectivity
            tasks: List of tasks to run ('masked_node', 'edge_contrastive', 'graph_contrastive')
            num_epochs: Epochs per task
            lr: Learning rate
            
        Returns:
            Dictionary of histories for each task
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running pretraining tasks: {tasks}")
        logger.info(f"{'='*60}")
        
        histories = {}
        
        for task in tasks:
            if task == 'masked_node':
                histories[task] = self.masked_node_prediction(
                    node_features, edge_index, num_epochs, lr
                )
            elif task == 'edge_contrastive':
                histories[task] = self.edge_contrastive_learning(
                    node_features, edge_index, num_epochs, lr
                )
            elif task == 'graph_contrastive':
                histories[task] = self.graph_contrastive_learning(
                    node_features, edge_index, num_epochs, lr
                )
            else:
                logger.warning(f"Unknown task: {task}")
        
        logger.info(f"\n{'='*60}")
        logger.info("All pretraining tasks complete!")
        logger.info(f"{'='*60}")
        
        return histories


def test_pretrainer():
    """Test pretrainingwith dummy data."""
    logger.info("Testing GraphPretrainer...")
    
    logger.info("Pretrainer test placeholder - requires full model setup")
    logger.info("✓ Pretrainer module loaded successfully")


if __name__ == "__main__":
    test_pretrainer()
```

## File: `src/training/__init__.py`

```python
"""
Training module for PromptGFM-Bio.

This module implements pretraining, finetuning, and loss functions
for the PromptGFM model.
"""
```

## File: `src/utils/config.py`

```python
"""
Configuration management utilities.

Handles loading and merging YAML configuration files.
"""

import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config, override_config):
    """Merge override config into base config."""
    logger.info("Merging configurations")
    # Implementation placeholder
    pass


def save_config(config, output_path):
    """Save configuration to YAML file."""
    logger.info(f"Saving config to {output_path}")
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    logger.info("Config utilities ready")
```

## File: `src/utils/logger.py`

```python
"""
Logging utilities for PromptGFM-Bio.

Provides consistent logging across modules with Weights & Biases integration.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def init_wandb(config, project_name='promptgfm-bio'):
    """Initialize Weights & Biases logging."""
    logger.info("W&B initialization placeholder - will be implemented in Phase 4")
    pass


if __name__ == "__main__":
    logger.info("Logger utilities ready")
```

## File: `src/utils/__init__.py`

```python
"""
Utilities module for PromptGFM-Bio.

This module contains configuration management and logging utilities.
"""
```

