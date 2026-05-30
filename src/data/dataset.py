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
