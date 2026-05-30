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
