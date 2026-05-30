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
