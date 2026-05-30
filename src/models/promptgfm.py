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
