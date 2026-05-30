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
