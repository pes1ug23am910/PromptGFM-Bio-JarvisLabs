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
