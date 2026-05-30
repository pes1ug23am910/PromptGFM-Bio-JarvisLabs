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
