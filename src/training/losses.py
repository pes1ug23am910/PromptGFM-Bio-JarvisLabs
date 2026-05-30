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
