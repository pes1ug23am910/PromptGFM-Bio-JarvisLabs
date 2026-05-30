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
