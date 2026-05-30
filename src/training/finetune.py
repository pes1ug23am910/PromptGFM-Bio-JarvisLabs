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
