"""
Main training script for PromptGFM-Bio.

Usage:
    python scripts/train.py --config configs/pretrain_config.yaml  # Pretraining
    python scripts/train.py --config configs/finetune_config.yaml  # Fine-tuning

Dual-GPU (Kaggle T4 ×2):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/kaggle_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings as _w
_w.filterwarnings("ignore", message="An issue occurred while importing", category=UserWarning)

from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset
from src.models.promptgfm import PromptGFM, GNNOnlyBaseline
from src.training.finetune import PromptGFMTrainer, create_optimizer, create_scheduler
from src.training.pretrain import GraphPretrainer
from src.training.losses import BCELoss, MarginRankingLoss, CombinedLoss
from src.evaluation.metrics import GeneRankingEvaluator
from src.utils.config import load_config
from src.utils.logger import setup_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path 2 reproducibility: deterministic per-seed RNG for ALL stochastic sources.
# Called once from run_finetuning / run_pretraining with config['seed'].
# The data split is seeded SEPARATELY in create_dataloaders() using
# config['data']['random_seed'], which is held FIXED across all 10 seeds.
# ---------------------------------------------------------------------------
def _set_all_seeds(seed: int) -> None:
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"[reproducibility] all RNGs seeded with seed={seed} "
                f"(split seed comes from config['data']['random_seed'])")


def _seed_worker(worker_id):
    """Path 2: deterministic, decorrelated per-worker RNG.

    Negative sampling in the DataLoader collate_fn uses Python `random`
    (random.sample / random.choices). PyTorch reseeds each worker's torch and
    numpy RNG but NOT stdlib `random`; with multiprocessing_context="fork" all
    workers would otherwise inherit the SAME `random` state -> correlated
    negatives. This reseeds both `random` and numpy per worker from
    torch.initial_seed() (== main-process torch seed + worker_id, which
    _set_all_seeds fixed from config['seed']), so negatives are reproducible
    across runs AND independent across workers. Relies on
    persistent_workers=True (the reseed runs once per worker; negatives still
    vary across epochs because each worker's stream keeps advancing)."""
    import random as _py_random
    import numpy as _np
    worker_seed = torch.initial_seed() % (2 ** 32)
    _py_random.seed(worker_seed)
    _np.random.seed(worker_seed)


# Distributed training helpers ------------------------------------------------
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
IS_DDP = LOCAL_RANK != -1

def _get_rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def _is_main_process() -> bool:
    return _get_rank() == 0
# -----------------------------------------------------------------------------


def create_collate_fn(edges_df, graph, gene_to_idx, input_dim=128, num_negatives=5):
    """
    Create a collate function for batching gene-disease edges with negative sampling.
    
    Args:
        edges_df: DataFrame of edges (gene, disease, score, etc.)
        graph: PyTorch Geometric HeteroData graph
        gene_to_idx: Dict mapping gene names to graph node indices
        input_dim: Input feature dimension for GNN
        num_negatives: Number of negative samples per positive sample
    
    Returns:
        Collate function for DataLoader
    """
    # Get the actual number of gene nodes from the graph
    num_graph_genes = graph['gene'].num_nodes
    
    # Pre-compute disease to genes mapping for negative sampling
    disease_to_genes = {}
    for _, row in edges_df.iterrows():
        disease = row['disease']
        gene = row['gene']
        if disease not in disease_to_genes:
            disease_to_genes[disease] = set()
        if gene in gene_to_idx:
            idx = gene_to_idx[gene]
            if 0 <= idx < num_graph_genes:
                disease_to_genes[disease].add(idx)
    
    # Get all valid gene indices for negative sampling
    all_gene_indices = set(range(num_graph_genes))
    
    # Pre-compute node features once (will be reused across batches)
    # Use the graph's actual node count, not the dataset vocabulary
    if 'gene' in graph.node_types and hasattr(graph['gene'], 'x'):
        cached_node_features = graph['gene'].x
    else:
        # Create learnable random features (fixed per gene)
        # IMPORTANT: Use graph's node count, not dataset vocabulary size
        torch.manual_seed(42)  # For reproducibility
        cached_node_features = torch.randn(num_graph_genes, input_dim)
    
    # Pre-compute edge index once (will be reused across batches)
    # IMPORTANT: Only use gene-gene edges for message passing
    try:
        edge_types = graph.edge_types if hasattr(graph, 'edge_types') else []
        
        # Only try gene-gene edge types (PPI networks)
        gene_gene_edge_types = [
            ('gene', 'interacts', 'gene'),
            ('gene', 'protein_interaction', 'gene'),
            ('gene', 'ppi', 'gene')
        ]
        
        cached_edge_index = None
        for edge_type in gene_gene_edge_types:
            if edge_type in edge_types:
                cached_edge_index = graph[edge_type].edge_index
                break
        
        # Fallback to empty if no gene-gene edges found
        if cached_edge_index is None:
            import logging
            logging.info(f"No gene-gene edges found in graph. "
                       f"Available edge types: {edge_types}. "
                       f"Training without message passing.")
            cached_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # Validate edge indices are within bounds
            if cached_edge_index.numel() > 0:
                max_idx = cached_edge_index.max().item()
                if max_idx >= num_graph_genes:
                    # Edge index contains invalid node references
                    import logging
                    logging.warning(f"Edge index contains out-of-bounds indices "
                                  f"(max={max_idx}, num_nodes={num_graph_genes}). "
                                  f"Using empty edge index.")
                    cached_edge_index = torch.empty((2, 0), dtype=torch.long)
    except Exception as e:
        import logging
        logging.warning(f"Error loading edge index: {e}. Using empty edge index.")
        cached_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    def collate_fn(batch):
        """
        Convert batch of edge indices to model input format with negative sampling.
        
        Args:
            batch: List of tensors containing edge indices
            
        Returns:
            Dict with keys: node_features, edge_index, disease_texts, gene_indices, labels
        """
        # Extract edge indices from batch (DataLoader wraps in list)
        if isinstance(batch[0], torch.Tensor):
            edge_indices = batch[0].tolist()
        else:
            edge_indices = [item[0].item() if isinstance(item[0], torch.Tensor) else item[0] for item in batch]
        
        # Get edges for this batch
        batch_edges = edges_df.iloc[edge_indices]
        
        # Extract gene names and diseases
        gene_names = batch_edges['gene'].tolist()
        disease_ids = batch_edges['disease'].tolist()
        
        # Map gene names to graph indices (POSITIVE samples)
        positive_gene_indices = []
        positive_disease_texts = []
        num_genes = cached_node_features.shape[0]
        
        for gene, disease in zip(gene_names, disease_ids):
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                # Validate index is in bounds
                if 0 <= idx < num_genes:
                    positive_gene_indices.append(idx)
                    positive_disease_texts.append(disease)
                # else: skip out-of-bounds indices
            # else: skip genes not in mapping
        
        # Sample NEGATIVE genes for each disease
        negative_gene_indices = []
        negative_disease_texts = []
        
        import random
        for disease in positive_disease_texts:
            # Get genes known to be associated with this disease
            known_genes = disease_to_genes.get(disease, set())
            
            # Available negative samples: all genes NOT associated with this disease
            available_negatives = list(all_gene_indices - known_genes)
            
            # Sample num_negatives random genes
            if len(available_negatives) >= num_negatives:
                neg_samples = random.sample(available_negatives, num_negatives)
            else:
                # If not enough, sample with replacement
                neg_samples = random.choices(available_negatives, k=num_negatives) if available_negatives else []
            
            negative_gene_indices.extend(neg_samples)
            # Repeat disease text for each negative sample
            negative_disease_texts.extend([disease] * len(neg_samples))
        
        # Combine positive and negative samples
        combined_gene_indices = positive_gene_indices + negative_gene_indices
        combined_disease_texts = positive_disease_texts + negative_disease_texts
        
        # Convert to tensors
        gene_indices = torch.tensor(combined_gene_indices, dtype=torch.long)
        
        # Labels: 1 for positive, 0 for negative
        positive_labels = torch.ones(len(positive_gene_indices), dtype=torch.float32)
        negative_labels = torch.zeros(len(negative_gene_indices), dtype=torch.float32)
        labels = torch.cat([positive_labels, negative_labels])
        
        # Use cached node features and edge index
        node_features = cached_node_features
        edge_index = cached_edge_index
        
        # node_features and edge_index are deliberately NOT included here.
        # They are graph-wide constants (~40 MB combined) that never change
        # between batches.  Returning them through the DataLoader queue forces
        # workers to serialize them into /dev/shm on every prefetch, which
        # exhausts shared memory and causes ENOSPC Bus Error crashes.
        # The trainer receives them once via set_graph_tensors() below.
        return {
            'disease_texts': combined_disease_texts,
            'gene_indices': gene_indices,
            'labels': labels
        }
    
    return collate_fn


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    logger.info("Creating dataloaders...")
    
    # Load gene-disease dataset (it will load the graph internally)
    dataset = GeneDiseaseDataset(
        graph_path=config['data']['graph_file'],
        edges_path=config['data']['edge_file'],
        min_score=config['data'].get('min_score', 0.3)
    )
    
    # Split data — Path 2: split seed comes from data.random_seed (fixed across
    # all model-init seeds), NOT from top-level config['seed']. Backward-compat
    # fallback: if data.random_seed is missing, use the top-level seed.
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed
    )
    logger.info(f"[reproducibility] split seed = {split_seed} "
                f"(should be FIXED across all 10 model-init seeds)")
    
    logger.info(f"  Train: {len(train_edges)} edges")
    logger.info(f"  Val:   {len(val_edges)} edges")
    logger.info(f"  Test:  {len(test_edges)} edges")
    
    # Get input dimension from config
    input_dim = config.get('model', {}).get('gnn_input_dim', 128)
    
    # Debug: Check graph and dataset consistency
    logger.info(f"  Gene nodes in graph: {dataset.graph['gene'].num_nodes}")
    logger.info(f"  Genes in dataset vocabulary: {len(dataset.gene_to_idx)}")
    logger.info(f"  Input feature dim: {input_dim}")
    
    # Create collate functions with negative sampling
    num_negatives = config['data'].get('num_negatives', 5)
    logger.info(f"  Using {num_negatives} negative samples per positive sample")
    
    train_collate_fn = create_collate_fn(train_edges, dataset.graph, dataset.gene_to_idx, input_dim, num_negatives)
    val_collate_fn = create_collate_fn(val_edges, dataset.graph, dataset.gene_to_idx, input_dim, num_negatives)
    
    # Create dataloaders with proper collate function and optimizations
    from torch.utils.data import TensorDataset
    import os
    import platform
    
    # Optimize DataLoader settings for better CPU/GPU overlap
    # Note: Windows has issues pickling nested functions, so disable workers on Windows
    cpu_count = os.cpu_count()
    is_windows = platform.system() == 'Windows'
    
    # Read num_workers from config when explicitly set; otherwise auto-detect.
    _cfg_nw = config.get("data", {}).get("num_workers", None)
    cfg_nw = int(_cfg_nw) if _cfg_nw is not None else None
    if is_windows:
        num_workers = 0
        logger.info(f"  DataLoader workers: {num_workers} (Windows)")
    elif cfg_nw is not None:
        num_workers = cfg_nw
        logger.info(f"  DataLoader workers: {num_workers} (from config)")
    else:
        num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
        logger.info(f"  DataLoader workers: {num_workers} (auto-detected)")

    # In distributed training use DistributedSampler so each rank gets a unique data shard
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(TensorDataset(torch.arange(len(train_edges))),
                                       shuffle=True) if IS_DDP else None
    val_sampler   = DistributedSampler(TensorDataset(torch.arange(len(val_edges))),
                                       shuffle=False) if IS_DDP else None

    train_loader = DataLoader(
        TensorDataset(torch.arange(len(train_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),  # DistributedSampler handles shuffling
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=_seed_worker,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.arange(len(val_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=_seed_worker,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    
    return train_loader, val_loader, dataset


def run_pretraining(config):
    """Run self-supervised pretraining."""
    logger.info("\n" + "="*60)
    logger.info("Starting Self-Supervised Pretraining")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
    
    # Load graph
    graph_dataset = BiomedicalGraphDataset(config['data']['graph_file'])
    graph = graph_dataset.graph
    
    # Create model
    model = PromptGFM(**config['model'])
    
    # Get device
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create pretrainer
    pretrainer = GraphPretrainer(
        model=model,
        device=device,
        mask_rate=config.get('pretraining', {}).get('mask_ratio', 0.15),
        negative_samples=config.get('pretraining', {}).get('num_negatives', 1)
    )
    
    # Run pretraining tasks
    histories = pretrainer.pretrain_all(
        node_features=graph.x_dict['gene'],  # Assuming gene nodes
        edge_index=graph.edge_index_dict[('gene', 'interacts', 'gene')],
        tasks=config.get('pretraining', {}).get('tasks', {}),
        num_epochs=config.get('pretraining', {}).get('num_epochs', 50),
        lr=config['training'].get('learning_rate', 0.001)
    )
    
    # Save pretrained model
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'histories': histories
    }, checkpoint_dir / 'pretrained_model.pt')
    
    logger.info(f"\n✓ Pretraining complete! Model saved to {checkpoint_dir / 'pretrained_model.pt'}")


def run_finetuning(config):
    """Run supervised fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("Starting Supervised Fine-tuning")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
    
    # Create dataloaders (GeneDiseaseDataset will load the graph internally)
    train_loader, val_loader, dataset = create_dataloaders(config)
    
    # Prepare model parameters from config
    model_config = config['model']
    
    # Create model
    if model_config.get('baseline', False) or not model_config.get('use_prompt', True):
        logger.info("Training GNN-Only Baseline")
        model_params = {
            'gnn_type': model_config.get('gnn_type', 'graphsage'),
            'gnn_hidden_dim': model_config.get('hidden_dim', 256),
            'gnn_num_layers': model_config.get('num_layers', 3),
            'gnn_dropout': model_config.get('dropout', 0.3),
            'hidden_dim': model_config.get('hidden_dim', 256)
        }
        model = GNNOnlyBaseline(**model_params)
    else:
        logger.info("Training PromptGFM")
        # Flatten nested config for PromptGFM
        model_params = {
            # GNN parameters
            'gnn_input_dim': model_config.get('gene_feature_dim', 256),
            'gnn_hidden_dim': model_config.get('hidden_dim', 256),
            'gnn_output_dim': model_config.get('hidden_dim', 256),
            'gnn_num_layers': model_config.get('num_layers', 3),
            'gnn_type': model_config.get('gnn_type', 'graphsage'),
            'gnn_dropout': model_config.get('dropout', 0.3),
            # Prompt encoder parameters
            'prompt_model_name': model_config.get('prompt_encoder', {}).get('model_name', 
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
            'prompt_pooling': model_config.get('prompt_encoder', {}).get('pooling_strategy', 'cls'),
            'prompt_max_length': model_config.get('prompt_encoder', {}).get('max_length', 512),
            'freeze_prompt': model_config.get('prompt_encoder', {}).get('freeze_encoder', False),
            # Conditioning parameters
            'conditioning_type': model_config.get('conditioning_type', 'film'),
            'conditioning_hidden_dim': model_config.get('hidden_dim', 256),
            # Predictor parameters
            'predictor_hidden_dim': model_config.get('prediction_hidden_dim', 128),
            'predictor_dropout': model_config.get('dropout', 0.3),
            # ── FIX (Bug 1, 2026-04-17): forward ablation flags so YAML overrides
            #    reach the model.  Without these, every ablation trained the full
            #    model silently (defaults use_gnn=True, use_conditioning=True).
            'use_gnn':          model_config.get('use_gnn', True),
            'use_conditioning': model_config.get('use_conditioning', True),
        }
        model = PromptGFM(**model_params)
        logger.info(
            f"  Ablation flags active: use_gnn={model_params['use_gnn']}, "
            f"use_conditioning={model_params['use_conditioning']}"
        )
        
        # Load pretrained weights if available
        pretrained_path = config['training'].get('pretrained_checkpoint')
        if pretrained_path and Path(pretrained_path).exists():
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Create loss function
    loss_type = config['training'].get('loss_type', 'bce')
    if loss_type == 'bce':
        loss_fn = BCELoss()
    elif loss_type == 'ranking':
        loss_fn = MarginRankingLoss(
            margin=config['training'].get('margin', 0.5)
        )
    elif loss_type == 'combined':
        loss_weights = config['training'].get('loss_weights', {})
        loss_fn = CombinedLoss(
            bce_weight=loss_weights.get('bce', 1.0),
            ranking_weight=loss_weights.get('ranking', 0.5),
            contrastive_weight=loss_weights.get('contrastive', 0.1)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        lr=config['training'].get('learning_rate', 0.001),
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config['training'].get('lr_scheduler', 'cosine'),
        num_epochs=config['training'].get('num_epochs', 100)
    )
    
    # Create evaluator
    evaluator = GeneRankingEvaluator(
        k_values=config.get('evaluation', {}).get('k_values', [10, 20, 50])
    )
    
    # Get device and checkpoint directory
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if IS_DDP:
        device = f'cuda:{LOCAL_RANK}'
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')

    # --- Wrap model for DDP if running multi-GPU ---
    if IS_DDP:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = model.to(device)
        # find_unused_parameters=True required when freeze_encoder=True:
        # frozen BioBERT params have no gradient hooks; with False DDP hangs
        # on the first backward waiting for a reducer bucket that never fires.
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
        logger.info(f"  DDP enabled — rank {LOCAL_RANK}/{WORLD_SIZE}, device {device}")
    
    # --- Pre-compute BioBERT embeddings for all unique disease texts ---
    # BioBERT is frozen (freeze_encoder=true in config) so embeddings never change.
    # Running BioBERT once upfront and caching by text string eliminates the most
    # expensive forward pass from every training step.
    inner_model = model.module if hasattr(model, 'module') else model
    freeze_encoder = config.get('model', {}).get('prompt_encoder', {}).get('freeze_encoder', False)
    if freeze_encoder and hasattr(inner_model, 'prompt_encoder'):
        logger.info("\n  Pre-computing frozen BioBERT embeddings (runs once, then cached)...")
        all_disease_texts = list(dataset.diseases)  # dataset.diseases = unique disease names

        prompt_enc = inner_model.prompt_encoder
        prompt_enc.eval()
        emb_cache: dict = {}
        encode_bs = 64  # encode in small chunks to avoid OOM during pre-computation
        with torch.no_grad():
            for i in range(0, len(all_disease_texts), encode_bs):
                chunk = all_disease_texts[i:i + encode_bs]
                embs = prompt_enc(chunk).cpu()  # store on CPU; moved to device in _forward_batch
                for txt, emb in zip(chunk, embs):
                    emb_cache[txt] = emb
        logger.info(f"  ✅ Cached {len(emb_cache)} disease embeddings — BioBERT skipped per batch")
    else:
        emb_cache = None
        if not freeze_encoder:
            logger.info("  BioBERT not frozen — gradients needed, skipping pre-computation")

    # Create trainer
    trainer = PromptGFMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        evaluator=evaluator,
        checkpoint_dir=checkpoint_dir,
        max_epochs=config['training'].get('num_epochs', 100),
        patience=config['training'].get('early_stopping_patience', 10),
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        use_wandb=config.get('logging', {}).get('use_wandb', False),
        use_amp=config['training'].get('use_amp', True)  # Enable mixed precision by default
    )

    if emb_cache:
        trainer.set_prompt_cache(emb_cache)

    # ── Graph tensors: pass once to trainer, never through DataLoader queue ──
    # Resolve node features from the graph (same logic as create_collate_fn).
    _graph = dataset.graph
    _num_g = _graph['gene'].num_nodes
    if hasattr(_graph['gene'], 'x') and _graph['gene'].x is not None:
        _nf = _graph['gene'].x
    else:
        import torch as _t
        _t.manual_seed(42)
        _nf = _t.randn(_num_g, config.get('model', {}).get('gene_feature_dim', 128))

    # Resolve edge index (gene-gene PPI edges).
    _et = _graph.edge_types if hasattr(_graph, 'edge_types') else []
    _ei = None
    for _candidate in [('gene','interacts','gene'),
                        ('gene','protein_interaction','gene'),
                        ('gene','ppi','gene')]:
        if _candidate in _et:
            _ei = _graph[_candidate].edge_index
            break
    if _ei is None:
        import torch as _t
        _ei = _t.empty((2, 0), dtype=_t.long)

    trainer.set_graph_tensors(_nf, _ei)
    logger.info(f"  ✅ Graph tensors set on trainer "
                f"(node_features={list(_nf.shape)}, "
                f"edge_index={list(_ei.shape)}) — removed from DataLoader queue")
    
    # Check for resume checkpoint
    resume_checkpoint = config['training'].get('resume_checkpoint')
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info("RESUMING FROM CHECKPOINT")
            logger.info(f"{'='*60}")
            trainer.load_checkpoint(resume_path, load_optimizer=True)
        else:
            logger.warning(f"Resume checkpoint not found: {resume_checkpoint}")
            logger.info("Starting training from scratch...")
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        val_metric=config['training'].get('val_metric', 'auroc')  # Default to AUROC
    )
    
    logger.info("\n✓ Fine-tuning complete!")


def main():
    parser = argparse.ArgumentParser(description='Train PromptGFM-Bio')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pretrain', 'finetune', 'auto'],
        default='auto',
        help='Training mode (auto: detect from config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision training (AMP)'
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # CLI --resume-checkpoint overrides anything in the YAML
    if args.resume_checkpoint:
        if 'training' not in config:
            config['training'] = {}
        config['training']['resume_checkpoint'] = args.resume_checkpoint
    
    # Enable cuDNN autotuning for optimized convolution algorithms
    # This finds the fastest algorithm for your specific hardware (one-time cost)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("✓ cuDNN autotuning enabled (first epoch may be slightly slower)")

    # --- DDP init (activated by torchrun --nproc_per_node=N) ---
    if IS_DDP:
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend='nccl')
        logger.info(f"✓ DDP process group initialised — rank {LOCAL_RANK}/{WORLD_SIZE}")
    
    # Ensure hardware section exists
    if 'hardware' not in config:
        config['hardware'] = {}
    
    # Override device if specified
    if args.device:
        config['hardware']['device'] = args.device
    elif 'device' not in config['hardware']:
        config['hardware']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine mode
    mode = args.mode
    if mode == 'auto':
        mode = config.get('mode', 'finetune')
    
    logger.info(f"Mode: {mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {config['hardware']['device']}")
    
    # Store AMP setting in config for trainer
    if 'training' not in config:
        config['training'] = {}
    config['training']['use_amp'] = not args.no_amp
    if config['training']['use_amp'] and torch.cuda.is_available():
        logger.info("✓ Mixed precision (AMP) enabled (1.5-2× speedup expected)")
    
    # Run training
    if mode == 'pretrain':
        run_pretraining(config)
    elif mode == 'finetune':
        run_finetuning(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
