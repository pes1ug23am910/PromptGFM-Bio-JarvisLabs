"""
Resume training script with interactive checkpoint management.

Provides options to:
A) Resume from last checkpoint
B) Start fresh (archive old checkpoints)
C) Resume from custom epoch
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_checkpoints(checkpoint_dir):
    """List all available checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for ckpt_file in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
        epoch = int(ckpt_file.stem.split('_')[-1])
        
        # Load checkpoint to get metrics
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            metrics = ckpt.get('current_metrics', {})
            checkpoints.append({
                'path': ckpt_file,
                'epoch': epoch,
                'auroc': metrics.get('auroc', 0),
                'aupr': metrics.get('aupr', 0),
                'loss': metrics.get('loss', 0)
            })
        except Exception as e:
            logger.warning(f"Could not load {ckpt_file}: {e}")
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    return checkpoints


def archive_checkpoints(checkpoint_dir):
    """Archive existing checkpoints to timestamped folder."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_archive_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Move all checkpoint files
    moved = 0
    for ckpt_file in checkpoint_dir.glob("*.pt"):
        shutil.move(str(ckpt_file), str(archive_dir / ckpt_file.name))
        moved += 1
    
    # Move JSON files too
    for json_file in checkpoint_dir.glob("*.json"):
        shutil.move(str(json_file), str(archive_dir / json_file.name))
        moved += 1
    
    logger.info(f"\u2713 Archived {moved} files to: {archive_dir}")
    return archive_dir


def resume_training(config_path, checkpoint_path, start_epoch=None):
    """Resume training from checkpoint.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint to resume from
        start_epoch: Override starting epoch (for custom resume)
    """
    # Import training modules
    from scripts.train import run_finetuning
    from src.training.finetune import PromptGFMTrainer
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set resume checkpoint in config
    config['training']['resume_checkpoint'] = str(checkpoint_path)
    if start_epoch is not None:
        config['training']['resume_epoch'] = start_epoch
    
    logger.info(f"\n{'='*70}")
    logger.info("RESUMING TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    if start_epoch:
        logger.info(f"Starting from epoch: {start_epoch}")
    logger.info(f"{'='*70}\n")
    
    # Run training
    run_finetuning(config)


def interactive_resume():
    """Interactive mode to choose resume option."""
    import sys
    
    print("\n" + "="*70)
    print("  PROMPTGFM-BIO TRAINING RESUME")
    print("="*70)
    
    # Get config file
    config_path = input("\nEnter config file path [configs/finetune_config.yaml]: ").strip()
    if not config_path:
        config_path = "configs/finetune_config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"\n\u274c Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config to get checkpoint directory
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    
    # List available checkpoints
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"\n\u26a0 No checkpoints found in {checkpoint_dir}")
        print("Starting fresh training...\n")
        from scripts.train import run_finetuning
        run_finetuning(config)
        return
    
    # Display options
    print(f"\nFound {len(checkpoints)} checkpoint(s) in {checkpoint_dir}:")
    print("\n" + "-"*70)
    for ckpt in checkpoints[-5:]:  # Show last 5
        print(f"  Epoch {ckpt['epoch']:3d} | AUROC: {ckpt['auroc']:.4f} | "
              f"AUPR: {ckpt['aupr']:.4f} | Loss: {ckpt['loss']:.6f}")
    if len(checkpoints) > 5:
        print(f"  ... and {len(checkpoints) - 5} more")
    print("-"*70)
    
    print("\nChoose an option:")
    print("  A) Resume from last checkpoint (Epoch {})".format(checkpoints[-1]['epoch']))
    print("  B) Start fresh (archive current checkpoints)")
    print("  C) Resume from custom epoch")
    print("  Q) Quit")
    
    choice = input("\nYour choice [A/B/C/Q]: ").strip().upper()
    
    if choice == 'A':
        # Resume from last checkpoint
        last_ckpt = checkpoints[-1]
        print(f"\n\u2713 Resuming from Epoch {last_ckpt['epoch']}")
        print(f"  AUROC: {last_ckpt['auroc']:.4f}, AUPR: {last_ckpt['aupr']:.4f}")
        resume_training(config_path, last_ckpt['path'])
        
    elif choice == 'B':
        # Archive and start fresh
        confirm = input("\n\u26a0 Archive checkpoints and start fresh? [y/N]: ").strip().lower()
        if confirm == 'y':
            archive_dir = archive_checkpoints(checkpoint_dir)
            print(f"\n\u2713 Starting fresh training...")
            from scripts.train import run_finetuning
            run_finetuning(config)
        else:
            print("\nCancelled.")
            
    elif choice == 'C':
        # Custom epoch
        print("\nAvailable epochs:")
        for ckpt in checkpoints:
            print(f"  {ckpt['epoch']}")
        
        try:
            epoch = int(input("\nEnter epoch number to resume from: ").strip())
            matching = [c for c in checkpoints if c['epoch'] == epoch]
            
            if not matching:
                print(f"\n\u274c Error: No checkpoint found for epoch {epoch}")
                sys.exit(1)
            
            ckpt = matching[0]
            print(f"\n\u2713 Resuming from Epoch {ckpt['epoch']}")
            print(f"  AUROC: {ckpt['auroc']:.4f}, AUPR: {ckpt['aupr']:.4f}")
            resume_training(config_path, ckpt['path'])
            
        except ValueError:
            print("\n\u274c Invalid epoch number")
            sys.exit(1)
            
    elif choice == 'Q':
        print("\nExiting...")
        sys.exit(0)
        
    else:
        print("\n\u274c Invalid choice")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Resume PromptGFM-Bio training with checkpoint management'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/finetune_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Specific checkpoint to resume from'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        help='Specific epoch to resume from'
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive existing checkpoints and start fresh'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode (default if no args provided)'
    )
    
    args = parser.parse_args()
    
    # If no specific args, go interactive
    if not any([args.checkpoint, args.epoch, args.archive]) or args.interactive:
        interactive_resume()
        return
    
    # Archive mode
    if args.archive:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
        archive_checkpoints(checkpoint_dir)
        print("\n\u2713 Starting fresh training...")
        from scripts.train import run_finetuning
        run_finetuning(config)
        return
    
    # Resume from specific checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        resume_training(args.config, checkpoint_path, args.epoch)
        return
    
    # Resume from specific epoch
    if args.epoch:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{args.epoch}.pt"
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        resume_training(args.config, checkpoint_path, args.epoch)
        return
    
    # Default: interactive
    interactive_resume()


if __name__ == "__main__":
    main()
