"""
Test script to verify negative sampling works correctly.
"""
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import GeneDiseaseDataset
from scripts.train import create_collate_fn

def test_negative_sampling():
    """Test that negative sampling generates proper labels."""
    print("Testing negative sampling implementation...")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'finetune_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = GeneDiseaseDataset(
        graph_path=config['data']['graph_file'],
        edges_path=config['data']['edge_file'],
        min_score=config['data'].get('min_score', 0.3)
    )
    print(f"   ✓ Loaded {len(dataset.edges_df)} edges")
    print(f"   ✓ Graph has {dataset.graph['gene'].num_nodes} gene nodes")
    
    # Create train split
    print("\n2. Creating train split...")
    train_edges, _, _ = dataset.create_train_val_test_split(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42
    )
    print(f"   ✓ Train edges: {len(train_edges)}")
    
    # Create collate function with negative sampling
    print("\n3. Creating collate function...")
    num_negatives = config['data'].get('num_negatives', 5)
    print(f"   Num negatives per positive: {num_negatives}")
    
    input_dim = config.get('model', {}).get('gnn_input_dim', 128)
    collate_fn = create_collate_fn(
        train_edges, dataset.graph, dataset.gene_to_idx, 
        input_dim, num_negatives
    )
    print(f"   ✓ Collate function created")
    
    # Test with a small batch
    print("\n4. Testing with batch of 4 edges...")
    batch_indices = torch.arange(4)
    batch_data = collate_fn([batch_indices])
    
    print(f"\n5. Batch results:")
    print(f"   Gene indices shape: {batch_data['gene_indices'].shape}")
    print(f"   Labels shape: {batch_data['labels'].shape}")
    print(f"   Disease texts count: {len(batch_data['disease_texts'])}")
    
    # Count positive and negative samples
    labels = batch_data['labels']
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    
    print(f"\n6. Label distribution:")
    print(f"   Positive samples (label=1): {num_pos}")
    print(f"   Negative samples (label=0): {num_neg}")
    print(f"   Expected ratio: 1:{num_negatives}")
    print(f"   Actual ratio: 1:{num_neg/num_pos:.1f}")
    
    # Show sample of labels
    print(f"\n7. First 10 labels: {labels[:10].tolist()}")
    print(f"   Last 10 labels: {labels[-10:].tolist()}")
    
    # Validation
    print(f"\n8. Validation:")
    expected_total = num_pos * (1 + num_negatives)
    actual_total = len(labels)
    
    if num_pos > 0 and num_neg > 0:
        print(f"   ✓ Both positive and negative samples present")
    else:
        print(f"   ✗ Missing positive or negative samples!")
        
    if abs(num_neg / num_pos - num_negatives) < 0.5:
        print(f"   ✓ Ratio approximately correct")
    else:
        print(f"   ✗ Ratio incorrect (expected 1:{num_negatives})")
    
    if actual_total == expected_total:
        print(f"   ✓ Total sample count correct ({actual_total})")
    else:
        print(f"   ⚠ Total samples: {actual_total} (expected ~{expected_total})")
    
    print(f"\n{'='*60}")
    if num_pos > 0 and num_neg > 0:
        print("✓ NEGATIVE SAMPLING WORKING CORRECTLY!")
        print(f"  Training will use mixed positive/negative samples")
        print(f"  Loss should be non-zero and decrease over epochs")
    else:
        print("✗ NEGATIVE SAMPLING FAILED!")
        print(f"  Check implementation in scripts/train.py")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_negative_sampling()
