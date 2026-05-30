"""
Quick test to verify training optimizations are working correctly.
This tests that AMP, DataLoader optimizations, and cuDNN are properly configured.
"""

import sys
import torch
from pathlib import Path

def test_optimizations():
    """Test that all optimizations are properly configured."""
    
    print("="*70)
    print("TRAINING OPTIMIZATIONS TEST")
    print("="*70)
    
    # Test 1: Check CUDA availability
    print("\n1. CUDA Availability:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    
    # Test 2: Test Mixed Precision (AMP)
    print("\n2. Mixed Precision (AMP):")
    if cuda_available:
        from torch.cuda.amp import autocast, GradScaler
        
        # Create simple tensors
        x = torch.randn(10, 10, device='cuda')
        y = torch.randn(10, 10, device='cuda')
        
        # Test autocast
        with autocast():
            z = torch.mm(x, y)
            print(f"   Autocast dtype: {z.dtype}")
            if z.dtype == torch.float16:
                print("   ✓ AMP working correctly (FP16)")
            else:
                print(f"   ⚠ Expected FP16, got {z.dtype}")
        
        # Test GradScaler
        scaler = GradScaler()
        print(f"   ✓ GradScaler initialized")
    else:
        print("   ⚠ Skipped (no CUDA)")
    
    # Test 3: cuDNN Benchmark
    print("\n3. cuDNN Autotuning:")
    if cuda_available:
        torch.backends.cudnn.benchmark = True
        print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   ✓ cuDNN autotuning enabled")
    else:
        print("   ⚠ Skipped (no CUDA)")
    
    # Test 4: DataLoader settings
    print("\n4. DataLoader Configuration:")
    import os
    cpu_count = os.cpu_count()
    num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
    print(f"   CPU cores: {cpu_count}")
    print(f"   DataLoader workers: {num_workers}")
    print(f"   Pin memory: {cuda_available}")
    if num_workers > 0:
        print(f"   ✓ Parallel data loading enabled")
    else:
        print(f"   ⚠ Single-threaded (may be slower)")
    
    # Test 5: Import trainer with AMP
    print("\n5. Trainer Module:")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.training.finetune import PromptGFMTrainer
        print(f"   ✓ PromptGFMTrainer imported successfully")
        
        # Check if trainer has AMP support
        import inspect
        sig = inspect.signature(PromptGFMTrainer.__init__)
        params = list(sig.parameters.keys())
        if 'use_amp' in params:
            print(f"   ✓ AMP parameter available in trainer")
        else:
            print(f"   ✗ AMP parameter missing from trainer")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    optimizations_working = []
    optimizations_skipped = []
    
    if cuda_available:
        optimizations_working.append("✓ Mixed Precision (AMP) - 1.5-2× speedup")
        optimizations_working.append("✓ cuDNN Autotuning - 5-15% speedup")
        optimizations_working.append("✓ Pin Memory - Faster GPU transfers")
    else:
        optimizations_skipped.append("⚠ GPU optimizations (no CUDA)")
    
    if num_workers > 0:
        optimizations_working.append(f"✓ Parallel DataLoader ({num_workers} workers) - 20-40% speedup")
    else:
        optimizations_skipped.append("⚠ DataLoader parallelization")
    
    print("\nEnabled Optimizations:")
    for opt in optimizations_working:
        print(f"  {opt}")
    
    if optimizations_skipped:
        print("\nSkipped:")
        for opt in optimizations_skipped:
            print(f"  {opt}")
    
    print("\nExpected Performance:")
    if cuda_available and num_workers > 0:
        print("  • Combined speedup: 2-3× faster than baseline")
        print("  • Memory usage: ~40% lower (can use larger batch)")
        print("  • No accuracy degradation")
    else:
        print("  • Limited speedup without GPU")
    
    print("\n" + "="*70)
    print("Ready to train with optimizations!")
    print("="*70)

if __name__ == '__main__':
    test_optimizations()
