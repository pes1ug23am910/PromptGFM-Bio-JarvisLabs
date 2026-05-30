"""
Setup verification script for PromptGFM-Bio.

Checks that all required packages are installed and working correctly.
"""

import sys
import importlib


def check_package(package_name, import_name=None, check_version=None):
    """Check if a package is installed and optionally verify its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        status = "OK"
        if check_version and version != check_version:
            status = f"WARN (found {version}, expected {check_version})"
        else:
            status = f"OK (version {version})"
        
        print(f"  {status} {package_name}")
        return True
    except ImportError:
        print(f"  MISSING {package_name} - NOT FOUND")
        return False


def main():
    """Run all verification checks."""
    print("=" * 50)
    print("PromptGFM-Bio Setup Verification")
    print("=" * 50)
    print()
    
    # Check Python version
    print("Python Environment:")
    print(f"  OK Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  WARN Python 3.10+ recommended")
    print()
    
    # Check core packages
    print("Core Deep Learning:")
    all_ok = True
    all_ok &= check_package("torch", check_version="2.1.0")
    all_ok &= check_package("torchvision", check_version="0.16.0")
    all_ok &= check_package("torchaudio", check_version="2.1.0")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  OK CUDA available (version {torch.version.cuda})")
            print(f"  OK GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  OK VRAM: {props.total_memory / 1024**3:.2f} GB")
            print(f"  OK Compute Capability: {props.major}.{props.minor}")
        else:
            print("  WARN CUDA not available (CPU-only mode)")
    except:
        pass
    print()
    
    # Check PyTorch Geometric
    print("PyTorch Geometric:")
    all_ok &= check_package("torch-geometric", "torch_geometric", "2.4.0")
    all_ok &= check_package("torch-scatter", "torch_scatter")
    all_ok &= check_package("torch-sparse", "torch_sparse")
    all_ok &= check_package("torch-cluster", "torch_cluster")
    print()
    
    # Check NLP packages
    print("NLP & Transformers:")
    all_ok &= check_package("transformers", check_version="4.35.0")
    all_ok &= check_package("sentence-transformers", "sentence_transformers")
    all_ok &= check_package("datasets")
    print()
    
    # Check data science packages
    print("Data Science:")
    all_ok &= check_package("numpy")
    all_ok &= check_package("pandas")
    all_ok &= check_package("scipy")
    all_ok &= check_package("scikit-learn", "sklearn")
    all_ok &= check_package("networkx")
    print()
    
    # Check biomedical packages
    print("Biomedical:")
    all_ok &= check_package("biopython", "Bio")
    print()
    
    # Check visualization
    print("Visualization:")
    all_ok &= check_package("matplotlib")
    all_ok &= check_package("seaborn")
    print()
    
    # Check utilities
    print("Utilities:")
    all_ok &= check_package("tqdm")
    all_ok &= check_package("wandb")
    all_ok &= check_package("yaml")
    all_ok &= check_package("requests")
    print()
    
    # Check development tools
    print("Development (optional):")
    check_package("pytest")
    check_package("jupyter")
    print()
    
    # Final status
    print("=" * 50)
    if all_ok:
        print("OK Setup verification PASSED")
        print("All required packages are installed correctly!")
    else:
        print("FAILED Setup verification")
        print("Some required packages are missing or have incorrect versions.")
        print("\nPlease run:")
        print("  pip install -r requirements.txt")
    print("=" * 50)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
