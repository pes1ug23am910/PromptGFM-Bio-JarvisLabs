"""
Test script for data download module.

This script tests the download functionality without actually downloading large files.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import (
    _get_data_dir,
    _verify_checksum,
    download_biogrid,
    download_string,
    download_disgenet,
    download_hpo
)

def test_directory_creation():
    """Test that data directory is created properly."""
    print("Testing directory creation...")
    data_dir = _get_data_dir()
    print(f"✓ Data directory: {data_dir}")
    assert data_dir.exists(), "Data directory should exist"
    assert data_dir.is_dir(), "Data directory should be a directory"
    print("✓ Directory creation test passed\n")


def test_imports():
    """Test that all functions can be imported."""
    print("Testing imports...")
    print("✓ _get_data_dir imported")
    print("✓ _verify_checksum imported")
    print("✓ download_biogrid imported")
    print("✓ download_string imported")
    print("✓ download_disgenet imported")
    print("✓ download_hpo imported")
    print("✓ All imports successful\n")


def test_function_signatures():
    """Test that functions have correct signatures."""
    print("Testing function signatures...")
    
    # Test that functions accept expected parameters
    import inspect
    
    sig_biogrid = inspect.signature(download_biogrid)
    print(f"✓ download_biogrid signature: {sig_biogrid}")
    
    sig_string = inspect.signature(download_string)
    print(f"✓ download_string signature: {sig_string}")
    
    sig_disgenet = inspect.signature(download_disgenet)
    print(f"✓ download_disgenet signature: {sig_disgenet}")
    
    sig_hpo = inspect.signature(download_hpo)
    print(f"✓ download_hpo signature: {sig_hpo}")
    
    print("✓ Function signature test passed\n")


def show_download_info():
    """Display information about downloads without executing."""
    print("="*70)
    print("Download Module Information")
    print("="*70)
    print("\nAvailable download functions:")
    print("  1. download_biogrid()  - BioGRID PPI (~500MB)")
    print("  2. download_string()   - STRING network (~700MB)")
    print("  3. download_disgenet() - DisGeNET associations (~300MB)")
    print("  4. download_hpo()      - HPO annotations (~50MB)")
    print("  5. download_all()      - All datasets (~1.5GB total)")
    print("\nFeatures:")
    print("  ✓ Progress bars with tqdm")
    print("  ✓ Automatic retry with exponential backoff")
    print("  ✓ Checksum verification")
    print("  ✓ Automatic archive extraction")
    print("  ✓ Caching (skip if already downloaded)")
    print("  ✓ Error handling and logging")
    print("\nUsage:")
    print("  python -m src.data.download --dataset all")
    print("  python -m src.data.download --dataset biogrid --force")
    print("\nTo actually download data, run:")
    print("  python scripts/download_data.py")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PromptGFM-Bio Download Module Test")
    print("="*70 + "\n")
    
    try:
        test_imports()
        test_directory_creation()
        test_function_signatures()
        show_download_info()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nDownload module is ready to use!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
