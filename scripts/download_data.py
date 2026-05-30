"""
Main script to download all biomedical datasets for PromptGFM-Bio.

This script orchestrates the download of all required datasets:
- BioGRID protein-protein interactions
- STRING database PPI
- DisGeNET gene-disease associations
- Human Phenotype Ontology (HPO)

Usage:
    python scripts/download_data.py                    # Download all datasets
    python scripts/download_data.py --dataset biogrid  # Download specific dataset
    python scripts/download_data.py --force            # Force re-download
"""

import sys
from pathlib import Path
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import (
    download_all,
    download_biogrid,
    download_string,
    download_disgenet,
    download_hpo
)


def main():
    parser = argparse.ArgumentParser(
        description="Download biomedical datasets for PromptGFM-Bio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Download all datasets
  %(prog)s --dataset string         Download only STRING database
  %(prog)s --force                  Re-download all (overwrite existing)
  %(prog)s --dataset hpo --force    Re-download HPO only

Datasets:
  all       - All datasets (~1.5GB total)
  biogrid   - BioGRID protein interactions (~500MB)
  string    - STRING protein network (~700MB)
  disgenet  - DisGeNET gene-disease associations (~300MB)
  hpo       - Human Phenotype Ontology (~50MB)
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['all', 'biogrid', 'string', 'disgenet', 'hpo'],
        default='all',
        help='Which dataset to download (default: all)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files already exist'
    )
    
    parser.add_argument(
        '--skip-failing',
        action='store_true',
        default=True,
        help='Continue downloading other datasets if one fails (default: True)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PromptGFM-Bio Data Download")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Force re-download: {args.force}")
    print(f"Skip failing: {args.skip_failing}")
    print()
    
    try:
        if args.dataset == 'all':
            results = download_all(force=args.force, skip_failing=args.skip_failing)
            
            # Check results
            success_count = sum(1 for files in results.values() if files)
            total_count = len(results)
            
            print(f"\n✓ Successfully downloaded {success_count}/{total_count} datasets")
            
            if success_count < total_count:
                print("\n⚠ Some downloads failed. You can:")
                print("  1. Try again: python scripts/download_data.py --force")
                print("  2. Download specific dataset: python scripts/download_data.py --dataset <name>")
                print("  3. Download manually to data/raw/ directory")
                sys.exit(1)
            
        elif args.dataset == 'biogrid':
            results = download_biogrid(force=args.force)
            if not results:
                sys.exit(1)
                
        elif args.dataset == 'string':
            results = download_string(force=args.force)
            if not results:
                sys.exit(1)
                
        elif args.dataset == 'disgenet':
            results = download_disgenet(force=args.force)
            if not results:
                sys.exit(1)
                
        elif args.dataset == 'hpo':
            results = download_hpo(force=args.force)
            if not results:
                sys.exit(1)
        
        print("\n" + "="*70)
        print("✓ DOWNLOAD COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run preprocessing: python scripts/preprocess_all.py")
        print("  2. Check downloaded files in: data/raw/")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
