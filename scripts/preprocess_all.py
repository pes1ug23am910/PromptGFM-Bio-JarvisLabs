"""
Preprocessing script for PromptGFM-Bio.

This script orchestrates the complete preprocessing pipeline:
1. Parse BioGRID and STRING PPI networks
2. Create gene-disease edges using enhanced methods:
   - HPO Bridge: IDF-weighted phenotype overlap (primary method)
   - Orphadata: Gold standard rare disease associations (validation)
   - DisGeNET: Backup method (if available)
3. Parse HPO phenotype annotations
4. Optionally add UniProt gene descriptions (Week 4+)
5. Optionally add Reactome pathway annotations (Week 4+)
6. Build heterogeneous graph
7. Save processed graph for model training

Usage:
    # Basic usage (HPO bridge + Orphadata)
    python scripts/preprocess_all.py
    
    # Force reprocess
    python scripts/preprocess_all.py --force
    
    # Week 4+ enhancements
    python scripts/preprocess_all.py --with-uniprot --with-pathways
    
    # Minimal (HPO bridge only)
    python scripts/preprocess_all.py --no-orphadata
"""

import sys
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import preprocess_all
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess biomedical data into knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes raw biomedical datasets into a heterogeneous knowledge graph:
  
  Input files (in data/raw/):
    - BioGRID: protein-protein interactions
    - STRING: protein network database  
    - HPO: Human Phenotype Ontology (genes_to_phenotype.txt, phenotype.hpoa)
    - Orphadata: Gold standard rare disease gene associations (optional)
    - UniProt: Gene descriptions (optional, Week 4+)
    - Reactome: Pathway annotations (optional, Week 4+)
    
  Output (in data/processed/):
    - biomedical_graph.pt: PyTorch Geometric HeteroData graph
    - biomedical_graph_stats.txt: Graph statistics
    - hpo_gene_disease_edges.csv: HPO bridge edges (if enabled)
    - uniprot_gene_descriptions.csv: Gene descriptions (if enabled)
    - reactome_gene_pathways.csv: Pathway annotations (if enabled)
    
  Graph structure:
    Node types: [gene, disease, phenotype]
    Edge types: [gene-gene, gene-disease, disease-phenotype]
  
  Gene-Disease Edge Methods:
    1. HPO Bridge (Primary): IDF-weighted phenotype overlap
    2. Orphadata (Secondary): Gold standard validation
    3. DisGeNET (Backup): Direct associations (if available)

Examples:
  %(prog)s                              Process with HPO bridge + Orphadata
  %(prog)s --force                      Force reprocess
  %(prog)s --no-orphadata               Use HPO bridge only (MVP)
  %(prog)s --with-uniprot               Add UniProt gene descriptions
  %(prog)s --with-uniprot --with-pathways   Full enhancement (Week 4+)
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output already exists'
    )
    
    parser.add_argument(
        '--no-hpo-bridge',
        action='store_true',
        help='Disable HPO phenotype bridge (primary gene-disease method)'
    )
    
    parser.add_argument(
        '--no-orphadata',
        action='store_true',
        help='Disable Orphadata gold standard integration'
    )
    
    parser.add_argument(
        '--with-uniprot',
        action='store_true',
        help='Enable UniProt gene descriptions (Week 4+)'
    )
    
    parser.add_argument(
        '--with-pathways',
        action='store_true',
        help='Enable Reactome pathway annotations (Week 4+)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PromptGFM-Bio Enhanced Preprocessing Pipeline")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Force reprocess: {args.force}")
    print(f"  HPO Bridge: {not args.no_hpo_bridge}")
    print(f"  Orphadata: {not args.no_orphadata}")
    print(f"  UniProt: {args.with_uniprot}")
    print(f"  Pathways: {args.with_pathways}")
    print()
    
    try:
        preprocess_all(
            force=args.force,
            use_hpo_bridge=not args.no_hpo_bridge,
            use_orphadata=not args.no_orphadata,
            use_uniprot=args.with_uniprot,
            use_pathways=args.with_pathways
        )
        
        print("\n" + "="*70)
        print("✓ PREPROCESSING COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Create dataset splits: python -m src.data.dataset")
        print("  2. Check graph file: data/processed/biomedical_graph.pt")
        print("  3. View statistics: data/processed/biomedical_graph_stats.txt")
        print()
        
    except FileNotFoundError as e:
        print(f"\n✗ Preprocessing failed: Required data file not found")
        print(f"   {e}")
        print("\nMake sure you've downloaded all datasets:")
        print("  python scripts/download_data.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
