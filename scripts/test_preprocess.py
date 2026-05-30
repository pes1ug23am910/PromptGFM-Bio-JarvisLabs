"""
Test script for preprocessing module.

This script tests the preprocessing functions without requiring full datasets.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import (
    _normalize_gene_symbol,
    _get_data_dirs,
    build_heterogeneous_graph,
)


def test_gene_normalization():
    """Test gene symbol normalization."""
    print("Testing gene symbol normalization...")
    
    test_cases = [
        ("TP53", "TP53"),
        ("tp53", "TP53"),
        ("  BRCA1  ", "BRCA1"),
        ("HUMAN_MYC", "MYC"),
        ("", None),
        (None, None),
    ]
    
    for input_symbol, expected in test_cases:
        result = _normalize_gene_symbol(input_symbol)
        assert result == expected, f"Failed for {input_symbol}: got {result}, expected {expected}"
        print(f"  ✓ {input_symbol!r} -> {result!r}")
    
    print("✓ Gene normalization test passed\n")


def test_directory_structure():
    """Test directory creation."""
    print("Testing directory structure...")
    
    dirs = _get_data_dirs()
    print(f"  Raw data dir: {dirs['raw']}")
    print(f"  Processed dir: {dirs['processed']}")
    
    assert dirs['raw'].exists(), "Raw directory should exist"
    assert dirs['processed'].parent.exists(), "Data directory should exist"
    
    print("✓ Directory structure test passed\n")


def test_graph_building():
    """Test heterogeneous graph construction with toy data."""
    print("Testing graph construction with toy data...")
    
    # Create toy PPI data
    ppi_data = {
        'gene_a': ['TP53', 'BRCA1', 'MYC'],
        'gene_b': ['MDM2', 'TP53', 'MAX'],
        'confidence': [0.9, 0.8, 0.95],
        'source': ['biogrid', 'biogrid', 'string']
    }
    ppi_edges = pd.DataFrame(ppi_data)
    
    # Create toy gene-disease data
    gene_disease_data = {
        'gene': ['TP53', 'BRCA1', 'TP53'],
        'diseaseId': ['C0006826', 'C0006826', 'C0024623'],
        'diseaseName': ['Cancer', 'Cancer', 'Li-Fraumeni Syndrome'],
        'confidence': [0.95, 0.9, 0.98],
        'source': ['disgenet', 'disgenet', 'disgenet']
    }
    gene_disease_edges = pd.DataFrame(gene_disease_data)
    
    disease_info = {
        'C0006826': 'Cancer',
        'C0024623': 'Li-Fraumeni Syndrome'
    }
    
    # Build graph
    graph = build_heterogeneous_graph(
        ppi_edges=ppi_edges,
        gene_disease_edges=gene_disease_edges,
        disease_info=disease_info
    )
    
    # Verify graph structure
    assert 'gene' in graph.node_types, "Graph should have 'gene' nodes"
    assert 'disease' in graph.node_types, "Graph should have 'disease' nodes"
    
    print(f"  Gene nodes: {graph['gene'].num_nodes}")
    print(f"  Disease nodes: {graph['disease'].num_nodes}")
    print(f"  Edge types: {graph.edge_types}")
    
    assert graph['gene'].num_nodes > 0, "Should have gene nodes"
    assert graph['disease'].num_nodes > 0, "Should have disease nodes"
    
    print("✓ Graph construction test passed\n")


def test_imports():
    """Test that all functions can be imported."""
    print("Testing imports...")
    
    from src.data.preprocess import (
        parse_biogrid,
        parse_string,
        parse_ppi_network,
        parse_disgenet,
        parse_hpo,
        build_heterogeneous_graph,
        save_graph,
        preprocess_all
    )
    
    print("  ✓ parse_biogrid imported")
    print("  ✓ parse_string imported")
    print("  ✓ parse_ppi_network imported")
    print("  ✓ parse_disgenet imported")
    print("  ✓ parse_hpo imported")
    print("  ✓ build_heterogeneous_graph imported")
    print("  ✓ save_graph imported")
    print("  ✓ preprocess_all imported")
    
    print("✓ All imports successful\n")


def show_preprocessing_info():
    """Display information about preprocessing."""
    print("="*70)
    print("Preprocessing Module Information")
    print("="*70)
    print("\nFunctions:")
    print("  1. parse_biogrid()       - Parse BioGRID PPI data")
    print("  2. parse_string()        - Parse STRING network")
    print("  3. parse_ppi_network()   - Parse and combine PPI networks")
    print("  4. parse_disgenet()      - Parse gene-disease associations")
    print("  5. parse_hpo()           - Parse phenotype annotations")
    print("  6. build_heterogeneous_graph() - Construct PyG HeteroData")
    print("  7. save_graph()          - Save graph to disk")
    print("  8. preprocess_all()      - Run complete pipeline")
    print("\nFeatures:")
    print("  ✓ HGNC gene symbol normalization")
    print("  ✓ Filter to Homo sapiens only")
    print("  ✓ Confidence score filtering")
    print("  ✓ Rare disease filtering (<= N known genes)")
    print("  ✓ Heterogeneous graph construction")
    print("  ✓ PyTorch Geometric HeteroData format")
    print("  ✓ Comprehensive logging and statistics")
    print("\nUsage:")
    print("  python scripts/preprocess_all.py")
    print("  python scripts/preprocess_all.py --force")
    print("\nOutput:")
    print("  data/processed/biomedical_graph.pt")
    print("  data/processed/biomedical_graph_stats.txt")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PromptGFM-Bio Preprocessing Module Test")
    print("="*70 + "\n")
    
    try:
        test_imports()
        test_gene_normalization()
        test_directory_structure()
        test_graph_building()
        show_preprocessing_info()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nPreprocessing module is ready to use!")
        print("\nOnce data download completes, run:")
        print("  python scripts/preprocess_all.py")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
