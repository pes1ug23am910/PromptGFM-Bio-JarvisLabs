"""
Orphadata Integration Module

Orphadata provides authoritative rare disease gene associations.
This module handles download, parsing, and integration into the graph.

Orphadata Files:
- en_product6.xml: Gene-disease associations
- en_product1.xml: Rare disease classifications
- en_product4.xml: Disease prevalence data
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def download_orphadata(output_dir: Path = None) -> Dict[str, Path]:
    """
    Download Orphadata XML files.
    
    Files downloaded:
    1. en_product6.xml - Gene-disease associations (PRIMARY)
    2. en_product1.xml - Disease classifications
    3. en_product4.xml - Prevalence data
    
    Args:
        output_dir: Directory to save files
        
    Returns:
        Dict mapping filenames to paths
    """
    if output_dir is None:
        # Get default data directory
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "raw" / "orphanet"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = {
        'en_product6.xml': 'http://www.orphadata.com/data/xml/en_product6.xml',  # Genes
        'en_product1.xml': 'http://www.orphadata.com/data/xml/en_product1.xml',  # Classifications
        'en_product4.xml': 'http://www.orphadata.com/data/xml/en_product4.xml',  # Prevalence
    }
    
    downloaded_files = {}
    
    for filename, url in files_to_download.items():
        output_file = output_dir / filename
        
        if output_file.exists():
            logger.info(f"✓ {filename} already exists, skipping")
            downloaded_files[filename] = output_file
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / 1024 / 1024
            logger.info(f"✓ Downloaded {filename} ({file_size_mb:.1f} MB)")
            downloaded_files[filename] = output_file
            
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
    
    logger.info(f"\nOrphadata files saved to {output_dir}")
    return downloaded_files


def parse_orphadata_gene_associations(xml_path: Path) -> pd.DataFrame:
    """
    Parse en_product6.xml to extract gene-disease associations.
    
    XML structure (simplified):
    <JDBOR>
        <DisorderList>
            <Disorder id="...">
                <OrphaCode>ORPHA:166024</OrphaCode>
                <Name>Angelman syndrome</Name>
                <DisorderGeneAssociationList>
                    <DisorderGeneAssociation>
                        <Gene>
                            <Symbol>UBE3A</Symbol>
                            <ExternalReferenceList>
                                <ExternalReference>
                                    <Source>HGNC</Source>
                                    <Reference>12496</Reference>
                                </ExternalReference>
                            </ExternalReferenceList>
                        </Gene>
                        <DisorderGeneAssociationType>
                            <Name>Disease-causing germline mutation(s) in</Name>
                        </DisorderGeneAssociationType>
                        <DisorderGeneAssociationStatus>
                            <Name>Assessed</Name>
                        </DisorderGeneAssociationStatus>
                    </DisorderGeneAssociation>
                </DisorderGeneAssociationList>
            </Disorder>
        </DisorderList>
    </JDBOR>
    
    Returns:
        DataFrame with columns: [orpha_code, disease_name, gene_symbol, hgnc_id, 
                                 association_type, association_status]
    """
    logger.info(f"Parsing Orphadata gene associations from {xml_path}...")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        associations = []
        
        # Find all disorders
        for disorder in root.findall('.//Disorder'):
            # Extract disease info
            orpha_code = disorder.find('OrphaCode').text if disorder.find('OrphaCode') is not None else None
            disease_name = disorder.find('Name').text if disorder.find('Name') is not None else None
            
            # Extract gene associations
            gene_assoc_list = disorder.find('DisorderGeneAssociationList')
            if gene_assoc_list is None:
                continue
            
            for gene_assoc in gene_assoc_list.findall('DisorderGeneAssociation'):
                # Gene info
                gene = gene_assoc.find('Gene')
                if gene is None:
                    continue
                
                gene_symbol = gene.find('Symbol').text if gene.find('Symbol') is not None else None
                
                # Extract HGNC ID
                hgnc_id = None
                ext_refs = gene.find('ExternalReferenceList')
                if ext_refs is not None:
                    for ext_ref in ext_refs.findall('ExternalReference'):
                        source = ext_ref.find('Source')
                        if source is not None and source.text == 'HGNC':
                            hgnc_id = ext_ref.find('Reference').text
                            break
                
                # Association type
                assoc_type_elem = gene_assoc.find('.//DisorderGeneAssociationType/Name')
                assoc_type = assoc_type_elem.text if assoc_type_elem is not None else None
                
                # Association status
                assoc_status_elem = gene_assoc.find('.//DisorderGeneAssociationStatus/Name')
                assoc_status = assoc_status_elem.text if assoc_status_elem is not None else None
                
                associations.append({
                    'orpha_code': orpha_code,
                    'disease_name': disease_name,
                    'gene_symbol': gene_symbol.upper() if gene_symbol else None,  # Normalize
                    'hgnc_id': hgnc_id,
                    'association_type': assoc_type,
                    'association_status': assoc_status
                })
        
        df = pd.DataFrame(associations)
        logger.info(f"Extracted {len(df)} gene-disease associations from Orphadata")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse Orphadata: {e}")
        return pd.DataFrame()


def filter_high_confidence_orphadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to high-confidence gene-disease associations.
    
    Criteria:
    - Association status = 'Assessed' or 'Validated'
    - Association type includes 'Disease-causing' or 'Major'
    
    Args:
        df: DataFrame from parse_orphadata_gene_associations
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    # Filter by status
    valid_statuses = ['Assessed', 'Validated']
    df_filtered = df[df['association_status'].isin(valid_statuses)].copy()
    
    # Filter by type (disease-causing mutations)
    disease_causing_keywords = ['Disease-causing', 'disease-causing', 'Major', 'major']
    df_filtered = df_filtered[
        df_filtered['association_type'].str.contains('|'.join(disease_causing_keywords), na=False)
    ]
    
    logger.info(f"Filtered to {len(df_filtered)} high-confidence associations")
    logger.info(f"  ({len(df)} total → {len(df_filtered)} after filtering)")
    
    return df_filtered


def merge_orphadata_with_hpo(orphadata_df: pd.DataFrame,
                             hpo_edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Orphadata and HPO-derived gene-disease edges.
    
    Strategy:
    1. Use Orphadata as gold standard (score = 1.0, provenance = 'Orphadata')
    2. Add HPO edges that are NOT in Orphadata (provenance = 'HPO_bridge')
    3. Tag overlapping edges (provenance = 'Orphadata+HPO')
    
    Args:
        orphadata_df: From parse_orphadata_gene_associations (filtered)
        hpo_edges_df: From HPOGeneDiseaseBuilder
        
    Returns:
        Merged DataFrame with all edges
    """
    if orphadata_df.empty:
        logger.warning("Orphadata DataFrame is empty, returning HPO edges only")
        return hpo_edges_df
    
    if hpo_edges_df.empty:
        logger.warning("HPO DataFrame is empty, returning Orphadata edges only")
        # Prepare Orphadata format
        orpha_edges = orphadata_df[['gene_symbol', 'orpha_code', 'disease_name']].copy()
        orpha_edges.rename(columns={'gene_symbol': 'gene', 'orpha_code': 'disease'}, inplace=True)
        orpha_edges['score'] = 1.0
        orpha_edges['provenance'] = 'Orphadata'
        orpha_edges['supporting_phenotypes'] = ''
        orpha_edges['num_shared_phenotypes'] = 0
        return orpha_edges
    
    # Prepare Orphadata edges
    orpha_edges = orphadata_df[['gene_symbol', 'orpha_code', 'disease_name']].copy()
    orpha_edges.rename(columns={'gene_symbol': 'gene', 'orpha_code': 'disease'}, inplace=True)
    orpha_edges['score'] = 1.0  # Gold standard
    orpha_edges['provenance'] = 'Orphadata'
    orpha_edges['supporting_phenotypes'] = ''
    orpha_edges['num_shared_phenotypes'] = 0
    
    # Create lookup set for Orphadata pairs
    orpha_pairs = set(zip(orpha_edges['gene'], orpha_edges['disease']))
    
    # Tag HPO edges
    hpo_edges_df = hpo_edges_df.copy()
    hpo_edges_df['in_orphadata'] = hpo_edges_df.apply(
        lambda row: (row['gene'], row['disease']) in orpha_pairs, axis=1
    )
    
    # Separate HPO edges
    hpo_only = hpo_edges_df[~hpo_edges_df['in_orphadata']].copy()
    hpo_overlap = hpo_edges_df[hpo_edges_df['in_orphadata']].copy()
    
    # Update provenance for overlapping edges in Orphadata
    for idx, row in hpo_overlap.iterrows():
        mask = (orpha_edges['gene'] == row['gene']) & (orpha_edges['disease'] == row['disease'])
        orpha_edges.loc[mask, 'provenance'] = 'Orphadata+HPO'
        orpha_edges.loc[mask, 'supporting_phenotypes'] = row['supporting_phenotypes']
        orpha_edges.loc[mask, 'num_shared_phenotypes'] = row['num_shared_phenotypes']
    
    # Drop the temporary column
    hpo_only = hpo_only.drop('in_orphadata', axis=1)
    
    # Combine all edges
    merged = pd.concat([orpha_edges, hpo_only], ignore_index=True)
    
    logger.info(f"\nMerge summary:")
    logger.info(f"  Orphadata edges: {len(orpha_edges)}")
    logger.info(f"  HPO-only edges: {len(hpo_only)}")
    logger.info(f"  Overlap: {len(hpo_overlap)} edges confirmed by both sources")
    logger.info(f"  Total merged edges: {len(merged)}")
    
    return merged


def validate_hpo_against_orphadata(hpo_edges_df: pd.DataFrame,
                                    orphadata_df: pd.DataFrame) -> Dict[str, float]:
    """
    Validate HPO-derived edges against Orphadata gold standard.
    
    Computes:
    - Precision: What fraction of HPO edges are in Orphadata?
    - Recall: What fraction of Orphadata edges are recovered?
    - F1 score
    
    Args:
        hpo_edges_df: DataFrame with 'gene' and 'disease' columns
        orphadata_df: DataFrame with 'gene_symbol' and 'orpha_code' columns
        
    Returns:
        Dict with precision, recall, F1, and counts
    """
    if hpo_edges_df.empty or orphadata_df.empty:
        logger.warning("Cannot validate: One or both DataFrames are empty")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Create sets of (gene, disease) tuples
    hpo_pairs = set(zip(hpo_edges_df['gene'], hpo_edges_df['disease']))
    
    # Map Orphadata to same format
    orpha_pairs = set(zip(
        orphadata_df['gene_symbol'], 
        orphadata_df['orpha_code']
    ))
    
    # Compute metrics
    true_positives = len(hpo_pairs & orpha_pairs)
    precision = true_positives / len(hpo_pairs) if hpo_pairs else 0
    recall = true_positives / len(orpha_pairs) if orpha_pairs else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hpo_edges': len(hpo_pairs),
        'orphadata_edges': len(orpha_pairs),
        'overlap': true_positives
    }
    
    logger.info(f"\nHPO Bridge Validation vs Orphadata:")
    logger.info(f"  Precision: {precision:.3f} ({true_positives}/{len(hpo_pairs)})")
    logger.info(f"  Recall: {recall:.3f} ({true_positives}/{len(orpha_pairs)})")
    logger.info(f"  F1: {f1:.3f}")
    
    return results


def get_orphadata_gene_disease_edges(
    download_dir: Optional[Path] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Complete pipeline: Download, parse, and filter Orphadata.
    
    Args:
        download_dir: Where to save/load Orphadata files
        force_download: Re-download even if files exist
        
    Returns:
        DataFrame of high-confidence gene-disease edges
    """
    if download_dir is None:
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "raw" / "orphanet"
    
    # Download if needed
    if force_download or not (download_dir / 'en_product6.xml').exists():
        logger.info("Downloading Orphadata...")
        download_orphadata(download_dir)
    
    # Parse gene associations
    xml_path = download_dir / 'en_product6.xml'
    if not xml_path.exists():
        logger.error(f"Orphadata file not found: {xml_path}")
        return pd.DataFrame()
    
    orphadata_df = parse_orphadata_gene_associations(xml_path)
    
    # Filter to high confidence
    orphadata_filtered = filter_high_confidence_orphadata(orphadata_df)
    
    return orphadata_filtered
