"""
UniProt and Reactome Pathways Integration

This module adds:
1. UniProt gene descriptions for better text-based features
2. Reactome pathway annotations for biological validation

Purpose:
- UniProt: Provides protein function descriptions
- Reactome: Provides pathway membership for enrichment analysis
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Set, Optional
from collections import defaultdict
import gzip
import json

logger = logging.getLogger(__name__)


def download_uniprot_human_proteins(output_dir: Path = None) -> Path:
    """
    Download UniProt SwissProt human protein annotations.
    
    SwissProt = manually curated, high-quality subset of UniProt
    Focus on Homo sapiens (organism 9606)
    
    Args:
        output_dir: Where to save file
        
    Returns:
        Path to downloaded file
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "raw" / "uniprot"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "uniprot_human_swissprot.tsv.gz"
    
    if output_file.exists():
        logger.info(f"✓ UniProt file already exists: {output_file}")
        return output_file
    
    logger.info("Downloading UniProt human SwissProt annotations...")
    
    # UniProt REST API for Homo sapiens SwissProt entries
    # Query: organism:9606 AND reviewed:true
    # Fields: accession, gene names, protein names, function
    url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        "format=tsv&"
        "query=(organism_id:9606)%20AND%20(reviewed:true)&"
        "fields=accession,id,gene_names,protein_name,cc_function,organism_name"
    )
    
    try:
        logger.info("Requesting data from UniProt REST API (this may take a few minutes)...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        # Write gzipped
        import gzip
        with gzip.open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"✓ Downloaded UniProt data ({file_size_mb:.1f} MB)")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download UniProt: {e}")
        logger.info("Alternative: Download manually from https://www.uniprot.org/")
        return None


def parse_uniprot_descriptions(filepath: Path) -> pd.DataFrame:
    """
    Parse UniProt TSV file to extract gene descriptions.
    
    Args:
        filepath: Path to uniprot_human_swissprot.tsv.gz
        
    Returns:
        DataFrame with columns: [gene_symbol, protein_name, function_description]
    """
    logger.info(f"Parsing UniProt descriptions from {filepath}...")
    
    try:
        # Read gzipped TSV
        df = pd.read_csv(filepath, sep='\t', compression='gzip')
        
        logger.info(f"Loaded {len(df)} UniProt entries")
        
        # Extract gene symbols from 'Gene Names' column
        # Format: "SYMBOL synonym1 synonym2"
        gene_data = []
        
        for _, row in df.iterrows():
            gene_names = row.get('Gene Names', '') or row.get('Gene names', '')
            if not gene_names or pd.isna(gene_names):
                continue
            
            # First name is primary symbol
            gene_symbol = gene_names.split()[0].upper() if gene_names else None
            
            if gene_symbol:
                protein_name = row.get('Protein names', '') or ''
                function_desc = row.get('Function [CC]', '') or ''
                
                gene_data.append({
                    'gene_symbol': gene_symbol,
                    'protein_name': protein_name,
                    'function_description': function_desc
                })
        
        result_df = pd.DataFrame(gene_data)
        
        # Remove duplicates (keep first entry per gene)
        result_df = result_df.drop_duplicates(subset=['gene_symbol'], keep='first')
        
        logger.info(f"Extracted descriptions for {len(result_df)} genes")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to parse UniProt: {e}")
        return pd.DataFrame()


def download_reactome_pathways(output_dir: Path = None) -> Dict[str, Path]:
    """
    Download Reactome pathway annotations for human.
    
    Files:
    1. UniProt2Reactome: Maps proteins to pathways
    2. ReactomePathways: Pathway names and descriptions
    
    Args:
        output_dir: Where to save files
        
    Returns:
        Dict mapping filenames to paths
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "raw" / "reactome"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = {
        'UniProt2Reactome_All_Levels.txt': (
            'https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt'
        ),
        'ReactomePathways.txt': (
            'https://reactome.org/download/current/ReactomePathways.txt'
        )
    }
    
    downloaded = {}
    
    for filename, url in files_to_download.items():
        output_file = output_dir / filename
        
        if output_file.exists():
            logger.info(f"✓ {filename} already exists")
            downloaded[filename] = output_file
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_kb = len(response.content) / 1024
            logger.info(f"✓ Downloaded {filename} ({file_size_kb:.1f} KB)")
            downloaded[filename] = output_file
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
    
    return downloaded


def parse_reactome_pathways(uniprot2reactome_path: Path,
                            reactome_pathways_path: Path,
                            organism: str = 'Homo sapiens') -> pd.DataFrame:
    """
    Parse Reactome pathway annotations.
    
    Args:
        uniprot2reactome_path: Path to UniProt2Reactome_All_Levels.txt
        reactome_pathways_path: Path to ReactomePathways.txt
        organism: Filter to this organism
        
    Returns:
        DataFrame with columns: [gene_symbol, pathway_id, pathway_name]
    """
    logger.info("Parsing Reactome pathways...")
    
    try:
        # Parse pathway names
        pathway_names = {}
        with open(reactome_pathways_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pathway_id = parts[0]
                    pathway_name = parts[1]
                    # Filter to human pathways (format: R-HSA-...)
                    if pathway_id.startswith('R-HSA'):
                        pathway_names[pathway_id] = pathway_name
        
        logger.info(f"Loaded {len(pathway_names)} human pathway names")
        
        # Parse gene-pathway mappings
        # Format: UniProtID  PathwayID  URL  PathwayName  Evidence  Organism
        gene_pathways = []
        
        with open(uniprot2reactome_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                
                uniprot_id = parts[0]
                pathway_id = parts[1]
                pathway_name = parts[3]
                organism_name = parts[5]
                
                # Filter to human
                if organism_name != organism:
                    continue
                
                # We need to map UniProt ID to gene symbol
                # For simplicity, we'll store UniProt ID and map later
                # Or extract from UniProt data
                gene_pathways.append({
                    'uniprot_id': uniprot_id,
                    'pathway_id': pathway_id,
                    'pathway_name': pathway_name
                })
        
        df = pd.DataFrame(gene_pathways)
        logger.info(f"Parsed {len(df)} gene-pathway associations")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse Reactome: {e}")
        return pd.DataFrame()


def map_uniprot_to_gene_symbols(gene_pathways_df: pd.DataFrame,
                                uniprot_descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UniProt IDs in pathway data to gene symbols using UniProt descriptions.
    
    Args:
        gene_pathways_df: From parse_reactome_pathways
        uniprot_descriptions_df: From parse_uniprot_descriptions
        
    Returns:
        DataFrame with gene_symbol column added
    """
    # This is a simplified approach
    # In production, you'd want a proper UniProt ID -> gene symbol mapping file
    
    logger.info("Mapping UniProt IDs to gene symbols...")
    
    # For now, we'll return the pathways dataframe as-is
    # In a full implementation, you'd use the UniProt API or mapping files
    
    logger.warning("UniProt ID -> gene symbol mapping not fully implemented")
    logger.warning("Use gene_symbol from UniProt descriptions or download ID mapping separately")
    
    return gene_pathways_df


def get_uniprot_gene_descriptions(download_dir: Optional[Path] = None,
                                  force_download: bool = False) -> pd.DataFrame:
    """
    Complete pipeline: Download and parse UniProt gene descriptions.
    
    Args:
        download_dir: Where to save/load files
        force_download: Re-download even if files exist
        
    Returns:
        DataFrame with gene descriptions
    """
    if download_dir is None:
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "raw" / "uniprot"
    
    # Download if needed
    if force_download or not (download_dir / 'uniprot_human_swissprot.tsv.gz').exists():
        filepath = download_uniprot_human_proteins(download_dir)
        if not filepath:
            return pd.DataFrame()
    else:
        filepath = download_dir / 'uniprot_human_swissprot.tsv.gz'
    
    # Parse descriptions
    descriptions_df = parse_uniprot_descriptions(filepath)
    
    return descriptions_df


def get_reactome_gene_pathways(download_dir: Optional[Path] = None,
                               force_download: bool = False) -> pd.DataFrame:
    """
    Complete pipeline: Download and parse Reactome pathways.
    
    Args:
        download_dir: Where to save/load files
        force_download: Re-download even if files exist
        
    Returns:
        DataFrame with gene-pathway associations
    """
    if download_dir is None:
        project_root = Path(__file__).parent.parent.parent
        download_dir = project_root / "data" / "raw" / "reactome"
    
    # Download if needed
    required_files = [
        'UniProt2Reactome_All_Levels.txt',
        'ReactomePathways.txt'
    ]
    
    if force_download or not all((download_dir / f).exists() for f in required_files):
        downloaded = download_reactome_pathways(download_dir)
        if not downloaded:
            return pd.DataFrame()
    
    # Parse pathways
    uniprot2reactome = download_dir / 'UniProt2Reactome_All_Levels.txt'
    pathway_names = download_dir / 'ReactomePathways.txt'
    
    if not uniprot2reactome.exists() or not pathway_names.exists():
        logger.error("Required Reactome files not found")
        return pd.DataFrame()
    
    pathways_df = parse_reactome_pathways(uniprot2reactome, pathway_names)
    
    return pathways_df


def create_pathway_enrichment_validator(graph_gene_pathways: pd.DataFrame,
                                        background_pathways: pd.DataFrame) -> Dict:
    """
    Create a pathway enrichment validator for predicted gene lists.
    
    Given a list of predicted genes, compute enriched pathways using
    hypergeometric test.
    
    Args:
        graph_gene_pathways: Gene-pathway mappings for genes in your graph
        background_pathways: All gene-pathway mappings (background)
        
    Returns:
        Dict with pathway enrichment functions
    """
    # This is a placeholder for a more complete implementation
    # You would use scipy.stats.hypergeom for enrichment testing
    
    logger.info("Pathway enrichment validator created")
    logger.info("For full implementation, use scipy.stats.hypergeom or GSEApy")
    
    return {
        'gene_pathways': graph_gene_pathways,
        'background': background_pathways,
        'method': 'hypergeometric_test'
    }


# Example usage functions
def integrate_uniprot_into_graph(gene_list: List[str],
                                descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add UniProt descriptions as node features for genes in graph.
    
    Args:
        gene_list: List of gene symbols in your graph
        descriptions_df: From get_uniprot_gene_descriptions
        
    Returns:
        DataFrame with gene features
    """
    # Filter to genes in graph
    gene_features = descriptions_df[descriptions_df['gene_symbol'].isin(gene_list)].copy()
    
    logger.info(f"Matched {len(gene_features)}/{len(gene_list)} genes with UniProt descriptions")
    
    return gene_features


def integrate_pathways_into_graph(gene_list: List[str],
                                  pathways_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pathway edges for genes in graph.
    
    Args:
        gene_list: List of gene symbols in your graph
        pathways_df: From get_reactome_gene_pathways
        
    Returns:
        DataFrame with gene-pathway edges for graph genes
    """
    # This would require mapping UniProt IDs to gene symbols first
    # For now, return empty DataFrame as placeholder
    
    logger.warning("Pathway integration requires UniProt ID mapping")
    return pd.DataFrame()
