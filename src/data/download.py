"""
Data download module for PromptGFM-Bio.

This module handles downloading biomedical datasets including:
- BioGRID protein-protein interactions
- STRING database PPI
- DisGeNET gene-disease associations
- Human Phenotype Ontology (HPO)
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import logging
import hashlib
import time
from typing import Optional, Dict
import zipfile
import gzip
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_data_dir() -> Path:
    """Get the data directory path, creating it if necessary."""
    # Get project root (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_file_with_progress(url: str, output_path: Path, 
                                  max_retries: int = 3, 
                                  timeout: int = 300) -> bool:
    """
    Download a file with progress bar and retry logic.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for the request
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading from {url} (attempt {attempt + 1}/{max_retries})")
            
            # Stream the download
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded to {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
    
    return False


def _verify_checksum(file_path: Path, expected_hash: Optional[str] = None, 
                     algorithm: str = 'md5') -> bool:
    """
    Verify file integrity using checksum.
    
    Args:
        file_path: Path to the file to verify
        expected_hash: Expected hash value (if None, only compute and log)
        algorithm: Hash algorithm to use ('md5' or 'sha256')
        
    Returns:
        True if verification passes or no expected_hash provided
    """
    if not file_path.exists():
        return False
    
    hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    computed_hash = hash_func.hexdigest()
    logger.info(f"File {file_path.name} {algorithm}: {computed_hash}")
    
    if expected_hash:
        if computed_hash == expected_hash:
            logger.info("Checksum verification passed!")
            return True
        else:
            logger.error(f"Checksum mismatch! Expected {expected_hash}, got {computed_hash}")
            return False
    
    return True


def _extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract zip or gz archives.
    
    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            logger.info(f"Extracting {archive_path.name}...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info("Extraction complete!")
            
        elif archive_path.suffix == '.gz':
            logger.info(f"Extracting {archive_path.name}...")
            output_file = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted to {output_file}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def download_biogrid(force: bool = False) -> Dict[str, Path]:
    """
    Download BioGRID protein-protein interaction database.
    
    BioGRID is a biomedical interaction repository containing protein and 
    genetic interactions from major model organisms.
    
    Args:
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    biogrid_dir = data_dir / "biogrid"
    biogrid_dir.mkdir(parents=True, exist_ok=True)
    
    # BioGRID latest release (tab-delimited format)
    # Note: This URL may need updating - check https://downloads.thebiogrid.org/
    url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.224/BIOGRID-ALL-4.4.224.tab3.zip"
    filename = "BIOGRID-ALL-4.4.224.tab3.zip"
    output_path = biogrid_dir / filename
    
    if output_path.exists() and not force:
        logger.info(f"BioGRID file already exists at {output_path}")
        logger.info("Use force=True to re-download")
        return {"biogrid_zip": output_path}
    
    logger.info("Downloading BioGRID database (~500MB)...")
    success = _download_file_with_progress(url, output_path)
    
    if success:
        _verify_checksum(output_path)
        # Extract the archive
        _extract_archive(output_path, biogrid_dir)
        return {"biogrid_zip": output_path, "biogrid_dir": biogrid_dir}
    
    return {}


def download_string(organism: str = "9606", score_threshold: int = 400, 
                    force: bool = False) -> Dict[str, Path]:
    """
    Download STRING protein-protein interaction network.
    
    STRING is a database of known and predicted protein-protein interactions.
    
    Args:
        organism: NCBI taxonomy ID (9606 = Homo sapiens)
        score_threshold: Minimum confidence score (0-1000, default 400 = medium confidence)
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    string_dir = data_dir / "string"
    string_dir.mkdir(parents=True, exist_ok=True)
    
    # STRING v11.5 protein links (physical interactions)
    base_url = "https://stringdb-downloads.org/download"
    version = "v12.0"
    filename = f"{organism}.protein.links.{version}.txt.gz"
    url = f"{base_url}/protein.links.{version}/{filename}"
    info_filename = f"{organism}.protein.info.{version}.txt.gz"
    info_url = f"{base_url}/protein.info.{version}/{info_filename}"
    
    output_path = string_dir / filename
    info_path = string_dir / info_filename

    def _ensure_info_file() -> bool:
        if info_path.exists() and not force:
            return True

        logger.info("Downloading STRING protein info for gene name mapping...")
        success_info = _download_file_with_progress(info_url, info_path)
        if success_info:
            _extract_archive(info_path, string_dir)
            return True

        logger.warning("Failed to download STRING protein info file")
        return False
    
    if output_path.exists() and not force:
        logger.info(f"STRING file already exists at {output_path}")
        _ensure_info_file()
        logger.info("Use force=True to re-download")
        return {
            "string_links": output_path,
            "string_info": info_path if info_path.exists() else None,
            "string_dir": string_dir
        }
    
    logger.info(f"Downloading STRING database for organism {organism} (~700MB)...")
    success = _download_file_with_progress(url, output_path)
    
    if success:
        _verify_checksum(output_path)
        # Extract the gz file
        _extract_archive(output_path, string_dir)
        _ensure_info_file()
        
        return {
            "string_links": output_path,
            "string_info": info_path if info_path.exists() else None,
            "string_dir": string_dir
        }
    
    return {}


def download_disgenet(version: str = "v7.0", force: bool = False) -> Dict[str, Path]:
    """
    Download DisGeNET gene-disease associations.
    
    DisGeNET is a comprehensive platform integrating information on gene-disease
    associations from various expert curated databases and text-mining.
    
    NOTE: DisGeNET requires registration and authentication for some datasets.
    This function downloads the publicly available core dataset.
    For full access, register at: https://www.disgenet.org/signup/
    
    Args:
        version: DisGeNET version to download
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    disgenet_dir = data_dir / "disgenet"
    disgenet_dir.mkdir(parents=True, exist_ok=True)
    
    # Public DisGeNET curated gene-disease associations
    # Note: For private/full data, you need authentication
    base_url = "https://www.disgenet.org/static/disgenet_ap1/files/downloads"
    filename = "curated_gene_disease_associations.tsv.gz"
    url = f"{base_url}/{filename}"
    
    output_path = disgenet_dir / filename
    
    if output_path.exists() and not force:
        logger.info(f"DisGeNET file already exists at {output_path}")
        logger.info("Use force=True to re-download")
        return {"disgenet_gz": output_path}
    
    logger.info("Downloading DisGeNET curated associations (~300MB)...")
    logger.info("Note: This is the public dataset. For full access, register at disgenet.org")
    
    success = _download_file_with_progress(url, output_path)
    
    if success:
        _verify_checksum(output_path)
        _extract_archive(output_path, disgenet_dir)
        return {"disgenet_gz": output_path, "disgenet_dir": disgenet_dir}
    else:
        # Provide fallback message
        logger.warning("Public DisGeNET download may require authentication.")
        logger.info("Alternative: Download manually from https://www.disgenet.org/downloads")
        logger.info(f"Save the file to: {disgenet_dir}")
    
    return {}


def download_hpo(force: bool = False) -> Dict[str, Path]:
    """
    Download Human Phenotype Ontology (HPO) annotations.
    
    HPO provides a standardized vocabulary of phenotypic abnormalities
    encountered in human disease.
    
    Args:
        force: If True, re-download even if file exists
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_dir = _get_data_dir()
    hpo_dir = data_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    
    # HPO annotations
    files_to_download = {
        "phenotype_to_genes": "http://purl.obolibrary.org/obo/hp/hpoa/phenotype_to_genes.txt",
        "genes_to_phenotype": "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
        "phenotype_annotations": "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
    }
    
    downloaded_files = {}
    
    for file_key, url in files_to_download.items():
        filename = url.split('/')[-1]
        output_path = hpo_dir / filename
        
        if output_path.exists() and not force:
            logger.info(f"HPO file {filename} already exists")
            downloaded_files[file_key] = output_path
            continue
        
        logger.info(f"Downloading HPO {file_key}...")
        success = _download_file_with_progress(url, output_path)
        
        if success:
            downloaded_files[file_key] = output_path
            _verify_checksum(output_path)
    
    downloaded_files["hpo_dir"] = hpo_dir
    return downloaded_files


def download_all(force: bool = False, skip_failing: bool = True) -> Dict[str, Dict[str, Path]]:
    """
    Download all biomedical datasets.
    
    Args:
        force: If True, re-download even if files exist
        skip_failing: If True, continue even if some downloads fail
        
    Returns:
        Dictionary mapping dataset names to their download results
    """
    logger.info("="*70)
    logger.info("Starting download of all biomedical datasets...")
    logger.info("This may take 30-60 minutes depending on your connection")
    logger.info("Total size: ~1.5 GB")
    logger.info("="*70)
    
    results = {}
    
    # Download BioGRID
    try:
        logger.info("\n[1/4] BioGRID Protein-Protein Interactions")
        results['biogrid'] = download_biogrid(force=force)
    except Exception as e:
        logger.error(f"BioGRID download failed: {e}")
        if not skip_failing:
            raise
        results['biogrid'] = {}
    
    # Download STRING
    try:
        logger.info("\n[2/4] STRING Protein Network")
        results['string'] = download_string(force=force)
    except Exception as e:
        logger.error(f"STRING download failed: {e}")
        if not skip_failing:
            raise
        results['string'] = {}
    
    # Download DisGeNET
    try:
        logger.info("\n[3/4] DisGeNET Gene-Disease Associations")
        results['disgenet'] = download_disgenet(force=force)
    except Exception as e:
        logger.error(f"DisGeNET download failed: {e}")
        if not skip_failing:
            raise
        results['disgenet'] = {}
    
    # Download HPO
    try:
        logger.info("\n[4/4] Human Phenotype Ontology")
        results['hpo'] = download_hpo(force=force)
    except Exception as e:
        logger.error(f"HPO download failed: {e}")
        if not skip_failing:
            raise
        results['hpo'] = {}
    
    logger.info("\n" + "="*70)
    logger.info("Download Summary:")
    for dataset, files in results.items():
        if files:
            logger.info(f"✓ {dataset.upper()}: {len(files)} files downloaded")
        else:
            logger.warning(f"✗ {dataset.upper()}: Download failed or incomplete")
    logger.info("="*70)
    
    return results


if __name__ == "__main__":
    """Command-line interface for data download."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download biomedical datasets for PromptGFM-Bio")
    parser.add_argument('--dataset', type=str, choices=['all', 'biogrid', 'string', 'disgenet', 'hpo'],
                       default='all', help='Which dataset to download')
    parser.add_argument('--force', action='store_true', help='Re-download even if files exist')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        download_all(force=args.force)
    elif args.dataset == 'biogrid':
        download_biogrid(force=args.force)
    elif args.dataset == 'string':
        download_string(force=args.force)
    elif args.dataset == 'disgenet':
        download_disgenet(force=args.force)
    elif args.dataset == 'hpo':
        download_hpo(force=args.force)


if __name__ == "__main__":
    download_all()
