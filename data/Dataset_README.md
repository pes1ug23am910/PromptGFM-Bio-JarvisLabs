# Data Directory

This directory contains the datasets used for training and evaluating PromptGFM-Bio.

**⚠️ Important**: Large data files are excluded from version control. You must download them separately.

---

## Directory Structure

```
data/
├── raw/              # Raw downloaded datasets (~10GB)
│   ├── string/       # STRING protein-protein interactions
│   ├── biogrid/      # BioGRID interaction data
│   ├── disgenet/     # DisGeNET gene-disease associations
│   ├── hpo/          # Human Phenotype Ontology
│   └── orphanet/     # Orphanet rare disease data
├── processed/        # Preprocessed data (~2GB)
│   ├── biomedical_graph.pt      # Main knowledge graph
│   ├── gene_disease_edges.csv   # Processed associations
│   └── graph_stats.txt          # Statistics
└── splits/           # Train/val/test splits
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## Downloading Data

### Automatic Download (Recommended)

Use the provided download script to fetch all required datasets:

```bash
# Download all datasets
python scripts/download_data.py --all

# Or download specific datasets
python scripts/download_data.py --datasets string biogrid disgenet hpo orphanet
```

### Manual Download

If automatic download fails, you can manually download from these sources:

#### 1. STRING Database
- **URL**: https://string-db.org/cgi/download
- **File**: `9606.protein.links.v12.0.txt.gz` (Human protein links)
- **Size**: ~4GB
- **Place in**: `data/raw/string/`

#### 2. BioGRID
- **URL**: https://downloads.thebiogrid.org/BioGRID/Release-Archive/
- **File**: `BIOGRID-ALL-4.4.224.tab3.txt` (or latest version)
- **Size**: ~500MB
- **Place in**: `data/raw/biogrid/`

#### 3. DisGeNET
- **URL**: https://www.disgenet.org/downloads
- **File**: `all_gene_disease_associations.tsv.gz`
- **Size**: ~200MB
- **Note**: Requires free registration
- **Place in**: `data/raw/disgenet/`

#### 4. Human Phenotype Ontology (HPO)
- **URL**: https://hpo.jax.org/app/download/annotation
- **Files**:
  - `genes_to_phenotype.txt`
  - `phenotype_to_genes.txt`
  - `phenotype.hpoa`
- **Size**: ~50MB
- **Place in**: `data/raw/hpo/`

#### 5. Orphanet
- **URL**: http://www.orphadata.org/cgi-bin/index.php
- **Files**:
  - `en_product1.xml` (Rare diseases and classifications)
  - `en_product4.xml` (Genes associated with rare diseases)
  - `en_product6.xml` (Epidemiological data)
- **Size**: ~100MB
- **Place in**: `data/raw/orphanet/`

---

## Preprocessing Data

After downloading raw data, preprocess it to build the knowledge graph:

```bash
# Run preprocessing pipeline
python scripts/preprocess_all.py

# This will:
# 1. Parse raw data files
# 2. Map gene/protein identifiers
# 3. Construct heterogeneous graph
# 4. Generate train/val/test splits
# 5. Save processed data to data/processed/
```

**Expected output**:
- `data/processed/biomedical_graph.pt` - PyTorch Geometric graph object
- `data/processed/merged_gene_disease_edges.csv` - Processed associations
- `data/processed/biomedical_graph_stats.txt` - Graph statistics

---

## Data Statistics

After preprocessing, you should have:

| Component | Count |
|-----------|-------|
| **Genes/Proteins** | ~20,000 nodes |
| **Diseases** | ~10,000 nodes |
| **Phenotypes** | ~15,000 nodes |
| **Protein-Protein Interactions** | ~9.7M edges |
| **Gene-Disease Associations** | ~87K edges |
| **Gene-Phenotype Mappings** | ~200K edges |

---

## Disk Space Requirements

- **Raw data**: ~10 GB
- **Processed data**: ~2 GB
- **Temporary files (during preprocessing)**: ~3 GB
- **Total**: ~15 GB

Ensure you have sufficient disk space before downloading.

---

## Data Versions

The current implementation uses:

- **STRING v12.0** (2021)
- **BioGRID 4.4.224** (2023)
- **DisGeNET v7.0** (2020)
- **HPO** (monthly releases)
- **Orphanet** (updated quarterly)

To update to newer versions:
1. Download new files to `data/raw/`
2. Update file paths in `src/data/download.py`
3. Re-run preprocessing

---

## Data License and Attribution

All datasets are publicly available but have specific licenses:

- **STRING**: Creative Commons BY 4.0
- **BioGRID**: MIT License
- **DisGeNET**: ODbL 1.0 (Open Database License)
- **HPO**: Free for research use
- **Orphanet**: Creative Commons BY 4.0

**Important**: If you use this data in publications, cite the original sources (see [LICENSE](../LICENSE)).

---

## Troubleshooting

### Download Failures

If downloads fail:
1. Check internet connection
2. Verify URLs are still valid
3. Try manual download (links above)
4. Check if registration is required (DisGeNET)

### Preprocessing Errors

Common issues:
- **Missing files**: Ensure all raw data is downloaded
- **Parsing errors**: Check file formats match expected versions
- **Memory errors**: Preprocessing requires ~16GB RAM
- **Disk space**: Ensure sufficient space for temporary files

See [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) for more help.

---

## Using Your Own Data

To integrate additional datasets:

1. **Add downloader**: Modify `src/data/download.py`
2. **Add parser**: Extend `src/data/preprocess.py`
3. **Update graph construction**: Modify `build_biomedical_graph()`
4. **Test**: Verify graph structure and statistics

Example:
```python
# In src/data/preprocess.py
def parse_custom_dataset(file_path):
    df = pd.read_csv(file_path)
    # Process and return edges
    return edges
```

---

## Data Privacy and Ethics

This project uses only publicly available biomedical data:
- No patient-level information
- No protected health information (PHI)
- De-identified, aggregated data only

**For clinical applications**: Additional validation and ethical review required. This software is for research purposes only.

---

For more information on data handling, see:
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Data pipeline details
- [SETUP.md](../SETUP.md) - Complete setup instructions
- [scripts/download_data.py](../scripts/download_data.py) - Download script source
