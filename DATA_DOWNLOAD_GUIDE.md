# Data Download Guide

This guide documents the current downloader behavior in scripts/download_data.py and src/data/download.py.

## 1. Supported CLI

Download all supported datasets:

```bash
python scripts/download_data.py --dataset all
```

Download one dataset:

```bash
python scripts/download_data.py --dataset biogrid
python scripts/download_data.py --dataset string
python scripts/download_data.py --dataset disgenet
python scripts/download_data.py --dataset hpo
```

Force re-download:

```bash
python scripts/download_data.py --dataset all --force
```

## 2. Dataset Scope

Current downloader supports:

- BioGRID
- STRING
- DisGeNET (may require manual fallback)
- HPO

Current downloader does not include Orphadata download in this script.
Orphadata is handled by preprocessing integration logic.

## 3. Expected Raw Directory Layout

After successful downloads:

```text
data/raw/
  biogrid/
  string/
  disgenet/
  hpo/
```

## 4. Known Download Caveats

- DisGeNET public download may fail or return non-data content depending on upstream access/auth changes.
- If DisGeNET fails, the current project can still run the validated path through HPO bridge based gene-disease edges.

## 5. Unsupported or Removed Command Patterns

Do not use stale patterns such as:

- --datasets (plural)
- --sample
- --test-mode

These are not supported by scripts/download_data.py.

## 6. Next Step

Run preprocessing after downloads:

```bash
python scripts/preprocess_all.py
```

Then proceed to training/evaluation using TRAINING_GUIDE.md.
