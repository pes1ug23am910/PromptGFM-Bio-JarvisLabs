# Deployment Guide

Updated: 2026-04-01

This project is currently deployed as reproducible training/evaluation jobs, not as a production API service.

## Primary Deployment Path (Workstation)

- Notebook: notebooks/workstation_training_v7_fixed_v5_Clean_Completed.ipynb
- Config: configs/workstation_config.yaml
- Checkpoint path: checkpoints/promptgfm_film/
- Results path: results/evaluation_results.json

## CLI Deployment Path

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess_all.py
python scripts/train.py --config configs/workstation_config.yaml
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## Cloud Deployment Notes

For cloud environments, see ../CLOUD_TRAINING_GUIDE.md.

## Persistence Requirements

Always persist these paths between sessions:

- checkpoints/
- results/
- configs/workstation_config.yaml

## Current Benchmark Reference

See ../LATEST_EVALUATION_SUMMARY.md.
