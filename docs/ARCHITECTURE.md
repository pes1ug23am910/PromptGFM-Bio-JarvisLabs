# Architecture Overview

Updated: 2026-04-01

## High-Level Pipeline

1. Data acquisition
   - scripts/download_data.py
2. Graph preprocessing
   - scripts/preprocess_all.py
   - output: data/processed/biomedical_graph.pt
3. Training
   - scripts/train.py with configs/workstation_config.yaml
4. Evaluation
   - scripts/evaluate.py
   - output: results/evaluation_results.json

## Core Modules

- src/data: download, preprocess, dataset logic
- src/models: prompt encoder, GNN backbone, conditioning, PromptGFM
- src/training: finetune loop, pretraining, losses
- src/evaluation: ranking metrics

## Current Validated Runtime Profile

- Config: configs/workstation_config.yaml
- Conditioning: FiLM
- GNN: GraphSAGE
- Prompt encoder: PubMedBERT

## Notes for Design Iteration

- The latest run used no gene-gene message passing edges; improving this is a primary architecture-level optimization target.
- Use LATEST_EVALUATION_SUMMARY.md for up-to-date benchmark reference.
