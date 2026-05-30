# PromptGFM-Bio Training Guide


### Validated Workstation Environment Details (v6)

- **Notebook filename:** TEF_MP_New_workstation_training_v7_fixed_v6.ipynb
- **PROJECT_ROOT:** /home/mluser/projects_yash/new_project/PromptGFM-Bio
- **Hardware:** Intel i9-14900K, 128GB RAM, RTX 4090 (24GB VRAM), CUDA 13.0
- **Secrets:** Loaded from .env (GITHUB_TOKEN + WANDB_API_KEY)
- **Features:** VRAM-aware batch sizing (auto-detected)
- **Warning:** 5-day data retention window on workstation; backup required.


## 1. Quick Start

Updated: 2026-04-01

## Fastest Path (Notebook)

1. Open notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb.
2. Run all cells in order.
3. Confirm generated config: configs/workstation_config.yaml.
4. Confirm best checkpoint: checkpoints/promptgfm_film/best_model.pt.
5. Confirm evaluation file: results/evaluation_results.json.

## CLI Path

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess_all.py
python scripts/train.py --config configs/workstation_config.yaml
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## Baseline Metrics to Match

From results/evaluation_results.json:

- AUROC: 0.8130
- AUPR: 0.4618
- Precision@10: 1.0000
- NDCG@100: 0.9286

## If You Need Resume

```bash
python scripts/resume_training.py --interactive
```

or

```bash
python scripts/train.py \
  --config configs/workstation_config.yaml \
  --resume-checkpoint checkpoints/promptgfm_film/checkpoint_epoch_61.pt
```

## Reference

- LATEST_EVALUATION_SUMMARY.md
- TRAINING_GUIDE.md




---

## 2. Preprocessing Guide

This guide reflects the current preprocessing code path in scripts/preprocess_all.py and src/data/preprocess.py.

## 1. What Preprocessing Does

The preprocessing pipeline builds a PyTorch Geometric HeteroData graph for training.

High-level steps:

1. Parse PPI sources (BioGRID and STRING) when available.
2. Build gene-disease edges using enhanced methods:
   - HPO bridge (primary)
   - Orphadata merge (optional, enabled by default)
   - DisGeNET path exists but is not the default operational path.
3. Parse HPO phenotype files when available.
4. Build and save data/processed/biomedical_graph.pt and graph stats.

## 2. Supported Commands

Default run:

```bash
python scripts/preprocess_all.py
```

Force rebuild:

```bash
python scripts/preprocess_all.py --force
```

Flags supported by the current script:

```bash
python scripts/preprocess_all.py --no-hpo-bridge
python scripts/preprocess_all.py --no-orphadata
python scripts/preprocess_all.py --with-uniprot
python scripts/preprocess_all.py --with-pathways
```

## 3. Output Files

Expected under data/processed/:

- biomedical_graph.pt
- biomedical_graph_stats.txt
- hpo_gene_disease_edges.csv
- merged_gene_disease_edges.csv (when merge outputs are produced)

Optional outputs when enabled:

- uniprot_gene_descriptions.csv
- reactome_gene_pathways.csv

## 4. Sanity Checks

Check graph edge types:

```bash
python -c "import torch; g=torch.load('data/processed/biomedical_graph.pt'); print(g.edge_types)"
```

Check edge counts per type:

```bash
python -c "import torch; g=torch.load('data/processed/biomedical_graph.pt'); print({str(et): int(g[et].edge_index.shape[1]) for et in g.edge_types})"
```

## 5. Important Current Limitation

In the validated workstation run, training logs show no gene-gene edges were used for message passing.
That means the validated baseline was trained without active PPI message passing, even though the code supports it when those edges exist.

## 6. Notes on Data Splits

- Train/val/test split CSV files are not produced by scripts/preprocess_all.py.
- Splits are generated at runtime by dataset loading logic in training/evaluation.

## 7. Removed Stale Guidance

This guide intentionally avoids outdated assumptions such as:

- DisGeNET-only default graph construction.
- Guaranteed disease-phenotype edge presence in every processed graph.
- Guaranteed gene-gene edge presence in every run.

## 8. Related Docs

- DATA_DOWNLOAD_GUIDE.md
- TRAINING_GUIDE.md
- docs/TROUBLESHOOTING.md


---

## 3. Full Training Reference

This guide documents the currently validated training and evaluation workflow.

## 1. Validated Path

Primary validated artifact:

- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

Primary validated config:

- configs/workstation_config.yaml

Primary validated checkpoint:

- checkpoints/promptgfm_film/best_model.pt

## 2. Supported Training Commands

Train:

```bash
python scripts/train.py --config configs/workstation_config.yaml
```

Resume:

```bash
python scripts/train.py --config configs/workstation_config.yaml --resume-checkpoint checkpoints/promptgfm_film/checkpoint_epoch_61.pt
```

Or use helper:

```bash
python scripts/resume_training.py --interactive
```

## 3. Supported Evaluation Commands

Main evaluation:

```bash
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

Stratified evaluation:

```bash
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt --stratified
```

Note on few-shot flag:

- scripts/evaluate.py currently parses --few-shot, but there is no active execution branch for it in main().
- Treat few-shot CLI output as unsupported until implemented.

## 4. Current Baseline Metrics

Canonical source:

- LATEST_EVALUATION_SUMMARY.md (mirrors results/evaluation_results.json)

This file intentionally does not duplicate numeric metric values.

## 5. Plain-Language Interpretation

- The model is good at ranking many true positives near the top (high precision@K).
- Recall remains low because K is small compared with the size of the candidate set.
- This baseline is usable for top-rank screening, but it is not high-coverage retrieval yet.

## 6. Validated Run Limitations

- No gene-gene message passing in the validated run:
  notebook logs show no gene-gene edges were available during training.
- Baseline constructor/signature mismatch:
  baseline model construction path currently has parameter/signature mismatch risk.
- Cross-attention shape risk:
  current PromptGFM forward path is not shape-safe for cross-attention mode.
- Runtime vs pinned dependency drift:
  validated runtime used torch 2.6.0+cu124, while requirements.txt pins torch 2.1.0.

## 7. Config Status

- Validated:
  - configs/workstation_config.yaml (GraphSAGE + FiLM)
- Experimental:
  - configs/baseline_config.yaml
  - configs/cross_attention_config.yaml
  - configs/kaggle_config.yaml

## 8. Recommended Next Checks Before New Experiments

1. Confirm graph edge types before training (especially gene-gene edges).
2. Keep config and checkpoint paired (same run profile).
3. Record runtime versions with each result artifact.

## 9. Related Docs

- LATEST_EVALUATION_SUMMARY.md
- PREPROCESSING_GUIDE.md
- docs/TROUBLESHOOTING.md




---

## 4. Optimization Guide

Updated: 2026-04-01

This guide defines optimization work after the current baseline run.

Baseline reference:

- LATEST_EVALUATION_SUMMARY.md
- AUROC: 0.8130
- AUPR: 0.4618

## Priority 1: Graph Message Passing Coverage

Observation from notebook logs:

- No gene-gene edges were used for message passing in the validated run.

Action:

1. Rebuild/verify graph with gene-gene interaction edges.
2. Confirm train.py consumes those edges.
3. Re-run same protocol and compare metrics.

Expected impact:

- Potential improvement in ranking quality over current baseline.

## Priority 2: Controlled Config Sweep

Run one-factor sweeps from configs/workstation_config.yaml:

- batch_size and grad accumulation
- learning_rate
- hidden_dim
- conditioning_type

Record per run:

- AUROC, AUPR, Precision@K, NDCG@K
- runtime per epoch
- GPU memory behavior

## Priority 3: Evaluation Depth

Add consistent exports for:

- stratified rarity metrics
- few-shot metrics
- confidence intervals across seeds

## Priority 4: Stability and Throughput

- Keep mixed precision enabled on CUDA.
- Use checkpoint resume for long experiments.
- Track run metadata (config hash + checkpoint path + seed).

## Suggested Milestone Targets

- M1: Reproduce AUROC 0.8130 in clean rerun.
- M2: Beat AUROC 0.82 under same test protocol.
- M3: Publish full stratified/few-shot tables for paper.


---

## 5. Resume Guide

Updated: 2026-04-01

This guide documents checkpoint resume with the current workstation configuration.

## 1. Checkpoint Location

Current run checkpoint directory:

- checkpoints/promptgfm_film/

Typical files:

- best_model.pt
- checkpoint_epoch_<N>.pt

## 2. Resume Options

### Option A: Interactive helper

```bash
python scripts/resume_training.py --interactive
```

### Option B: Resume from explicit checkpoint

```bash
python scripts/train.py \
  --config configs/workstation_config.yaml \
  --resume-checkpoint checkpoints/promptgfm_film/checkpoint_epoch_61.pt
```

### Option C: Use helper with explicit checkpoint

```bash
python scripts/resume_training.py \
  --config configs/workstation_config.yaml \
  --checkpoint checkpoints/promptgfm_film/checkpoint_epoch_61.pt
```

## 3. Validate Resume Result

After training resumes/completes:

```bash
python scripts/evaluate.py \
  --config configs/workstation_config.yaml \
  --checkpoint checkpoints/promptgfm_film/best_model.pt
```

Check:

- results/evaluation_results.json

## 4. Troubleshooting

- File not found:
  - Verify checkpoint path and filename.
- Config mismatch:
  - Use configs/workstation_config.yaml for workstation checkpoints.
- CUDA/device mismatch:
  - Re-run with correct environment and GPU visibility.

## 5. Current Reference Run

Latest known run resumed from checkpoint_epoch_61.pt and early-stopped at epoch 62.
See LATEST_EVALUATION_SUMMARY.md for exact metrics.


---

## 6. Cloud Training

Updated: 2026-04-01

This guide maps the workstation flow to cloud environments.

## Recommended Cloud Options

| Platform | Typical GPU | VRAM | Best Use |
|---|---|---:|---|
| Kaggle Notebooks | T4 or P100 | 16 GB | Free training runs |
| Colab Pro | A100/L4/T4 | 16-40 GB | Faster experiments |
| Lightning AI | T4 | 16 GB | VS Code style cloud workflow |
| GitHub Codespaces | CPU | - | Editing/debugging only |

## Migration from Workstation Notebook

Use the same high-level sequence as notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb:

1. Setup project root and secrets
2. Install deps
3. Download data
4. Preprocess graph
5. Train with cloud-appropriate config
6. Evaluate and export results JSON

## Config Notes

- For 16 GB GPUs, start from configs/kaggle_config.yaml.
- For high-VRAM cloud GPUs, prefer workstation-style larger batch settings.
- Keep checkpoint directory under persistent storage.

## Cloud CLI Example

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess_all.py
python scripts/train.py --config configs/kaggle_config.yaml
python scripts/evaluate.py --config configs/kaggle_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## Persistence

Always persist:

- checkpoints/
- results/
- generated config files

before instance/session termination.


---

## 7. GPU Training

Updated: 2026-04-01

## Target Hardware Profile

Validated run profile:

- GPU: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- Driver/CUDA (system): 580.65.06 / 13.0
- PyTorch runtime: 2.6.0+cu124

## VRAM-Aware Batch Sizing

The workstation notebook generates configs/workstation_config.yaml from free VRAM.

Current policy used in notebook:

| Free VRAM | batch_size | grad_accum | effective_batch | workers |
|---|---:|---:|---:|---:|
| >= 20 GB | 768 | 1 | 768 | 8 |
| 16-20 GB | 512 | 1 | 512 | 6 |
| 12-16 GB | 384 | 1 | 384 | 4 |
| 8-12 GB | 256 | 1 | 256 | 4 |
| 5-8 GB | 128 | 2 | 256 | 2 |
| < 5 GB | 64 | 4 | 256 | 2 |

## Commands

Check GPU:

```bash
nvidia-smi
```

Train:

```bash
python scripts/train.py --config configs/workstation_config.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## Notes

- The notebook path is preferred because it auto-tunes config for current VRAM.
- Keep a safety buffer in VRAM to avoid OOM during long runs.
- Use checkpoint resume for interrupted sessions.
