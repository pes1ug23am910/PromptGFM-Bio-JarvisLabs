# PromptGFM-Bio

PromptGFM-Bio is a graph-and-text learning project for rare-disease gene ranking.

## Current Validated Run (Source of Truth)

Validated workflow artifact:

- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

Validated metrics artifacts:

- results/evaluation_results.json
- LATEST_EVALUATION_SUMMARY.md
- STATUS.md

Canonical metric source for live docs:

- LATEST_EVALUATION_SUMMARY.md
- STATUS.md

Do not treat this README as a second metric source of truth.

## Plain-Language Result Interpretation

- AUROC and AUPR show the model can separate many true associations from negatives better than random.
- Precision@K is high because many of the top-ranked genes are relevant.
- Recall@K is very low because the candidate space is large and top-K only covers a tiny fraction of all positives.
- In short: the top of the ranking is often useful, but coverage is still limited.

## Validated Run Limitations (Important)

- No gene-gene message passing in the validated run:
  training logs show gene-gene edges were not present, so training proceeded without PPI message passing.
- Baseline constructor/signature mismatch:
  baseline code path is currently not reliable due argument/signature mismatch in model construction.
- Cross-attention shape risk:
  cross-attention path is not validated and has tensor-shape mismatch risk in current forward flow.
- Runtime vs pinned dependency drift:
  validated workstation run used torch 2.6.0+cu124 while requirements.txt pins torch 2.1.0.

## Recommended Workflow

### Option A: Workstation Notebook (Recommended)

1. Open notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb.
2. Run cells top to bottom.
3. Confirm generated config: configs/workstation_config.yaml.
4. Confirm checkpoint: checkpoints/promptgfm_film/best_model.pt.
5. Confirm metrics output: results/evaluation_results.json.

### Option B: CLI

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess_all.py
python scripts/train.py --config configs/workstation_config.yaml
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## Config Status

- Validated:
  - configs/workstation_config.yaml (GraphSAGE + FiLM)
- Experimental (not currently validated end-to-end):
  - configs/baseline_config.yaml
  - configs/cross_attention_config.yaml
  - configs/kaggle_config.yaml

## Live Documentation

- SETUP.md
- QUICKSTART.md
- TRAINING_GUIDE.md
- DATA_DOWNLOAD_GUIDE.md
- docs/TROUBLESHOOTING.md
- LATEST_EVALUATION_SUMMARY.md
- STATUS.md

### Validated Workstation Environment Details (v6)

- **Notebook filename:** TEF_MP_New_workstation_training_v7_fixed_v6.ipynb
- **PROJECT_ROOT:** /home/mluser/projects_yash/new_project/PromptGFM-Bio
- **Hardware:** Intel i9-14900K, 128GB RAM, RTX 4090 (24GB VRAM), CUDA 13.0
- **Secrets:** Loaded from .env (GITHUB_TOKEN + WANDB_API_KEY)
- **Features:** VRAM-aware batch sizing (auto-detected)
- **Warning:** 5-day data retention window on workstation; backup required.
