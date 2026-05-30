# Quick Start

This quick start follows the currently validated workstation flow.

## 1. Complete Setup

Follow SETUP.md first.

## 2. Run the Validated Notebook Path

Open and run:

- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

Run all cells in order.

## 3. Verify Artifacts

After training and evaluation, confirm:

- configs/workstation_config.yaml
- checkpoints/promptgfm_film/best_model.pt
- results/evaluation_results.json

## 4. Optional CLI Re-run

```bash
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## 5. Expected Baseline Metrics

Canonical source:

- LATEST_EVALUATION_SUMMARY.md (mirrors results/evaluation_results.json)

This quickstart intentionally avoids duplicating numeric metric values.

## 6. Plain-Language Metric Read

- High precision@K means many top-ranked genes are relevant.
- Low recall@K means top-K still covers only a small fraction of all positives.
- This is expected in large-candidate ranking settings and indicates room to improve coverage.

## 7. Current Run Limitations

- No gene-gene message passing was active in the validated run.
- Baseline training path has constructor/signature mismatch risk.
- Cross-attention path has tensor-shape mismatch risk.
- Runtime dependency versions can drift from requirements pins.

## 8. Next References

- TRAINING_GUIDE.md
- PREPROCESSING_GUIDE.md
- docs/TROUBLESHOOTING.md
- LATEST_EVALUATION_SUMMARY.md


### Validated Workstation Environment Details (v6)

- **Notebook filename:** TEF_MP_New_workstation_training_v7_fixed_v6.ipynb
- **PROJECT_ROOT:** /home/mluser/projects_yash/new_project/PromptGFM-Bio
- **Hardware:** Intel i9-14900K, 128GB RAM, RTX 4090 (24GB VRAM), CUDA 13.0
- **Secrets:** Loaded from .env (GITHUB_TOKEN + WANDB_API_KEY)
- **Features:** VRAM-aware batch sizing (auto-detected)
- **Warning:** 5-day data retention window on workstation; backup required.
