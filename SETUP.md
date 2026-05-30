# Setup Guide

This guide is aligned with the current validated workstation run.

## 1. Create Environment

Recommended Python: 3.10+

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

## 2. Install Project Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## 3. Choose Runtime Profile (Important)

There is dependency drift between pinned requirements and the validated workstation runtime:

- requirements.txt pins torch==2.1.0
- validated workstation run used torch 2.6.0+cu124

If you want to reproduce the validated workstation runtime, install this stack after step 2:

```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade --force-reinstall -f https://data.pyg.org/whl/torch-2.6.0+cu124.html torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

If you are on CPU only:

```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade --force-reinstall torch-geometric
```

## 4. Verify Installation

```bash
python -c "import torch, torch_geometric; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('pyg', torch_geometric.__version__)"
python scripts/verify_setup.py
```

## 5. Configure Optional Secrets

Create .env in repo root if needed:

```bash
GITHUB_TOKEN=ghp_example_personal_access_token
WANDB_API_KEY=wandb_example_api_key
```

## 6. First Run (Validated Path)

Preferred:

- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

CLI alternative:

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess_all.py
python scripts/train.py --config configs/workstation_config.yaml
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## 7. Known Limitations

- Validated run had no gene-gene message passing.
- Baseline config path currently has constructor/signature mismatch risk.
- Cross-attention path currently has tensor-shape mismatch risk.
- Runtime stack can drift from requirements pins unless you select one profile explicitly.

## 8. References

- QUICKSTART.md
- TRAINING_GUIDE.md
- PREPROCESSING_GUIDE.md
- DATA_DOWNLOAD_GUIDE.md
- docs/TROUBLESHOOTING.md
- LATEST_EVALUATION_SUMMARY.md


### Validated Workstation Environment Details (v6)

- **Notebook filename:** TEF_MP_New_workstation_training_v7_fixed_v6.ipynb
- **PROJECT_ROOT:** /home/mluser/projects_yash/new_project/PromptGFM-Bio
- **Hardware:** Intel i9-14900K, 128GB RAM, RTX 4090 (24GB VRAM), CUDA 13.0
- **Secrets:** Loaded from .env (GITHUB_TOKEN + WANDB_API_KEY)
- **Features:** VRAM-aware batch sizing (auto-detected)
- **Warning:** 5-day data retention window on workstation; backup required.
