# Notebook Re-Run Guide (Remote GPU)

Notebook path:
- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

This guide tells you exactly what to run after uploading the changed files.

## 1) Before opening notebook

From repo root:

```powershell
cd M:/Extra/PromptGMF-Bio
. ./.venv/Scripts/Activate.ps1
```

## 2) Re-upload these files first

- src/data/preprocess.py
- src/data/download.py
- add_string_ppi_edges.py
- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

## 3) Which cells to re-run

Important: this notebook depends on variables defined in earlier cells (like PROJECT_ROOT, DATA_DIR, SCRIPTS_DIR), so do not run Cell 23 alone in a fresh kernel.

### Full safe re-run (recommended after reconnect/new kernel)
Run these code cells in order:
- Cell 3
- Cell 5
- Cell 7
- Cell 9
- Cell 11
- Cell 16
- Cell 18
- Cell 20
- Cell 22
- Cell 23
- Cell 26
- Cell 28
- Cell 30
- Cell 31

Then optional:
- Cell 33 (post-training GPU check)
- Cell 38 (evaluation)

### Quick re-run (same active kernel, already initialized)
If your current kernel already ran setup and path cells in this session, run:
- Cell 20 (download data, ensures latest assets)
- Cell 23 (preprocess graph + auto-check/fallback patch for missing gene-gene edges)
- Cell 31 (train)

Then optional:
- Cell 38 (evaluation)

## 4) What changed in behavior

Cell 23 now does the following automatically:
- Rebuilds graph with --force when an old graph exists (unless RESUME_DATA skip condition applies).
- Verifies gene-gene PPI edges exist after preprocessing.
- If missing, runs add_string_ppi_edges.py to patch the graph.
- Re-validates that gene-gene edges are present before continuing.

## 5) Expected success signal before training

In Cell 23 output, you should see either:
- Gene-gene edges found: ...

or
- Repaired gene-gene edges: ...

If neither appears and the cell errors, stop and fix before running Cell 31.


### Validated Workstation Environment Details (v6)

- **Notebook filename:** TEF_MP_New_workstation_training_v7_fixed_v6.ipynb
- **PROJECT_ROOT:** /home/mluser/projects_yash/new_project/PromptGFM-Bio
- **Hardware:** Intel i9-14900K, 128GB RAM, RTX 4090 (24GB VRAM), CUDA 13.0
- **Secrets:** Loaded from .env (GITHUB_TOKEN + WANDB_API_KEY)
- **Features:** VRAM-aware batch sizing (auto-detected)
- **Warning:** 5-day data retention window on workstation; backup required.
