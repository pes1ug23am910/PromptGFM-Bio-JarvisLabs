"""
Rebuilds notebooks/kaggle_training.ipynb with full cross-session resume support.
Run from project root:  python scripts/rebuild_kaggle_notebook.py
"""
import json, pathlib, sys

OUT = pathlib.Path("notebooks/kaggle_training.ipynb")

def md(src): return {"cell_type":"markdown","id":None,"metadata":{},"source":[src]}
def code(lines): return {"cell_type":"code","id":None,"execution_count":None,"metadata":{},"outputs":[],"source":lines}

# ── helper: assign sequential ids ────────────────────────────────────────────
def build(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell{i:02d}abcd"
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{
                "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.10.0"},
                "kaggle":{"accelerator":"gpu","dataSources":[],"dockerImageVersionId":30761,
                          "isInternetEnabled":True,"language":"python","sourceType":"notebook"}
            },
            "cells":cells}

cells = []

# ── 0 · Title ─────────────────────────────────────────────────────────────────
cells.append(md("""\
# PromptGFM-Bio — Kaggle Training Notebook
**Gene-Phenotype Prediction for Rare Diseases**

### ✅ Resumable Across Sessions & Accounts
This notebook saves **three Kaggle Datasets** after training so any future session — \
or a different Kaggle account — can skip all expensive steps:

| Dataset name (you choose) | What it stores | Skips |
|---|---|---|
| `promptgfm-model-cache` | HuggingFace BioBERT weights | ~5 min download |
| `promptgfm-data` | Raw + processed graph | ~25 min download + preprocess |
| `promptgfm-checkpoints` | Per-epoch checkpoints | training from epoch 0 |

**Setup once → add as Dataset inputs on every future session.**

> ⚙️ Before running: Settings → Accelerator → **GPU T4 x2** · Internet → **On**\
"""))

# ── 1 · Env check ─────────────────────────────────────────────────────────────
cells.append(md("## 1. Environment Check"))
cells.append(code([
    "import sys, subprocess, os\n",
    "import torch\n",
    "\n",
    "print(f'Python  : {sys.version}')\n",
    "print(f'PyTorch : {torch.__version__}')\n",
    "print(f'CUDA    : {torch.version.cuda}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU     : {torch.cuda.get_device_name(0)}')\n",
    "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print(f'VRAM    : {vram:.1f} GB  (expect ~15-16 GB on T4)')\n",
    "else:\n",
    "    print('⚠️  No GPU — enable in Notebook Settings → Accelerator → GPU T4')\n",
]))

# ── 2 · Session configuration ─────────────────────────────────────────────────
cells.append(md("""\
## 2. ⚙️ Session Configuration
Edit the variables below **before running any other cell**.

**`RESUME_*` flags**: set to `True` if you have added the corresponding Kaggle Dataset as input.  
**Dataset input paths**: change if you named your datasets differently.\
"""))
cells.append(code([
    "# ─── RESUME FLAGS ────────────────────────────────────────────────────────\n",
    "# Set True when you have added the matching dataset as notebook input\n",
    "RESUME_HF_CACHE     = False  # True → skip BioBERT download (saves ~5 min)\n",
    "RESUME_DATA         = False  # True → skip raw download + preprocessing (~25 min)\n",
    "RESUME_CHECKPOINTS  = False  # True → resume training from last saved epoch\n",
    "\n",
    "# ─── INPUT DATASET PATHS (Kaggle mounts datasets under /kaggle/input/) ───\n",
    "# After you create and add them, the paths will match these names:\n",
    "INPUT_HF_CACHE    = '/kaggle/input/promptgfm-model-cache'\n",
    "INPUT_DATA        = '/kaggle/input/promptgfm-data'\n",
    "INPUT_CHECKPOINTS = '/kaggle/input/promptgfm-checkpoints'\n",
    "\n",
    "# ─── OUTPUT DATASET NAMES (used in Step 12 instructions) ─────────────────\n",
    "OUTPUT_HF_CACHE    = 'promptgfm-model-cache'\n",
    "OUTPUT_DATA        = 'promptgfm-data'\n",
    "OUTPUT_CHECKPOINTS = 'promptgfm-checkpoints'\n",
    "\n",
    "# ─── HF MODEL CACHE LOCATION ─────────────────────────────────────────────\n",
    "# Point HuggingFace to a path inside /kaggle/working/ so we can save it\n",
    "HF_CACHE_DIR = '/kaggle/working/hf_cache'\n",
    "os.environ['HF_HOME']              = HF_CACHE_DIR\n",
    "os.environ['TRANSFORMERS_CACHE']   = HF_CACHE_DIR\n",
    "os.environ['HUGGINGFACE_HUB_CACHE']= HF_CACHE_DIR\n",
    "\n",
    "print('Configuration:')\n",
    "print(f'  RESUME_HF_CACHE    = {RESUME_HF_CACHE}')\n",
    "print(f'  RESUME_DATA        = {RESUME_DATA}')\n",
    "print(f'  RESUME_CHECKPOINTS = {RESUME_CHECKPOINTS}')\n",
    "print(f'  HF cache dir       = {HF_CACHE_DIR}')\n",
]))

# ── 3 · Install PyG ───────────────────────────────────────────────────────────
cells.append(md("## 3. Install PyTorch Geometric"))
cells.append(code([
    "import torch, subprocess, sys\n",
    "\n",
    "TORCH_VER = torch.__version__.split('+')[0]\n",
    "CUDA_VER  = torch.version.cuda.replace('.', '')\n",
    "WHEEL_URL = f'https://data.pyg.org/whl/torch-{TORCH_VER}+cu{CUDA_VER}.html'\n",
    "print(f'PyG wheel source: {WHEEL_URL}')\n",
    "\n",
    "subprocess.run(\n",
    "    [sys.executable, '-m', 'pip', 'install', '--quiet',\n",
    "     '-f', WHEEL_URL,\n",
    "     'torch-scatter', 'torch-sparse', 'torch-cluster',\n",
    "     'torch-spline-conv', 'torch-geometric'],\n",
    "    check=True\n",
    ")\n",
    "print('✅ PyTorch Geometric installed')\n",
]))

# ── 4 · Install extras ────────────────────────────────────────────────────────
cells.append(md("## 4. Install Extra Dependencies"))
cells.append(code([
    "# Upgrade build tools first — prevents metadata-generation-failed on Python 3.12\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',\n",
    "                '--upgrade', 'setuptools', 'wheel', 'pip'], check=True)\n",
    "\n",
    "extra = [\n",
    "    'transformers>=4.40.0',\n",
    "    'sentence-transformers>=2.7.0',\n",
    "    'biopython>=1.84',\n",
    "    'networkx>=3.2',\n",
    "    'wandb>=0.17.0',\n",
    "    'python-dotenv>=1.0.0',\n",
    "]\n",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + extra, check=True)\n",
    "print('✅ Extra packages installed')\n",
]))

# ── 5 · Clone repo ────────────────────────────────────────────────────────────
cells.append(md("## 5. Clone Project Code from GitHub"))
cells.append(code([
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "GITHUB_URL  = 'https://github.com/pes1ug23am910/PROMPTGMF-Bio-Kaggle.git'\n",
    "PROJECT_DIR = '/kaggle/working/PromptGFM-Bio'\n",
    "\n",
    "if not os.path.exists(PROJECT_DIR):\n",
    "    subprocess.run(['git', 'clone', '--depth', '1', GITHUB_URL, PROJECT_DIR], check=True)\n",
    "    print(f'✅ Cloned to {PROJECT_DIR}')\n",
    "else:\n",
    "    subprocess.run(['git', '-C', PROJECT_DIR, 'pull'], check=True)\n",
    "    print(f'✅ Pulled latest changes')\n",
    "\n",
    "os.chdir(PROJECT_DIR)\n",
    "sys.path.insert(0, PROJECT_DIR)\n",
    "print(f'Working directory: {os.getcwd()}')\n",
]))

# ── 6 · Create dirs ───────────────────────────────────────────────────────────
cells.append(md("## 6. Create Directory Structure"))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "dirs = [\n",
    "    'data/raw', 'data/processed', 'data/splits',\n",
    "    'checkpoints/promptgfm_film',\n",
    "    'logs',\n",
    "]\n",
    "for d in dirs:\n",
    "    Path(d).mkdir(parents=True, exist_ok=True)\n",
    "print('✅ Directories created')\n",
]))

# ── 7 · Restore assets ────────────────────────────────────────────────────────
cells.append(md("""\
## 7. Restore Assets from Previous Session
Restores HuggingFace model cache, preprocessed data, and training checkpoints \
from saved Kaggle Datasets — skipping all expensive steps below.\

**First-time run**: all three blocks will print "not found — will download/preprocess fresh".\
"""))
cells.append(code([
    "import shutil, tarfile\n",
    "from pathlib import Path\n",
    "\n",
    "def restore_tar(src_path, dest_dir, label):\n",
    "    \"\"\"Extract a .tar.gz archive if it exists.\"\"\"\n",
    "    src = Path(src_path)\n",
    "    if src.exists():\n",
    "        dest = Path(dest_dir)\n",
    "        dest.mkdir(parents=True, exist_ok=True)\n",
    "        with tarfile.open(src, 'r:gz') as tar:\n",
    "            tar.extractall(dest)\n",
    "        print(f'✅ {label} restored from {src}')\n",
    "        return True\n",
    "    return False\n",
    "\n",
    "def restore_dir(src_path, dest_dir, label):\n",
    "    \"\"\"Copy directory tree if source exists.\"\"\"\n",
    "    src = Path(src_path)\n",
    "    if src.exists() and any(src.iterdir()):\n",
    "        dest = Path(dest_dir)\n",
    "        if dest.exists():\n",
    "            shutil.rmtree(dest)\n",
    "        shutil.copytree(src, dest)\n",
    "        print(f'✅ {label} restored from {src}')\n",
    "        return True\n",
    "    return False\n",
    "\n",
    "# ── A. HuggingFace model cache ────────────────────────────────────────────\n",
    "if RESUME_HF_CACHE:\n",
    "    ok = restore_tar(f'{INPUT_HF_CACHE}/hf_cache.tar.gz', HF_CACHE_DIR, 'HF model cache')\n",
    "    if not ok:\n",
    "        ok = restore_dir(f'{INPUT_HF_CACHE}/hf_cache', HF_CACHE_DIR, 'HF model cache')\n",
    "    if not ok:\n",
    "        print(f'⚠️  HF cache not found at {INPUT_HF_CACHE} — BioBERT will be re-downloaded')\n",
    "else:\n",
    "    print('HF cache: skipped (RESUME_HF_CACHE=False)')\n",
    "\n",
    "# ── B. Preprocessed graph + raw data ─────────────────────────────────────\n",
    "if RESUME_DATA:\n",
    "    ok = restore_tar(f'{INPUT_DATA}/data.tar.gz', 'data', 'Graph data')\n",
    "    if not ok:\n",
    "        ok = restore_dir(f'{INPUT_DATA}/processed', 'data/processed', 'Processed graph')\n",
    "        restore_dir(f'{INPUT_DATA}/splits', 'data/splits', 'Data splits')\n",
    "    graph = Path('data/processed/biomedical_graph.pt')\n",
    "    if graph.exists():\n",
    "        print(f'✅ Graph ready ({graph.stat().st_size/1e6:.0f} MB)')\n",
    "    else:\n",
    "        print('⚠️  Graph not found in restored data — will preprocess fresh')\n",
    "        RESUME_DATA = False   # force re-preprocessing below\n",
    "else:\n",
    "    print('Data: skipped (RESUME_DATA=False)')\n",
    "\n",
    "# ── C. Training checkpoints ───────────────────────────────────────────────\n",
    "if RESUME_CHECKPOINTS:\n",
    "    ckpt_src = Path(INPUT_CHECKPOINTS)\n",
    "    ckpt_dst = Path('checkpoints/promptgfm_film')\n",
    "    ckpt_dst.mkdir(parents=True, exist_ok=True)\n",
    "    files_copied = 0\n",
    "    if ckpt_src.exists():\n",
    "        for f in ckpt_src.glob('*'):\n",
    "            shutil.copy2(f, ckpt_dst / f.name)\n",
    "            files_copied += 1\n",
    "    if files_copied:\n",
    "        epochs = sorted([f.stem.replace('checkpoint_epoch_','') for f in ckpt_dst.glob('checkpoint_epoch_*.pt')])\n",
    "        print(f'✅ Checkpoints restored ({files_copied} files). Epochs available: {epochs}')\n",
    "    else:\n",
    "        print(f'⚠️  No checkpoints found at {INPUT_CHECKPOINTS} — will train from scratch')\n",
    "        RESUME_CHECKPOINTS = False\n",
    "else:\n",
    "    print('Checkpoints: skipped (RESUME_CHECKPOINTS=False)')\n",
]))

# ── 8 · Download data ─────────────────────────────────────────────────────────
cells.append(md("""\
## 8. Download Biomedical Datasets
Skipped automatically if `RESUME_DATA=True` and graph was restored successfully.  
Total download: ~1.5 GB · takes ~10 min.\
"""))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "graph_exists = Path('data/processed/biomedical_graph.pt').exists()\n",
    "\n",
    "if RESUME_DATA and graph_exists:\n",
    "    print('⏭️  Download skipped — restored from Kaggle Dataset')\n",
    "else:\n",
    "    print('Downloading datasets...')\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, 'scripts/download_data.py', '--dataset', 'all'],\n",
    "        capture_output=False\n",
    "    )\n",
    "    print('Download exit code:', result.returncode)\n",
]))

# ── 9 · Preprocess ────────────────────────────────────────────────────────────
cells.append(md("""\
## 9. Preprocess Data (Build Knowledge Graph)
Skipped automatically if `RESUME_DATA=True` and graph was restored successfully.\
"""))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "graph_path = Path('data/processed/biomedical_graph.pt')\n",
    "\n",
    "if RESUME_DATA and graph_path.exists():\n",
    "    print(f'⏭️  Preprocessing skipped — graph ready ({graph_path.stat().st_size/1e6:.0f} MB)')\n",
    "else:\n",
    "    print('Building knowledge graph...')\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, 'scripts/preprocess_all.py'],\n",
    "        capture_output=False\n",
    "    )\n",
    "    print('Preprocessing exit code:', result.returncode)\n",
    "    if graph_path.exists():\n",
    "        print(f'✅ Graph ready ({graph_path.stat().st_size/1e6:.0f} MB)')\n",
    "    else:\n",
    "        raise RuntimeError('Graph file not created — check logs above')\n",
]))

# ── 10 · W&B ──────────────────────────────────────────────────────────────────
cells.append(md("## 10. W&B Login (Optional)"))
cells.append(code([
    "WANDB_API_KEY = ''   # paste your key here, or leave empty to disable\n",
    "\n",
    "if WANDB_API_KEY:\n",
    "    import wandb\n",
    "    wandb.login(key=WANDB_API_KEY)\n",
    "    print('✅ W&B logged in')\n",
    "else:\n",
    "    os.environ['WANDB_MODE'] = 'disabled'\n",
    "    print('W&B disabled — metrics logged to stdout only')\n",
]))

# ── 11 · Train ────────────────────────────────────────────────────────────────
cells.append(md("""\
## 11. Train
Uses `configs/kaggle_config.yaml` (T4-tuned: batch_size=64, hidden_dim=512).  
Set `RESUME=True` to continue from the last restored checkpoint.\
"""))
cells.append(code([
    "RESUME = RESUME_CHECKPOINTS   # auto-set from config above; override here if needed\n",
    "\n",
    "if RESUME:\n",
    "    cmd = [sys.executable, 'scripts/resume_training.py',\n",
    "           '--config', 'configs/kaggle_config.yaml']\n",
    "else:\n",
    "    cmd = [sys.executable, 'scripts/train.py',\n",
    "           '--config', 'configs/kaggle_config.yaml']\n",
    "\n",
    "print('Running:', ' '.join(cmd))\n",
    "result = subprocess.run(cmd, capture_output=False)\n",
    "print('Training exit code:', result.returncode)\n",
]))

# ── 12 · Save everything ──────────────────────────────────────────────────────
cells.append(md("""\
## 12. 💾 Save ALL Assets as Kaggle Output

Run this cell **before the session ends** (set a reminder before the 9-hour limit).

It saves three directories under `/kaggle/working/`:

| Directory | Contents | Create Dataset named |
|---|---|---|
| `out_model_cache/` | BioBERT weights (~440 MB) | `promptgfm-model-cache` |
| `out_data/` | Raw + processed graph (~600 MB) | `promptgfm-data` |
| `out_checkpoints/` | Per-epoch `.pt` files | `promptgfm-checkpoints` |

### After this cell completes:
1. Click **Output** tab (right panel) → you'll see these three folders
2. For **each** folder → click the ⊕ icon → **New Dataset** → use the names above
3. Make the datasets **Public** (or **Private** if you want them only for yourself)
4. Next session: **Add Data** → **Your Datasets** → add all three → set `RESUME_*=True`

### Using from a different Kaggle account:
Make the datasets **Public**, then the other account can find them by searching  
`pes1ug23am910/promptgfm-model-cache` etc. in **Add Data**.\
"""))
cells.append(code([
    "import shutil, tarfile, os\n",
    "from pathlib import Path\n",
    "\n",
    "def make_tar(src_dir, out_file, label):\n",
    "    src = Path(src_dir)\n",
    "    if not src.exists() or not any(src.rglob('*')):\n",
    "        print(f'⚠️  {label}: source empty or missing ({src}) — skipped')\n",
    "        return\n",
    "    out = Path(out_file)\n",
    "    out.parent.mkdir(parents=True, exist_ok=True)\n",
    "    with tarfile.open(out, 'w:gz') as tar:\n",
    "        tar.add(src, arcname=src.name)\n",
    "    size_mb = out.stat().st_size / 1e6\n",
    "    print(f'✅ {label} → {out}  ({size_mb:.0f} MB)')\n",
    "\n",
    "def copy_dir(src_dir, out_dir, label):\n",
    "    src = Path(src_dir)\n",
    "    out = Path(out_dir)\n",
    "    if not src.exists():\n",
    "        print(f'⚠️  {label}: missing ({src}) — skipped')\n",
    "        return\n",
    "    if out.exists():\n",
    "        shutil.rmtree(out)\n",
    "    shutil.copytree(src, out)\n",
    "    files = list(out.rglob('*.*'))\n",
    "    total_mb = sum(f.stat().st_size for f in files) / 1e6\n",
    "    print(f'✅ {label} → {out}  ({len(files)} files, {total_mb:.0f} MB)')\n",
    "\n",
    "print('=== Saving assets for next session ===')\n",
    "\n",
    "# A. HuggingFace model cache (BioBERT) ─────────────────────────────────────\n",
    "make_tar(HF_CACHE_DIR,\n",
    "         '/kaggle/working/out_model_cache/hf_cache.tar.gz',\n",
    "         'HF model cache')\n",
    "\n",
    "# B. Data (raw + processed graph + splits) ─────────────────────────────────\n",
    "make_tar('data/processed',\n",
    "         '/kaggle/working/out_data/data.tar.gz',\n",
    "         'Processed graph data')\n",
    "\n",
    "# C. Training checkpoints ──────────────────────────────────────────────────\n",
    "copy_dir('checkpoints/promptgfm_film',\n",
    "         '/kaggle/working/out_checkpoints',\n",
    "         'Training checkpoints')\n",
    "\n",
    "print()\n",
    "print('=== Output summary ===')\n",
    "for d in ['out_model_cache', 'out_data', 'out_checkpoints']:\n",
    "    p = Path('/kaggle/working') / d\n",
    "    if p.exists():\n",
    "        files = list(p.rglob('*.*'))\n",
    "        total_mb = sum(f.stat().st_size for f in files) / 1e6\n",
    "        print(f'  /kaggle/working/{d}/  →  {len(files)} files, {total_mb:.0f} MB')\n",
    "\n",
    "print()\n",
    "print('Next steps:')\n",
    "print('  1. Output tab → create 3 datasets from out_model_cache, out_data, out_checkpoints')\n",
    "print('  2. Next session: add those datasets as input, set RESUME_*=True in cell 2')\n",
]))

# ── 13 · Evaluate ─────────────────────────────────────────────────────────────
cells.append(md("## 13. Quick Evaluation"))
cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "best = Path('checkpoints/promptgfm_film/best_model.pt')\n",
    "if not best.exists():\n",
    "    print('No best_model.pt yet — run more training epochs first')\n",
    "else:\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, 'scripts/evaluate.py',\n",
    "         '--config', 'configs/kaggle_config.yaml',\n",
    "         '--checkpoint', str(best)],\n",
    "        capture_output=False\n",
    "    )\n",
    "    print('Evaluation exit code:', result.returncode)\n",
]))

print(f"Built {len(cells)} cells")
nb = build(cells)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written → {OUT}")
