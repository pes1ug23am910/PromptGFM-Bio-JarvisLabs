---
jupyter:
  kernelspec:
    display_name: 🧬 PromptGFM-Bio (py3.12)
    language: python
    name: promptgfm
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.13
  nbformat: 4
  nbformat_minor: 5
---

::: {#c6c5bd4c .cell .markdown}
# PromptGFM-Bio --- ARC Labs Workstation Notebook {#promptgfm-bio--arc-labs-workstation-notebook}

**Gene-Phenotype Prediction for Rare Diseases**

  Spec   Value
  ------ --------------------------------------
  CPU    Intel i9-14900K
  RAM    128 GB
  GPU    NVIDIA GeForce RTX 4090 (24 GB VRAM)
  CUDA   13.0 · Driver 580.65.06
  Disk   512 GB

### How this notebook works

-   **Single project root**: every cell uses `PROJECT_ROOT` --- no path
    confusion.
-   **VRAM-aware**: auto-detects free GPU memory and scales batch size
    accordingly.
-   **Resumable**: data persists on disk between Jupyter sessions (5-day
    window).
-   **Secrets**: GitHub token + W&B key loaded from `.env` in the
    project root.

> ⚠️ **5-day data retention** --- back up to GitHub/HuggingFace before
> your window expires.
:::

::: {#141e2164 .cell .markdown}
## 0. 🔒 Master Path Setup (RUN THIS FIRST --- ALWAYS) {#0--master-path-setup-run-this-first--always}

This cell sets `PROJECT_ROOT` and `os.chdir()` so **every cell below**
uses the same directory. Nothing else in this notebook defines paths
independently.
:::

::: {#76f9549d .cell .code execution_count="1"}
``` python
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EDIT THIS ONE LINE if your project is in a different location         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, sys, subprocess
from pathlib import Path

GITHUB_URL   = "https://github.com/pes1ug23am910/PromptGFM-Bio.git"
PROJECT_ROOT = Path("/home/mluser/projects_yash/new_project/PromptGFM-Bio").resolve()

# ── Ensure directory exists ───────────────────────────────────────────────
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-CLONE: if scripts/ and configs/ are missing, pull the repo code
# Safe for non-empty directories — won't clobber .env, data/, hf_cache/
# ═══════════════════════════════════════════════════════════════════════════
if not (PROJECT_ROOT / "scripts").is_dir() or not (PROJECT_ROOT / "configs").is_dir():
    print("⚠️  scripts/ and/or configs/ not found — cloning repo code...")
    print()

    # Load .env early to get GITHUB_TOKEN for private repo
    _env = PROJECT_ROOT / ".env"
    if _env.is_file():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ[_k.strip()] = _v.strip()

    _token = os.environ.get("GITHUB_TOKEN", "")
    _auth_url = GITHUB_URL.replace("https://", f"https://{_token}@") if _token else GITHUB_URL

    if (PROJECT_ROOT / ".git").is_dir():
        # Already a git repo but missing files — just pull
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull"], check=True)
    else:
        # Not a git repo — init and pull (safe for non-empty dirs)
        subprocess.run(["git", "init"], cwd=str(PROJECT_ROOT), check=True,
                       capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", _auth_url],
                       cwd=str(PROJECT_ROOT), capture_output=True)

        result = subprocess.run(
            ["git", "fetch", "origin"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"   ❌ git fetch failed: {result.stderr.strip()}")
            if not _token:
                print("   Repo is private — add GITHUB_TOKEN to .env")
            raise RuntimeError("Cannot fetch repo code")

        # Detect default branch
        _branch_result = subprocess.run(
            ["git", "remote", "show", "origin"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True
        )
        _default_branch = "main"
        for _bl in _branch_result.stdout.splitlines():
            if "HEAD branch" in _bl:
                _default_branch = _bl.split(":")[-1].strip()
                break

        subprocess.run(
            ["git", "checkout", "-f", f"origin/{_default_branch}", "-b", _default_branch],
            cwd=str(PROJECT_ROOT), capture_output=True
        )

    if (PROJECT_ROOT / "scripts").is_dir() and (PROJECT_ROOT / "configs").is_dir():
        print("   ✅ Repo code cloned successfully")
    else:
        print("   ❌ Clone finished but scripts/configs still missing!")
        raise RuntimeError("Repo doesn't contain expected scripts/ and configs/")
else:
    print("✅ Repo code already present (scripts/ and configs/ found)")

# ── Derived path variables ────────────────────────────────────────────────
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
CONFIGS_DIR  = PROJECT_ROOT / "configs"
DATA_DIR     = PROJECT_ROOT / "data"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "promptgfm_film"
LOGS_DIR     = PROJECT_ROOT / "logs"
ENV_FILE     = PROJECT_ROOT / ".env"

# ── HF cache ─────────────────────────────────────────────────────────────
HF_CACHE_DIR = str(PROJECT_ROOT / "hf_cache")
os.environ["HF_HOME"]               = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]     = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR

# ── Resume flags ──────────────────────────────────────────────────────────
RESUME_HF_CACHE     = False
RESUME_DATA         = False
RESUME_CHECKPOINTS  = False

# ── Status report ─────────────────────────────────────────────────────────
print()
print(f"PROJECT_ROOT   = {PROJECT_ROOT}")
print(f"CWD            = {Path.cwd()}")
print(f"  scripts/     : {'✅' if SCRIPTS_DIR.is_dir() else '❌'}")
print(f"  configs/     : {'✅' if CONFIGS_DIR.is_dir() else '❌'}")
print(f"  data/        : {'✅' if DATA_DIR.is_dir() else '⬜ will create'}")
print(f"  .env         : {'✅' if ENV_FILE.is_file() else '❌ MISSING'}")
print(f"  .git/        : {'✅' if (PROJECT_ROOT / '.git').is_dir() else '❌'}")
print(f"  hf_cache/    : {'✅' if Path(HF_CACHE_DIR).is_dir() else '⬜ will create'}")
print(f"  RESUME_HF_CACHE    = {RESUME_HF_CACHE}")
print(f"  RESUME_DATA        = {RESUME_DATA}")
print(f"  RESUME_CHECKPOINTS = {RESUME_CHECKPOINTS}")
```

::: {.output .stream .stdout}
    ✅ Repo code already present (scripts/ and configs/ found)

    PROJECT_ROOT   = /home/mluser/projects_yash/new_project/PromptGFM-Bio
    CWD            = /home/mluser/projects_yash/new_project/PromptGFM-Bio
      scripts/     : ✅
      configs/     : ✅
      data/        : ✅
      .env         : ✅
      .git/        : ❌
      hf_cache/    : ✅
      RESUME_HF_CACHE    = False
      RESUME_DATA        = False
      RESUME_CHECKPOINTS = False
:::
:::

::: {#0fee6c6b .cell .markdown}
## 1. Load Secrets from `.env` {#1-load-secrets-from-env}

Your `.env` file should be at:

    /home/mluser/projects_yash/new_project/PromptGFM-Bio/.env

Contents:

    GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
    WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
:::

::: {#db636d27 .cell .code execution_count="2"}
``` python
import os

print(f"Loading: {ENV_FILE}")

if ENV_FILE.is_file():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip()
    print("✅ .env loaded")
else:
    print(f"❌ .env not found at {ENV_FILE}")
    print(f'   Create it:  echo -e "GITHUB_TOKEN=ghp_xxx\nWANDB_API_KEY=xxx" > {ENV_FILE}')

print(f"  GITHUB_TOKEN  : {'✅ set' if os.environ.get('GITHUB_TOKEN') else '❌ missing'}")
print(f"  WANDB_API_KEY : {'✅ set' if os.environ.get('WANDB_API_KEY') else '❌ missing'}")
```

::: {.output .stream .stdout}
    Loading: /home/mluser/projects_yash/new_project/PromptGFM-Bio/.env
    ✅ .env loaded
      GITHUB_TOKEN  : ✅ set
      WANDB_API_KEY : ✅ set
:::
:::

::: {#87a3153c .cell .markdown}
## 2. GPU Diagnostics & VRAM Budget {#2-gpu-diagnostics--vram-budget}
:::

::: {#b5980bbd .cell .code execution_count="3"}
``` python
import subprocess, os

subprocess.run(["nvidia-smi"])
print()

try:
    import torch
    if torch.cuda.is_available():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True
        )
        smi_used, smi_total, smi_free = [int(x.strip()) for x in result.stdout.strip().split(",")]
        usable_mb = smi_free - 2048

        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM    : {smi_total:,} MiB")
        print(f"Used (idle)   : {smi_used:,} MiB")
        print(f"Free VRAM     : {smi_free:,} MiB")
        print(f"Usable        : {usable_mb:,} MiB (after 2 GB safety margin)")
        print(f"PyTorch       : {torch.__version__}")
        print(f"CUDA          : {torch.version.cuda}")

        os.environ["_VRAM_FREE_MB"]   = str(smi_free)
        os.environ["_VRAM_USABLE_MB"] = str(usable_mb)
    else:
        print("⚠️  No GPU detected")
except ImportError:
    print("⚠️  PyTorch not installed yet — run Step 3 first")
```

::: {.output .stream .stdout}
    Thu Apr  2 16:50:44 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
    | 34%   39C    P8             17W /  450W |     117MiB /  24564MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

    GPU           : NVIDIA GeForce RTX 4090
    Total VRAM    : 24,564 MiB
    Used (idle)   : 120 MiB
    Free VRAM     : 23,953 MiB
    Usable        : 21,905 MiB (after 2 GB safety margin)
    PyTorch       : 2.6.0+cu124
    CUDA          : 12.4
:::
:::

::: {#e103dec9 .cell .markdown}
## 3. Install PyTorch & PyTorch Geometric {#3-install-pytorch--pytorch-geometric}

Installs into your existing venv. CUDA 12.4 wheels are compatible with
your CUDA 13.0 driver.
:::

::: {#613dccfe .cell .code execution_count="4"}
``` python
import subprocess, sys

# ── PyTorch ───────────────────────────────────────────────────────────────
try:
    import torch
    print(f"PyTorch {torch.__version__} already installed (CUDA {torch.version.cuda})")
except ImportError:
    print("Installing PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ], check=True)
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")

# ── PyG ───────────────────────────────────────────────────────────────────
TORCH_VER = torch.__version__.split("+")[0]
CUDA_VER  = torch.version.cuda.replace(".", "")
WHEEL_URL = f"https://data.pyg.org/whl/torch-{TORCH_VER}+cu{CUDA_VER}.html"
print(f"PyG wheel source: {WHEEL_URL}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "-f", WHEEL_URL,
     "torch-scatter", "torch-sparse", "torch-cluster",
     "torch-spline-conv", "torch-geometric"],
    check=True
)
print("✅ PyTorch Geometric installed")
```

::: {.output .stream .stdout}
    PyTorch 2.6.0+cu124 already installed (CUDA 12.4)
    PyG wheel source: https://data.pyg.org/whl/torch-2.6.0+cu124.html
    ✅ PyTorch Geometric installed
:::
:::

::: {#41f56eea .cell .markdown}
## 4. Install Extra Dependencies {#4-install-extra-dependencies}
:::

::: {#d321f853 .cell .code execution_count="5"}
``` python
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "--upgrade", "setuptools", "wheel", "pip"], check=True)

extra = [
    "transformers>=4.40.0",
    "sentence-transformers>=2.7.0",
    "biopython>=1.84",
    "networkx>=3.2",
    "wandb>=0.17.0",
    "python-dotenv>=1.0.0",
    "huggingface_hub",
    "pyyaml",
]
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + extra, check=True)
print("✅ Extra packages installed")
```

::: {.output .stream .stdout}
    ✅ Extra packages installed
:::
:::

::: {#f294a5d6 .cell .markdown}
## 5. Git Pull Latest Code {#5-git-pull-latest-code}

Your repo is already cloned at `PROJECT_ROOT`. This just pulls the
latest changes.
:::

::: {#b273f478 .cell .code execution_count="6"}
``` python
import subprocess, os

os.environ["GIT_TERMINAL_PROMPT"] = "0"

if (PROJECT_ROOT / ".git").is_dir():
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ Latest code pulled")
        if result.stdout.strip():
            print(f"   {result.stdout.strip()}")
    else:
        print("⚠️  git pull failed — continuing with existing code")
        print(f"   {result.stderr.strip()}")
else:
    print("⚠️  Not a git repo — skipping pull (code was set up in Cell 0)")
```

::: {.output .stream .stdout}
    ⚠️  Not a git repo — skipping pull (code was set up in Cell 0)
:::
:::

::: {#9356370c-064c-48ec-b0c7-90b6e1fcaef5 .cell .code execution_count="7"}
``` python
'''
import os, sys, subprocess
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ EDIT THIS LINE to match your home directory / preferred location ║
# ╚══════════════════════════════════════════════════════════════════════════╝
GITHUB_URL = 'https://github.com/pes1ug23am910/PROMPTGMF-Bio-Kaggle.git'
PROJECT_DIR = Path('/home/mluser/projects_yash/new_project/PromptGFM-Bio').resolve() # ← change this

# ── Ensure parent directory exists ───────────────────────────────────────
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# ── Clone or pull ─────────────────────────────────────────────────────────
if not (PROJECT_DIR / '.git').is_dir():
 subprocess.run(['git', 'clone', '--depth', '1', GITHUB_URL, str(PROJECT_DIR)], check=True)
 print(f' Cloned to {PROJECT_DIR}')
else:
 subprocess.run(['git', '-C', str(PROJECT_DIR), 'pull'], check=True)
 print(f' Pulled latest changes')

# ── Set working directory & Python path ──────────────────────────────────
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print(f'Working directory: {os.getcwd()}')

'''
```

::: {.output .execute_result execution_count="7"}
    "\nimport os, sys, subprocess\nfrom pathlib import Path\n\n# ╔══════════════════════════════════════════════════════════════════════════╗\n# ║ EDIT THIS LINE to match your home directory / preferred location ║\n# ╚══════════════════════════════════════════════════════════════════════════╝\nGITHUB_URL = 'https://github.com/pes1ug23am910/PROMPTGMF-Bio-Kaggle.git'\nPROJECT_DIR = Path('/home/mluser/projects_yash/new_project/PromptGFM-Bio').resolve() # ← change this\n\n# ── Ensure parent directory exists ───────────────────────────────────────\nPROJECT_DIR.mkdir(parents=True, exist_ok=True)\n\n# ── Clone or pull ─────────────────────────────────────────────────────────\nif not (PROJECT_DIR / '.git').is_dir():\n subprocess.run(['git', 'clone', '--depth', '1', GITHUB_URL, str(PROJECT_DIR)], check=True)\n print(f' Cloned to {PROJECT_DIR}')\nelse:\n subprocess.run(['git', '-C', str(PROJECT_DIR), 'pull'], check=True)\n print(f' Pulled latest changes')\n\n# ── Set working directory & Python path ──────────────────────────────────\nos.chdir(PROJECT_DIR)\nsys.path.insert(0, str(PROJECT_DIR))\nprint(f'Working directory: {os.getcwd()}')\n\n"
:::
:::

::: {#75839c24 .cell .markdown}
## 6. Create Directory Structure {#6-create-directory-structure}
:::

::: {#33d41662 .cell .code execution_count="8"}
``` python
for d in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "splits", CKPT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print("✅ Directories created")
```

::: {.output .stream .stdout}
    ✅ Directories created
:::
:::

::: {#31bdb8d0 .cell .markdown}
## 7. Pre-download BioBERT Model {#7-pre-download-biobert-model}

Downloads \~440 MB of PubMedBERT weights. Only needed once --- cache
persists on disk.
:::

::: {#caa9fd07 .cell .code execution_count="9"}
``` python
from huggingface_hub import snapshot_download
from pathlib import Path
import warnings, os

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

hub_dir = Path(HF_CACHE_DIR) / "hub"
model_cache_name = "models--" + MODEL_NAME.replace("/", "--")
model_snapshots = hub_dir / model_cache_name / "snapshots"

if RESUME_HF_CACHE and model_snapshots.exists() and any(model_snapshots.iterdir()):
    print("⏭️  BioBERT already cached — skipping download")
else:
    print(f"Downloading {MODEL_NAME}")
    print(f"  Cache dir : {HF_CACHE_DIR}")
    print(f"  Size      : ~440 MB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        snapshot_download(
            repo_id=MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
        )
    print("\n✅ BioBERT fully downloaded and cached")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
print("✅ Offline mode enabled")
```

::: {.output .stream .stdout}
    Downloading microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
      Cache dir : /home/mluser/projects_yash/new_project/PromptGFM-Bio/hf_cache
      Size      : ~440 MB
:::

::: {.output .display_data}
``` json
{"model_id":"640441b263d24d898fa6ce3b3e90c092","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"f44d890084c24c87a22079c51d2bdf6f","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}

    ✅ BioBERT fully downloaded and cached
    ✅ Offline mode enabled
:::
:::

::: {#684b6f18 .cell .markdown}
## 8. Download Biomedical Datasets {#8-download-biomedical-datasets}

Skipped if `RESUME_DATA=True` and graph exists. Total download: \~1.5
GB.
:::

::: {#62c76a19 .cell .code execution_count="10"}
``` python
import subprocess, sys

graph_exists = (DATA_DIR / "processed" / "biomedical_graph.pt").exists()

if RESUME_DATA and graph_exists:
    print("⏭️  Download skipped — data already on disk")
else:
    print("Downloading datasets...")
    script = str(SCRIPTS_DIR / "download_data.py")
    print(f"  Running: {script}")
    result = subprocess.run(
        [sys.executable, script, "--dataset", "all"],
        cwd=str(PROJECT_ROOT),
    )
    print("Download exit code:", result.returncode)
    if result.returncode != 0:
        print("⚠️  Download may have failed — check output above")
```

::: {.output .stream .stdout}
    Downloading datasets...
      Running: /home/mluser/projects_yash/new_project/PromptGFM-Bio/scripts/download_data.py

    ======================================================================
    PromptGFM-Bio Data Download
    ======================================================================

    Dataset: all
    Force re-download: False
    Skip failing: True


    ✓ Successfully downloaded 4/4 datasets

    ======================================================================
    ✓ DOWNLOAD COMPLETE
    ======================================================================

    Next steps:
      1. Run preprocessing: python scripts/preprocess_all.py
      2. Check downloaded files in: data/raw/

    Download exit code: 0
:::

::: {.output .stream .stderr}
    INFO:src.data.download:======================================================================
    INFO:src.data.download:Starting download of all biomedical datasets...
    INFO:src.data.download:This may take 30-60 minutes depending on your connection
    INFO:src.data.download:Total size: ~1.5 GB
    INFO:src.data.download:======================================================================
    INFO:src.data.download:
    [1/4] BioGRID Protein-Protein Interactions
    INFO:src.data.download:BioGRID file already exists at /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.zip
    INFO:src.data.download:Use force=True to re-download
    INFO:src.data.download:
    [2/4] STRING Protein Network
    INFO:src.data.download:STRING file already exists at /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/string/9606.protein.links.v12.0.txt.gz
    INFO:src.data.download:Use force=True to re-download
    INFO:src.data.download:
    [3/4] DisGeNET Gene-Disease Associations
    INFO:src.data.download:DisGeNET file already exists at /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/disgenet/curated_gene_disease_associations.tsv.gz
    INFO:src.data.download:Use force=True to re-download
    INFO:src.data.download:
    [4/4] Human Phenotype Ontology
    INFO:src.data.download:HPO file phenotype_to_genes.txt already exists
    INFO:src.data.download:HPO file genes_to_phenotype.txt already exists
    INFO:src.data.download:HPO file phenotype.hpoa already exists
    INFO:src.data.download:
    ======================================================================
    INFO:src.data.download:Download Summary:
    INFO:src.data.download:✓ BIOGRID: 1 files downloaded
    INFO:src.data.download:✓ STRING: 1 files downloaded
    INFO:src.data.download:✓ DISGENET: 1 files downloaded
    INFO:src.data.download:✓ HPO: 4 files downloaded
    INFO:src.data.download:======================================================================
:::
:::

::: {#74491f55 .cell .markdown}
## 9. Preprocess Data (Build Knowledge Graph) {#9-preprocess-data-build-knowledge-graph}

Skipped if `RESUME_DATA=True` and graph exists.
:::

::: {#f9801652-9d68-442e-9297-cf16726b6301 .cell .code execution_count="11"}
``` python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
```

::: {.output .stream .stdout}
    Requirement already satisfied: pandas in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (3.0.2)
    Requirement already satisfied: numpy>=1.26.0 in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (from pandas) (2.4.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
:::

::: {.output .execute_result execution_count="11"}
    CompletedProcess(args=['/home/mluser/micromamba/envs/promptgfm/bin/python', '-m', 'pip', 'install', 'pandas'], returncode=0)
:::
:::

::: {#6ae6b758 .cell .code execution_count="12"}
``` python
import subprocess, sys

graph_path = DATA_DIR / "processed" / "biomedical_graph.pt"

if RESUME_DATA and graph_path.exists():
    print(f"⏭️  Preprocessing skipped — graph ready ({graph_path.stat().st_size/1e6:.0f} MB)")
else:
    print("Building knowledge graph...")
    script = str(SCRIPTS_DIR / "preprocess_all.py")
    print(f"  Running: {script}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=str(PROJECT_ROOT),
    )
    print("Preprocessing exit code:", result.returncode)
    if graph_path.exists():
        print(f"✅ Graph ready ({graph_path.stat().st_size/1e6:.0f} MB)")
    else:
        raise RuntimeError("Graph file not created — check logs above")
```

::: {.output .stream .stdout}
    Building knowledge graph...
      Running: /home/mluser/projects_yash/new_project/PromptGFM-Bio/scripts/preprocess_all.py
:::

::: {.output .stream .stderr}
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/libpyg.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_scatter/_scatter_cuda.so: undefined symbol: _ZN2at4_ops16div__Tensor_mode4callERNS_6TensorERKS2_St8optionalIN3c1017basic_string_viewIcEEE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-spline-conv'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_spline_conv/_basis_cuda.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_sparse/_spmm_cuda.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    INFO:src.data.preprocess:Processed graph already exists at /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/processed/biomedical_graph.pt
    INFO:src.data.preprocess:Use force=True to reprocess
:::

::: {.output .stream .stdout}

    ======================================================================
    PromptGFM-Bio Enhanced Preprocessing Pipeline
    ======================================================================

    Configuration:
      Force reprocess: False
      HPO Bridge: True
      Orphadata: True
      UniProt: False
      Pathways: False


    ======================================================================
    ✓ PREPROCESSING COMPLETE
    ======================================================================

    Next steps:
      1. Create dataset splits: python -m src.data.dataset
      2. Check graph file: data/processed/biomedical_graph.pt
      3. View statistics: data/processed/biomedical_graph_stats.txt

    Preprocessing exit code: 0
    ✅ Graph ready (313 MB)
:::
:::

::: {#3a47a00a-9801-4512-a5a9-918e355a4ca2 .cell .code execution_count="13"}
``` python
'''
#use if needed/ previous cell fails
# ─────────────────────────────────────────────────────────────
# 4. Install & Verify All Dependencies (ROBUST VERSION)
# ─────────────────────────────────────────────────────────────

import subprocess, sys, importlib

def pip_install(packages):
    print(f"Installing: {packages}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        check=True
    )

def ensure_import(pkg_name, pip_name=None):
    """Ensure a package is installed and importable"""
    pip_name = pip_name or pkg_name
    try:
        importlib.import_module(pkg_name)
        print(f"✅ {pkg_name} already available")
    except ImportError:
        print(f"⚠️  {pkg_name} missing → installing...")
        pip_install([pip_name])
        try:
            importlib.import_module(pkg_name)
            print(f"✅ {pkg_name} installed successfully")
        except ImportError:
            raise RuntimeError(f"❌ Failed to import {pkg_name} even after install")

# ── Upgrade core tooling ──────────────────────────────────────
pip_install(["--upgrade", "pip", "setuptools", "wheel"])

# ── Critical deps (MUST exist for scripts to run) ─────────────
critical_deps = {
    "pandas": "pandas>=2.2.0",        # 🔴 REQUIRED (your failure)
    "numpy": "numpy",
    "yaml": "pyyaml",
}

# ── Project deps ─────────────────────────────────────────────
project_deps = {
    "transformers": "transformers>=4.40.0",
    "sentence_transformers": "sentence-transformers>=2.7.0",
    "Bio": "biopython>=1.84",
    "networkx": "networkx>=3.2",
    "wandb": "wandb>=0.17.0",
    "dotenv": "python-dotenv>=1.0.0",
    "huggingface_hub": "huggingface_hub",
}

# ── Ensure everything is importable ───────────────────────────
print("\n🔍 Checking critical dependencies...")
for module, pip_pkg in critical_deps.items():
    ensure_import(module, pip_pkg)

print("\n🔍 Checking project dependencies...")
for module, pip_pkg in project_deps.items():
    ensure_import(module, pip_pkg)

print("\n✅ All dependencies verified and ready")
'''
```

::: {.output .execute_result execution_count="13"}
    '\n#use if needed/ previous cell fails\n# ─────────────────────────────────────────────────────────────\n# 4. Install & Verify All Dependencies (ROBUST VERSION)\n# ─────────────────────────────────────────────────────────────\n\nimport subprocess, sys, importlib\n\ndef pip_install(packages):\n    print(f"Installing: {packages}")\n    subprocess.run(\n        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,\n        check=True\n    )\n\ndef ensure_import(pkg_name, pip_name=None):\n    """Ensure a package is installed and importable"""\n    pip_name = pip_name or pkg_name\n    try:\n        importlib.import_module(pkg_name)\n        print(f"✅ {pkg_name} already available")\n    except ImportError:\n        print(f"⚠️  {pkg_name} missing → installing...")\n        pip_install([pip_name])\n        try:\n            importlib.import_module(pkg_name)\n            print(f"✅ {pkg_name} installed successfully")\n        except ImportError:\n            raise RuntimeError(f"❌ Failed to import {pkg_name} even after install")\n\n# ── Upgrade core tooling ──────────────────────────────────────\npip_install(["--upgrade", "pip", "setuptools", "wheel"])\n\n# ── Critical deps (MUST exist for scripts to run) ─────────────\ncritical_deps = {\n    "pandas": "pandas>=2.2.0",        # 🔴 REQUIRED (your failure)\n    "numpy": "numpy",\n    "yaml": "pyyaml",\n}\n\n# ── Project deps ─────────────────────────────────────────────\nproject_deps = {\n    "transformers": "transformers>=4.40.0",\n    "sentence_transformers": "sentence-transformers>=2.7.0",\n    "Bio": "biopython>=1.84",\n    "networkx": "networkx>=3.2",\n    "wandb": "wandb>=0.17.0",\n    "dotenv": "python-dotenv>=1.0.0",\n    "huggingface_hub": "huggingface_hub",\n}\n\n# ── Ensure everything is importable ───────────────────────────\nprint("\n🔍 Checking critical dependencies...")\nfor module, pip_pkg in critical_deps.items():\n    ensure_import(module, pip_pkg)\n\nprint("\n🔍 Checking project dependencies...")\nfor module, pip_pkg in project_deps.items():\n    ensure_import(module, pip_pkg)\n\nprint("\n✅ All dependencies verified and ready")\n'
:::
:::

::: {#6844e8a3 .cell .markdown}
## 10. W&B Login {#10-wb-login}
:::

::: {#536a9cfe .cell .code execution_count="14"}
``` python
import os

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

if WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)
    print("✅ W&B logged in")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print("W&B disabled — add WANDB_API_KEY to .env to enable")
```

::: {.output .stream .stderr}
    wandb: WARNING If you're specifying your api key in code, ensure this code is not shared publicly.
    wandb: WARNING Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
    wandb: [wandb.login()] Using explicit session credentials for https://api.wandb.ai.
    wandb: Appending key for api.wandb.ai to your netrc file: /home/mluser/.netrc
    wandb: Currently logged in as: pes1ug23am910 to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
:::

::: {.output .stream .stdout}
    ✅ W&B logged in
:::
:::

::: {#c727c6ab .cell .markdown}
## 11. 🧠 Generate VRAM-Aware Config {#11--generate-vram-aware-config}

Reads `configs/kaggle_config.yaml` and patches it based on free VRAM
**right now**.

  Free VRAM   batch_size   grad_accum   effective_batch   workers
  ----------- ------------ ------------ ----------------- ---------
  ≥ 20 GB     768          1            768               8
  16--20 GB   512          1            512               6
  12--16 GB   384          1            384               4
  8--12 GB    256          1            256               4
  5--8 GB     128          2            256               2
  \< 5 GB     64           4            256               2
:::

::: {#e80d9ed4 .cell .code execution_count="15"}
``` python
import subprocess, os, yaml
import torch

# ── 1. Probe free VRAM ───────────────────────────────────────────────────
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
     "--format=csv,nounits,noheader"],
    capture_output=True, text=True
)
smi_used, smi_total, smi_free = [int(x.strip()) for x in result.stdout.strip().split(",")]
usable_mb = smi_free - 2048

print(f"GPU VRAM: {smi_total:,} MiB total · {smi_used:,} MiB used · {smi_free:,} MiB free")
print(f"Usable for training: {usable_mb:,} MiB (after 2 GB safety margin)")
print()

# ── 2. Pick batch_size + gradient accumulation ───────────────────────────
if usable_mb >= 20000:
    batch_size, grad_accum, num_workers = 768, 1, 8
elif usable_mb >= 16000:
    batch_size, grad_accum, num_workers = 512, 1, 6
elif usable_mb >= 12000:
    batch_size, grad_accum, num_workers = 384, 1, 4
elif usable_mb >= 8000:
    batch_size, grad_accum, num_workers = 256, 1, 4
elif usable_mb >= 5000:
    batch_size, grad_accum, num_workers = 128, 2, 2
else:
    batch_size, grad_accum, num_workers = 64, 4, 2

effective_batch = batch_size * grad_accum

print(f"╔══════════════════════════════════════════════╗")
print(f"║  batch_size       = {batch_size:<6}                   ║")
print(f"║  grad_accum       = {grad_accum:<6}                   ║")
print(f"║  effective_batch   = {effective_batch:<6}                  ║")
print(f"║  num_workers      = {num_workers:<6}                   ║")
print(f"╚══════════════════════════════════════════════╝")
print()

# ── 3. Read base config and patch ────────────────────────────────────────
base_cfg_path = CONFIGS_DIR / "kaggle_config.yaml"
ws_cfg_path   = CONFIGS_DIR / "workstation_config.yaml"

if base_cfg_path.exists():
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    def patch_dict(d, patches):
        if not isinstance(d, dict):
            return
        for k, v in patches.items():
            if k in d:
                print(f"  patching {k}: {d[k]} → {v}")
                d[k] = v
        for child in d.values():
            if isinstance(child, dict):
                patch_dict(child, patches)

    patches = {
        "batch_size":                   batch_size,
        "num_workers":                  num_workers,
        "gradient_accumulation_steps":  grad_accum,
        "pin_memory":                   True,
        "persistent_workers":           num_workers > 0,
    }

    print("Patching config:")
    patch_dict(cfg, patches)

    if "gradient_accumulation_steps" not in str(cfg):
        target = cfg.get("training", cfg)
        target["gradient_accumulation_steps"] = grad_accum
        print(f"  added gradient_accumulation_steps: {grad_accum} (new key)")

    with open(ws_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Wrote {ws_cfg_path}")
else:
    print(f"⚠️  {base_cfg_path} not found")

# ── 4. RTX 4090 performance flags ────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
print("✅ TF32 matmul ✓ · TF32 cuDNN ✓ · cuDNN benchmark ✓")
```

::: {.output .stream .stdout}
    GPU VRAM: 24,564 MiB total · 120 MiB used · 23,953 MiB free
    Usable for training: 21,905 MiB (after 2 GB safety margin)

    ╔══════════════════════════════════════════════╗
    ║  batch_size       = 768                      ║
    ║  grad_accum       = 1                        ║
    ║  effective_batch   = 768                     ║
    ║  num_workers      = 8                        ║
    ╚══════════════════════════════════════════════╝

    Patching config:
      patching batch_size: 256 → 768
      patching num_workers: 4 → 8
      patching pin_memory: True → True
      added gradient_accumulation_steps: 1 (new key)

    ✅ Wrote /home/mluser/projects_yash/new_project/PromptGFM-Bio/configs/workstation_config.yaml
    ✅ TF32 matmul ✓ · TF32 cuDNN ✓ · cuDNN benchmark ✓
:::
:::

::: {#1c94ce39 .cell .markdown}
## 12. Train {#12-train}

Uses `configs/workstation_config.yaml` auto-generated above with
VRAM-aware batch_size.
:::

::: {#3e6e57ff .cell .code execution_count="16"}
``` python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
print("✅ Offline mode confirmed")
```

::: {.output .stream .stdout}
    ✅ Offline mode confirmed
:::
:::

::: {#739e7817 .cell .code execution_count="17"}
``` python
import torch, subprocess, sys

# ── Config ────────────────────────────────────────────────────────────────
ws_cfg = CONFIGS_DIR / "workstation_config.yaml"
config = str(ws_cfg) if ws_cfg.exists() else str(CONFIGS_DIR / "kaggle_config.yaml")
print(f"Using config: {config}")

# ── Auto-detect checkpoints for resume ────────────────────────────────────
ckpts = sorted(
    CKPT_DIR.glob("checkpoint_epoch_*.pt"),
    key=lambda p: int(p.stem.split("_")[-1])
) if CKPT_DIR.exists() else []

RESUME = RESUME_CHECKPOINTS or bool(ckpts)

base_args = [str(SCRIPTS_DIR / "train.py"), "--config", config]
if RESUME and ckpts:
    last_ckpt = str(ckpts[-1])
    base_args += ["--resume-checkpoint", last_ckpt]
    print(f"Resuming from: {last_ckpt}")
elif RESUME:
    print("RESUME_CHECKPOINTS=True but no checkpoints found — starting fresh")
    RESUME = False

# ── Launch ────────────────────────────────────────────────────────────────
n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus}")

if n_gpus > 1:
    import shutil
    torchrun = shutil.which("torchrun") or "torchrun"
    cmd = [torchrun, f"--nproc_per_node={n_gpus}", "--master_port=29500"] + base_args
    print(f"Launching DDP on {n_gpus} GPUs")
else:
    cmd = [sys.executable] + base_args
    print("Single-GPU launch (RTX 4090)")

print("Running:", " ".join(cmd))
print()
result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
print("\nTraining exit code:", result.returncode)
```

::: {.output .stream .stdout}
    Using config: /home/mluser/projects_yash/new_project/PromptGFM-Bio/configs/workstation_config.yaml
    GPUs available: 1
    Single-GPU launch (RTX 4090)
    Running: /home/mluser/micromamba/envs/promptgfm/bin/python /home/mluser/projects_yash/new_project/PromptGFM-Bio/scripts/train.py --config /home/mluser/projects_yash/new_project/PromptGFM-Bio/configs/workstation_config.yaml
:::

::: {.output .stream .stderr}
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/libpyg.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_scatter/_scatter_cuda.so: undefined symbol: _ZN2at4_ops16div__Tensor_mode4callERNS_6TensorERKS2_St8optionalIN3c1017basic_string_viewIcEEE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-spline-conv'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_spline_conv/_basis_cuda.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_sparse/_spmm_cuda.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    INFO:__main__:✓ cuDNN autotuning enabled (first epoch may be slightly slower)
    INFO:__main__:Mode: finetune
    INFO:__main__:Config: /home/mluser/projects_yash/new_project/PromptGFM-Bio/configs/workstation_config.yaml
    INFO:__main__:Device: cuda
    INFO:__main__:✓ Mixed precision (AMP) enabled (1.5-2× speedup expected)
    INFO:__main__:
    ============================================================
    INFO:__main__:Starting Supervised Fine-tuning
    INFO:__main__:============================================================
    INFO:__main__:Creating dataloaders...
    INFO:src.data.dataset:Loading graph from data/processed/biomedical_graph.pt
    INFO:src.data.dataset:Graph loaded: gene=5363, disease=16841, phenotype=11794, ('gene', 'associated_with', 'disease')=9741610, ('disease', 'rev_associated_with', 'gene')=9741610
    INFO:src.data.dataset:Loading gene-disease edges from data/processed/hpo_gene_disease_edges.csv
    INFO:src.data.dataset:Vocabulary: 5251 genes, 12714 diseases
    INFO:src.data.dataset:Loaded 1170143 edges ({'HPO_phenotype_bridge': 1170143})
    INFO:src.data.dataset:Split sizes: train=936114, val=117014, test=117015
    INFO:__main__:  Train: 936114 edges
    INFO:__main__:  Val:   117014 edges
    INFO:__main__:  Test:  117015 edges
    INFO:__main__:  Gene nodes in graph: 5363
    INFO:__main__:  Genes in dataset vocabulary: 5251
    INFO:__main__:  Input feature dim: 128
    INFO:__main__:  Using 5 negative samples per positive sample
    INFO:root:No gene-gene edges found in graph. Available edge types: [('gene', 'associated_with', 'disease'), ('disease', 'rev_associated_with', 'gene')]. Training without message passing.
    INFO:root:No gene-gene edges found in graph. Available edge types: [('gene', 'associated_with', 'disease'), ('disease', 'rev_associated_with', 'gene')]. Training without message passing.
    INFO:__main__:  DataLoader workers: 4 (parallel data loading enabled)
    INFO:__main__:Training PromptGFM
    INFO:src.models.promptgfm:Initializing PromptGFM model...
    INFO:src.models.prompt_encoder:Initializing PromptEncoder with model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Loading weights: 100%|██████████| 199/199 [00:00<00:00, 108708.84it/s]
    BertModel LOAD REPORT from: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Key                                        | Status     |  | 
    -------------------------------------------+------------+--+-
    cls.seq_relationship.bias                  | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight                | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
    cls.predictions.bias                       | UNEXPECTED |  | 
    cls.predictions.decoder.weight             | UNEXPECTED |  | 
    cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
    cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
    cls.predictions.decoder.bias               | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    INFO:src.models.prompt_encoder:Freezing BERT parameters
    INFO:src.models.prompt_encoder:PromptEncoder initialized - hidden_size: 768, pooling: cls, max_length: 512
    INFO:src.models.gnn_backbone:GNN Backbone: graphsage, 3 layers, 128→512→512
    INFO:src.models.conditioning:FiLM Conditioning: node_dim=512, prompt_dim=768
    INFO:src.models.promptgfm:PromptGFM initialized:
    INFO:src.models.promptgfm:  GNN: graphsage, layers=3, dim=512
    INFO:src.models.promptgfm:  Prompt: cls pooling, dim=768
    INFO:src.models.promptgfm:  Conditioning: film
    INFO:src.models.promptgfm:  Predictor: hidden_dim=256
    INFO:src.evaluation.metrics:GeneRankingEvaluator initialized with K values: [10, 20, 50, 100]
    INFO:__main__:
      Pre-computing frozen BioBERT embeddings (runs once, then cached)...
    INFO:__main__:  ✅ Cached 12714 disease embeddings — BioBERT skipped per batch
    /home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:89: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler() if self.use_amp else None
    INFO:src.training.finetune:PromptGFMTrainer initialized:
    INFO:src.training.finetune: Device: cuda
    INFO:src.training.finetune: Max epochs: 100
    INFO:src.training.finetune: Patience: 15
    INFO:src.training.finetune: Gradient clip: 1.0
    INFO:src.training.finetune: Mixed precision (AMP): enabled
    INFO:src.training.finetune: Checkpoint dir: checkpoints/promptgfm_film
    INFO:src.training.finetune: Prompt embedding cache: 12714 unique disease texts — BioBERT will be skipped per batch
    INFO:src.training.finetune:
    Starting training from epoch 1 to 100
    INFO:src.training.finetune:Early stopping on: auroc
    Epoch 1/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 1/100: 100%|██████████| 1219/1219 [00:15<00:00, 81.25it/s, loss=0.517]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.75it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 1/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.9s (Avg: 16.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 27m (for 99 epochs)
    INFO:src.training.finetune: Train Loss: 0.545578
    INFO:src.training.finetune: Val Loss: 0.529018
    INFO:src.training.finetune: Val AUROC: 0.7764
    INFO:src.training.finetune: Val AUPR: 0.4307
    INFO:src.training.finetune: ✓ New best auroc: 0.7764 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 5)
    INFO:src.training.finetune:======================================================================

    Epoch 2/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 2/100: 100%|██████████| 1219/1219 [00:14<00:00, 82.48it/s, loss=0.496]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 107.50it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 2/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.5s (Avg: 16.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 27m (for 98 epochs)
    INFO:src.training.finetune: Train Loss: 0.518698
    INFO:src.training.finetune: Val Loss: 0.524172
    INFO:src.training.finetune: Val AUROC: 0.7806
    INFO:src.training.finetune: Val AUPR: 0.4384
    INFO:src.training.finetune: ✓ New best auroc: 0.7806 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 5)
    INFO:src.training.finetune:======================================================================

    Epoch 3/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 3/100: 100%|██████████| 1219/1219 [00:14<00:00, 85.08it/s, loss=0.518]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 109.95it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 3/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.1s (Avg: 16.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 26m (for 97 epochs)
    INFO:src.training.finetune: Train Loss: 0.510806
    INFO:src.training.finetune: Val Loss: 0.520407
    INFO:src.training.finetune: Val AUROC: 0.7835
    INFO:src.training.finetune: Val AUPR: 0.4400
    INFO:src.training.finetune: ✓ New best auroc: 0.7835 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 5)
    INFO:src.training.finetune:======================================================================

    Epoch 4/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 4/100: 100%|██████████| 1219/1219 [00:14<00:00, 82.42it/s, loss=0.511]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 106.57it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 4/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.5s (Avg: 16.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 26m (for 96 epochs)
    INFO:src.training.finetune: Train Loss: 0.506204
    INFO:src.training.finetune: Val Loss: 0.516819
    INFO:src.training.finetune: Val AUROC: 0.7863
    INFO:src.training.finetune: Val AUPR: 0.4437
    INFO:src.training.finetune: ✓ New best auroc: 0.7863 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 5)
    INFO:src.training.finetune:======================================================================

    Epoch 5/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 5/100: 100%|██████████| 1219/1219 [00:14<00:00, 84.51it/s, loss=0.52] 
    Validating: 100%|██████████| 153/153 [00:01<00:00, 114.19it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 5/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.1s (Avg: 16.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 25m (for 95 epochs)
    INFO:src.training.finetune: Train Loss: 0.502112
    INFO:src.training.finetune: Val Loss: 0.514044
    INFO:src.training.finetune: Val AUROC: 0.7885
    INFO:src.training.finetune: Val AUPR: 0.4478
    INFO:src.training.finetune: ✓ New best auroc: 0.7885 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_5.pt
    INFO:src.training.finetune:======================================================================

    Epoch 6/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 6/100: 100%|██████████| 1219/1219 [00:14<00:00, 84.77it/s, loss=0.502]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 115.77it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 6/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.0s (Avg: 16.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 25m (for 94 epochs)
    INFO:src.training.finetune: Train Loss: 0.498305
    INFO:src.training.finetune: Val Loss: 0.510430
    INFO:src.training.finetune: Val AUROC: 0.7921
    INFO:src.training.finetune: Val AUPR: 0.4520
    INFO:src.training.finetune: ✓ New best auroc: 0.7921 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 10)
    INFO:src.training.finetune:======================================================================

    Epoch 7/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 7/100: 100%|██████████| 1219/1219 [00:14<00:00, 83.37it/s, loss=0.488]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.75it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 7/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.6s (Avg: 16.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 25m (for 93 epochs)
    INFO:src.training.finetune: Train Loss: 0.494541
    INFO:src.training.finetune: Val Loss: 0.507857
    INFO:src.training.finetune: Val AUROC: 0.7941
    INFO:src.training.finetune: Val AUPR: 0.4540
    INFO:src.training.finetune: ✓ New best auroc: 0.7941 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 10)
    INFO:src.training.finetune:======================================================================

    Epoch 8/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 8/100: 100%|██████████| 1219/1219 [00:14<00:00, 82.91it/s, loss=0.485]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 113.45it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 8/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.4s (Avg: 16.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 25m (for 92 epochs)
    INFO:src.training.finetune: Train Loss: 0.490523
    INFO:src.training.finetune: Val Loss: 0.504921
    INFO:src.training.finetune: Val AUROC: 0.7973
    INFO:src.training.finetune: Val AUPR: 0.4563
    INFO:src.training.finetune: ✓ New best auroc: 0.7973 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 10)
    INFO:src.training.finetune:======================================================================

    Epoch 9/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 9/100: 100%|██████████| 1219/1219 [00:14<00:00, 84.82it/s, loss=0.498]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 107.53it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 9/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.1s (Avg: 16.3s/epoch)
    INFO:src.training.finetune: ETA: 0h 24m (for 91 epochs)
    INFO:src.training.finetune: Train Loss: 0.485451
    INFO:src.training.finetune: Val Loss: 0.502210
    INFO:src.training.finetune: Val AUROC: 0.7997
    INFO:src.training.finetune: Val AUPR: 0.4589
    INFO:src.training.finetune: ✓ New best auroc: 0.7997 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 10)
    INFO:src.training.finetune:======================================================================

    Epoch 10/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 10/100: 100%|██████████| 1219/1219 [00:14<00:00, 84.03it/s, loss=0.475]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 102.67it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 10/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.3s (Avg: 16.3s/epoch)
    INFO:src.training.finetune: ETA: 0h 24m (for 90 epochs)
    INFO:src.training.finetune: Train Loss: 0.480749
    INFO:src.training.finetune: Val Loss: 0.499468
    INFO:src.training.finetune: Val AUROC: 0.8027
    INFO:src.training.finetune: Val AUPR: 0.4588
    INFO:src.training.finetune: ✓ New best auroc: 0.8027 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_10.pt
    INFO:src.training.finetune:======================================================================

    Epoch 11/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 11/100: 100%|██████████| 1219/1219 [00:14<00:00, 82.92it/s, loss=0.473]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 114.75it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 11/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.4s (Avg: 16.3s/epoch)
    INFO:src.training.finetune: ETA: 0h 24m (for 89 epochs)
    INFO:src.training.finetune: Train Loss: 0.476769
    INFO:src.training.finetune: Val Loss: 0.498087
    INFO:src.training.finetune: Val AUROC: 0.8035
    INFO:src.training.finetune: Val AUPR: 0.4601
    INFO:src.training.finetune: ✓ New best auroc: 0.8035 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 15)
    INFO:src.training.finetune:======================================================================

    Epoch 12/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 12/100: 100%|██████████| 1219/1219 [00:14<00:00, 83.26it/s, loss=0.462]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 101.66it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 12/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 16.5s (Avg: 16.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 88 epochs)
    INFO:src.training.finetune: Train Loss: 0.472888
    INFO:src.training.finetune: Val Loss: 0.495032
    INFO:src.training.finetune: Val AUROC: 0.8064
    INFO:src.training.finetune: Val AUPR: 0.4646
    INFO:src.training.finetune: ✓ New best auroc: 0.8064 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 15)
    INFO:src.training.finetune:======================================================================

    Epoch 13/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 13/100: 100%|██████████| 1219/1219 [00:16<00:00, 76.18it/s, loss=0.459]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.09it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 13/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.9s (Avg: 16.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 87 epochs)
    INFO:src.training.finetune: Train Loss: 0.470378
    INFO:src.training.finetune: Val Loss: 0.494434
    INFO:src.training.finetune: Val AUROC: 0.8066
    INFO:src.training.finetune: Val AUPR: 0.4636
    INFO:src.training.finetune: ✓ New best auroc: 0.8066 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 15)
    INFO:src.training.finetune:======================================================================

    Epoch 14/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 14/100: 100%|██████████| 1219/1219 [00:15<00:00, 77.13it/s, loss=0.463]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 100.96it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 14/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.7s (Avg: 16.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 86 epochs)
    INFO:src.training.finetune: Train Loss: 0.467551
    INFO:src.training.finetune: Val Loss: 0.492311
    INFO:src.training.finetune: Val AUROC: 0.8087
    INFO:src.training.finetune: Val AUPR: 0.4636
    INFO:src.training.finetune: ✓ New best auroc: 0.8087 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 15)
    INFO:src.training.finetune:======================================================================

    Epoch 15/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 15/100: 100%|██████████| 1219/1219 [00:15<00:00, 77.27it/s, loss=0.461]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 97.45it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 15/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.7s (Avg: 16.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 85 epochs)
    INFO:src.training.finetune: Train Loss: 0.465270
    INFO:src.training.finetune: Val Loss: 0.491957
    INFO:src.training.finetune: Val AUROC: 0.8096
    INFO:src.training.finetune: Val AUPR: 0.4651
    INFO:src.training.finetune: ✓ New best auroc: 0.8096 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_15.pt
    INFO:src.training.finetune:======================================================================

    Epoch 16/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 16/100: 100%|██████████| 1219/1219 [00:16<00:00, 76.16it/s, loss=0.475]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.60it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 16/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.9s (Avg: 16.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 84 epochs)
    INFO:src.training.finetune: Train Loss: 0.463763
    INFO:src.training.finetune: Val Loss: 0.490987
    INFO:src.training.finetune: Val AUROC: 0.8094
    INFO:src.training.finetune: Val AUPR: 0.4621
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 20)
    INFO:src.training.finetune:======================================================================

    Epoch 17/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 17/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.71it/s, loss=0.461]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 100.66it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 17/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.2s (Avg: 16.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 83 epochs)
    INFO:src.training.finetune: Train Loss: 0.462147
    INFO:src.training.finetune: Val Loss: 0.490108
    INFO:src.training.finetune: Val AUROC: 0.8106
    INFO:src.training.finetune: Val AUPR: 0.4666
    INFO:src.training.finetune: ✓ New best auroc: 0.8106 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 20)
    INFO:src.training.finetune:======================================================================

    Epoch 18/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 18/100: 100%|██████████| 1219/1219 [00:15<00:00, 76.97it/s, loss=0.455]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.71it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 18/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.8s (Avg: 16.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 23m (for 82 epochs)
    INFO:src.training.finetune: Train Loss: 0.460529
    INFO:src.training.finetune: Val Loss: 0.489117
    INFO:src.training.finetune: Val AUROC: 0.8112
    INFO:src.training.finetune: Val AUPR: 0.4658
    INFO:src.training.finetune: ✓ New best auroc: 0.8112 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 20)
    INFO:src.training.finetune:======================================================================

    Epoch 19/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 19/100: 100%|██████████| 1219/1219 [00:15<00:00, 76.81it/s, loss=0.454]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.52it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 19/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.8s (Avg: 16.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 22m (for 81 epochs)
    INFO:src.training.finetune: Train Loss: 0.458461
    INFO:src.training.finetune: Val Loss: 0.487610
    INFO:src.training.finetune: Val AUROC: 0.8128
    INFO:src.training.finetune: Val AUPR: 0.4695
    INFO:src.training.finetune: ✓ New best auroc: 0.8128 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 20)
    INFO:src.training.finetune:======================================================================

    Epoch 20/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 20/100: 100%|██████████| 1219/1219 [00:16<00:00, 75.10it/s, loss=0.453]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 97.88it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 20/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.1s (Avg: 17.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 22m (for 80 epochs)
    INFO:src.training.finetune: Train Loss: 0.456807
    INFO:src.training.finetune: Val Loss: 0.486506
    INFO:src.training.finetune: Val AUROC: 0.8130
    INFO:src.training.finetune: Val AUPR: 0.4629
    INFO:src.training.finetune: ✓ New best auroc: 0.8130 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_20.pt
    INFO:src.training.finetune:======================================================================

    Epoch 21/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 21/100: 100%|██████████| 1219/1219 [00:15<00:00, 76.66it/s, loss=0.453]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.84it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 21/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.8s (Avg: 17.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 22m (for 79 epochs)
    INFO:src.training.finetune: Train Loss: 0.455712
    INFO:src.training.finetune: Val Loss: 0.486663
    INFO:src.training.finetune: Val AUROC: 0.8139
    INFO:src.training.finetune: Val AUPR: 0.4714
    INFO:src.training.finetune: ✓ New best auroc: 0.8139 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 25)
    INFO:src.training.finetune:======================================================================

    Epoch 22/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 22/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.88it/s, loss=0.458]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.70it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 22/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.2s (Avg: 17.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 22m (for 78 epochs)
    INFO:src.training.finetune: Train Loss: 0.454663
    INFO:src.training.finetune: Val Loss: 0.486504
    INFO:src.training.finetune: Val AUROC: 0.8145
    INFO:src.training.finetune: Val AUPR: 0.4697
    INFO:src.training.finetune: ✓ New best auroc: 0.8145 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 25)
    INFO:src.training.finetune:======================================================================

    Epoch 23/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 23/100: 100%|██████████| 1219/1219 [00:15<00:00, 76.36it/s, loss=0.455]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.42it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 23/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.9s (Avg: 17.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 21m (for 77 epochs)
    INFO:src.training.finetune: Train Loss: 0.454152
    INFO:src.training.finetune: Val Loss: 0.485646
    INFO:src.training.finetune: Val AUROC: 0.8145
    INFO:src.training.finetune: Val AUPR: 0.4692
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 25)
    INFO:src.training.finetune:======================================================================

    Epoch 24/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 24/100: 100%|██████████| 1219/1219 [00:15<00:00, 76.41it/s, loss=0.47] 
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.18it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 24/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 17.8s (Avg: 17.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 21m (for 76 epochs)
    INFO:src.training.finetune: Train Loss: 0.453241
    INFO:src.training.finetune: Val Loss: 0.484737
    INFO:src.training.finetune: Val AUROC: 0.8151
    INFO:src.training.finetune: Val AUPR: 0.4730
    INFO:src.training.finetune: ✓ New best auroc: 0.8151 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 25)
    INFO:src.training.finetune:======================================================================

    Epoch 25/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 25/100: 100%|██████████| 1219/1219 [00:16<00:00, 75.42it/s, loss=0.464]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.98it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 25/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.1s (Avg: 17.2s/epoch)
    INFO:src.training.finetune: ETA: 0h 21m (for 75 epochs)
    INFO:src.training.finetune: Train Loss: 0.452156
    INFO:src.training.finetune: Val Loss: 0.484558
    INFO:src.training.finetune: Val AUROC: 0.8153
    INFO:src.training.finetune: Val AUPR: 0.4707
    INFO:src.training.finetune: ✓ New best auroc: 0.8153 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_25.pt
    INFO:src.training.finetune:======================================================================

    Epoch 26/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 26/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.94it/s, loss=0.442]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.39it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 26/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 17.2s/epoch)
    INFO:src.training.finetune: ETA: 0h 21m (for 74 epochs)
    INFO:src.training.finetune: Train Loss: 0.451053
    INFO:src.training.finetune: Val Loss: 0.484901
    INFO:src.training.finetune: Val AUROC: 0.8158
    INFO:src.training.finetune: Val AUPR: 0.4736
    INFO:src.training.finetune: ✓ New best auroc: 0.8158 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 30)
    INFO:src.training.finetune:======================================================================

    Epoch 27/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 27/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.80it/s, loss=0.468]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 94.89it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 27/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.3s/epoch)
    INFO:src.training.finetune: ETA: 0h 21m (for 73 epochs)
    INFO:src.training.finetune: Train Loss: 0.450218
    INFO:src.training.finetune: Val Loss: 0.482250
    INFO:src.training.finetune: Val AUROC: 0.8176
    INFO:src.training.finetune: Val AUPR: 0.4727
    INFO:src.training.finetune: ✓ New best auroc: 0.8176 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 30)
    INFO:src.training.finetune:======================================================================

    Epoch 28/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 28/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.68it/s, loss=0.453]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 94.84it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 28/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.3s/epoch)
    INFO:src.training.finetune: ETA: 0h 20m (for 72 epochs)
    INFO:src.training.finetune: Train Loss: 0.448985
    INFO:src.training.finetune: Val Loss: 0.482639
    INFO:src.training.finetune: Val AUROC: 0.8171
    INFO:src.training.finetune: Val AUPR: 0.4709
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 30)
    INFO:src.training.finetune:======================================================================

    Epoch 29/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 29/100: 100%|██████████| 1219/1219 [00:17<00:00, 71.57it/s, loss=0.449]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.10it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 29/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 19.0s (Avg: 17.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 20m (for 71 epochs)
    INFO:src.training.finetune: Train Loss: 0.447540
    INFO:src.training.finetune: Val Loss: 0.481957
    INFO:src.training.finetune: Val AUROC: 0.8178
    INFO:src.training.finetune: Val AUPR: 0.4705
    INFO:src.training.finetune: ✓ New best auroc: 0.8178 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 30)
    INFO:src.training.finetune:======================================================================

    Epoch 30/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 30/100: 100%|██████████| 1219/1219 [00:16<00:00, 71.71it/s, loss=0.437]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 94.32it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 30/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 19.0s (Avg: 17.4s/epoch)
    INFO:src.training.finetune: ETA: 0h 20m (for 70 epochs)
    INFO:src.training.finetune: Train Loss: 0.446727
    INFO:src.training.finetune: Val Loss: 0.483270
    INFO:src.training.finetune: Val AUROC: 0.8175
    INFO:src.training.finetune: Val AUPR: 0.4736
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_30.pt
    INFO:src.training.finetune:======================================================================

    Epoch 31/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 31/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.14it/s, loss=0.447]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 89.25it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 31/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 20m (for 69 epochs)
    INFO:src.training.finetune: Train Loss: 0.445632
    INFO:src.training.finetune: Val Loss: 0.480676
    INFO:src.training.finetune: Val AUROC: 0.8193
    INFO:src.training.finetune: Val AUPR: 0.4771
    INFO:src.training.finetune: ✓ New best auroc: 0.8193 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 35)
    INFO:src.training.finetune:======================================================================

    Epoch 32/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 32/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.69it/s, loss=0.458]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.20it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 32/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.2s (Avg: 17.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 19m (for 68 epochs)
    INFO:src.training.finetune: Train Loss: 0.444605
    INFO:src.training.finetune: Val Loss: 0.481019
    INFO:src.training.finetune: Val AUROC: 0.8182
    INFO:src.training.finetune: Val AUPR: 0.4731
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 35)
    INFO:src.training.finetune:======================================================================

    Epoch 33/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 33/100: 100%|██████████| 1219/1219 [00:16<00:00, 75.41it/s, loss=0.435]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 92.24it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 33/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.2s (Avg: 17.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 19m (for 67 epochs)
    INFO:src.training.finetune: Train Loss: 0.443769
    INFO:src.training.finetune: Val Loss: 0.482437
    INFO:src.training.finetune: Val AUROC: 0.8175
    INFO:src.training.finetune: Val AUPR: 0.4723
    INFO:src.training.finetune: No improvement for 2 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 35)
    INFO:src.training.finetune:======================================================================

    Epoch 34/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 34/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.95it/s, loss=0.436]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.82it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 34/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 17.5s/epoch)
    INFO:src.training.finetune: ETA: 0h 19m (for 66 epochs)
    INFO:src.training.finetune: Train Loss: 0.442731
    INFO:src.training.finetune: Val Loss: 0.481436
    INFO:src.training.finetune: Val AUROC: 0.8191
    INFO:src.training.finetune: Val AUPR: 0.4723
    INFO:src.training.finetune: No improvement for 3 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 35)
    INFO:src.training.finetune:======================================================================

    Epoch 35/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 35/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.48it/s, loss=0.428]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.79it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 35/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 19m (for 65 epochs)
    INFO:src.training.finetune: Train Loss: 0.441706
    INFO:src.training.finetune: Val Loss: 0.480591
    INFO:src.training.finetune: Val AUROC: 0.8193
    INFO:src.training.finetune: Val AUPR: 0.4753
    INFO:src.training.finetune: ✓ New best auroc: 0.8193 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_35.pt
    INFO:src.training.finetune:======================================================================

    Epoch 36/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 36/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.64it/s, loss=0.421]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.76it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 36/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.3s (Avg: 17.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 18m (for 64 epochs)
    INFO:src.training.finetune: Train Loss: 0.441067
    INFO:src.training.finetune: Val Loss: 0.480235
    INFO:src.training.finetune: Val AUROC: 0.8197
    INFO:src.training.finetune: Val AUPR: 0.4728
    INFO:src.training.finetune: ✓ New best auroc: 0.8197 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 40)
    INFO:src.training.finetune:======================================================================

    Epoch 37/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 37/100: 100%|██████████| 1219/1219 [00:16<00:00, 75.57it/s, loss=0.437]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.99it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 37/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.1s (Avg: 17.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 18m (for 63 epochs)
    INFO:src.training.finetune: Train Loss: 0.439834
    INFO:src.training.finetune: Val Loss: 0.479527
    INFO:src.training.finetune: Val AUROC: 0.8204
    INFO:src.training.finetune: Val AUPR: 0.4694
    INFO:src.training.finetune: ✓ New best auroc: 0.8204 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 40)
    INFO:src.training.finetune:======================================================================

    Epoch 38/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 38/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.07it/s, loss=0.427]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 97.39it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 38/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 17.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 18m (for 62 epochs)
    INFO:src.training.finetune: Train Loss: 0.438845
    INFO:src.training.finetune: Val Loss: 0.480347
    INFO:src.training.finetune: Val AUROC: 0.8203
    INFO:src.training.finetune: Val AUPR: 0.4751
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 40)
    INFO:src.training.finetune:======================================================================

    Epoch 39/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 39/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.66it/s, loss=0.465]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 97.66it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 39/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.3s (Avg: 17.6s/epoch)
    INFO:src.training.finetune: ETA: 0h 17m (for 61 epochs)
    INFO:src.training.finetune: Train Loss: 0.438285
    INFO:src.training.finetune: Val Loss: 0.479608
    INFO:src.training.finetune: Val AUROC: 0.8211
    INFO:src.training.finetune: Val AUPR: 0.4719
    INFO:src.training.finetune: ✓ New best auroc: 0.8211 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 40)
    INFO:src.training.finetune:======================================================================

    Epoch 40/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 40/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.85it/s, loss=0.433]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.16it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 40/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 17m (for 60 epochs)
    INFO:src.training.finetune: Train Loss: 0.437634
    INFO:src.training.finetune: Val Loss: 0.480373
    INFO:src.training.finetune: Val AUROC: 0.8205
    INFO:src.training.finetune: Val AUPR: 0.4731
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_40.pt
    INFO:src.training.finetune:======================================================================

    Epoch 41/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 41/100: 100%|██████████| 1219/1219 [00:16<00:00, 75.36it/s, loss=0.433]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.90it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 41/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.1s (Avg: 17.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 17m (for 59 epochs)
    INFO:src.training.finetune: Train Loss: 0.436105
    INFO:src.training.finetune: Val Loss: 0.480673
    INFO:src.training.finetune: Val AUROC: 0.8206
    INFO:src.training.finetune: Val AUPR: 0.4732
    INFO:src.training.finetune: No improvement for 2 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 45)
    INFO:src.training.finetune:======================================================================

    Epoch 42/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 42/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.11it/s, loss=0.425]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.67it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 42/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 17.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 17m (for 58 epochs)
    INFO:src.training.finetune: Train Loss: 0.435711
    INFO:src.training.finetune: Val Loss: 0.477975
    INFO:src.training.finetune: Val AUROC: 0.8225
    INFO:src.training.finetune: Val AUPR: 0.4753
    INFO:src.training.finetune: ✓ New best auroc: 0.8225 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 45)
    INFO:src.training.finetune:======================================================================

    Epoch 43/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 43/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.14it/s, loss=0.432]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 88.35it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 43/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 16m (for 57 epochs)
    INFO:src.training.finetune: Train Loss: 0.434747
    INFO:src.training.finetune: Val Loss: 0.479960
    INFO:src.training.finetune: Val AUROC: 0.8216
    INFO:src.training.finetune: Val AUPR: 0.4719
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 45)
    INFO:src.training.finetune:======================================================================

    Epoch 44/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 44/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.01it/s, loss=0.448]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.01it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 44/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.6s (Avg: 17.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 16m (for 56 epochs)
    INFO:src.training.finetune: Train Loss: 0.433897
    INFO:src.training.finetune: Val Loss: 0.480877
    INFO:src.training.finetune: Val AUROC: 0.8208
    INFO:src.training.finetune: Val AUPR: 0.4743
    INFO:src.training.finetune: No improvement for 2 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 45)
    INFO:src.training.finetune:======================================================================

    Epoch 45/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 45/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.20it/s, loss=0.432]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.30it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 45/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 17.7s/epoch)
    INFO:src.training.finetune: ETA: 0h 16m (for 55 epochs)
    INFO:src.training.finetune: Train Loss: 0.432753
    INFO:src.training.finetune: Val Loss: 0.481032
    INFO:src.training.finetune: Val AUROC: 0.8201
    INFO:src.training.finetune: Val AUPR: 0.4753
    INFO:src.training.finetune: No improvement for 3 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_45.pt
    INFO:src.training.finetune:======================================================================

    Epoch 46/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 46/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.88it/s, loss=0.423]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 88.89it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 46/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.6s (Avg: 17.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 15m (for 54 epochs)
    INFO:src.training.finetune: Train Loss: 0.431930
    INFO:src.training.finetune: Val Loss: 0.480628
    INFO:src.training.finetune: Val AUROC: 0.8213
    INFO:src.training.finetune: Val AUPR: 0.4747
    INFO:src.training.finetune: No improvement for 4 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 50)
    INFO:src.training.finetune:======================================================================

    Epoch 47/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 47/100: 100%|██████████| 1219/1219 [00:17<00:00, 71.69it/s, loss=0.434]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 89.99it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 47/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 19.1s (Avg: 17.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 15m (for 53 epochs)
    INFO:src.training.finetune: Train Loss: 0.431217
    INFO:src.training.finetune: Val Loss: 0.478690
    INFO:src.training.finetune: Val AUROC: 0.8224
    INFO:src.training.finetune: Val AUPR: 0.4756
    INFO:src.training.finetune: No improvement for 5 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 50)
    INFO:src.training.finetune:======================================================================

    Epoch 48/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 48/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.86it/s, loss=0.434]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.25it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 48/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 17.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 15m (for 52 epochs)
    INFO:src.training.finetune: Train Loss: 0.430565
    INFO:src.training.finetune: Val Loss: 0.478883
    INFO:src.training.finetune: Val AUROC: 0.8224
    INFO:src.training.finetune: Val AUPR: 0.4780
    INFO:src.training.finetune: No improvement for 6 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 50)
    INFO:src.training.finetune:======================================================================

    Epoch 49/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 49/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.24it/s, loss=0.431]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.59it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 49/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 17.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 15m (for 51 epochs)
    INFO:src.training.finetune: Train Loss: 0.429882
    INFO:src.training.finetune: Val Loss: 0.479884
    INFO:src.training.finetune: Val AUROC: 0.8227
    INFO:src.training.finetune: Val AUPR: 0.4738
    INFO:src.training.finetune: ✓ New best auroc: 0.8227 (saved as best_model.pt)
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 50)
    INFO:src.training.finetune:======================================================================

    Epoch 50/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 50/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.66it/s, loss=0.449]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.48it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 50/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.2s (Avg: 17.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 14m (for 50 epochs)
    INFO:src.training.finetune: Train Loss: 0.428959
    INFO:src.training.finetune: Val Loss: 0.483267
    INFO:src.training.finetune: Val AUROC: 0.8202
    INFO:src.training.finetune: Val AUPR: 0.4711
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_50.pt
    INFO:src.training.finetune:======================================================================

    Epoch 51/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 51/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.47it/s, loss=0.429]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.12it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 51/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.8s/epoch)
    INFO:src.training.finetune: ETA: 0h 14m (for 49 epochs)
    INFO:src.training.finetune: Train Loss: 0.428368
    INFO:src.training.finetune: Val Loss: 0.481396
    INFO:src.training.finetune: Val AUROC: 0.8220
    INFO:src.training.finetune: Val AUPR: 0.4706
    INFO:src.training.finetune: No improvement for 2 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 55)
    INFO:src.training.finetune:======================================================================

    Epoch 52/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 52/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.63it/s, loss=0.431]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.79it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 52/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 14m (for 48 epochs)
    INFO:src.training.finetune: Train Loss: 0.427319
    INFO:src.training.finetune: Val Loss: 0.483793
    INFO:src.training.finetune: Val AUROC: 0.8205
    INFO:src.training.finetune: Val AUPR: 0.4734
    INFO:src.training.finetune: No improvement for 3 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 55)
    INFO:src.training.finetune:======================================================================

    Epoch 53/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 53/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.13it/s, loss=0.437]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.79it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 53/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.3s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 13m (for 47 epochs)
    INFO:src.training.finetune: Train Loss: 0.426568
    INFO:src.training.finetune: Val Loss: 0.482181
    INFO:src.training.finetune: Val AUROC: 0.8217
    INFO:src.training.finetune: Val AUPR: 0.4740
    INFO:src.training.finetune: No improvement for 4 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 55)
    INFO:src.training.finetune:======================================================================

    Epoch 54/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 54/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.93it/s, loss=0.439]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.72it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 54/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.6s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 13m (for 46 epochs)
    INFO:src.training.finetune: Train Loss: 0.425577
    INFO:src.training.finetune: Val Loss: 0.481271
    INFO:src.training.finetune: Val AUROC: 0.8222
    INFO:src.training.finetune: Val AUPR: 0.4691
    INFO:src.training.finetune: No improvement for 5 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 55)
    INFO:src.training.finetune:======================================================================

    Epoch 55/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 55/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.89it/s, loss=0.424]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.73it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 55/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.2s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 13m (for 45 epochs)
    INFO:src.training.finetune: Train Loss: 0.425304
    INFO:src.training.finetune: Val Loss: 0.480562
    INFO:src.training.finetune: Val AUROC: 0.8228
    INFO:src.training.finetune: Val AUPR: 0.4748
    INFO:src.training.finetune: ✓ New best auroc: 0.8228 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_55.pt
    INFO:src.training.finetune:======================================================================

    Epoch 56/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 56/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.71it/s, loss=0.443]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.81it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 56/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.5s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 13m (for 44 epochs)
    INFO:src.training.finetune: Train Loss: 0.424246
    INFO:src.training.finetune: Val Loss: 0.481646
    INFO:src.training.finetune: Val AUROC: 0.8222
    INFO:src.training.finetune: Val AUPR: 0.4764
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 60)
    INFO:src.training.finetune:======================================================================

    Epoch 57/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 57/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.06it/s, loss=0.411]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.07it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 57/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.6s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 12m (for 43 epochs)
    INFO:src.training.finetune: Train Loss: 0.423677
    INFO:src.training.finetune: Val Loss: 0.480899
    INFO:src.training.finetune: Val AUROC: 0.8224
    INFO:src.training.finetune: Val AUPR: 0.4669
    INFO:src.training.finetune: No improvement for 2 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 60)
    INFO:src.training.finetune:======================================================================

    Epoch 58/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 58/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.39it/s, loss=0.445]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.67it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 58/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.8s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 12m (for 42 epochs)
    INFO:src.training.finetune: Train Loss: 0.422907
    INFO:src.training.finetune: Val Loss: 0.482688
    INFO:src.training.finetune: Val AUROC: 0.8220
    INFO:src.training.finetune: Val AUPR: 0.4676
    INFO:src.training.finetune: No improvement for 3 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 60)
    INFO:src.training.finetune:======================================================================

    Epoch 59/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 59/100: 100%|██████████| 1219/1219 [00:16<00:00, 75.41it/s, loss=0.428]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.92it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 59/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.1s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 12m (for 41 epochs)
    INFO:src.training.finetune: Train Loss: 0.422341
    INFO:src.training.finetune: Val Loss: 0.482319
    INFO:src.training.finetune: Val AUROC: 0.8226
    INFO:src.training.finetune: Val AUPR: 0.4697
    INFO:src.training.finetune: No improvement for 4 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 60)
    INFO:src.training.finetune:======================================================================

    Epoch 60/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 60/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.13it/s, loss=0.421]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.60it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 60/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 11m (for 40 epochs)
    INFO:src.training.finetune: Train Loss: 0.421451
    INFO:src.training.finetune: Val Loss: 0.479343
    INFO:src.training.finetune: Val AUROC: 0.8245
    INFO:src.training.finetune: Val AUPR: 0.4710
    INFO:src.training.finetune: ✓ New best auroc: 0.8245 (saved as best_model.pt)
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_60.pt
    INFO:src.training.finetune:======================================================================

    Epoch 61/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 61/100: 100%|██████████| 1219/1219 [00:17<00:00, 71.70it/s, loss=0.409]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 101.28it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 61/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.9s (Avg: 17.9s/epoch)
    INFO:src.training.finetune: ETA: 0h 11m (for 39 epochs)
    INFO:src.training.finetune: Train Loss: 0.420726
    INFO:src.training.finetune: Val Loss: 0.481696
    INFO:src.training.finetune: Val AUROC: 0.8232
    INFO:src.training.finetune: Val AUPR: 0.4670
    INFO:src.training.finetune: No improvement for 1 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 65)
    INFO:src.training.finetune:======================================================================

    Epoch 62/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 62/100: 100%|██████████| 1219/1219 [00:17<00:00, 71.20it/s, loss=0.425]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 101.00it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 62/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 19.0s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 11m (for 38 epochs)
    INFO:src.training.finetune: Train Loss: 0.420062
    INFO:src.training.finetune: Val Loss: 0.482365
    INFO:src.training.finetune: Val AUROC: 0.8230
    INFO:src.training.finetune: Val AUPR: 0.4699
    INFO:src.training.finetune: No improvement for 2 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 65)
    INFO:src.training.finetune:======================================================================

    Epoch 63/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 63/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.96it/s, loss=0.419]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 98.02it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 63/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 11m (for 37 epochs)
    INFO:src.training.finetune: Train Loss: 0.419561
    INFO:src.training.finetune: Val Loss: 0.482982
    INFO:src.training.finetune: Val AUROC: 0.8231
    INFO:src.training.finetune: Val AUPR: 0.4687
    INFO:src.training.finetune: No improvement for 3 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 65)
    INFO:src.training.finetune:======================================================================

    Epoch 64/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 64/100: 100%|██████████| 1219/1219 [00:16<00:00, 74.72it/s, loss=0.403]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.69it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 64/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.3s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 10m (for 36 epochs)
    INFO:src.training.finetune: Train Loss: 0.418861
    INFO:src.training.finetune: Val Loss: 0.483881
    INFO:src.training.finetune: Val AUROC: 0.8224
    INFO:src.training.finetune: Val AUPR: 0.4659
    INFO:src.training.finetune: No improvement for 4 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 65)
    INFO:src.training.finetune:======================================================================

    Epoch 65/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 65/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.46it/s, loss=0.41] 
    Validating: 100%|██████████| 153/153 [00:01<00:00, 97.97it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 65/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.8s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 10m (for 35 epochs)
    INFO:src.training.finetune: Train Loss: 0.417996
    INFO:src.training.finetune: Val Loss: 0.482130
    INFO:src.training.finetune: Val AUROC: 0.8236
    INFO:src.training.finetune: Val AUPR: 0.4643
    INFO:src.training.finetune: No improvement for 5 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_65.pt
    INFO:src.training.finetune:======================================================================

    Epoch 66/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 66/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.25it/s, loss=0.42] 
    Validating: 100%|██████████| 153/153 [00:01<00:00, 99.16it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 66/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.6s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 10m (for 34 epochs)
    INFO:src.training.finetune: Train Loss: 0.417394
    INFO:src.training.finetune: Val Loss: 0.483532
    INFO:src.training.finetune: Val AUROC: 0.8233
    INFO:src.training.finetune: Val AUPR: 0.4683
    INFO:src.training.finetune: No improvement for 6 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 70)
    INFO:src.training.finetune:======================================================================

    Epoch 67/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 67/100: 100%|██████████| 1219/1219 [00:16<00:00, 71.94it/s, loss=0.398]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 90.51it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 67/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 19.0s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 9m (for 33 epochs)
    INFO:src.training.finetune: Train Loss: 0.416771
    INFO:src.training.finetune: Val Loss: 0.484469
    INFO:src.training.finetune: Val AUROC: 0.8228
    INFO:src.training.finetune: Val AUPR: 0.4629
    INFO:src.training.finetune: No improvement for 7 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 70)
    INFO:src.training.finetune:======================================================================

    Epoch 68/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 68/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.94it/s, loss=0.417]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.38it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 68/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 9m (for 32 epochs)
    INFO:src.training.finetune: Train Loss: 0.416176
    INFO:src.training.finetune: Val Loss: 0.486267
    INFO:src.training.finetune: Val AUROC: 0.8224
    INFO:src.training.finetune: Val AUPR: 0.4652
    INFO:src.training.finetune: No improvement for 8 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 70)
    INFO:src.training.finetune:======================================================================

    Epoch 69/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 69/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.01it/s, loss=0.399]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.11it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 69/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 9m (for 31 epochs)
    INFO:src.training.finetune: Train Loss: 0.415788
    INFO:src.training.finetune: Val Loss: 0.483794
    INFO:src.training.finetune: Val AUROC: 0.8233
    INFO:src.training.finetune: Val AUPR: 0.4661
    INFO:src.training.finetune: No improvement for 9 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 70)
    INFO:src.training.finetune:======================================================================

    Epoch 70/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 70/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.85it/s, loss=0.409]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 96.30it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 70/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 9m (for 30 epochs)
    INFO:src.training.finetune: Train Loss: 0.415071
    INFO:src.training.finetune: Val Loss: 0.485067
    INFO:src.training.finetune: Val AUROC: 0.8229
    INFO:src.training.finetune: Val AUPR: 0.4652
    INFO:src.training.finetune: No improvement for 10 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_70.pt
    INFO:src.training.finetune:======================================================================

    Epoch 71/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 71/100: 100%|██████████| 1219/1219 [00:16<00:00, 73.90it/s, loss=0.41] 
    Validating: 100%|██████████| 153/153 [00:01<00:00, 97.62it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 71/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.4s (Avg: 18.0s/epoch)
    INFO:src.training.finetune: ETA: 0h 8m (for 29 epochs)
    INFO:src.training.finetune: Train Loss: 0.414489
    INFO:src.training.finetune: Val Loss: 0.484366
    INFO:src.training.finetune: Val AUROC: 0.8234
    INFO:src.training.finetune: Val AUPR: 0.4666
    INFO:src.training.finetune: No improvement for 11 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 75)
    INFO:src.training.finetune:======================================================================

    Epoch 72/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 72/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.83it/s, loss=0.422]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 93.55it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 72/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 18.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 8m (for 28 epochs)
    INFO:src.training.finetune: Train Loss: 0.413937
    INFO:src.training.finetune: Val Loss: 0.485557
    INFO:src.training.finetune: Val AUROC: 0.8232
    INFO:src.training.finetune: Val AUPR: 0.4649
    INFO:src.training.finetune: No improvement for 12 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 75)
    INFO:src.training.finetune:======================================================================

    Epoch 73/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 73/100: 100%|██████████| 1219/1219 [00:17<00:00, 71.32it/s, loss=0.423]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.95it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 73/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 19.0s (Avg: 18.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 8m (for 27 epochs)
    INFO:src.training.finetune: Train Loss: 0.413428
    INFO:src.training.finetune: Val Loss: 0.485355
    INFO:src.training.finetune: Val AUROC: 0.8231
    INFO:src.training.finetune: Val AUPR: 0.4663
    INFO:src.training.finetune: No improvement for 13 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 75)
    INFO:src.training.finetune:======================================================================

    Epoch 74/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 74/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.43it/s, loss=0.436]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.40it/s] 
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 74/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.8s (Avg: 18.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 7m (for 26 epochs)
    INFO:src.training.finetune: Train Loss: 0.412644
    INFO:src.training.finetune: Val Loss: 0.486607
    INFO:src.training.finetune: Val AUROC: 0.8235
    INFO:src.training.finetune: Val AUPR: 0.4625
    INFO:src.training.finetune: No improvement for 14 epochs
    INFO:src.training.finetune: Skipping checkpoint (next at epoch 75)
    INFO:src.training.finetune:======================================================================

    Epoch 75/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:146: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 75/100: 100%|██████████| 1219/1219 [00:16<00:00, 72.87it/s, loss=0.442]
    Validating: 100%|██████████| 153/153 [00:01<00:00, 95.59it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 75/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune: Time: 18.7s (Avg: 18.1s/epoch)
    INFO:src.training.finetune: ETA: 0h 7m (for 25 epochs)
    INFO:src.training.finetune: Train Loss: 0.412377
    INFO:src.training.finetune: Val Loss: 0.486655
    INFO:src.training.finetune: Val AUROC: 0.8224
    INFO:src.training.finetune: Val AUPR: 0.4647
    INFO:src.training.finetune: No improvement for 15 epochs
    INFO:src.training.finetune: Periodic checkpoint saved: checkpoint_epoch_75.pt
    INFO:src.training.finetune:======================================================================

    INFO:src.training.finetune:
    Early stopping after 75 epochs
    INFO:src.training.finetune:
    Training complete!
    INFO:src.training.finetune:Best auroc: 0.8245
    INFO:src.training.finetune:Loading best model from checkpoints/promptgfm_film/best_model.pt
    INFO:src.training.finetune:Loading checkpoint: checkpoints/promptgfm_film/best_model.pt
    INFO:src.training.finetune:✓ Resumed from epoch 60
    INFO:src.training.finetune: Best val metric: 0.8245
    INFO:src.training.finetune: Global step: 73140
    INFO:__main__:
    ✓ Fine-tuning complete!
:::

::: {.output .stream .stdout}

    Training exit code: 0
:::
:::

::: {#dbf94fae .cell .markdown}
## 12.5. 📊 Post-Training GPU Check {#125--post-training-gpu-check}

See peak memory usage --- helps decide if batch_size can go higher next
time.
:::

::: {#f3a077a4 .cell .code execution_count="18"}
``` python
import subprocess, torch

subprocess.run(["nvidia-smi"])

if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    reserved = torch.cuda.max_memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print()
    print(f"PyTorch peak allocated : {peak:.1f} GB")
    print(f"PyTorch peak reserved  : {reserved:.1f} GB")
    print(f"Total VRAM             : {total:.1f} GB")
    print(f"Headroom               : {total - reserved:.1f} GB")
    if total - reserved > 4.0:
        print("💡 You have headroom — try a larger batch_size next run.")
    elif total - reserved < 1.0:
        print("⚠️  Very tight — reduce batch_size if you see OOM errors.")
```

::: {.output .stream .stdout}
    Thu Apr  2 17:14:27 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
    | 34%   55C    P2             66W /  450W |     120MiB /  24564MiB |     24%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

    PyTorch peak allocated : 0.0 GB
    PyTorch peak reserved  : 0.0 GB
    Total VRAM             : 25.2 GB
    Headroom               : 25.2 GB
    💡 You have headroom — try a larger batch_size next run.
:::
:::

::: {#dbe9715c .cell .markdown}
## 13. 💾 Backup to GitHub / HuggingFace {#13--backup-to-github--huggingface}

> **CRITICAL**: Workstation wipes all data after 5 days!
:::

::: {#3e0b4a72 .cell .code execution_count="19"}
``` python
import subprocess, os

os.environ["GIT_TERMINAL_PROMPT"] = "0"

if not (PROJECT_ROOT / ".git").is_dir():
    print("⚠️  Not a git repo — cannot push. Back up manually:")
    print(f"   scp -r {CKPT_DIR} your-laptop:~/backups/")
else:
    for pattern in ["checkpoints/", "logs/"]:
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", "-f", pattern])

    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True, text=True
    )

    if status.stdout.strip():
        ckpts = sorted(
            CKPT_DIR.glob("checkpoint_epoch_*.pt"),
            key=lambda f: int(f.stem.split("_")[-1])
        ) if CKPT_DIR.exists() else []
        epoch = ckpts[-1].stem.split("_")[-1] if ckpts else "?"

        subprocess.run([
            "git", "-C", str(PROJECT_ROOT), "commit", "-m",
            f"Workstation training: epoch {epoch} checkpoints"
        ])
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "push"])
        if result.returncode == 0:
            print("✅ Pushed to GitHub")
        else:
            print("❌ Git push failed — check GITHUB_TOKEN in .env")
    else:
        print("Nothing new to commit")

print()
print("For large files, consider HuggingFace Hub:")
print(f"  huggingface-cli upload your-username/promptgfm-bio {HF_CACHE_DIR} --repo-type model")
```

::: {.output .stream .stdout}
    ⚠️  Not a git repo — cannot push. Back up manually:
       scp -r /home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film your-laptop:~/backups/

    For large files, consider HuggingFace Hub:
      huggingface-cli upload your-username/promptgfm-bio /home/mluser/projects_yash/new_project/PromptGFM-Bio/hf_cache --repo-type model
:::
:::

::: {#e03b8aa3 .cell .markdown}
## 14. Quick Evaluation {#14-quick-evaluation}
:::

::: {#84659f92 .cell .code execution_count="20"}
``` python
'''
import subprocess, sys

ws_cfg = CONFIGS_DIR / "workstation_config.yaml"
config = str(ws_cfg) if ws_cfg.exists() else str(CONFIGS_DIR / "kaggle_config.yaml")

best = CKPT_DIR / "best_model.pt"
if not best.exists():
    print("No best_model.pt yet — run more training epochs first")
else:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "evaluate.py"),
         "--config", config,
         "--checkpoint", str(best)],
        cwd=str(PROJECT_ROOT),
    )
    print("Evaluation exit code:", result.returncode)
'''
```

::: {.output .execute_result execution_count="20"}
    '\nimport subprocess, sys\n\nws_cfg = CONFIGS_DIR / "workstation_config.yaml"\nconfig = str(ws_cfg) if ws_cfg.exists() else str(CONFIGS_DIR / "kaggle_config.yaml")\n\nbest = CKPT_DIR / "best_model.pt"\nif not best.exists():\n    print("No best_model.pt yet — run more training epochs first")\nelse:\n    result = subprocess.run(\n        [sys.executable, str(SCRIPTS_DIR / "evaluate.py"),\n         "--config", config,\n         "--checkpoint", str(best)],\n        cwd=str(PROJECT_ROOT),\n    )\n    print("Evaluation exit code:", result.returncode)\n'
:::
:::

::: {#21e4a58b-8c03-4a27-b84c-2e57a9b119bf .cell .code execution_count="21"}
``` python
import subprocess, sys

# 🔴 FORCE correct config (no auto-switching)
config = str(CONFIGS_DIR / "workstation_config.yaml")

best = CKPT_DIR / "best_model.pt"

if not best.exists():
    print("No best_model.pt yet — run more training epochs first")
else:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate.py"),
            "--config", config,
            "--checkpoint", str(best),
        ],
        cwd=str(PROJECT_ROOT),
    )
    print("Evaluation exit code:", result.returncode)
```

::: {.output .stream .stderr}
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/libpyg.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_scatter/_scatter_cuda.so: undefined symbol: _ZN2at4_ops16div__Tensor_mode4callERNS_6TensorERKS2_St8optionalIN3c1017basic_string_viewIcEEE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-spline-conv'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_spline_conv/_basis_cuda.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_geometric/__init__.py:4: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages/torch_sparse/_spmm_cuda.so: undefined symbol: _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
      import torch_geometric.typing
    INFO:__main__:Device: cuda
    INFO:__main__:Building PromptGFM for evaluation
    INFO:__main__:  Model params: gnn_input=128, gnn_hidden=512, predictor_hidden=256
    INFO:src.models.promptgfm:Initializing PromptGFM model...
    INFO:src.models.prompt_encoder:Initializing PromptEncoder with model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Loading weights: 100%|██████████| 199/199 [00:00<00:00, 107090.90it/s]
    BertModel LOAD REPORT from: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Key                                        | Status     |  | 
    -------------------------------------------+------------+--+-
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
    cls.predictions.decoder.bias               | UNEXPECTED |  | 
    cls.predictions.bias                       | UNEXPECTED |  | 
    cls.predictions.decoder.weight             | UNEXPECTED |  | 
    cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
    cls.seq_relationship.weight                | UNEXPECTED |  | 
    cls.seq_relationship.bias                  | UNEXPECTED |  | 
    cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    INFO:src.models.prompt_encoder:Freezing BERT parameters
    INFO:src.models.prompt_encoder:PromptEncoder initialized - hidden_size: 768, pooling: cls, max_length: 512
    INFO:src.models.gnn_backbone:GNN Backbone: graphsage, 3 layers, 128→512→512
    INFO:src.models.conditioning:FiLM Conditioning: node_dim=512, prompt_dim=768
    INFO:src.models.promptgfm:PromptGFM initialized:
    INFO:src.models.promptgfm:  GNN: graphsage, layers=3, dim=512
    INFO:src.models.promptgfm:  Prompt: cls pooling, dim=768
    INFO:src.models.promptgfm:  Conditioning: film
    INFO:src.models.promptgfm:  Predictor: hidden_dim=256
    INFO:__main__:Loading checkpoint from /home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film/best_model.pt
    INFO:__main__:✓ Model loaded and set to eval mode
    INFO:src.data.dataset:Loading graph from data/processed/biomedical_graph.pt
    INFO:src.data.dataset:Graph loaded: gene=5363, disease=16841, phenotype=11794, ('gene', 'associated_with', 'disease')=9741610, ('disease', 'rev_associated_with', 'gene')=9741610
    INFO:src.data.dataset:Loading gene-disease edges from data/processed/hpo_gene_disease_edges.csv
    INFO:src.data.dataset:Vocabulary: 5251 genes, 12714 diseases
    INFO:src.data.dataset:Loaded 1170143 edges ({'HPO_phenotype_bridge': 1170143})
    INFO:src.data.dataset:Split sizes: train=936114, val=117014, test=117015
    INFO:__main__:  Train: 936114 edges, Val: 117014 edges, Test: 117015 edges
    INFO:__main__:
    ============================================================
    INFO:__main__:Evaluating TEST split
    INFO:__main__:============================================================
    INFO:__main__:  Ranking 5363 genes for each of 10267 disease queries
    Evaluating queries: 100%|██████████| 10267/10267 [44:46<00:00,  3.82it/s]
    INFO:__main__:  Evaluated 10267 queries with at least one test positive
    INFO:src.evaluation.metrics:GeneRankingEvaluator initialized with K values: [10, 20, 50, 100]
    INFO:src.evaluation.metrics:GeneRankingEvaluator initialized with K values: [10, 20, 50, 100]
    INFO:src.evaluation.metrics:
    test:
    INFO:src.evaluation.metrics:  AUROC: 0.8459
    INFO:src.evaluation.metrics:  AUPR:  0.0166
    INFO:src.evaluation.metrics:  Precision@10: 0.0570
    INFO:src.evaluation.metrics:  Recall@10: 0.0740
    INFO:src.evaluation.metrics:  NDCG@10: 0.0815
    INFO:src.evaluation.metrics:  Hit Rate@10: 0.3277
    INFO:src.evaluation.metrics:  Precision@20: 0.0473
    INFO:src.evaluation.metrics:  Recall@20: 0.1149
    INFO:src.evaluation.metrics:  NDCG@20: 0.0914
    INFO:src.evaluation.metrics:  Hit Rate@20: 0.4238
    INFO:src.evaluation.metrics:  Precision@50: 0.0361
    INFO:src.evaluation.metrics:  Recall@50: 0.1999
    INFO:src.evaluation.metrics:  NDCG@50: 0.1165
    INFO:src.evaluation.metrics:  Hit Rate@50: 0.5585
    INFO:src.evaluation.metrics:  Precision@100: 0.0281
    INFO:src.evaluation.metrics:  Recall@100: 0.2851
    INFO:src.evaluation.metrics:  NDCG@100: 0.1413
    INFO:src.evaluation.metrics:  Hit Rate@100: 0.6526
    INFO:src.evaluation.metrics:  MAP:   0.0580
    INFO:src.evaluation.metrics:  MRR:   0.1600
    INFO:__main__:
    ✓ Results saved to results/evaluation_results.json
    INFO:__main__:
    ✓ Evaluation complete!
:::

::: {.output .stream .stdout}
    Evaluation exit code: 0
:::
:::

::: {#58a2ca52 .cell .markdown}
## 15. Disk Usage Check {#15-disk-usage-check}
:::

::: {#7a982682 .cell .code execution_count="22"}
``` python
import subprocess
from pathlib import Path

subprocess.run(["df", "-h", str(PROJECT_ROOT)])
print()

for label, path in [("hf_cache", HF_CACHE_DIR), ("data", DATA_DIR),
                     ("checkpoints", CKPT_DIR), ("logs", LOGS_DIR)]:
    p = Path(path)
    if p.exists():
        result = subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True)
        print(result.stdout.strip())
```

::: {.output .stream .stdout}
    Filesystem      Size  Used Avail Use% Mounted on
    overlay         492G  171G  296G  37% /

    421M	/home/mluser/projects_yash/new_project/PromptGFM-Bio/hf_cache
    3.7G	/home/mluser/projects_yash/new_project/PromptGFM-Bio/data
    7.1G	/home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film
    4.0K	/home/mluser/projects_yash/new_project/PromptGFM-Bio/logs
:::
:::

::: {#7a3da1a5 .cell .markdown}
## 16. 🔧 Manual Override (Optional) {#16--manual-override-optional}

If auto-detected batch_size isn\'t right, uncomment and edit below, then
re-run Step 12.
:::

::: {#807b492d .cell .code execution_count="23"}
``` python
# import yaml
#
# MANUAL_BATCH_SIZE  = 384
# MANUAL_GRAD_ACCUM  = 1
# MANUAL_NUM_WORKERS = 4
#
# ws_cfg_path = PROJECT_ROOT / "configs" / "workstation_config.yaml"
# if ws_cfg_path.exists():
#     with open(ws_cfg_path) as f:
#         cfg = yaml.safe_load(f)
#
#     def patch(d, k, v):
#         if isinstance(d, dict):
#             if k in d: d[k] = v
#             for child in d.values(): patch(child, k, v)
#
#     patch(cfg, "batch_size", MANUAL_BATCH_SIZE)
#     patch(cfg, "gradient_accumulation_steps", MANUAL_GRAD_ACCUM)
#     patch(cfg, "num_workers", MANUAL_NUM_WORKERS)
#
#     with open(ws_cfg_path, "w") as f:
#         yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
#     print(f"✅ Updated: batch={MANUAL_BATCH_SIZE}, accum={MANUAL_GRAD_ACCUM}")
```
:::
