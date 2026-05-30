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
      .git/        : ✅
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
    Fri Apr  3 17:40:14 2026       
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

::: {#49203eb7-2563-40ee-8b2e-bbf3bef36f2f .cell .code execution_count="5"}
``` python
'''
#New in TE v6
# ══════════════════════════════════════════════════════════════════════════
# 3.5  PyG Extension Compatibility Check
# Detects ABI mismatches so we know whether scatter/sparse ops use
# fast CUDA kernels or fall back to pure-PyTorch paths.
# ══════════════════════════════════════════════════════════════════════════
import warnings, importlib, sys, subprocess
import torch

_EXT = {
    "pyg-lib":            "pyg_lib",
    "torch-scatter":      "torch_scatter",
    "torch-sparse":       "torch_sparse",
    "torch-cluster":      "torch_cluster",
    "torch-spline-conv":  "torch_spline_conv",
}

_broken, _missing = [], []
for pkg, mod in _EXT.items():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            importlib.import_module(mod)
        print(f"  ✅ {pkg:<24} C++ kernels active")
    except UserWarning:
        _broken.append(pkg)
        print(f"  ⚠️  {pkg:<24} ABI mismatch — pure-PyTorch fallback")
    except ImportError:
        _missing.append(pkg)
        print(f"  ❌ {pkg:<24} not installed")

_bad = _broken + _missing
if _bad:
    print(f"\n{len(_bad)} extension(s) degraded.  Impact on this project:")
    print("  • torch-scatter / torch-sparse: scatter_add falls back to native PyTorch")
    print("    → GraphSAGE trains correctly; scatter ops ~10-20% slower")
    print("  • pyg-lib / torch-spline-conv:  not used by GraphSAGE — no impact")
    print("\nAttempting force-reinstall from PyG CDN ...")
    _whl = (f"https://data.pyg.org/whl/"
            f"torch-{torch.__version__.split('+')[0]}"
            f"+cu{torch.version.cuda.replace('.','')}.html")
    _r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
         "-f", _whl,
         "torch-scatter", "torch-sparse", "torch-cluster",
         "torch-spline-conv", "torch-geometric"],
        capture_output=True, text=True,
    )
    if _r.returncode == 0:
        print("✅ Reinstall succeeded — restart kernel then re-run this cell to verify")
    else:
        print("⚠️  Pre-compiled wheels not yet available for this PyTorch/CUDA version.")
        print("   Training will proceed with pure-PyTorch fallbacks (fully correct).")
        print("   To silence the flood of UserWarnings during training, the patch cell")
        print("   below will suppress them at import time.")
else:
    print("\n✅ All PyG extensions functional")
'''
```

::: {.output .execute_result execution_count="5"}
    '\n#New in TE v6\n# ══════════════════════════════════════════════════════════════════════════\n# 3.5  PyG Extension Compatibility Check\n# Detects ABI mismatches so we know whether scatter/sparse ops use\n# fast CUDA kernels or fall back to pure-PyTorch paths.\n# ══════════════════════════════════════════════════════════════════════════\nimport warnings, importlib, sys, subprocess\nimport torch\n\n_EXT = {\n    "pyg-lib":            "pyg_lib",\n    "torch-scatter":      "torch_scatter",\n    "torch-sparse":       "torch_sparse",\n    "torch-cluster":      "torch_cluster",\n    "torch-spline-conv":  "torch_spline_conv",\n}\n\n_broken, _missing = [], []\nfor pkg, mod in _EXT.items():\n    try:\n        with warnings.catch_warnings():\n            warnings.simplefilter("error", UserWarning)\n            importlib.import_module(mod)\n        print(f"  ✅ {pkg:<24} C++ kernels active")\n    except UserWarning:\n        _broken.append(pkg)\n        print(f"  ⚠️  {pkg:<24} ABI mismatch — pure-PyTorch fallback")\n    except ImportError:\n        _missing.append(pkg)\n        print(f"  ❌ {pkg:<24} not installed")\n\n_bad = _broken + _missing\nif _bad:\n    print(f"\n{len(_bad)} extension(s) degraded.  Impact on this project:")\n    print("  • torch-scatter / torch-sparse: scatter_add falls back to native PyTorch")\n    print("    → GraphSAGE trains correctly; scatter ops ~10-20% slower")\n    print("  • pyg-lib / torch-spline-conv:  not used by GraphSAGE — no impact")\n    print("\nAttempting force-reinstall from PyG CDN ...")\n    _whl = (f"https://data.pyg.org/whl/"\n            f"torch-{torch.__version__.split(\'+\')[0]}"\n            f"+cu{torch.version.cuda.replace(\'.\',\'\')}.html")\n    _r = subprocess.run(\n        [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",\n         "-f", _whl,\n         "torch-scatter", "torch-sparse", "torch-cluster",\n         "torch-spline-conv", "torch-geometric"],\n        capture_output=True, text=True,\n    )\n    if _r.returncode == 0:\n        print("✅ Reinstall succeeded — restart kernel then re-run this cell to verify")\n    else:\n        print("⚠️  Pre-compiled wheels not yet available for this PyTorch/CUDA version.")\n        print("   Training will proceed with pure-PyTorch fallbacks (fully correct).")\n        print("   To silence the flood of UserWarnings during training, the patch cell")\n        print("   below will suppress them at import time.")\nelse:\n    print("\n✅ All PyG extensions functional")\n'
:::
:::

::: {#97e73cc3-2869-4273-adc5-c1824440c428 .cell .code execution_count="6"}
``` python
#New in TE v6
# ══════════════════════════════════════════════════════════════════════════
# 3.5  PyG Extension Compatibility Check
# Detects ABI mismatches so we know whether scatter/sparse ops use
# fast CUDA kernels or fall back to pure-PyTorch paths.
# ══════════════════════════════════════════════════════════════════════════
import warnings, importlib, sys, subprocess
import torch

_EXT = {
    "pyg-lib":           "pyg_lib",
    "torch-scatter":     "torch_scatter",
    "torch-sparse":      "torch_sparse",
    "torch-cluster":     "torch_cluster",
    "torch-spline-conv": "torch_spline_conv",
}

_broken, _missing = [], []

for pkg, mod in _EXT.items():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            importlib.import_module(mod)
        print(f"  ✅ {pkg:<24} C++ kernels active")
    except (UserWarning, OSError):
        _broken.append(pkg)
        print(f"  ⚠️  {pkg:<24} ABI mismatch — pure-PyTorch fallback")
    except ImportError:
        _missing.append(pkg)
        print(f"  ❌ {pkg:<24} not installed")

_bad = _broken + _missing

if _bad:
    print(f"\n{len(_bad)} extension(s) degraded. Impact on this project:")
    print("  • torch-scatter / torch-sparse: scatter_add falls back to native PyTorch")
    print("    → GraphSAGE trains correctly; scatter ops ~10-20% slower")
    print("  • pyg-lib / torch-spline-conv: not used by GraphSAGE — no impact")

    print("\nAttempting force-reinstall from PyG CDN ...")

    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda

    if cuda_version is None:
        wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
    else:
        wheel_url = (
            f"https://data.pyg.org/whl/"
            f"torch-{torch_version}+cu{cuda_version.replace('.', '')}.html"
        )

    _r = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--quiet", "--force-reinstall", "--no-cache-dir",
            "-f", wheel_url,
            "pyg_lib", "torch_scatter", "torch_sparse",
            "torch_cluster", "torch_spline_conv", "torch_geometric",
        ],
        capture_output=True,
        text=True,
    )

    if _r.returncode == 0:
        print("✅ Reinstall succeeded — restart kernel then re-run this cell to verify")
    else:
        print("⚠️  Pre-compiled wheels not yet available for this PyTorch/CUDA version.")
        print("   Training will proceed with pure-PyTorch fallbacks (fully correct).")
        print("   To silence the flood of UserWarnings during training, the patch cell")
        print("   below will suppress them at import time.")
else:
    print("\n✅ All PyG extensions functional")
```

::: {.output .stream .stdout}
      ✅ pyg-lib                  C++ kernels active
      ✅ torch-scatter            C++ kernels active
      ✅ torch-sparse             C++ kernels active
      ✅ torch-cluster            C++ kernels active
      ✅ torch-spline-conv        C++ kernels active

    ✅ All PyG extensions functional
:::
:::

::: {#41f56eea .cell .markdown}
## 4. Install Extra Dependencies {#4-install-extra-dependencies}
:::

::: {#d321f853 .cell .code execution_count="7"}
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

::: {#b273f478 .cell .code execution_count="8"}
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
    ✅ Latest code pulled
       Already up to date.
:::
:::

::: {#9356370c-064c-48ec-b0c7-90b6e1fcaef5 .cell .code execution_count="9"}
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

::: {.output .execute_result execution_count="9"}
    "\nimport os, sys, subprocess\nfrom pathlib import Path\n\n# ╔══════════════════════════════════════════════════════════════════════════╗\n# ║ EDIT THIS LINE to match your home directory / preferred location ║\n# ╚══════════════════════════════════════════════════════════════════════════╝\nGITHUB_URL = 'https://github.com/pes1ug23am910/PROMPTGMF-Bio-Kaggle.git'\nPROJECT_DIR = Path('/home/mluser/projects_yash/new_project/PromptGFM-Bio').resolve() # ← change this\n\n# ── Ensure parent directory exists ───────────────────────────────────────\nPROJECT_DIR.mkdir(parents=True, exist_ok=True)\n\n# ── Clone or pull ─────────────────────────────────────────────────────────\nif not (PROJECT_DIR / '.git').is_dir():\n subprocess.run(['git', 'clone', '--depth', '1', GITHUB_URL, str(PROJECT_DIR)], check=True)\n print(f' Cloned to {PROJECT_DIR}')\nelse:\n subprocess.run(['git', '-C', str(PROJECT_DIR), 'pull'], check=True)\n print(f' Pulled latest changes')\n\n# ── Set working directory & Python path ──────────────────────────────────\nos.chdir(PROJECT_DIR)\nsys.path.insert(0, str(PROJECT_DIR))\nprint(f'Working directory: {os.getcwd()}')\n\n"
:::
:::

::: {#75839c24 .cell .markdown}
## 6. Create Directory Structure {#6-create-directory-structure}
:::

::: {#33d41662 .cell .code execution_count="10"}
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

::: {#caa9fd07 .cell .code execution_count="11"}
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
{"model_id":"6187ca362dc446149b4bafebdf27cdf2","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"23d6e8d5eb3a4922be1becdf71703ce5","version_major":2,"version_minor":0}
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

::: {#62c76a19 .cell .code execution_count="12"}
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
    INFO:src.data.download:✓ STRING: 3 files downloaded
    INFO:src.data.download:✓ DISGENET: 1 files downloaded
    INFO:src.data.download:✓ HPO: 4 files downloaded
    INFO:src.data.download:======================================================================
:::
:::

::: {#74491f55 .cell .markdown}
## 9. Preprocess Data (Build Knowledge Graph) {#9-preprocess-data-build-knowledge-graph}

Skipped if `RESUME_DATA=True` and graph exists.
:::

::: {#f9801652-9d68-442e-9297-cf16726b6301 .cell .code execution_count="13"}
``` python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
```

::: {.output .stream .stdout}
    Requirement already satisfied: pandas in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (3.0.2)
    Requirement already satisfied: numpy>=1.26.0 in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (from pandas) (2.4.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in /home/mluser/micromamba/envs/promptgfm/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
:::

::: {.output .execute_result execution_count="13"}
    CompletedProcess(args=['/home/mluser/micromamba/envs/promptgfm/bin/python', '-m', 'pip', 'install', 'pandas'], returncode=0)
:::
:::

::: {#6ae6b758 .cell .code execution_count="14"}
``` python
import subprocess, sys
from pathlib import Path

graph_path = DATA_DIR / "processed" / "biomedical_graph.pt"
script = str(SCRIPTS_DIR / "preprocess_all.py")

if RESUME_DATA and graph_path.exists():
    print(f"⏭️  Preprocessing skipped — graph ready ({graph_path.stat().st_size/1e6:.0f} MB)")
else:
    cmd = [sys.executable, script]

    # If a previous graph exists, force rebuild so STRING path/mapping fixes are applied.
    if graph_path.exists():
        cmd.append("--force")
        print("Existing graph found — forcing rebuild to refresh PPI edges")

    print("Building knowledge graph...")
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    print("Preprocessing exit code:", result.returncode)
    if result.returncode != 0:
        raise RuntimeError("Preprocessing failed — check logs above")

if not graph_path.exists():
    raise RuntimeError("Graph file not created — check logs above")

print(f"✅ Graph ready ({graph_path.stat().st_size/1e6:.0f} MB)")

# Validate that gene-gene edges exist for GNN message passing.
import torch

def _load_graph_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _count_ppi_edges(graph_obj):
    edge_types = [
        ("gene", "interacts", "gene"),
        ("gene", "protein_interaction", "gene"),
        ("gene", "ppi", "gene"),
    ]
    for edge_type in edge_types:
        if edge_type in graph_obj.edge_types:
            return int(graph_obj[edge_type].edge_index.shape[1]), edge_type
    return 0, None

graph = _load_graph_cpu(graph_path)
ppi_edge_count, matched_edge_type = _count_ppi_edges(graph)

if ppi_edge_count > 0:
    print(f"✅ Gene-gene edges found: {ppi_edge_count:,} ({matched_edge_type})")
else:
    print("⚠️  No gene-gene edges found after preprocessing.")
    patch_script = PROJECT_ROOT / "add_string_ppi_edges.py"

    if patch_script.exists():
        print(f"  Running fallback patch: {patch_script}")
        patch_result = subprocess.run(
            [sys.executable, str(patch_script)],
            cwd=str(PROJECT_ROOT),
        )
        if patch_result.returncode != 0:
            raise RuntimeError("Fallback PPI patch failed — check logs above")

        graph = _load_graph_cpu(graph_path)
        repaired_count, repaired_edge_type = _count_ppi_edges(graph)
        if repaired_count > 0:
            print(f"✅ Repaired gene-gene edges: {repaired_count:,} ({repaired_edge_type})")
        else:
            raise RuntimeError("Graph still has no gene-gene edges after fallback patch")
    else:
        raise RuntimeError(f"Missing fallback script: {patch_script}")
```

::: {.output .stream .stdout}
    Existing graph found — forcing rebuild to refresh PPI edges
    Building knowledge graph...
      Running: /home/mluser/micromamba/envs/promptgfm/bin/python /home/mluser/projects_yash/new_project/PromptGFM-Bio/scripts/preprocess_all.py --force
:::

::: {.output .stream .stderr}
    INFO:src.data.preprocess:======================================================================
    INFO:src.data.preprocess:Starting enhanced preprocessing pipeline...
    INFO:src.data.preprocess:======================================================================
    INFO:src.data.preprocess:Options:
    INFO:src.data.preprocess:  HPO Bridge: True
    INFO:src.data.preprocess:  Orphadata: True
    INFO:src.data.preprocess:  UniProt: False
    INFO:src.data.preprocess:  Pathways: False
    INFO:src.data.preprocess:======================================================================
    INFO:src.data.preprocess:
    [Step 1] Parsing PPI networks...
    INFO:src.data.preprocess:Using STRING links file: /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/string/9606.protein.links.v12.0.txt
    INFO:src.data.preprocess:Using STRING protein info file: /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/string/9606.protein.info.v12.0.txt
    INFO:src.data.preprocess:Parsing BioGRID from /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.txt...
    ERROR:src.data.preprocess:Failed to parse BioGRID: Usecols do not match columns, columns expected but not found: ['Organism Interactor A', 'Organism Interactor B']
    INFO:src.data.preprocess:Parsing STRING from /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/string/9606.protein.links.v12.0.txt...
    INFO:src.data.preprocess:Loaded 13715404 STRING interactions
    INFO:src.data.preprocess:Filtered to 1858944 interactions with score >= 400
    INFO:src.data.preprocess:Loading gene name mapping from /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/string/9606.protein.info.v12.0.txt...
    INFO:src.data.preprocess:Loaded 19559 protein-gene mappings
    INFO:src.data.preprocess:STRING mapping coverage after filtering: 99.7%
    INFO:src.data.preprocess:Found 19412 unique genes in STRING
    INFO:src.data.preprocess:Combined PPI network: 1854012 edges, 19412 genes
    INFO:src.data.preprocess:
    [Step 2] Creating gene-disease edges...
    INFO:src.data.preprocess:
    [Method 1] Creating gene-disease edges from HPO bridge...
    INFO:src.data.hpo_bridge:Parsing HPO gene-phenotype annotations...
    INFO:src.data.hpo_bridge:🔍 [HPO_BRIDGE_V2] Parsing gene phenotypes from /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/hpo/genes_to_phenotype.txt
    INFO:src.data.hpo_bridge:🔍 [HPO_BRIDGE_V2] Skipped header: ncbi_gene_id	gene_symbol	hpo_id	hpo_name	frequency...
    INFO:src.data.hpo_bridge:🔍 [HPO_BRIDGE_V2] Line 0: gene=NAT2, hpo_id=HP:0000007
    INFO:src.data.hpo_bridge:🔍 [HPO_BRIDGE_V2] Line 1: gene=NAT2, hpo_id=HP:0001939
    INFO:src.data.hpo_bridge:🔍 [HPO_BRIDGE_V2] Line 2: gene=AARS1, hpo_id=HP:0002460
    INFO:src.data.hpo_bridge:Parsed 5251 genes with phenotype annotations
    INFO:src.data.hpo_bridge:Parsed 5251 genes with phenotype annotations
    INFO:src.data.hpo_bridge:  Sample gene: NAT2 → ['HP:0001939', 'HP:0000007']
    INFO:src.data.hpo_bridge:  Sample gene: AARS1 → ['HP:0045075', 'HP:0007495', 'HP:0012196']
    INFO:src.data.hpo_bridge:  Sample gene: ABAT → ['HP:0007266', 'HP:0008872', 'HP:0001321']
    INFO:src.data.hpo_bridge:Parsing HPO disease-phenotype annotations...
    INFO:src.data.hpo_bridge:Using columns: database_id=0, hpo_id=3
    INFO:src.data.hpo_bridge:Parsed 12996 diseases with phenotype annotations
    INFO:src.data.hpo_bridge:Parsed 12996 diseases with phenotype annotations
    INFO:src.data.hpo_bridge:  Sample disease: OMIM:619340 → ['HP:0032792', 'HP:0001518', 'HP:0001789']
    INFO:src.data.hpo_bridge:  Sample disease: OMIM:609153 → ['HP:0003394', 'HP:0003324', 'HP:0003768']
    INFO:src.data.hpo_bridge:  Sample disease: OMIM:614102 → ['HP:0011463', 'HP:6000261', 'HP:0002719']
    INFO:src.data.hpo_bridge:Computing phenotype IDF scores...
    INFO:src.data.hpo_bridge:Computed IDF for 11613 phenotypes
    INFO:src.data.hpo_bridge:Filtering 0 overly common phenotypes (>50.0% frequency)
    INFO:src.data.hpo_bridge:Scoring gene-disease associations (5251 genes x 12996 diseases)...
    INFO:src.data.hpo_bridge:Total possible pairs: 68,241,996
    INFO:src.data.hpo_bridge:  Debug: NAT2 x OMIM:619340 → 0 shared phenotypes out of 2 gene / 11 disease
    INFO:src.data.hpo_bridge:  Debug: NAT2 x OMIM:609153 → 0 shared phenotypes out of 2 gene / 7 disease
    INFO:src.data.hpo_bridge:  Debug: NAT2 x OMIM:614102 → 1 shared phenotypes out of 2 gene / 7 disease
    INFO:src.data.hpo_bridge:    Shared: ['HP:0000007']
    INFO:src.data.hpo_bridge:  Debug: NAT2 x OMIM:619426 → 0 shared phenotypes out of 2 gene / 49 disease
    INFO:src.data.hpo_bridge:  Debug: NAT2 x OMIM:610370 → 1 shared phenotypes out of 2 gene / 32 disease
    INFO:src.data.hpo_bridge:    Shared: ['HP:0000007']
    INFO:src.data.hpo_bridge:Created 9734247 gene-disease edges (score >= 0.1)
    INFO:src.data.hpo_bridge:Pairs with any phenotype intersection: 39,433,962 (57.79%)
    INFO:src.data.hpo_bridge:
    Score distribution:
    INFO:src.data.hpo_bridge:  Mean: 0.200
    INFO:src.data.hpo_bridge:  Median: 0.160
    INFO:src.data.hpo_bridge:  Min: 0.100
    INFO:src.data.hpo_bridge:  Max: 1.000
    INFO:src.data.hpo_bridge:Saved 9734247 HPO-derived edges to /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/processed/hpo_gene_disease_edges.csv
    INFO:src.data.preprocess:✓ HPO bridge created 9734247 edges
    INFO:src.data.preprocess:
    [Method 2] Downloading and parsing Orphadata...
    INFO:src.data.orphadata:Parsing Orphadata gene associations from /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/orphanet/en_product6.xml...
    INFO:src.data.orphadata:Extracted 8374 gene-disease associations from Orphadata
    INFO:src.data.orphadata:Filtered to 7363 high-confidence associations
    INFO:src.data.orphadata:  (8374 total → 7363 after filtering)
    INFO:src.data.orphadata:
    Merge summary:
    INFO:src.data.orphadata:  Orphadata edges: 7363
    INFO:src.data.orphadata:  HPO-only edges: 9734247
    INFO:src.data.orphadata:  Overlap: 0 edges confirmed by both sources
    INFO:src.data.orphadata:  Total merged edges: 9741610
    INFO:src.data.preprocess:✓ Orphadata added 7363 gold-standard edges
    INFO:src.data.preprocess:
    ✓ Total gene-disease edges: 9741610
    INFO:src.data.preprocess:✓ Unique genes: 5363
    INFO:src.data.preprocess:✓ Unique diseases: 16841
    INFO:src.data.preprocess:
    [Step 3] Parsing HPO phenotypes...
    INFO:src.data.preprocess:Parsing HPO from /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/raw/hpo/phenotype_to_genes.txt...
    INFO:src.data.preprocess:Loaded 1080402 phenotype-gene annotations
    INFO:src.data.preprocess:Found 11794 unique phenotypes
    INFO:src.data.preprocess:Created 1080402 gene-phenotype associations
    INFO:src.data.preprocess:
    [Step 4] Building heterogeneous graph...
    INFO:src.data.preprocess:Building heterogeneous graph...
    INFO:src.data.preprocess:Created 19576 gene nodes
    INFO:src.data.preprocess:Created 16841 disease nodes
    INFO:src.data.preprocess:Added 1854012 gene-gene edges
    INFO:src.data.preprocess:Added 9741610 gene-disease edges
    INFO:src.data.preprocess:Added 11794 phenotype nodes
    INFO:src.data.preprocess:
    ======================================================================
    INFO:src.data.preprocess:Graph Statistics:
    INFO:src.data.preprocess:  Gene nodes: 19576
    INFO:src.data.preprocess:  Disease nodes: 16841
    INFO:src.data.preprocess:  Phenotype nodes: 11794
    INFO:src.data.preprocess:  Edge types: [('gene', 'interacts', 'gene'), ('gene', 'associated_with', 'disease'), ('disease', 'rev_associated_with', 'gene')]
    INFO:src.data.preprocess:    ('gene', 'interacts', 'gene'): 1854012 edges
    INFO:src.data.preprocess:    ('gene', 'associated_with', 'disease'): 9741610 edges
    INFO:src.data.preprocess:    ('disease', 'rev_associated_with', 'gene'): 9741610 edges
    INFO:src.data.preprocess:======================================================================
    INFO:src.data.preprocess:Saved graph to /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/processed/biomedical_graph.pt
    INFO:src.data.preprocess:Saved statistics to /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/processed/biomedical_graph_stats.txt
    INFO:src.data.preprocess:
    ======================================================================
    INFO:src.data.preprocess:✓ Enhanced preprocessing complete!
    INFO:src.data.preprocess:Graph saved to: /home/mluser/projects_yash/new_project/PromptGFM-Bio/data/processed/biomedical_graph.pt
    INFO:src.data.preprocess:======================================================================
:::

::: {.output .stream .stdout}

    ======================================================================
    PromptGFM-Bio Enhanced Preprocessing Pipeline
    ======================================================================

    Configuration:
      Force reprocess: True
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
    ✅ Graph ready (351 MB)
    ✅ Gene-gene edges found: 1,854,012 (('gene', 'interacts', 'gene'))
:::
:::

::: {#3a47a00a-9801-4512-a5a9-918e355a4ca2 .cell .code execution_count="15"}
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

::: {.output .execute_result execution_count="15"}
    '\n#use if needed/ previous cell fails\n# ─────────────────────────────────────────────────────────────\n# 4. Install & Verify All Dependencies (ROBUST VERSION)\n# ─────────────────────────────────────────────────────────────\n\nimport subprocess, sys, importlib\n\ndef pip_install(packages):\n    print(f"Installing: {packages}")\n    subprocess.run(\n        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,\n        check=True\n    )\n\ndef ensure_import(pkg_name, pip_name=None):\n    """Ensure a package is installed and importable"""\n    pip_name = pip_name or pkg_name\n    try:\n        importlib.import_module(pkg_name)\n        print(f"✅ {pkg_name} already available")\n    except ImportError:\n        print(f"⚠️  {pkg_name} missing → installing...")\n        pip_install([pip_name])\n        try:\n            importlib.import_module(pkg_name)\n            print(f"✅ {pkg_name} installed successfully")\n        except ImportError:\n            raise RuntimeError(f"❌ Failed to import {pkg_name} even after install")\n\n# ── Upgrade core tooling ──────────────────────────────────────\npip_install(["--upgrade", "pip", "setuptools", "wheel"])\n\n# ── Critical deps (MUST exist for scripts to run) ─────────────\ncritical_deps = {\n    "pandas": "pandas>=2.2.0",        # 🔴 REQUIRED (your failure)\n    "numpy": "numpy",\n    "yaml": "pyyaml",\n}\n\n# ── Project deps ─────────────────────────────────────────────\nproject_deps = {\n    "transformers": "transformers>=4.40.0",\n    "sentence_transformers": "sentence-transformers>=2.7.0",\n    "Bio": "biopython>=1.84",\n    "networkx": "networkx>=3.2",\n    "wandb": "wandb>=0.17.0",\n    "dotenv": "python-dotenv>=1.0.0",\n    "huggingface_hub": "huggingface_hub",\n}\n\n# ── Ensure everything is importable ───────────────────────────\nprint("\n🔍 Checking critical dependencies...")\nfor module, pip_pkg in critical_deps.items():\n    ensure_import(module, pip_pkg)\n\nprint("\n🔍 Checking project dependencies...")\nfor module, pip_pkg in project_deps.items():\n    ensure_import(module, pip_pkg)\n\nprint("\n✅ All dependencies verified and ready")\n'
:::
:::

::: {#6844e8a3 .cell .markdown}
## 10. W&B Login {#10-wb-login}
:::

::: {#536a9cfe .cell .code execution_count="16"}
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

::: {#e80d9ed4 .cell .code execution_count="17"}
``` python
'''
#Modified to fix workers in TE v6
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
'''
```

::: {.output .execute_result execution_count="17"}
    '\n#Modified to fix workers in TE v6\nimport subprocess, os, yaml\nimport torch\n\n# ── 1. Probe free VRAM ───────────────────────────────────────────────────\nresult = subprocess.run(\n    ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",\n     "--format=csv,nounits,noheader"],\n    capture_output=True, text=True\n)\nsmi_used, smi_total, smi_free = [int(x.strip()) for x in result.stdout.strip().split(",")]\nusable_mb = smi_free - 2048\n\nprint(f"GPU VRAM: {smi_total:,} MiB total · {smi_used:,} MiB used · {smi_free:,} MiB free")\nprint(f"Usable for training: {usable_mb:,} MiB (after 2 GB safety margin)")\nprint()\n\n# ── 2. Pick batch_size + gradient accumulation ───────────────────────────\nif usable_mb >= 20000:\n    batch_size, grad_accum, num_workers = 768, 1, 8\nelif usable_mb >= 16000:\n    batch_size, grad_accum, num_workers = 512, 1, 6\nelif usable_mb >= 12000:\n    batch_size, grad_accum, num_workers = 384, 1, 4\nelif usable_mb >= 8000:\n    batch_size, grad_accum, num_workers = 256, 1, 4\nelif usable_mb >= 5000:\n    batch_size, grad_accum, num_workers = 128, 2, 2\nelse:\n    batch_size, grad_accum, num_workers = 64, 4, 2\n\neffective_batch = batch_size * grad_accum\n\nprint(f"╔══════════════════════════════════════════════╗")\nprint(f"║  batch_size       = {batch_size:<6}                   ║")\nprint(f"║  grad_accum       = {grad_accum:<6}                   ║")\nprint(f"║  effective_batch   = {effective_batch:<6}                  ║")\nprint(f"║  num_workers      = {num_workers:<6}                   ║")\nprint(f"╚══════════════════════════════════════════════╝")\nprint()\n\n# ── 3. Read base config and patch ────────────────────────────────────────\nbase_cfg_path = CONFIGS_DIR / "kaggle_config.yaml"\nws_cfg_path   = CONFIGS_DIR / "workstation_config.yaml"\n\nif base_cfg_path.exists():\n    with open(base_cfg_path) as f:\n        cfg = yaml.safe_load(f)\n\n    def patch_dict(d, patches):\n        if not isinstance(d, dict):\n            return\n        for k, v in patches.items():\n            if k in d:\n                print(f"  patching {k}: {d[k]} → {v}")\n                d[k] = v\n        for child in d.values():\n            if isinstance(child, dict):\n                patch_dict(child, patches)\n\n    patches = {\n        "batch_size":                   batch_size,\n        "num_workers":                  num_workers,\n        "gradient_accumulation_steps":  grad_accum,\n        "pin_memory":                   True,\n        "persistent_workers":           num_workers > 0,\n    }\n\n    print("Patching config:")\n    patch_dict(cfg, patches)\n\n    if "gradient_accumulation_steps" not in str(cfg):\n        target = cfg.get("training", cfg)\n        target["gradient_accumulation_steps"] = grad_accum\n        print(f"  added gradient_accumulation_steps: {grad_accum} (new key)")\n\n    with open(ws_cfg_path, "w") as f:\n        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)\n\n    print(f"\n✅ Wrote {ws_cfg_path}")\nelse:\n    print(f"⚠️  {base_cfg_path} not found")\n\n# ── 4. RTX 4090 performance flags ────────────────────────────────────────\ntorch.backends.cuda.matmul.allow_tf32 = True\ntorch.backends.cudnn.allow_tf32       = True\ntorch.backends.cudnn.benchmark        = True\nprint("✅ TF32 matmul ✓ · TF32 cuDNN ✓ · cuDNN benchmark ✓")\n'
:::
:::

::: {#ef8ae620-e2bd-4f73-ba35-256ba821033a .cell .code execution_count="18"}
``` python
#New in TE v6
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

# ── 2. Probe /dev/shm free space ─────────────────────────────────────────
# Each DataLoader worker serialises ~50 MB of tensors (node_features +
# edge_index) per batch prefetch via /dev/shm when multiprocessing_context
# is the default 'fork-server' / 'spawn'.  The patch cell below switches to
# 'fork', which inherits parent memory and bypasses /dev/shm entirely.
# We still record it here as a diagnostic.
_shm_r = subprocess.run(
    "df -m /dev/shm | awk 'NR==2 {print $2, $4}'",
    shell=True, capture_output=True, text=True
)
try:
    shm_total_mb, shm_free_mb = [int(x) for x in _shm_r.stdout.strip().split()]
except ValueError:
    shm_total_mb, shm_free_mb = 64, 64   # conservative fallback

WORKER_SHM_MB = 50   # approx shm consumption per worker (pre-fork-patch)
shm_safe_workers = max(0, shm_free_mb // WORKER_SHM_MB - 1)

print(f"/dev/shm  total={shm_total_mb} MB  free={shm_free_mb} MB")
if shm_free_mb < 256:
    print(f"  ⚠️  /dev/shm is tight ({shm_free_mb} MB free).  Without the fork-context")
    print(f"     patch, multi-worker DataLoading WILL crash (Bus Error / errno 28).")
    print(f"     Run the 'Patch train.py' cell below BEFORE launching training.")
else:
    print(f"  ✅ /dev/shm adequate for {shm_safe_workers} workers without fork patch")

# ── 3. Pick batch_size + gradient accumulation ───────────────────────────
if usable_mb >= 20000:
    batch_size, grad_accum = 768, 1
elif usable_mb >= 16000:
    batch_size, grad_accum = 512, 1
elif usable_mb >= 12000:
    batch_size, grad_accum = 384, 1
elif usable_mb >= 8000:
    batch_size, grad_accum = 256, 1
elif usable_mb >= 5000:
    batch_size, grad_accum = 128, 2
else:
    batch_size, grad_accum = 64, 4

effective_batch = batch_size * grad_accum

# ── 4. Choose num_workers ─────────────────────────────────────────────────
# After the fork-context patch, workers inherit parent memory and /dev/shm
# is never touched, so the CPU-based limit is the only relevant constraint.
# Before the patch, cap to shm_safe_workers as a safety net.
cpu_count = os.cpu_count() or 1
cpu_max_workers = min(4, max(0, cpu_count - 2))

# Detect whether the fork-context patch has already been applied.
_train_py_src = (PROJECT_ROOT / "scripts" / "train.py").read_text()
_fork_patch_applied = '"fork"' in _train_py_src and "cfg_nw" in _train_py_src

if _fork_patch_applied:
    num_workers = cpu_max_workers   # fork context → shm irrelevant
    print(f"\nFork-context patch detected → using full cpu_max_workers={num_workers}")
else:
    num_workers = min(cpu_max_workers, shm_safe_workers)
    print(f"\nFork-context patch NOT yet applied → capping workers to shm_safe={num_workers}")
    print("  Run the 'Patch train.py' cell then re-run Step 11 for full worker count.")

print()
print(f"╔══════════════════════════════════════════════╗")
print(f"║  batch_size       = {batch_size:<6}                   ║")
print(f"║  grad_accum       = {grad_accum:<6}                   ║")
print(f"║  effective_batch   = {effective_batch:<6}                  ║")
print(f"║  num_workers      = {num_workers:<6}   (written to YAML)   ║")
print(f"╚══════════════════════════════════════════════╝")
print()

# ── 5. Read base config and patch ────────────────────────────────────────
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
        "batch_size":                  batch_size,
        "num_workers":                 num_workers,   # train.py reads this after patch
        "gradient_accumulation_steps": grad_accum,
        "pin_memory":                  True,
        "persistent_workers":          num_workers > 0,
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

# ── 6. RTX 4090 performance flags ────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
print("✅ TF32 matmul ✓ · TF32 cuDNN ✓ · cuDNN benchmark ✓")
```

::: {.output .stream .stdout}
    GPU VRAM: 24,564 MiB total · 120 MiB used · 23,953 MiB free
    Usable for training: 21,905 MiB (after 2 GB safety margin)

    /dev/shm  total=63 MB  free=63 MB
      ⚠️  /dev/shm is tight (63 MB free).  Without the fork-context
         patch, multi-worker DataLoading WILL crash (Bus Error / errno 28).
         Run the 'Patch train.py' cell below BEFORE launching training.

    Fork-context patch detected → using full cpu_max_workers=4

    ╔══════════════════════════════════════════════╗
    ║  batch_size       = 768                      ║
    ║  grad_accum       = 1                        ║
    ║  effective_batch   = 768                     ║
    ║  num_workers      = 4        (written to YAML)   ║
    ╚══════════════════════════════════════════════╝

    Patching config:
      patching batch_size: 256 → 768
      patching num_workers: 4 → 4
      patching pin_memory: True → True
      added gradient_accumulation_steps: 1 (new key)

    ✅ Wrote /home/mluser/projects_yash/new_project/PromptGFM-Bio/configs/workstation_config.yaml
    ✅ TF32 matmul ✓ · TF32 cuDNN ✓ · cuDNN benchmark ✓
:::
:::

::: {#9650942d-e10b-4844-aa09-7bfb5300bd63 .cell .code execution_count="19"}
``` python
# #New in TE v6

# # ══════════════════════════════════════════════════════════════════════════

# # 11.5  Patch train.py  (run once per session; idempotent)

# #

# # Bug A — train.py hard-codes num_workers from cpu_count and ignores

# # config['data']['num_workers'], so YAML changes have no effect.

# # Bug B — Default multiprocessing context uses /dev/shm for tensor IPC.

# # With ~50 MB × 4 workers × batch prefetch the 64 MB default shm

# # exhausts instantly → Bus Error / errno ENOSPC (28).

# #

# # Fix A  — Make create_dataloaders() respect config['data']['num_workers']

# # when the key is present.

# # Fix B  — Pass multiprocessing_context='fork' to both DataLoaders.

# # Fork workers *inherit* the parent's address space; the large

# # cached_node_features and cached_edge_index tensors are already

# # in memory and never go through /dev/shm.

# # ══════════════════════════════════════════════════════════════════════════

# import warnings\
# from pathlib import Path

# train_py = PROJECT_ROOT / "scripts" / "train.py"\
# src_orig = src = train_py.read_text()

# # ── Patch A: respect config num_workers ──────────────────────────────────

# _OLD_A = '''\
# if is_windows:\
# num_workers = 0  # Windows multiprocessing can't pickle nested collate_fn\
# logger.info(f"  DataLoader workers: {num_workers} (Windows: single-threaded to avoid pickling issues)")\
# else:\
# num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0\
# logger.info(f"  DataLoader workers: {num_workers} (parallel data loading enabled)")'''

# _NEW_A = '''\
# # Read num_workers from config when explicitly set; otherwise auto-detect.\
# _cfg_nw = config.get("data", {}).get("num_workers", None)\
# cfg_nw = int(_cfg_nw) if _cfg_nw is not None else None\
# if is_windows:\
# num_workers = 0\
# logger.info(f"  DataLoader workers: {num_workers} (Windows)")\
# elif cfg_nw is not None:\
# num_workers = cfg_nw\
# logger.info(f"  DataLoader workers: {num_workers} (from config)")\
# else:\
# num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0\
# logger.info(f"  DataLoader workers: {num_workers} (auto-detected)")'''

# if _OLD_A in src:\
# src = src.replace(_OLD_A, _NEW_A)\
# print("✅ Patch A applied — train.py now honours config['data']['num_workers']")\
# elif "cfg_nw" in src:\
# print("ℹ️  Patch A already applied")\
# else:\
# print("⚠️  Patch A: target string not found — train.py may have changed; skipping")\
# '''

# # ── Patch B: fork context for both DataLoaders ───────────────────────────

# # We target the closing lines that are identical in both train_loader and

# # val_loader definitions.

# _OLD_B = '''\
# pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer\
# persistent_workers=True if num_workers > 0 else False  # Keep workers alive\
# )'''

# _NEW_B = '''\
# pin_memory=True if torch.cuda.is_available() else False,\
# persistent_workers=True if num_workers > 0 else False,\
# # 'fork' workers inherit parent memory; large cached tensors (node_features,\
# # edge_index) are never serialised to /dev/shm — eliminates Bus Error / ENOSPC.\
# multiprocessing_context="fork" if num_workers > 0 else None,\
# )'''

# _count_b = src.count(_OLD_B)\
# if _count_b == 2:\
# src = src.replace(_OLD_B, _NEW_B)\
# print("✅ Patch B applied — both DataLoaders use fork context (bypasses /dev/shm)")\
# elif _count_b == 0 and '"fork"' in src:\
# print("ℹ️  Patch B already applied")\
# elif _count_b == 0:\
# print("⚠️  Patch B: target string not found — train.py may have changed; skipping")\
# else:\
# print(f"⚠️  Patch B: found {_count_b} matches (expected 2) — skipping to be safe")\
# '''

# # ── Patch B: fork context for both DataLoaders ───────────────────────────

# _OLD_B_VARIANTS = [\
# '''\
# pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer\
# persistent_workers=True if num_workers > 0 else False  # Keep workers alive\
# )''',

# '''
#     pin_memory=True if torch.cuda.is_available() else False,
#     persistent_workers=True if num_workers > 0 else False
# )'''
# ]

# _NEW_B = '''\
# pin_memory=True if torch.cuda.is_available() else False,\
# persistent_workers=True if num_workers > 0 else False,\
# multiprocessing_context="fork" if num_workers > 0 else None,\
# )'''

# patched = 0\
# for old in _OLD_B_VARIANTS:\
# if old in src:\
# src = src.replace(old, _NEW_B)\
# patched += 1

# if patched == 2:\
# print("✅ Patch B applied — both DataLoaders patched")\
# elif patched == 1:\
# print("⚠️  Patch B partially applied (1 block patched)")\
# elif '"fork"' in src:\
# print("ℹ️  Patch B already applied")\
# else:\
# print("⚠️  Patch B: no match found — train.py structure changed")

# # ── Suppress PyG ABI UserWarnings that flood stdout ──────────────────────

# # Insert a warnings filter immediately after the torch_geometric import block.

# _OLD_C = "from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset"\
# _NEW_C = """\
# import warnings as _w\
# _w.filterwarnings("ignore", message="An issue occurred while importing", category=UserWarning)

# from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset"""

# if _OLD_C in src and "_w.filterwarnings" not in src:\
# src = src.replace(_OLD_C, _NEW_C)\
# print("✅ Patch C applied — PyG ABI UserWarnings suppressed in train.py output")\
# elif "_w.filterwarnings" in src:\
# print("ℹ️  Patch C already applied")\
# else:\
# print("⚠️  Patch C: anchor not found — skipping")

# # ── Write & verify ────────────────────────────────────────────────────────

# if src != src_orig:\
# train_py.write_text(src)\
# print(f"\n✅ train.py saved ({train_py})")\
# print("   Re-run Step 11 (VRAM Config) to update num_workers in the YAML now")\
# print("   that the fork-patch is detected.")\
# else:\
# print("\nℹ️  No changes written — all patches already present")

# # Quick smoke-test: confirm multiprocessing context setting is present

# assert '"fork"' in train_py.read_text(), "Patch B verification failed"\
# print("✅ Patch verification passed")
```
:::

::: {#a9b25d0d-3fcb-4c8b-a5ac-4c80d6285020 .cell .code execution_count="20"}
``` python
# #New in TE v6 (CLEAN FIX)
# # ══════════════════════════════════════════════════════════════════════════
# # 11.5  Patch train.py (robust + no syntax issues)
# # ══════════════════════════════════════════════════════════════════════════

# from pathlib import Path

# train_py = PROJECT_ROOT / "scripts" / "train.py"
# src_orig = src = train_py.read_text()

# # ── Patch A: respect config num_workers ──────────────────────────────────
# OLD_A = """\
#     if is_windows:
#         num_workers = 0  # Windows multiprocessing can't pickle nested collate_fn
#         logger.info(f"  DataLoader workers: {num_workers} (Windows: single-threaded to avoid pickling issues)")
#     else:
#         num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
#         logger.info(f"  DataLoader workers: {num_workers} (parallel data loading enabled)")"""

# NEW_A = """\
#     # Read num_workers from config when explicitly set; otherwise auto-detect.
#     _cfg_nw = config.get("data", {}).get("num_workers", None)
#     cfg_nw = int(_cfg_nw) if _cfg_nw is not None else None
#     if is_windows:
#         num_workers = 0
#         logger.info(f"  DataLoader workers: {num_workers} (Windows)")
#     elif cfg_nw is not None:
#         num_workers = cfg_nw
#         logger.info(f"  DataLoader workers: {num_workers} (from config)")
#     else:
#         num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
#         logger.info(f"  DataLoader workers: {num_workers} (auto-detected)")"""

# if OLD_A in src:
#     src = src.replace(OLD_A, NEW_A)
#     print("✅ Patch A applied")
# elif "cfg_nw" in src:
#     print("ℹ️ Patch A already applied")
# else:
#     print("⚠️ Patch A not found")

# # ── Patch B: fork context (handles BOTH variants) ────────────────────────
# OLD_B_VARIANTS = [
#     """\
#         pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
#         persistent_workers=True if num_workers > 0 else False  # Keep workers alive
#     )""",
#     """\
#         pin_memory=True if torch.cuda.is_available() else False,
#         persistent_workers=True if num_workers > 0 else False
#     )"""
# ]

# NEW_B = """\
#         pin_memory=True if torch.cuda.is_available() else False,
#         persistent_workers=True if num_workers > 0 else False,
#         multiprocessing_context="fork" if num_workers > 0 else None,
#     )"""

# patched = 0
# for old in OLD_B_VARIANTS:
#     if old in src:
#         src = src.replace(old, NEW_B)
#         patched += 1

# if patched == 2:
#     print("✅ Patch B applied (both loaders)")
# elif patched == 1:
#     print("⚠️ Patch B applied partially (1 loader)")
# elif 'multiprocessing_context="fork"' in src:
#     print("ℹ️ Patch B already applied")
# else:
#     print("⚠️ Patch B not applied")

# # ── Patch C: suppress PyG warnings ───────────────────────────────────────
# OLD_C = "from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset"
# NEW_C = """\
# import warnings as _w
# _w.filterwarnings("ignore", message="An issue occurred while importing", category=UserWarning)

# from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset"""

# if OLD_C in src and "_w.filterwarnings" not in src:
#     src = src.replace(OLD_C, NEW_C)
#     print("✅ Patch C applied")
# elif "_w.filterwarnings" in src:
#     print("ℹ️ Patch C already applied")
# else:
#     print("⚠️ Patch C not found")

# # ── Write file ───────────────────────────────────────────────────────────
# if src != src_orig:
#     train_py.write_text(src)
#     print(f"\n✅ train.py updated")
# else:
#     print("\nℹ️ No changes needed")

# # ── Verification (robust) ────────────────────────────────────────────────
# content = train_py.read_text()

# if 'multiprocessing_context="fork"' not in content:
#     raise RuntimeError("❌ Fork patch missing")

# if "cfg_nw" not in content:
#     raise RuntimeError("❌ num_workers patch missing")

# print("✅ Patch verification passed")
```
:::

::: {#0af9ff2c-bda8-42f2-bdde-8d27c58cb831 .cell .code execution_count="21"}
``` python
# #New in TE v6 (FIXED)
# # ══════════════════════════════════════════════════════════════════════════
# # 11.5  Patch train.py  (robust + compatible with your file)
# # ══════════════════════════════════════════════════════════════════════════

# from pathlib import Path

# train_py = PROJECT_ROOT / "scripts" / "train.py"
# src_orig = src = train_py.read_text()

# # ── Patch A: respect config num_workers ──────────────────────────────────
# _OLD_A = '''\
#     if is_windows:
#         num_workers = 0  # Windows multiprocessing can't pickle nested collate_fn
#         logger.info(f"  DataLoader workers: {num_workers} (Windows: single-threaded to avoid pickling issues)")
#     else:
#         num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
#         logger.info(f"  DataLoader workers: {num_workers} (parallel data loading enabled)")'''

# _NEW_A = '''\
#     # Read num_workers from config when explicitly set; otherwise auto-detect.
#     _cfg_nw = config.get("data", {}).get("num_workers", None)
#     cfg_nw = int(_cfg_nw) if _cfg_nw is not None else None
#     if is_windows:
#         num_workers = 0
#         logger.info(f"  DataLoader workers: {num_workers} (Windows)")
#     elif cfg_nw is not None:
#         num_workers = cfg_nw
#         logger.info(f"  DataLoader workers: {num_workers} (from config)")
#     else:
#         num_workers = min(4, max(0, cpu_count - 2)) if cpu_count and cpu_count > 2 else 0
#         logger.info(f"  DataLoader workers: {num_workers} (auto-detected)")'''

# if _OLD_A in src:
#     src = src.replace(_OLD_A, _NEW_A)
#     print("✅ Patch A applied — train.py now honours config['data']['num_workers']")
# elif "cfg_nw" in src:
#     print("ℹ️  Patch A already applied")
# else:
#     print("⚠️  Patch A: target string not found — skipping")

# # ── Patch B: fork context (FIXED for your train.py) ──────────────────────
# _OLD_B_VARIANTS = [
#     '''\
#         pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
#         persistent_workers=True if num_workers > 0 else False  # Keep workers alive
#     )''',

#     '''\
#         pin_memory=True if torch.cuda.is_available() else False,
#         persistent_workers=True if num_workers > 0 else False
#     )'''
# ]

# _NEW_B = '''\
#         pin_memory=True if torch.cuda.is_available() else False,
#         persistent_workers=True if num_workers > 0 else False,
#         # Fork avoids /dev/shm → fixes Bus Error / ENOSPC
#         multiprocessing_context="fork" if num_workers > 0 else None,
#     )'''

# patched = 0
# for old in _OLD_B_VARIANTS:
#     if old in src:
#         src = src.replace(old, _NEW_B)
#         patched += 1

# if patched == 2:
#     print("✅ Patch B applied — both DataLoaders patched")
# elif patched == 1:
#     print("⚠️  Patch B partially applied (1 block patched)")
# elif '"fork"' in src:
#     print("ℹ️  Patch B already applied")
# else:
#     print("⚠️  Patch B: no match found — skipping")

# # ── Patch C: suppress PyG warnings ───────────────────────────────────────
# _OLD_C = "from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset"
# _NEW_C = """\
# import warnings as _w
# _w.filterwarnings("ignore", message="An issue occurred while importing", category=UserWarning)

# from src.data.dataset import BiomedicalGraphDataset, GeneDiseaseDataset"""

# if _OLD_C in src and "_w.filterwarnings" not in src:
#     src = src.replace(_OLD_C, _NEW_C)
#     print("✅ Patch C applied — warnings suppressed")
# elif "_w.filterwarnings" in src:
#     print("ℹ️  Patch C already applied")
# else:
#     print("⚠️  Patch C: anchor not found — skipping")

# # ── Write file ───────────────────────────────────────────────────────────
# if src != src_orig:
#     train_py.write_text(src)
#     print(f"\n✅ train.py saved ({train_py})")
# else:
#     print("\nℹ️  No changes written — already patched")

# # ── Robust verification (FIXED) ──────────────────────────────────────────
# content = train_py.read_text()

# if 'multiprocessing_context="fork"' not in content:
#     raise RuntimeError("❌ Patch B verification failed — fork context not found")

# if "cfg_nw" not in content:
#     raise RuntimeError("❌ Patch A verification failed")

# print("✅ Patch verification passed")
```
:::

::: {#56654e06-5a5a-48dd-9670-5448e060e0d5 .cell .code execution_count="22"}
``` python
# ══════════════════════════════════════════════════════════════════════════
# 11.6  Root-cause fix: remove graph tensors from DataLoader batch payload
#
# DIAGNOSIS
# ---------
# Even with fork context, PyTorch uses fd-based shared memory when workers
# send data back to the main process through the multiprocessing queue.
# The collate function currently returns two graph-wide constant tensors
# in EVERY batch:
#
#   node_features : shape [19576, 128] float32  →  ~10 MB per batch
#   edge_index    : shape [2, 1854012] int64    →  ~30 MB per batch
#
# With num_workers=4, prefetch_factor=2 that is:
#   8 in-flight batches × 40 MB = 320 MB through /dev/shm  → ENOSPC crash
#
# These tensors are IDENTICAL in every batch (they are the full graph).
# They should live on the trainer, not inside the DataLoader queue.
#
# PATCHES
# -------
# Patch D — train.py collate_fn:
#   Stop returning node_features / edge_index in the batch dict.
#
# Patch E — train.py run_finetuning:
#   After constructing the trainer, call trainer.set_graph_tensors()
#   to store the graph on the correct device once.
#
# Patch F — src/training/finetune.py:
#   • Add set_graph_tensors() method.
#   • In train_epoch: get node_features/edge_index from self when absent
#     from batch (backward-compatible: still works if present in batch).
# ══════════════════════════════════════════════════════════════════════════
from pathlib import Path

train_py    = PROJECT_ROOT / "scripts" / "train.py"
finetune_py = PROJECT_ROOT / "src" / "training" / "finetune.py"

# ─────────────────────────────────────────────────────────────────────────
# Patch D — remove large constants from collate return dict
# ─────────────────────────────────────────────────────────────────────────
src_train = train_py.read_text()

_OLD_D = """        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'disease_texts': combined_disease_texts,
            'gene_indices': gene_indices,
            'labels': labels
        }"""

_NEW_D = """        # node_features and edge_index are deliberately NOT included here.
        # They are graph-wide constants (~40 MB combined) that never change
        # between batches.  Returning them through the DataLoader queue forces
        # workers to serialize them into /dev/shm on every prefetch, which
        # exhausts shared memory and causes ENOSPC Bus Error crashes.
        # The trainer receives them once via set_graph_tensors() below.
        return {
            'disease_texts': combined_disease_texts,
            'gene_indices': gene_indices,
            'labels': labels
        }"""

if _OLD_D in src_train:
    src_train = src_train.replace(_OLD_D, _NEW_D)
    print("✅ Patch D applied — node_features/edge_index removed from batch dict")
elif "'node_features': node_features" not in src_train:
    print("ℹ️  Patch D already applied")
else:
    print("⚠️  Patch D: exact string not found — check indentation in collate_fn")

# ─────────────────────────────────────────────────────────────────────────
# Patch E — call set_graph_tensors() after trainer construction
# Insert after the existing  `if emb_cache: trainer.set_prompt_cache(...)` block
# ─────────────────────────────────────────────────────────────────────────
_OLD_E = """    if emb_cache:
        trainer.set_prompt_cache(emb_cache)"""

_NEW_E = """    if emb_cache:
        trainer.set_prompt_cache(emb_cache)

    # ── Graph tensors: pass once to trainer, never through DataLoader queue ──
    # Resolve node features from the graph (same logic as create_collate_fn).
    _graph = dataset.graph
    _num_g = _graph['gene'].num_nodes
    if hasattr(_graph['gene'], 'x') and _graph['gene'].x is not None:
        _nf = _graph['gene'].x
    else:
        import torch as _t
        _t.manual_seed(42)
        _nf = _t.randn(_num_g, config.get('model', {}).get('gene_feature_dim', 128))

    # Resolve edge index (gene-gene PPI edges).
    _et = _graph.edge_types if hasattr(_graph, 'edge_types') else []
    _ei = None
    for _candidate in [('gene','interacts','gene'),
                        ('gene','protein_interaction','gene'),
                        ('gene','ppi','gene')]:
        if _candidate in _et:
            _ei = _graph[_candidate].edge_index
            break
    if _ei is None:
        import torch as _t
        _ei = _t.empty((2, 0), dtype=_t.long)

    trainer.set_graph_tensors(_nf, _ei)
    logger.info(f"  ✅ Graph tensors set on trainer "
                f"(node_features={list(_nf.shape)}, "
                f"edge_index={list(_ei.shape)}) — removed from DataLoader queue")"""

if _OLD_E in src_train:
    src_train = src_train.replace(_OLD_E, _NEW_E)
    print("✅ Patch E applied — set_graph_tensors() called after trainer construction")
elif "set_graph_tensors" in src_train and "Graph tensors: pass once" in src_train:
    print("ℹ️  Patch E already applied")
else:
    print("⚠️  Patch E: anchor not found — trainer.set_prompt_cache() line may differ")

train_py.write_text(src_train)
print(f"   train.py saved")

# ─────────────────────────────────────────────────────────────────────────
# Patch F — finetune.py: add set_graph_tensors() + use in train_epoch
# ─────────────────────────────────────────────────────────────────────────
src_ft = finetune_py.read_text()

# F1 — add set_graph_tensors method alongside set_prompt_cache
_OLD_F1 = """    def set_prompt_cache(self, cache: dict):"""

_NEW_F1 = """    def set_graph_tensors(self, node_features, edge_index):
        \"\"\"Store graph-wide tensors on the trainer so the DataLoader never
        needs to carry them inside batch dicts.  Called once after construction.\"\"\"
        self.graph_node_features = node_features.to(self.device)
        self.graph_edge_index    = edge_index.to(self.device)
        logger.info(f"  Graph tensors moved to {self.device}: "
                    f"node_features={list(node_features.shape)}, "
                    f"edge_index={list(edge_index.shape)}")

    def set_prompt_cache(self, cache: dict):"""

if _OLD_F1 in src_ft:
    src_ft = src_ft.replace(_OLD_F1, _NEW_F1)
    print("✅ Patch F1 applied — set_graph_tensors() method added to PromptGFMTrainer")
elif "set_graph_tensors" in src_ft:
    print("ℹ️  Patch F1 already applied")
else:
    # Fallback: insert before set_prompt_cache regardless of exact signature
    _fallback_anchor = "def set_prompt_cache"
    if _fallback_anchor in src_ft:
        src_ft = src_ft.replace(
            _fallback_anchor,
            "def set_graph_tensors(self, node_features, edge_index):\n"
            "        self.graph_node_features = node_features.to(self.device)\n"
            "        self.graph_edge_index    = edge_index.to(self.device)\n\n"
            "    " + _fallback_anchor
        )
        print("✅ Patch F1 applied (fallback path)")
    else:
        print("⚠️  Patch F1: could not locate set_prompt_cache — add set_graph_tensors manually")

# F2 — in train_epoch, get node_features/edge_index from self when not in batch
# Locate the lines that extract these two keys from the batch dict.
_OLD_F2 = """            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)"""

_NEW_F2 = """            # Graph-wide constants live on the trainer (set_graph_tensors),
            # not in the batch dict — keeps DataLoader queue lean.
            node_features = (batch['node_features'].to(self.device)
                             if 'node_features' in batch
                             else self.graph_node_features)
            edge_index    = (batch['edge_index'].to(self.device)
                             if 'edge_index' in batch
                             else self.graph_edge_index)"""

if _OLD_F2 in src_ft:
    src_ft = src_ft.replace(_OLD_F2, _NEW_F2)
    print("✅ Patch F2 applied — train_epoch uses trainer-cached graph tensors")
elif "self.graph_node_features" in src_ft:
    print("ℹ️  Patch F2 already applied")
else:
    # Try alternate indentation (4-space vs 8-space outer function body)
    _OLD_F2b = """        node_features = batch['node_features'].to(self.device)
        edge_index = batch['edge_index'].to(self.device)"""
    _NEW_F2b = """        node_features = (batch['node_features'].to(self.device)
                         if 'node_features' in batch
                         else self.graph_node_features)
        edge_index    = (batch['edge_index'].to(self.device)
                         if 'edge_index' in batch
                         else self.graph_edge_index)"""
    if _OLD_F2b in src_ft:
        src_ft = src_ft.replace(_OLD_F2b, _NEW_F2b)
        print("✅ Patch F2 applied (alternate indentation)")
    else:
        print("⚠️  Patch F2: could not find node_features/edge_index extraction in train_epoch")
        print("   Open finetune.py and manually replace:")
        print("     batch['node_features'].to(self.device)")
        print("   with:")
        print("     batch.get('node_features', self.graph_node_features).to(self.device)  # etc.")

# Also apply the same backward-compatible pattern for validate_epoch if present
if "'node_features'" in src_ft and "validate_epoch" in src_ft:
    # Only patch lines inside validate_epoch that still do dict extraction
    _OLD_F3 = """            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)"""
    # This is the same as _OLD_F2 but may appear a second time in validate_epoch
    count_old = src_ft.count(_OLD_F3)
    if count_old > 0:
        src_ft = src_ft.replace(_OLD_F3, _NEW_F2)   # replace ALL remaining occurrences
        print(f"✅ Patch F3 applied — validate_epoch also updated ({count_old} site(s))")

finetune_py.write_text(src_ft)
print(f"   finetune.py saved")

# ─────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────
print()
print("─── Verification ───")
_t_src = train_py.read_text()
_f_src = finetune_py.read_text()

checks = {
    "batch dict no longer has node_features key":
        "'node_features': node_features" not in _t_src,
    "batch dict no longer has edge_index key":
        "'edge_index': edge_index" not in _t_src,
    "set_graph_tensors() called in run_finetuning":
        "trainer.set_graph_tensors(" in _t_src,
    "set_graph_tensors() defined in finetune.py":
        "def set_graph_tensors(" in _f_src,
    "train_epoch uses self.graph_node_features":
        "self.graph_node_features" in _f_src,
}

all_ok = True
for desc, result in checks.items():
    status = "✅" if result else "❌"
    if not result:
        all_ok = False
    print(f"  {status}  {desc}")

print()
if all_ok:
    print("✅ All patches verified — proceed to training")
    print()
    print("Expected outcome:")
    print("  • DataLoader queue payload per batch: ~4 KB (indices + labels only)")
    print("  • /dev/shm usage: ~0 bytes from DataLoader")
    print("  • Graph tensors moved to GPU once at trainer init, reused every batch")
else:
    print("⚠️  Some patches need manual review — check the lines marked ❌ above")
```

::: {.output .stream .stdout}
    ✅ Patch D applied — node_features/edge_index removed from batch dict
    ✅ Patch E applied — set_graph_tensors() called after trainer construction
       train.py saved
    ✅ Patch F1 applied (fallback path)
    ℹ️  Patch F2 already applied
       finetune.py saved

    ─── Verification ───
      ✅  batch dict no longer has node_features key
      ✅  batch dict no longer has edge_index key
      ✅  set_graph_tensors() called in run_finetuning
      ✅  set_graph_tensors() defined in finetune.py
      ✅  train_epoch uses self.graph_node_features

    ✅ All patches verified — proceed to training

    Expected outcome:
      • DataLoader queue payload per batch: ~4 KB (indices + labels only)
      • /dev/shm usage: ~0 bytes from DataLoader
      • Graph tensors moved to GPU once at trainer init, reused every batch
:::
:::

::: {#1c94ce39 .cell .markdown}
## 12. Train {#12-train}

Uses `configs/workstation_config.yaml` auto-generated above with
VRAM-aware batch_size.
:::

::: {#3e6e57ff .cell .code execution_count="23"}
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

::: {#739e7817 .cell .code execution_count="32"}
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
    Resuming from: /home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film/checkpoint_epoch_5.pt
    GPUs available: 1
    Single-GPU launch (RTX 4090)
    Running: /home/mluser/micromamba/envs/promptgfm/bin/python /home/mluser/projects_yash/new_project/PromptGFM-Bio/scripts/train.py --config /home/mluser/projects_yash/new_project/PromptGFM-Bio/configs/workstation_config.yaml --resume-checkpoint /home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film/checkpoint_epoch_5.pt
:::

::: {.output .stream .stderr}
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
    INFO:src.data.dataset:Graph loaded: gene=19576, disease=16841, phenotype=11794, ('gene', 'interacts', 'gene')=1854012, ('gene', 'associated_with', 'disease')=9741610, ('disease', 'rev_associated_with', 'gene')=9741610
    INFO:src.data.dataset:Loading gene-disease edges from data/processed/hpo_gene_disease_edges.csv
    INFO:src.data.dataset:Vocabulary: 5251 genes, 12714 diseases
    INFO:src.data.dataset:Loaded 1170143 edges ({'HPO_phenotype_bridge': 1170143})
    INFO:src.data.dataset:Split sizes: train=936114, val=117014, test=117015
    INFO:__main__:  Train: 936114 edges
    INFO:__main__:  Val:   117014 edges
    INFO:__main__:  Test:  117015 edges
    INFO:__main__:  Gene nodes in graph: 19576
    INFO:__main__:  Genes in dataset vocabulary: 5251
    INFO:__main__:  Input feature dim: 128
    INFO:__main__:  Using 5 negative samples per positive sample
    INFO:__main__:  DataLoader workers: 4 (from config)
    INFO:__main__:Training PromptGFM
    INFO:src.models.promptgfm:Initializing PromptGFM model...
    INFO:src.models.prompt_encoder:Initializing PromptEncoder with model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Loading weights: 100%|██████████| 199/199 [00:00<00:00, 113606.44it/s]
    BertModel LOAD REPORT from: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Key                                        | Status     |  | 
    -------------------------------------------+------------+--+-
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
    cls.predictions.bias                       | UNEXPECTED |  | 
    cls.seq_relationship.bias                  | UNEXPECTED |  | 
    cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
    cls.predictions.decoder.weight             | UNEXPECTED |  | 
    cls.seq_relationship.weight                | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
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
    /home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:91: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      self.scaler = GradScaler() if self.use_amp else None
    INFO:src.training.finetune:PromptGFMTrainer initialized:
    INFO:src.training.finetune:  Device: cuda
    INFO:src.training.finetune:  Max epochs: 100
    INFO:src.training.finetune:  Patience: 15
    INFO:src.training.finetune:  Gradient clip: 1.0
    INFO:src.training.finetune:  Mixed precision (AMP): enabled
    INFO:src.training.finetune:  Checkpoint dir: checkpoints/promptgfm_film
    INFO:src.training.finetune:  Prompt embedding cache: 12714 unique disease texts — BioBERT will be skipped per batch
    INFO:__main__:  ✅ Graph tensors set on trainer (node_features=[19576, 128], edge_index=[2, 1854012]) — removed from DataLoader queue
    INFO:__main__:
    ============================================================
    INFO:__main__:RESUMING FROM CHECKPOINT
    INFO:__main__:============================================================
    INFO:src.training.finetune:Loading checkpoint: /home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film/checkpoint_epoch_5.pt
    INFO:src.training.finetune:✓ Resumed from epoch 5
    INFO:src.training.finetune:  Best val metric: 0.9413
    INFO:src.training.finetune:  Global step: 6095
    INFO:src.training.finetune:
    Starting training from epoch 6 to 100
    INFO:src.training.finetune:Early stopping on: auroc
    Epoch 6/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 6/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.52it/s, loss=0.242]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.66it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 6/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.3s (Avg: 93.8s/epoch)
    INFO:src.training.finetune:  ETA: 2h 26m (for 94 epochs)
    INFO:src.training.finetune:  Train Loss: 0.247517
    INFO:src.training.finetune:  Val Loss:   0.257274
    INFO:src.training.finetune:  Val AUROC:  0.9418
    INFO:src.training.finetune:  Val AUPR:   0.7206
    INFO:src.training.finetune:  ✓ New best auroc: 0.9418 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 7/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 7/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.55it/s, loss=0.242]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.20it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 7/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 93.8s/epoch)
    INFO:src.training.finetune:  ETA: 2h 25m (for 93 epochs)
    INFO:src.training.finetune:  Train Loss: 0.245183
    INFO:src.training.finetune:  Val Loss:   0.259762
    INFO:src.training.finetune:  Val AUROC:  0.9416
    INFO:src.training.finetune:  Val AUPR:   0.7211
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 8/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 8/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.56it/s, loss=0.248]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.15it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 8/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.0s (Avg: 93.8s/epoch)
    INFO:src.training.finetune:  ETA: 2h 23m (for 92 epochs)
    INFO:src.training.finetune:  Train Loss: 0.243478
    INFO:src.training.finetune:  Val Loss:   0.257510
    INFO:src.training.finetune:  Val AUROC:  0.9420
    INFO:src.training.finetune:  Val AUPR:   0.7214
    INFO:src.training.finetune:  ✓ New best auroc: 0.9420 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 9/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 9/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.54it/s, loss=0.223]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.80it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 9/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 93.9s/epoch)
    INFO:src.training.finetune:  ETA: 2h 22m (for 91 epochs)
    INFO:src.training.finetune:  Train Loss: 0.242457
    INFO:src.training.finetune:  Val Loss:   0.255371
    INFO:src.training.finetune:  Val AUROC:  0.9430
    INFO:src.training.finetune:  Val AUPR:   0.7249
    INFO:src.training.finetune:  ✓ New best auroc: 0.9430 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 10/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 10/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.54it/s, loss=0.236]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.47it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 10/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 93.9s/epoch)
    INFO:src.training.finetune:  ETA: 2h 20m (for 90 epochs)
    INFO:src.training.finetune:  Train Loss: 0.240837
    INFO:src.training.finetune:  Val Loss:   0.252956
    INFO:src.training.finetune:  Val AUROC:  0.9436
    INFO:src.training.finetune:  Val AUPR:   0.7272
    INFO:src.training.finetune:  ✓ New best auroc: 0.9436 (saved as best_model.pt)
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_10.pt
    INFO:src.training.finetune:======================================================================

    Epoch 11/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 11/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.54it/s, loss=0.244]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.96it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 11/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 93.9s/epoch)
    INFO:src.training.finetune:  ETA: 2h 19m (for 89 epochs)
    INFO:src.training.finetune:  Train Loss: 0.240722
    INFO:src.training.finetune:  Val Loss:   0.252375
    INFO:src.training.finetune:  Val AUROC:  0.9439
    INFO:src.training.finetune:  Val AUPR:   0.7267
    INFO:src.training.finetune:  ✓ New best auroc: 0.9439 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 12/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 12/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.56it/s, loss=0.243]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.77it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 12/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 93.9s/epoch)
    INFO:src.training.finetune:  ETA: 2h 17m (for 88 epochs)
    INFO:src.training.finetune:  Train Loss: 0.239528
    INFO:src.training.finetune:  Val Loss:   0.251239
    INFO:src.training.finetune:  Val AUROC:  0.9448
    INFO:src.training.finetune:  Val AUPR:   0.7300
    INFO:src.training.finetune:  ✓ New best auroc: 0.9448 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 13/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 13/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.55it/s, loss=0.231]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.78it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 13/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 93.9s/epoch)
    INFO:src.training.finetune:  ETA: 2h 16m (for 87 epochs)
    INFO:src.training.finetune:  Train Loss: 0.237750
    INFO:src.training.finetune:  Val Loss:   0.250345
    INFO:src.training.finetune:  Val AUROC:  0.9450
    INFO:src.training.finetune:  Val AUPR:   0.7298
    INFO:src.training.finetune:  ✓ New best auroc: 0.9450 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 14/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 14/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.23] 
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.29it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 14/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 14m (for 86 epochs)
    INFO:src.training.finetune:  Train Loss: 0.236233
    INFO:src.training.finetune:  Val Loss:   0.249600
    INFO:src.training.finetune:  Val AUROC:  0.9453
    INFO:src.training.finetune:  Val AUPR:   0.7298
    INFO:src.training.finetune:  ✓ New best auroc: 0.9453 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 15/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 15/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.54it/s, loss=0.239]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.48it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 15/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 13m (for 85 epochs)
    INFO:src.training.finetune:  Train Loss: 0.235141
    INFO:src.training.finetune:  Val Loss:   0.248030
    INFO:src.training.finetune:  Val AUROC:  0.9460
    INFO:src.training.finetune:  Val AUPR:   0.7326
    INFO:src.training.finetune:  ✓ New best auroc: 0.9460 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 16/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 16/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.236]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.98it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 16/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.3s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 11m (for 84 epochs)
    INFO:src.training.finetune:  Train Loss: 0.233160
    INFO:src.training.finetune:  Val Loss:   0.246566
    INFO:src.training.finetune:  Val AUROC:  0.9467
    INFO:src.training.finetune:  Val AUPR:   0.7356
    INFO:src.training.finetune:  ✓ New best auroc: 0.9467 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 17/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 17/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.54it/s, loss=0.234]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.39it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 17/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 10m (for 83 epochs)
    INFO:src.training.finetune:  Train Loss: 0.231919
    INFO:src.training.finetune:  Val Loss:   0.246240
    INFO:src.training.finetune:  Val AUROC:  0.9471
    INFO:src.training.finetune:  Val AUPR:   0.7359
    INFO:src.training.finetune:  ✓ New best auroc: 0.9471 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 18/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 18/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.235]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.16it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 18/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 8m (for 82 epochs)
    INFO:src.training.finetune:  Train Loss: 0.230465
    INFO:src.training.finetune:  Val Loss:   0.244793
    INFO:src.training.finetune:  Val AUROC:  0.9476
    INFO:src.training.finetune:  Val AUPR:   0.7358
    INFO:src.training.finetune:  ✓ New best auroc: 0.9476 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 19/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 19/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.54it/s, loss=0.225]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.77it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 19/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 6m (for 81 epochs)
    INFO:src.training.finetune:  Train Loss: 0.229850
    INFO:src.training.finetune:  Val Loss:   0.244806
    INFO:src.training.finetune:  Val AUROC:  0.9476
    INFO:src.training.finetune:  Val AUPR:   0.7352
    INFO:src.training.finetune:  ✓ New best auroc: 0.9476 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 20/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 20/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.55it/s, loss=0.225]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.67it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 20/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 5m (for 80 epochs)
    INFO:src.training.finetune:  Train Loss: 0.228519
    INFO:src.training.finetune:  Val Loss:   0.244082
    INFO:src.training.finetune:  Val AUROC:  0.9476
    INFO:src.training.finetune:  Val AUPR:   0.7360
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_20.pt
    INFO:src.training.finetune:======================================================================

    Epoch 21/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 21/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.55it/s, loss=0.225]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.33it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 21/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 3m (for 79 epochs)
    INFO:src.training.finetune:  Train Loss: 0.227414
    INFO:src.training.finetune:  Val Loss:   0.244472
    INFO:src.training.finetune:  Val AUROC:  0.9479
    INFO:src.training.finetune:  Val AUPR:   0.7376
    INFO:src.training.finetune:  ✓ New best auroc: 0.9479 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 22/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 22/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.55it/s, loss=0.229]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 40.84it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 22/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.0s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 2m (for 78 epochs)
    INFO:src.training.finetune:  Train Loss: 0.226832
    INFO:src.training.finetune:  Val Loss:   0.242422
    INFO:src.training.finetune:  Val AUROC:  0.9484
    INFO:src.training.finetune:  Val AUPR:   0.7384
    INFO:src.training.finetune:  ✓ New best auroc: 0.9484 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 23/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 23/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.226]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 41.26it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 23/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.1s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 2h 0m (for 77 epochs)
    INFO:src.training.finetune:  Train Loss: 0.225653
    INFO:src.training.finetune:  Val Loss:   0.243119
    INFO:src.training.finetune:  Val AUROC:  0.9482
    INFO:src.training.finetune:  Val AUPR:   0.7374
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 24/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 24/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.229]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.40it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 24/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.3s (Avg: 94.0s/epoch)
    INFO:src.training.finetune:  ETA: 1h 59m (for 76 epochs)
    INFO:src.training.finetune:  Train Loss: 0.225156
    INFO:src.training.finetune:  Val Loss:   0.242056
    INFO:src.training.finetune:  Val AUROC:  0.9487
    INFO:src.training.finetune:  Val AUPR:   0.7389
    INFO:src.training.finetune:  ✓ New best auroc: 0.9487 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 25/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 25/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.49it/s, loss=0.219]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 41.47it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 25/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.4s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 57m (for 75 epochs)
    INFO:src.training.finetune:  Train Loss: 0.224125
    INFO:src.training.finetune:  Val Loss:   0.242379
    INFO:src.training.finetune:  Val AUROC:  0.9486
    INFO:src.training.finetune:  Val AUPR:   0.7377
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 26/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 26/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.238]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 38.77it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 26/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.4s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 56m (for 74 epochs)
    INFO:src.training.finetune:  Train Loss: 0.224044
    INFO:src.training.finetune:  Val Loss:   0.240600
    INFO:src.training.finetune:  Val AUROC:  0.9492
    INFO:src.training.finetune:  Val AUPR:   0.7387
    INFO:src.training.finetune:  ✓ New best auroc: 0.9492 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 27/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 27/100: 100%|██████████| 1219/1219 [01:29<00:00, 13.54it/s, loss=0.221]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 39.56it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 27/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.2s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 54m (for 73 epochs)
    INFO:src.training.finetune:  Train Loss: 0.222709
    INFO:src.training.finetune:  Val Loss:   0.240568
    INFO:src.training.finetune:  Val AUROC:  0.9493
    INFO:src.training.finetune:  Val AUPR:   0.7357
    INFO:src.training.finetune:  ✓ New best auroc: 0.9493 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 28/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 28/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.219]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 38.37it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 28/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.5s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 52m (for 72 epochs)
    INFO:src.training.finetune:  Train Loss: 0.221667
    INFO:src.training.finetune:  Val Loss:   0.240684
    INFO:src.training.finetune:  Val AUROC:  0.9493
    INFO:src.training.finetune:  Val AUPR:   0.7377
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 29/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 29/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.207]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.11it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 29/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.6s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 51m (for 71 epochs)
    INFO:src.training.finetune:  Train Loss: 0.221813
    INFO:src.training.finetune:  Val Loss:   0.239442
    INFO:src.training.finetune:  Val AUROC:  0.9496
    INFO:src.training.finetune:  Val AUPR:   0.7389
    INFO:src.training.finetune:  ✓ New best auroc: 0.9496 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 30/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 30/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.54it/s, loss=0.217]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.42it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 30/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.5s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 49m (for 70 epochs)
    INFO:src.training.finetune:  Train Loss: 0.220215
    INFO:src.training.finetune:  Val Loss:   0.239234
    INFO:src.training.finetune:  Val AUROC:  0.9499
    INFO:src.training.finetune:  Val AUPR:   0.7422
    INFO:src.training.finetune:  ✓ New best auroc: 0.9499 (saved as best_model.pt)
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_30.pt
    INFO:src.training.finetune:======================================================================

    Epoch 31/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 31/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.221]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.83it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 31/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.5s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 48m (for 69 epochs)
    INFO:src.training.finetune:  Train Loss: 0.219828
    INFO:src.training.finetune:  Val Loss:   0.239031
    INFO:src.training.finetune:  Val AUROC:  0.9499
    INFO:src.training.finetune:  Val AUPR:   0.7396
    INFO:src.training.finetune:  ✓ New best auroc: 0.9499 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 32/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 32/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.52it/s, loss=0.221]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.78it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 32/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.6s (Avg: 94.1s/epoch)
    INFO:src.training.finetune:  ETA: 1h 46m (for 68 epochs)
    INFO:src.training.finetune:  Train Loss: 0.218940
    INFO:src.training.finetune:  Val Loss:   0.238484
    INFO:src.training.finetune:  Val AUROC:  0.9499
    INFO:src.training.finetune:  Val AUPR:   0.7391
    INFO:src.training.finetune:  ✓ New best auroc: 0.9499 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 33/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 33/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.228]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.27it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 33/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.2s/epoch)
    INFO:src.training.finetune:  ETA: 1h 45m (for 67 epochs)
    INFO:src.training.finetune:  Train Loss: 0.218275
    INFO:src.training.finetune:  Val Loss:   0.236883
    INFO:src.training.finetune:  Val AUROC:  0.9509
    INFO:src.training.finetune:  Val AUPR:   0.7448
    INFO:src.training.finetune:  ✓ New best auroc: 0.9509 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 34/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 34/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.225]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.46it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 34/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.2s/epoch)
    INFO:src.training.finetune:  ETA: 1h 43m (for 66 epochs)
    INFO:src.training.finetune:  Train Loss: 0.217830
    INFO:src.training.finetune:  Val Loss:   0.237055
    INFO:src.training.finetune:  Val AUROC:  0.9504
    INFO:src.training.finetune:  Val AUPR:   0.7401
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 35/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 35/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.53it/s, loss=0.22] 
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.36it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 35/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.6s (Avg: 94.2s/epoch)
    INFO:src.training.finetune:  ETA: 1h 42m (for 65 epochs)
    INFO:src.training.finetune:  Train Loss: 0.217172
    INFO:src.training.finetune:  Val Loss:   0.238143
    INFO:src.training.finetune:  Val AUROC:  0.9502
    INFO:src.training.finetune:  Val AUPR:   0.7402
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 36/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 36/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.48it/s, loss=0.218]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.33it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 36/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.0s (Avg: 94.2s/epoch)
    INFO:src.training.finetune:  ETA: 1h 40m (for 64 epochs)
    INFO:src.training.finetune:  Train Loss: 0.216097
    INFO:src.training.finetune:  Val Loss:   0.237163
    INFO:src.training.finetune:  Val AUROC:  0.9505
    INFO:src.training.finetune:  Val AUPR:   0.7416
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 37/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 37/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.52it/s, loss=0.219]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.12it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 37/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.6s (Avg: 94.2s/epoch)
    INFO:src.training.finetune:  ETA: 1h 38m (for 63 epochs)
    INFO:src.training.finetune:  Train Loss: 0.215990
    INFO:src.training.finetune:  Val Loss:   0.236418
    INFO:src.training.finetune:  Val AUROC:  0.9510
    INFO:src.training.finetune:  Val AUPR:   0.7423
    INFO:src.training.finetune:  ✓ New best auroc: 0.9510 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 38/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 38/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.225]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.62it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 38/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.2s/epoch)
    INFO:src.training.finetune:  ETA: 1h 37m (for 62 epochs)
    INFO:src.training.finetune:  Train Loss: 0.215451
    INFO:src.training.finetune:  Val Loss:   0.236152
    INFO:src.training.finetune:  Val AUROC:  0.9510
    INFO:src.training.finetune:  Val AUPR:   0.7451
    INFO:src.training.finetune:  ✓ New best auroc: 0.9510 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 39/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 39/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.221]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.24it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 39/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 35m (for 61 epochs)
    INFO:src.training.finetune:  Train Loss: 0.214428
    INFO:src.training.finetune:  Val Loss:   0.235315
    INFO:src.training.finetune:  Val AUROC:  0.9512
    INFO:src.training.finetune:  Val AUPR:   0.7434
    INFO:src.training.finetune:  ✓ New best auroc: 0.9512 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 40/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 40/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.204]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.43it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 40/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 34m (for 60 epochs)
    INFO:src.training.finetune:  Train Loss: 0.213878
    INFO:src.training.finetune:  Val Loss:   0.235854
    INFO:src.training.finetune:  Val AUROC:  0.9513
    INFO:src.training.finetune:  Val AUPR:   0.7421
    INFO:src.training.finetune:  ✓ New best auroc: 0.9513 (saved as best_model.pt)
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_40.pt
    INFO:src.training.finetune:======================================================================

    Epoch 41/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 41/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.49it/s, loss=0.21] 
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.43it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 41/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.8s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 32m (for 59 epochs)
    INFO:src.training.finetune:  Train Loss: 0.213704
    INFO:src.training.finetune:  Val Loss:   0.236574
    INFO:src.training.finetune:  Val AUROC:  0.9508
    INFO:src.training.finetune:  Val AUPR:   0.7426
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 42/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 42/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.52it/s, loss=0.213]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 38.13it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 42/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.5s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 31m (for 58 epochs)
    INFO:src.training.finetune:  Train Loss: 0.212913
    INFO:src.training.finetune:  Val Loss:   0.234474
    INFO:src.training.finetune:  Val AUROC:  0.9516
    INFO:src.training.finetune:  Val AUPR:   0.7419
    INFO:src.training.finetune:  ✓ New best auroc: 0.9516 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 43/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 43/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.47it/s, loss=0.201]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.07it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 43/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.0s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 29m (for 57 epochs)
    INFO:src.training.finetune:  Train Loss: 0.212670
    INFO:src.training.finetune:  Val Loss:   0.235188
    INFO:src.training.finetune:  Val AUROC:  0.9514
    INFO:src.training.finetune:  Val AUPR:   0.7406
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 44/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 44/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.46it/s, loss=0.217]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.25it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 44/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 28m (for 56 epochs)
    INFO:src.training.finetune:  Train Loss: 0.211993
    INFO:src.training.finetune:  Val Loss:   0.235154
    INFO:src.training.finetune:  Val AUROC:  0.9513
    INFO:src.training.finetune:  Val AUPR:   0.7396
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 45/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 45/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.51it/s, loss=0.208]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.39it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 45/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.6s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 26m (for 55 epochs)
    INFO:src.training.finetune:  Train Loss: 0.211623
    INFO:src.training.finetune:  Val Loss:   0.235812
    INFO:src.training.finetune:  Val AUROC:  0.9513
    INFO:src.training.finetune:  Val AUPR:   0.7418
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 46/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 46/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.214]
    Validating: 100%|██████████| 153/153 [00:03<00:00, 38.55it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 46/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.6s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 24m (for 54 epochs)
    INFO:src.training.finetune:  Train Loss: 0.210955
    INFO:src.training.finetune:  Val Loss:   0.233566
    INFO:src.training.finetune:  Val AUROC:  0.9520
    INFO:src.training.finetune:  Val AUPR:   0.7457
    INFO:src.training.finetune:  ✓ New best auroc: 0.9520 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 47/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 47/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.48it/s, loss=0.2]  
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.06it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 47/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.9s (Avg: 94.3s/epoch)
    INFO:src.training.finetune:  ETA: 1h 23m (for 53 epochs)
    INFO:src.training.finetune:  Train Loss: 0.210288
    INFO:src.training.finetune:  Val Loss:   0.234391
    INFO:src.training.finetune:  Val AUROC:  0.9517
    INFO:src.training.finetune:  Val AUPR:   0.7430
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 48/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 48/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.51it/s, loss=0.211]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.50it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 48/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.8s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 21m (for 52 epochs)
    INFO:src.training.finetune:  Train Loss: 0.210323
    INFO:src.training.finetune:  Val Loss:   0.234217
    INFO:src.training.finetune:  Val AUROC:  0.9518
    INFO:src.training.finetune:  Val AUPR:   0.7452
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 49/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 49/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.47it/s, loss=0.222]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.48it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 49/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.9s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 20m (for 51 epochs)
    INFO:src.training.finetune:  Train Loss: 0.209681
    INFO:src.training.finetune:  Val Loss:   0.233890
    INFO:src.training.finetune:  Val AUROC:  0.9519
    INFO:src.training.finetune:  Val AUPR:   0.7461
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 50/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 50/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.51it/s, loss=0.209]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.17it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 50/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 18m (for 50 epochs)
    INFO:src.training.finetune:  Train Loss: 0.209063
    INFO:src.training.finetune:  Val Loss:   0.233823
    INFO:src.training.finetune:  Val AUROC:  0.9520
    INFO:src.training.finetune:  Val AUPR:   0.7440
    INFO:src.training.finetune:  No improvement for 4 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_50.pt
    INFO:src.training.finetune:======================================================================

    Epoch 51/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 51/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.50it/s, loss=0.199]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.17it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 51/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.8s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 17m (for 49 epochs)
    INFO:src.training.finetune:  Train Loss: 0.208470
    INFO:src.training.finetune:  Val Loss:   0.233567
    INFO:src.training.finetune:  Val AUROC:  0.9520
    INFO:src.training.finetune:  Val AUPR:   0.7401
    INFO:src.training.finetune:  ✓ New best auroc: 0.9520 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 52/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 52/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.51it/s, loss=0.222]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.81it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 52/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.7s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 15m (for 48 epochs)
    INFO:src.training.finetune:  Train Loss: 0.208200
    INFO:src.training.finetune:  Val Loss:   0.233678
    INFO:src.training.finetune:  Val AUROC:  0.9521
    INFO:src.training.finetune:  Val AUPR:   0.7441
    INFO:src.training.finetune:  ✓ New best auroc: 0.9521 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 53/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 53/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.49it/s, loss=0.209]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.32it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 53/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.8s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 13m (for 47 epochs)
    INFO:src.training.finetune:  Train Loss: 0.207846
    INFO:src.training.finetune:  Val Loss:   0.233731
    INFO:src.training.finetune:  Val AUROC:  0.9521
    INFO:src.training.finetune:  Val AUPR:   0.7442
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 54/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 54/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.47it/s, loss=0.22] 
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.10it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 54/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.0s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 12m (for 46 epochs)
    INFO:src.training.finetune:  Train Loss: 0.207607
    INFO:src.training.finetune:  Val Loss:   0.232550
    INFO:src.training.finetune:  Val AUROC:  0.9525
    INFO:src.training.finetune:  Val AUPR:   0.7450
    INFO:src.training.finetune:  ✓ New best auroc: 0.9525 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 55/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 55/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.48it/s, loss=0.21] 
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.16it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 55/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.9s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 10m (for 45 epochs)
    INFO:src.training.finetune:  Train Loss: 0.206759
    INFO:src.training.finetune:  Val Loss:   0.231853
    INFO:src.training.finetune:  Val AUROC:  0.9527
    INFO:src.training.finetune:  Val AUPR:   0.7469
    INFO:src.training.finetune:  ✓ New best auroc: 0.9527 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 56/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 56/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.48it/s, loss=0.213]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.51it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 56/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.9s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 9m (for 44 epochs)
    INFO:src.training.finetune:  Train Loss: 0.206314
    INFO:src.training.finetune:  Val Loss:   0.231732
    INFO:src.training.finetune:  Val AUROC:  0.9528
    INFO:src.training.finetune:  Val AUPR:   0.7479
    INFO:src.training.finetune:  ✓ New best auroc: 0.9528 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 57/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 57/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.47it/s, loss=0.209]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.82it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 57/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.0s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 7m (for 43 epochs)
    INFO:src.training.finetune:  Train Loss: 0.206041
    INFO:src.training.finetune:  Val Loss:   0.232329
    INFO:src.training.finetune:  Val AUROC:  0.9527
    INFO:src.training.finetune:  Val AUPR:   0.7467
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 58/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 58/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.46it/s, loss=0.217]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.98it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 58/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.4s/epoch)
    INFO:src.training.finetune:  ETA: 1h 6m (for 42 epochs)
    INFO:src.training.finetune:  Train Loss: 0.205550
    INFO:src.training.finetune:  Val Loss:   0.232051
    INFO:src.training.finetune:  Val AUROC:  0.9527
    INFO:src.training.finetune:  Val AUPR:   0.7457
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 59/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 59/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.44it/s, loss=0.192]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.81it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 59/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.2s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 1h 4m (for 41 epochs)
    INFO:src.training.finetune:  Train Loss: 0.205274
    INFO:src.training.finetune:  Val Loss:   0.231054
    INFO:src.training.finetune:  Val AUROC:  0.9531
    INFO:src.training.finetune:  Val AUPR:   0.7477
    INFO:src.training.finetune:  ✓ New best auroc: 0.9531 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 60/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 60/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.43it/s, loss=0.199]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.05it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 60/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 1h 2m (for 40 epochs)
    INFO:src.training.finetune:  Train Loss: 0.204578
    INFO:src.training.finetune:  Val Loss:   0.232239
    INFO:src.training.finetune:  Val AUROC:  0.9525
    INFO:src.training.finetune:  Val AUPR:   0.7446
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_60.pt
    INFO:src.training.finetune:======================================================================

    Epoch 61/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 61/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.187]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.05it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 61/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 1h 1m (for 39 epochs)
    INFO:src.training.finetune:  Train Loss: 0.204592
    INFO:src.training.finetune:  Val Loss:   0.231794
    INFO:src.training.finetune:  Val AUROC:  0.9528
    INFO:src.training.finetune:  Val AUPR:   0.7465
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 62/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 62/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.45it/s, loss=0.194]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.94it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 62/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.2s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 0h 59m (for 38 epochs)
    INFO:src.training.finetune:  Train Loss: 0.204040
    INFO:src.training.finetune:  Val Loss:   0.230550
    INFO:src.training.finetune:  Val AUROC:  0.9532
    INFO:src.training.finetune:  Val AUPR:   0.7451
    INFO:src.training.finetune:  ✓ New best auroc: 0.9532 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 63/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 63/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.47it/s, loss=0.21] 
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.78it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 63/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 0h 58m (for 37 epochs)
    INFO:src.training.finetune:  Train Loss: 0.203561
    INFO:src.training.finetune:  Val Loss:   0.231132
    INFO:src.training.finetune:  Val AUROC:  0.9530
    INFO:src.training.finetune:  Val AUPR:   0.7457
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 64/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 64/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.49it/s, loss=0.206]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.63it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 64/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.0s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 0h 56m (for 36 epochs)
    INFO:src.training.finetune:  Train Loss: 0.202961
    INFO:src.training.finetune:  Val Loss:   0.231399
    INFO:src.training.finetune:  Val AUROC:  0.9529
    INFO:src.training.finetune:  Val AUPR:   0.7461
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 65/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 65/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.46it/s, loss=0.204]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.61it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 65/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 0h 55m (for 35 epochs)
    INFO:src.training.finetune:  Train Loss: 0.202619
    INFO:src.training.finetune:  Val Loss:   0.230430
    INFO:src.training.finetune:  Val AUROC:  0.9534
    INFO:src.training.finetune:  Val AUPR:   0.7481
    INFO:src.training.finetune:  ✓ New best auroc: 0.9534 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 66/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 66/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.49it/s, loss=0.196]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.69it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 66/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 94.9s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 0h 53m (for 34 epochs)
    INFO:src.training.finetune:  Train Loss: 0.202444
    INFO:src.training.finetune:  Val Loss:   0.230079
    INFO:src.training.finetune:  Val AUROC:  0.9534
    INFO:src.training.finetune:  Val AUPR:   0.7460
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 67/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 67/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.42it/s, loss=0.203]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.15it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 67/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.5s/epoch)
    INFO:src.training.finetune:  ETA: 0h 51m (for 33 epochs)
    INFO:src.training.finetune:  Train Loss: 0.201658
    INFO:src.training.finetune:  Val Loss:   0.229597
    INFO:src.training.finetune:  Val AUROC:  0.9538
    INFO:src.training.finetune:  Val AUPR:   0.7459
    INFO:src.training.finetune:  ✓ New best auroc: 0.9538 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 68/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 68/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.39it/s, loss=0.202]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 37.70it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 68/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.4s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 50m (for 32 epochs)
    INFO:src.training.finetune:  Train Loss: 0.201373
    INFO:src.training.finetune:  Val Loss:   0.229284
    INFO:src.training.finetune:  Val AUROC:  0.9538
    INFO:src.training.finetune:  Val AUPR:   0.7482
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 69/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 69/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.46it/s, loss=0.209]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.08it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 69/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 48m (for 31 epochs)
    INFO:src.training.finetune:  Train Loss: 0.201151
    INFO:src.training.finetune:  Val Loss:   0.230564
    INFO:src.training.finetune:  Val AUROC:  0.9534
    INFO:src.training.finetune:  Val AUPR:   0.7478
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 70/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 70/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.44it/s, loss=0.206]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.95it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 70/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 47m (for 30 epochs)
    INFO:src.training.finetune:  Train Loss: 0.200996
    INFO:src.training.finetune:  Val Loss:   0.230245
    INFO:src.training.finetune:  Val AUROC:  0.9534
    INFO:src.training.finetune:  Val AUPR:   0.7478
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_70.pt
    INFO:src.training.finetune:======================================================================

    Epoch 71/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 71/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.41it/s, loss=0.198]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.62it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 71/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.4s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 45m (for 29 epochs)
    INFO:src.training.finetune:  Train Loss: 0.200367
    INFO:src.training.finetune:  Val Loss:   0.229411
    INFO:src.training.finetune:  Val AUROC:  0.9537
    INFO:src.training.finetune:  Val AUPR:   0.7469
    INFO:src.training.finetune:  No improvement for 4 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 72/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 72/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.46it/s, loss=0.189]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.55it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 72/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 44m (for 28 epochs)
    INFO:src.training.finetune:  Train Loss: 0.199976
    INFO:src.training.finetune:  Val Loss:   0.228902
    INFO:src.training.finetune:  Val AUROC:  0.9539
    INFO:src.training.finetune:  Val AUPR:   0.7488
    INFO:src.training.finetune:  ✓ New best auroc: 0.9539 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 73/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 73/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.41it/s, loss=0.193]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.89it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 73/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 42m (for 27 epochs)
    INFO:src.training.finetune:  Train Loss: 0.199847
    INFO:src.training.finetune:  Val Loss:   0.229043
    INFO:src.training.finetune:  Val AUROC:  0.9538
    INFO:src.training.finetune:  Val AUPR:   0.7469
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 74/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 74/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.41it/s, loss=0.208]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.24it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 74/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 41m (for 26 epochs)
    INFO:src.training.finetune:  Train Loss: 0.199485
    INFO:src.training.finetune:  Val Loss:   0.229764
    INFO:src.training.finetune:  Val AUROC:  0.9536
    INFO:src.training.finetune:  Val AUPR:   0.7441
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 75/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 75/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.39it/s, loss=0.207]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.21it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 75/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.6s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 39m (for 25 epochs)
    INFO:src.training.finetune:  Train Loss: 0.198827
    INFO:src.training.finetune:  Val Loss:   0.229699
    INFO:src.training.finetune:  Val AUROC:  0.9536
    INFO:src.training.finetune:  Val AUPR:   0.7477
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 76/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 76/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.46it/s, loss=0.193]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.73it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 76/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.1s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 37m (for 24 epochs)
    INFO:src.training.finetune:  Train Loss: 0.198536
    INFO:src.training.finetune:  Val Loss:   0.229080
    INFO:src.training.finetune:  Val AUROC:  0.9539
    INFO:src.training.finetune:  Val AUPR:   0.7489
    INFO:src.training.finetune:  No improvement for 4 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 77/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 77/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.43it/s, loss=0.198]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.83it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 77/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.6s/epoch)
    INFO:src.training.finetune:  ETA: 0h 36m (for 23 epochs)
    INFO:src.training.finetune:  Train Loss: 0.198235
    INFO:src.training.finetune:  Val Loss:   0.229748
    INFO:src.training.finetune:  Val AUROC:  0.9537
    INFO:src.training.finetune:  Val AUPR:   0.7471
    INFO:src.training.finetune:  No improvement for 5 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 78/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 78/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.43it/s, loss=0.21] 
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.28it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 78/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 34m (for 22 epochs)
    INFO:src.training.finetune:  Train Loss: 0.197921
    INFO:src.training.finetune:  Val Loss:   0.228065
    INFO:src.training.finetune:  Val AUROC:  0.9543
    INFO:src.training.finetune:  Val AUPR:   0.7509
    INFO:src.training.finetune:  ✓ New best auroc: 0.9543 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 79/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 79/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.41it/s, loss=0.188]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.36it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 79/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 33m (for 21 epochs)
    INFO:src.training.finetune:  Train Loss: 0.197866
    INFO:src.training.finetune:  Val Loss:   0.227802
    INFO:src.training.finetune:  Val AUROC:  0.9544
    INFO:src.training.finetune:  Val AUPR:   0.7483
    INFO:src.training.finetune:  ✓ New best auroc: 0.9544 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 80/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 80/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.191]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.46it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 80/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 31m (for 20 epochs)
    INFO:src.training.finetune:  Train Loss: 0.197321
    INFO:src.training.finetune:  Val Loss:   0.228712
    INFO:src.training.finetune:  Val AUROC:  0.9541
    INFO:src.training.finetune:  Val AUPR:   0.7498
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_80.pt
    INFO:src.training.finetune:======================================================================

    Epoch 81/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 81/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.44it/s, loss=0.201]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.47it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 81/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 29m (for 19 epochs)
    INFO:src.training.finetune:  Train Loss: 0.196993
    INFO:src.training.finetune:  Val Loss:   0.227793
    INFO:src.training.finetune:  Val AUROC:  0.9543
    INFO:src.training.finetune:  Val AUPR:   0.7496
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 82/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 82/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.41it/s, loss=0.194]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.43it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 82/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.4s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 28m (for 18 epochs)
    INFO:src.training.finetune:  Train Loss: 0.196810
    INFO:src.training.finetune:  Val Loss:   0.228688
    INFO:src.training.finetune:  Val AUROC:  0.9540
    INFO:src.training.finetune:  Val AUPR:   0.7486
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 83/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 83/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.43it/s, loss=0.202]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.13it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 83/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.4s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 26m (for 17 epochs)
    INFO:src.training.finetune:  Train Loss: 0.196756
    INFO:src.training.finetune:  Val Loss:   0.228050
    INFO:src.training.finetune:  Val AUROC:  0.9544
    INFO:src.training.finetune:  Val AUPR:   0.7503
    INFO:src.training.finetune:  ✓ New best auroc: 0.9544 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 84/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 84/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.2]  
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.65it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 84/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 25m (for 16 epochs)
    INFO:src.training.finetune:  Train Loss: 0.196115
    INFO:src.training.finetune:  Val Loss:   0.228722
    INFO:src.training.finetune:  Val AUROC:  0.9541
    INFO:src.training.finetune:  Val AUPR:   0.7488
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 85/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 85/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.39it/s, loss=0.197]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.84it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 85/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.6s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 23m (for 15 epochs)
    INFO:src.training.finetune:  Train Loss: 0.196045
    INFO:src.training.finetune:  Val Loss:   0.228101
    INFO:src.training.finetune:  Val AUROC:  0.9544
    INFO:src.training.finetune:  Val AUPR:   0.7499
    INFO:src.training.finetune:  ✓ New best auroc: 0.9544 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 86/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 86/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.198]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.37it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 86/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.5s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 22m (for 14 epochs)
    INFO:src.training.finetune:  Train Loss: 0.195721
    INFO:src.training.finetune:  Val Loss:   0.228428
    INFO:src.training.finetune:  Val AUROC:  0.9543
    INFO:src.training.finetune:  Val AUPR:   0.7491
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 87/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 87/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.41it/s, loss=0.195]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.81it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 87/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.4s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 20m (for 13 epochs)
    INFO:src.training.finetune:  Train Loss: 0.195711
    INFO:src.training.finetune:  Val Loss:   0.227299
    INFO:src.training.finetune:  Val AUROC:  0.9547
    INFO:src.training.finetune:  Val AUPR:   0.7507
    INFO:src.training.finetune:  ✓ New best auroc: 0.9547 (saved as best_model.pt)
    INFO:src.training.finetune:======================================================================

    Epoch 88/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 88/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.204]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.92it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 88/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.6s (Avg: 94.7s/epoch)
    INFO:src.training.finetune:  ETA: 0h 18m (for 12 epochs)
    INFO:src.training.finetune:  Train Loss: 0.195223
    INFO:src.training.finetune:  Val Loss:   0.228886
    INFO:src.training.finetune:  Val AUROC:  0.9540
    INFO:src.training.finetune:  Val AUPR:   0.7488
    INFO:src.training.finetune:  No improvement for 1 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 89/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 89/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.38it/s, loss=0.192]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.80it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 89/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.8s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 17m (for 11 epochs)
    INFO:src.training.finetune:  Train Loss: 0.195221
    INFO:src.training.finetune:  Val Loss:   0.228091
    INFO:src.training.finetune:  Val AUROC:  0.9545
    INFO:src.training.finetune:  Val AUPR:   0.7503
    INFO:src.training.finetune:  No improvement for 2 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 90/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 90/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.45it/s, loss=0.198]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.18it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 90/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.2s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 15m (for 10 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194932
    INFO:src.training.finetune:  Val Loss:   0.229030
    INFO:src.training.finetune:  Val AUROC:  0.9540
    INFO:src.training.finetune:  Val AUPR:   0.7480
    INFO:src.training.finetune:  No improvement for 3 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_90.pt
    INFO:src.training.finetune:======================================================================

    Epoch 91/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 91/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.38it/s, loss=0.191]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.28it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 91/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.7s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 14m (for 9 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194984
    INFO:src.training.finetune:  Val Loss:   0.227838
    INFO:src.training.finetune:  Val AUROC:  0.9545
    INFO:src.training.finetune:  Val AUPR:   0.7509
    INFO:src.training.finetune:  No improvement for 4 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 92/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 92/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.39it/s, loss=0.196]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.05it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 92/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.6s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 12m (for 8 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194769
    INFO:src.training.finetune:  Val Loss:   0.228050
    INFO:src.training.finetune:  Val AUROC:  0.9545
    INFO:src.training.finetune:  Val AUPR:   0.7506
    INFO:src.training.finetune:  No improvement for 5 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 93/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 93/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.38it/s, loss=0.187]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.83it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 93/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.7s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 11m (for 7 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194359
    INFO:src.training.finetune:  Val Loss:   0.228177
    INFO:src.training.finetune:  Val AUROC:  0.9546
    INFO:src.training.finetune:  Val AUPR:   0.7501
    INFO:src.training.finetune:  No improvement for 6 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 94/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 94/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.43it/s, loss=0.196]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.25it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 94/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 9m (for 6 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194631
    INFO:src.training.finetune:  Val Loss:   0.228050
    INFO:src.training.finetune:  Val AUROC:  0.9546
    INFO:src.training.finetune:  Val AUPR:   0.7519
    INFO:src.training.finetune:  No improvement for 7 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 95/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 95/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.43it/s, loss=0.209]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.61it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 95/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.3s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 7m (for 5 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194395
    INFO:src.training.finetune:  Val Loss:   0.228369
    INFO:src.training.finetune:  Val AUROC:  0.9544
    INFO:src.training.finetune:  Val AUPR:   0.7497
    INFO:src.training.finetune:  No improvement for 8 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 96/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 96/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.39it/s, loss=0.2]  
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.29it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 96/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.6s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 6m (for 4 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194337
    INFO:src.training.finetune:  Val Loss:   0.227850
    INFO:src.training.finetune:  Val AUROC:  0.9547
    INFO:src.training.finetune:  Val AUPR:   0.7508
    INFO:src.training.finetune:  No improvement for 9 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 97/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 97/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.192]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 34.93it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 97/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.7s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 4m (for 3 epochs)
    INFO:src.training.finetune:  Train Loss: 0.193974
    INFO:src.training.finetune:  Val Loss:   0.228992
    INFO:src.training.finetune:  Val AUROC:  0.9542
    INFO:src.training.finetune:  Val AUPR:   0.7492
    INFO:src.training.finetune:  No improvement for 10 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 98/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 98/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.37it/s, loss=0.195]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.89it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 98/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.8s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 3m (for 2 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194232
    INFO:src.training.finetune:  Val Loss:   0.228225
    INFO:src.training.finetune:  Val AUROC:  0.9544
    INFO:src.training.finetune:  Val AUPR:   0.7493
    INFO:src.training.finetune:  No improvement for 11 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 99/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 99/100: 100%|██████████| 1219/1219 [01:30<00:00, 13.40it/s, loss=0.2]  
    Validating: 100%|██████████| 153/153 [00:04<00:00, 35.81it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 99/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.6s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 1m (for 1 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194199
    INFO:src.training.finetune:  Val Loss:   0.229311
    INFO:src.training.finetune:  Val AUROC:  0.9541
    INFO:src.training.finetune:  Val AUPR:   0.7481
    INFO:src.training.finetune:  No improvement for 12 epochs
    INFO:src.training.finetune:======================================================================

    Epoch 100/100:   0%|          | 0/1219 [00:00<?, ?it/s]/home/mluser/projects_yash/new_project/PromptGFM-Bio/src/training/finetune.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():
    Epoch 100/100: 100%|██████████| 1219/1219 [01:31<00:00, 13.38it/s, loss=0.205]
    Validating: 100%|██████████| 153/153 [00:04<00:00, 36.27it/s]
    INFO:src.training.finetune:
    ======================================================================
    INFO:src.training.finetune:Epoch 100/100 Complete
    INFO:src.training.finetune:======================================================================
    INFO:src.training.finetune:  Time: 95.7s (Avg: 94.8s/epoch)
    INFO:src.training.finetune:  ETA: 0h 0m (for 0 epochs)
    INFO:src.training.finetune:  Train Loss: 0.194170
    INFO:src.training.finetune:  Val Loss:   0.228342
    INFO:src.training.finetune:  Val AUROC:  0.9543
    INFO:src.training.finetune:  Val AUPR:   0.7492
    INFO:src.training.finetune:  No improvement for 13 epochs
    INFO:src.training.finetune:  💾 Checkpoint saved: checkpoint_epoch_100.pt
    INFO:src.training.finetune:======================================================================

    INFO:src.training.finetune:
    Training complete!
    INFO:src.training.finetune:Best auroc: 0.9547
    INFO:src.training.finetune:Loading best model from checkpoints/promptgfm_film/best_model.pt
    INFO:src.training.finetune:Loading checkpoint: checkpoints/promptgfm_film/best_model.pt
    INFO:src.training.finetune:✓ Resumed from epoch 87
    INFO:src.training.finetune:  Best val metric: 0.9547
    INFO:src.training.finetune:  Global step: 106053
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

::: {#f3a077a4 .cell .code execution_count="25"}
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
    Fri Apr  3 17:47:07 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
    |  0%   46C    P8             19W /  450W |     120MiB /  24564MiB |      0%      Default |
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

::: {#3e0b4a72 .cell .code execution_count="26"}
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
    On branch main
    Your branch is ahead of 'origin/main' by 1 commit.
      (use "git push" to publish your local commits)

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
    	modified:   .env.example
    	modified:   .github/ISSUE_TEMPLATE/bug_report.md
    	modified:   .github/ISSUE_TEMPLATE/feature_request.md
    	modified:   .github/ISSUE_TEMPLATE/question.md
    	modified:   .github/PULL_REQUEST_TEMPLATE.md
    	modified:   .gitignore
    	modified:   CHANGELOG.md
    	modified:   CLOUD_TRAINING_GUIDE.md
    	modified:   CODE_OF_CONDUCT.md
    	modified:   CONTRIBUTING.md
    	modified:   DATA_DOWNLOAD_GUIDE.md
    	modified:   GPU_TRAINING_GUIDE.md
    	modified:   LATEST_EVALUATION_SUMMARY.md
    	modified:   LICENSE
    	modified:   PREPROCESSING_GUIDE.md
    	modified:   Project_Details.md
    	modified:   QUICKSTART.md
    	modified:   README.md
    	modified:   SECURITY.md
    	modified:   SETUP.md
    	modified:   TRAINING_GUIDE.md
    	modified:   TRAINING_OPTIMIZATION_GUIDE.md
    	modified:   TRAINING_QUICKSTART.md
    	modified:   TRAINING_RESUME_GUIDE.md
    	modified:   configs/base_config.yaml
    	modified:   configs/baseline_config.yaml
    	modified:   configs/cross_attention_config.yaml
    	modified:   configs/finetune_config.yaml
    	modified:   configs/kaggle_config.yaml
    	modified:   configs/pretrain_config.yaml
    	modified:   data/README.md
    	modified:   data/processed/.gitkeep
    	modified:   data/raw/.gitkeep
    	modified:   data/splits/.gitkeep
    	modified:   docs/ARCHITECTURE.md
    	modified:   docs/DEPLOYMENT.md
    	modified:   docs/README.md
    	modified:   docs/TROUBLESHOOTING.md
    	modified:   kaggle/kaggle_training.ipynb
    	modified:   kaggle/kaggle_training_to_use.ipynb
    	modified:   kaggle/latest_kaggle_training.ipynb
    	modified:   kaggle/notebookc3ec4f0373.ipynb
    	modified:   kaggle/notebookc6217a9ddc.ipynb
    	modified:   kaggle/now_kaggle_training.ipynb
    	modified:   kaggle/now_kaggle_training_v2.ipynb
    	modified:   kaggle/now_kaggle_training_v3.ipynb
    	modified:   kaggle/now_kaggle_training_v7.ipynb
    	modified:   notebooks/.gitkeep
    	modified:   notebooks/notebook5ed33cd62c.ipynb
    	modified:   notebooks/now_kaggle_training.ipynb
    	modified:   requirements-dev.txt
    	modified:   requirements.txt
    	modified:   scripts/.gitkeep
    	modified:   scripts/download_data.py
    	modified:   scripts/evaluate.py
    	modified:   scripts/preprocess_all.py
    	modified:   scripts/rebuild_kaggle_notebook.py
    	modified:   scripts/resume_training.py
    	modified:   scripts/test_download.py
    	modified:   scripts/test_gpu.py
    	modified:   scripts/test_negative_sampling.py
    	modified:   scripts/test_optimizations.py
    	modified:   scripts/test_preprocess.py
    	modified:   scripts/train.py
    	modified:   setup.py
    	modified:   setup_environment.ps1
    	modified:   src/__init__.py
    	modified:   src/data/__init__.py
    	modified:   src/data/dataset.py
    	modified:   src/data/download.py
    	modified:   src/data/hpo_bridge.py
    	modified:   src/data/orphadata.py
    	modified:   src/data/preprocess.py
    	modified:   src/data/uniprot_pathways.py
    	modified:   src/evaluation/__init__.py
    	modified:   src/evaluation/case_study.py
    	modified:   src/evaluation/metrics.py
    	modified:   src/models/__init__.py
    	modified:   src/models/conditioning.py
    	modified:   src/models/gnn_backbone.py
    	modified:   src/models/prompt_encoder.py
    	modified:   src/models/promptgfm.py
    	modified:   src/training/__init__.py
    	modified:   src/training/finetune.py
    	modified:   src/training/losses.py
    	modified:   src/training/pretrain.py
    	modified:   src/utils/__init__.py
    	modified:   src/utils/config.py
    	modified:   src/utils/logger.py
    	modified:   tests/test_models.py

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
    	Collab/
    	NOTEBOOK_RERUN_GUIDE.md
    	REUPLOAD_CHANGED_FILES.txt
    	add_string_ppi_edges.py
    	change-log/
    	claude_context/
    	configs/workstation_config.yaml
    	docs/Node-to-Self_DLG_OnePager.pdf
    	docs/Node_To_Self_GraphDL.pdf
    	docs/Node_To_Self_GraphDL.pptx
    	docs/PromptGFM_Publication_Strategy.md
    	hf_cache/
    	kaggle/now_kaggle_training_v4.ipynb
    	kaggle/now_kaggle_training_v5.ipynb
    	kaggle/now_kaggle_training_v8.ipynb
    	kaggle/promptgfm_kaggle_v2.ipynb
    	notebooks/New_workstation_training_v7_fixed_v5.5.ipynb
    	notebooks/New_workstation_training_v7_fixed_v5.5.md
    	notebooks/New_workstation_training_v7_fixed_v6.ipynb
    	notebooks/workstation_training_v7_fixed_v5_Clean_Completed.ipynb
    	scripts/evaluate_Old.py
    	workstation_training_v7_fixed_v5_Clean_Completed.ipynb

    no changes added to commit (use "git add" and/or "git commit -a")
    ❌ Git push failed — check GITHUB_TOKEN in .env

    For large files, consider HuggingFace Hub:
      huggingface-cli upload your-username/promptgfm-bio /home/mluser/projects_yash/new_project/PromptGFM-Bio/hf_cache --repo-type model
:::

::: {.output .stream .stderr}
    fatal: could not read Password for 'https://[REDACTED_TOKEN]@github.com': terminal prompts disabled
:::
:::

::: {#e03b8aa3 .cell .markdown}
## 14. Quick Evaluation {#14-quick-evaluation}
:::

::: {#84659f92 .cell .code execution_count="27"}
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

::: {.output .execute_result execution_count="27"}
    '\nimport subprocess, sys\n\nws_cfg = CONFIGS_DIR / "workstation_config.yaml"\nconfig = str(ws_cfg) if ws_cfg.exists() else str(CONFIGS_DIR / "kaggle_config.yaml")\n\nbest = CKPT_DIR / "best_model.pt"\nif not best.exists():\n    print("No best_model.pt yet — run more training epochs first")\nelse:\n    result = subprocess.run(\n        [sys.executable, str(SCRIPTS_DIR / "evaluate.py"),\n         "--config", config,\n         "--checkpoint", str(best)],\n        cwd=str(PROJECT_ROOT),\n    )\n    print("Evaluation exit code:", result.returncode)\n'
:::
:::

::: {#21e4a58b-8c03-4a27-b84c-2e57a9b119bf .cell .code execution_count="33"}
``` python
import subprocess, sys

# 🔴 FORCE correct config (no auto-switching)
#config = str(CONFIGS_DIR / "workstation_config.yaml")

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
    INFO:__main__:Device: cuda
    INFO:__main__:Building PromptGFM for evaluation
    INFO:__main__:  Model params: gnn_input=128, gnn_hidden=512, predictor_hidden=256
    INFO:src.models.promptgfm:Initializing PromptGFM model...
    INFO:src.models.prompt_encoder:Initializing PromptEncoder with model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Loading weights: 100%|██████████| 199/199 [00:00<00:00, 88530.60it/s]
    BertModel LOAD REPORT from: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Key                                        | Status     |  | 
    -------------------------------------------+------------+--+-
    cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
    cls.predictions.bias                       | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
    cls.seq_relationship.weight                | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
    cls.predictions.decoder.weight             | UNEXPECTED |  | 
    cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
    cls.seq_relationship.bias                  | UNEXPECTED |  | 
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
    INFO:__main__:Loading checkpoint from /home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film/best_model.pt
    INFO:__main__:✓ Model loaded and set to eval mode
    INFO:src.data.dataset:Loading graph from data/processed/biomedical_graph.pt
    INFO:src.data.dataset:Graph loaded: gene=19576, disease=16841, phenotype=11794, ('gene', 'interacts', 'gene')=1854012, ('gene', 'associated_with', 'disease')=9741610, ('disease', 'rev_associated_with', 'gene')=9741610
    INFO:src.data.dataset:Loading gene-disease edges from data/processed/hpo_gene_disease_edges.csv
    INFO:src.data.dataset:Vocabulary: 5251 genes, 12714 diseases
    INFO:src.data.dataset:Loaded 1170143 edges ({'HPO_phenotype_bridge': 1170143})
    INFO:src.data.dataset:Split sizes: train=936114, val=117014, test=117015
    INFO:__main__:  Train: 936114 edges, Val: 117014 edges, Test: 117015 edges
    INFO:__main__:
    ============================================================
    INFO:__main__:Evaluating TEST split
    INFO:__main__:============================================================
    INFO:__main__:  Ranking 19576 genes for each of 10267 disease queries
    Evaluating queries: 100%|██████████| 10267/10267 [6:40:01<00:00,  2.34s/it] 
    INFO:__main__:  Evaluated 10267 queries with at least one test positive
    INFO:src.evaluation.metrics:GeneRankingEvaluator initialized with K values: [10, 20, 50, 100]
    INFO:src.evaluation.metrics:GeneRankingEvaluator initialized with K values: [10, 20, 50, 100]
    INFO:src.evaluation.metrics:
    test:
    INFO:src.evaluation.metrics:  AUROC: 0.9626
    INFO:src.evaluation.metrics:  AUPR:  0.0188
    INFO:src.evaluation.metrics:  Precision@10: 0.0557
    INFO:src.evaluation.metrics:  Recall@10: 0.0695
    INFO:src.evaluation.metrics:  NDCG@10: 0.0795
    INFO:src.evaluation.metrics:  Hit Rate@10: 0.3218
    INFO:src.evaluation.metrics:  Precision@20: 0.0459
    INFO:src.evaluation.metrics:  Recall@20: 0.1090
    INFO:src.evaluation.metrics:  NDCG@20: 0.0884
    INFO:src.evaluation.metrics:  Hit Rate@20: 0.4175
    INFO:src.evaluation.metrics:  Precision@50: 0.0358
    INFO:src.evaluation.metrics:  Recall@50: 0.1941
    INFO:src.evaluation.metrics:  NDCG@50: 0.1137
    INFO:src.evaluation.metrics:  Hit Rate@50: 0.5573
    INFO:src.evaluation.metrics:  Precision@100: 0.0279
    INFO:src.evaluation.metrics:  Recall@100: 0.2801
    INFO:src.evaluation.metrics:  NDCG@100: 0.1384
    INFO:src.evaluation.metrics:  Hit Rate@100: 0.6537
    INFO:src.evaluation.metrics:  MAP:   0.0569
    INFO:src.evaluation.metrics:  MRR:   0.1588
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

::: {#7a982682 .cell .code execution_count="29"}
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
    overlay         492G  165G  302G  36% /

    421M	/home/mluser/projects_yash/new_project/PromptGFM-Bio/hf_cache
    3.7G	/home/mluser/projects_yash/new_project/PromptGFM-Bio/data
    4.0K	/home/mluser/projects_yash/new_project/PromptGFM-Bio/checkpoints/promptgfm_film
    4.0K	/home/mluser/projects_yash/new_project/PromptGFM-Bio/logs
:::
:::

::: {#7a3da1a5 .cell .markdown}
## 16. 🔧 Manual Override (Optional) {#16--manual-override-optional}

If auto-detected batch_size isn\'t right, uncomment and edit below, then
re-run Step 12.
:::

::: {#807b492d .cell .code execution_count="30"}
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
