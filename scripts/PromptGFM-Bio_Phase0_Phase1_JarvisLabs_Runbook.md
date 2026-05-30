# PromptGFM-Bio — Phase 0 & Phase 1 on JarvisLabs (Path 2 edition)

**Standalone runbook.** You are on **Windows 11**. Project at
`E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio`.
GitHub user `pes1ug23am910`. Cloud repo
`https://github.com/pes1ug23am910/PromptGFM-Bio-JarvisLabs.git`.

**Scope:** Phase 0 (environment bring-up + smoke gate on an A100 Template) and
Phase 1 (external baselines on an L4 VM). The 40-run study itself is Phase 2 and
is a separate runbook.

> **Path 2 reconciliation — read this first.** Your older `Phase0_Phase1_Guide`
> defined Phase 0 as a *parity check* whose only job was to justify pooling the 3
> old workstation seeds (42/43/44) with new ones. **Path 2 discards the old runs
> and trains all 10 seeds fresh on the A100**, so there is nothing to "match" —
> the parity check is obsolete (the Path 2 deployment guide explicitly supersedes
> it). Under Path 2, **Phase 0 = bring the A100 up and pass the smoke test**
> (which validates the environment *and* same-seed determinism). **Do not run
> `parity_check-jarvis.sh`** — it would cost ~₹126 and answer a question Path 2
> no longer asks. **Phase 1 (baselines) is unchanged and still mandatory** — it's
> the single biggest acceptance risk for a top venue.

---

## 0. Prerequisites (do these once, before any instance)

1. **Apply the two pre-launch patches** from `PATH2_PATCHES_v2.md` to your local
   `scripts\train.py` and `scripts\evaluate.py` (worker_init_fn + per-disease
   dump). Run the two `Select-String` verifies in that doc. These must be in the
   code **before** you push, so the instance clones the patched version.
2. **Place the new files** in your project (then they ride along in the push):
   - `scripts\smoke_test_path2_v2-jarvis.sh`
   - `scripts\aggregate_results.py`
   - `scripts\export_promptgfm_rankings.py`
   - `requirements.txt`  → **project root** (replaces the old one)
3. **JarvisLabs → Settings → add your SSH public key** now. The VM (Phase 1)
   refuses to launch without it; doing it now covers both phases.
4. **Recharge.** You have ₹1,000. Phase 0 + the start of Phase 1 fit in that.
   Recharge to **~₹5,000–7,000 before Phase 2** (the 40 runs). Not needed yet.

### Generate an SSH key on Windows (if you don't have one)
```powershell
ssh-keygen -t ed25519 -C "pes1ug23am910@jarvislabs"   # press Enter through prompts
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub       # paste this into JarvisLabs Settings
```

---

## 1. One-time Windows prep & push to GitHub

Open **PowerShell** in the project folder:

```powershell
cd E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio
```

### 1a. Make the repo private OR guarantee no secrets/data are committed

> ⚠️ **Your repo is currently public.** That is acceptable **only** if you never
> commit secrets or data. The safe move for a research repo is to make it
> **private**: GitHub → repo → Settings → General → Danger Zone → *Change
> visibility → Private*. Either way, the hard rules below are non-negotiable.

### 1b. `.gitignore` — keep secrets and big binaries out of git

Merge `scripts\gitignore_additions.txt` into `.gitignore`, then **prove** the
sensitive paths are ignored before any push:

```powershell
git status                                   # .env, *.pt, data\raw must NOT be staged
git check-ignore .env data\processed\biomedical_graph.pt
# both should be echoed back (= ignored). If .env is NOT echoed, STOP and fix .gitignore.
```

Never commit: `.env`, API keys, `data\processed\*.pt`, `data\raw\` archives,
`checkpoints\`, large media. Git carries **code only**; data goes up separately
(Section 2c / 3c).

### 1c. Push to your JarvisLabs repo

```powershell
# If this folder is not yet a git repo:
git init
git branch -M main

# Point at your cloud repo (use a dedicated remote name so it won't clash with
# any existing 'origin'):
git remote add jarvis https://github.com/pes1ug23am910/PromptGFM-Bio-JarvisLabs.git

git add .
git commit -m "Path 2 cloud prep: patches applied, smoke v2, aggregator, export, requirements"
git push -u jarvis main
```

On GitHub, confirm: `src/`, `scripts/`, `configs/`, `requirements.txt` are
present; `.env`, `data/raw/`, and `*.pt` are **absent**.

---

## 2. PHASE 0 — A100 bring-up + smoke gate

**Goal:** stand up the exact training environment on the A100 and prove it is
correctly wired and deterministic. Cost: a few rupees of bring-up + **~₹17** for
the smoke test. This same instance is what you'll later use for the 40 runs.

### 2a. Launch the instance (maps to your "PYTORCH" template screen)

JarvisLabs → **Templates → PyTorch**:
- **Step 1 · Pick GPU:** select the **A100 · AMPERE · ₹84.24/hr · 40 GB · IN2**
  tile. (Not RTX PRO 6000 — Blackwell, wheel-risk; not H100/H200 — overkill; not
  A100-80GB — you won't OOM at batch 768.)
- **Step 2 · Configure:** 1 GPU.
- **Step 3 · Pricing:** **On-Demand** (you do not want the smoke interrupted).
- **Step 4 · Storage:** **100 GB** (`/home` is persistent; `/root` is wiped on
  pause — keep everything under `/home`).
- Name it (e.g. `promptgfm-a100`), **Launch instance**. Open **JupyterLab** from
  the instance card.

### 2b. Bootstrap (JupyterLab → Terminal)

```bash
cd /home
git clone https://github.com/pes1ug23am910/PromptGFM-Bio-JarvisLabs.git promptgfm-bio
cd /home/promptgfm-bio        # the *-jarvis.sh scripts default to exactly this path

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# PyG companion wheels — install from the CUDA-matched index (NOT source build):
CUDA=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
echo "instance CUDA tag: ${CUDA}"
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html

export WANDB_MODE=offline        # so training never stalls on a wandb login

python scripts/test_gpu.py       # expect: CUDA available, GPU = A100
python scripts/verify_setup.py   # expect torch 2.1.0 / PyG 2.4.0 / transformers 4.35.0
```

> If `verify_setup.py` flags a version mismatch, fix it before spending on the
> smoke test. If the PyG wheel index 404s for your `${CUDA}` tag, try `cu118`
> (the other common A100 build) — match whatever `torch.version.cuda` printed.

### 2c. Upload the data (gitignored, so transferred separately)

**Easiest on Windows — JupyterLab drag-drop.** In the JupyterLab file browser,
go to `/home/promptgfm-bio/data/processed/` and drop:
- `biomedical_graph.pt`
- `hpo_gene_disease_edges.csv`

`data/splits/zero_shot_rare_diseases.json` is small and already came via git
(confirm it's there; if your `.gitignore` excluded it, upload it to
`/home/promptgfm-bio/data/splits/`).

**Or via scp** (HOST/PORT from the instance's SSH panel), from PowerShell:
```powershell
$H="<HOST>"; $P="<PORT>"; $U="<user>"   # from JarvisLabs SSH panel
$base="E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio"
scp -P $P "$base\data\processed\biomedical_graph.pt"        ${U}@${H}:/home/promptgfm-bio/data/processed/
scp -P $P "$base\data\processed\hpo_gene_disease_edges.csv" ${U}@${H}:/home/promptgfm-bio/data/processed/
```

### 2d. Run the smoke gate (the whole point of Phase 0 under Path 2)

```bash
cd /home/promptgfm-bio && source .venv/bin/activate && export WANDB_MODE=offline
mkdir -p logs
chmod +x scripts/*jarvis*.sh
bash scripts/smoke_test_path2_v2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log
```

This trains `ablation_4_full_model` for 3 epochs at seed 42, seed 43, and seed
42 **again**, then checks five invariants. **Require the final line:**

```
>>> SMOKE PASS — Path 2 is wired correctly AND deterministic. Safe to launch run_all_seeds-jarvis.sh.
```

- If invariant **(E)** fails → the worker_init_fn patch (PATH2_PATCHES_v2.md,
  Patch A) wasn't applied; apply it, re-push, re-clone or `git pull`, re-run.
- If **(A)–(D)** fail → the base Path 2 patches (`PATH2_PATCHES.md`) aren't in
  `train.py`/`evaluate.py`; fix before spending another rupee.

### 2e. Done with Phase 0

- **Do NOT** run `parity_check-jarvis.sh` (obsolete — see the reconciliation
  note up top).
- If you're going straight into Phase 2, leave it running. Otherwise **Pause**
  the instance (not Destroy) — `/home` persists, you pay only ~₹1.13/hr storage.

**Phase 0 cost:** ~₹17 smoke + a little bring-up/idle ≈ **under ₹100.**

---

## 3. PHASE 1 — External baselines (L4 VM)

**Goal:** SHEPHERD, Phrank, LIRICAL, PubMedBERT-cosine, LLM-direct, and your own
model, all scored on the 117 zero-shot diseases in one uniform table
(Hit@10 / Hit@50 / MRR). This runs on a **VM + L4** because SHEPHERD needs root
for its conda env and LIRICAL needs Java; the workload is inference, so a cheap
GPU is right.

> **Fairness design (state this in the paper):** every method receives the
> **identical HPO term list** per disease — the same Orphanet/HPO terms your
> `PromptEncoder.create_prompt()` consumes — so there is no LLM
> phenotype-extraction confound. `prepare_baseline_inputs.py` builds these shared
> inputs once.

### 3a. Launch the VM (maps to your "Launch VM" screen)

JarvisLabs → **VM** → Select Resource: **GPU** → **1 × L4 · ₹41.31/hr · IN2**
(24 GB). Storage **100 GB**. Pricing **On-Demand**. Your SSH key must already be
in Settings (Section 0.3) — otherwise the red *"Add an SSH key before launching"*
blocker appears. **Launch VM**, then SSH in from PowerShell using the VM's
HOST/PORT:

```powershell
ssh -p <PORT> <user>@<HOST>
```

### 3b. Two isolated environments (both under persistent `/home`)

SHEPHERD targets an older torch/PyG stack and **will conflict** with your torch
2.1 venv — keep them separate.

```bash
# (i) YOUR code env — prep scripts, PubMedBERT baseline, your-model export
cd /home
git clone https://github.com/pes1ug23am910/PromptGFM-Bio-JarvisLabs.git promptgfm-bio || true
cd /home/promptgfm-bio
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install mygene          # SYMBOL -> Ensembl mapping (SHEPHERD needs Ensembl)
```

### 3c. Upload the raw phenotype inputs

Via JupyterLab (open it on the VM) drag-drop or scp, into the matching folders:
- `data/raw/hpo/phenotype.hpoa`
- `data/raw/hpo/phenotype_to_genes.txt`
- `data/raw/hpo/genes_to_phenotype.txt`
- `data/raw/orphanet/en_product4.xml`
- `data/processed/biomedical_graph.pt`, `data/processed/hpo_gene_disease_edges.csv`
- `data/splits/zero_shot_rare_diseases.json`

### 3d. Build the shared baseline inputs (once) — then STOP and read the report

```bash
python scripts/prepare_baseline_inputs.py
cat data/baselines/prep_report.txt
```

This writes `data/baselines/`: `disease_hpo_terms.json`,
`disease_true_genes.json`, `symbol_to_ensembl.json`,
`all_candidate_genes_{symbols,ensembl}.json`, and `prep_report.txt`.

> **Gate:** in `prep_report.txt`, *"with ≥1 HPO term"* must be **~117**. If many
> OMIM diseases are empty, your `phenotype.hpoa` column names differ from what
> the script auto-detected — open the header (`head -5 data/raw/hpo/phenotype.hpoa`)
> and adjust, then re-run. **Do not proceed with empty phenotype lists** — it
> would unfairly cripple the baselines and invalidate the comparison.

### 3e. SHEPHERD (the headline baseline — budget one focused day)

SHEPHERD uses a **pretrained model on its own KG; you do NOT retrain it.**

```bash
# miniconda (VM has root)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /home/miniconda3
source /home/miniconda3/bin/activate

cd /home
git clone https://github.com/mims-harvard/SHEPHERD.git
cd SHEPHERD
conda env create -f environment.yml
conda activate shepherd
bash install_pyg.sh          # the fragile step; if it fails, open a repo issue
```

Download their **data + pretrained checkpoint** from Harvard Dataverse
(`doi:10.7910/DVN/TZTPFL`, linked in their README) into the path their README
specifies, preserving directory structure.

Build the SHEPHERD input (in **your** venv, not the shepherd env):
```bash
source /home/promptgfm-bio/.venv/bin/activate
cd /home/promptgfm-bio
python scripts/make_shepherd_input.py     # -> data/baselines/shepherd_patients.jsonl
```

Run SHEPHERD's **causal-gene-discovery** entry point (in the `shepherd` env) with
`shepherd_patients.jsonl` + the pretrained checkpoint. **Copy the exact command
from their README's "causal gene discovery" section** — entry-point names change
between releases, so don't trust a hard-coded one. Capture per-disease ranked
Ensembl genes, then map Ensembl→symbol (invert `symbol_to_ensembl.json`) into
`data/baselines/rankings/shepherd_symbols.json` shaped `{disease_id: [SYMBOLS]}`.

> Pause the VM overnight between sessions (you keep `/home`, pay storage only).

### 3f. The other baselines → `data/baselines/rankings/<name>.json`

Each writes `{disease_id: [ranked SYMBOLS, best_first]}`:

- **Phrank** (`bitbucket.org/bejerano/phrank`, pure-Python, no GPU, ~1 day): give
  it the HPO DAG + `phenotype_to_genes.txt`; rank genes by phenotype match →
  `rankings/phrank.json`.
- **LIRICAL** (Java, CPU, ~1 day):
  ```bash
  sudo apt-get update && sudo apt-get install -y default-jdk
  # download the LIRICAL release + its data per its docs; run phenotype-only mode per disease
  ```
  → `rankings/lirical.json`.
- **PubMedBERT-cosine** (your frozen `PromptEncoder`, ~hours): embed each disease
  prompt + each candidate gene description, rank by cosine →
  `rankings/pubmedbert_cosine.json`. This is the static strawman your FiLM
  conditioning is designed to beat.
- **LLM-direct** (Kim et al. protocol, API, ~$ few): prompt GPT-4/Claude for
  likely causal genes per disease, parse to a ranked list →
  `rankings/llm_direct.json`. Set the API key as an **environment variable**
  (`export OPENAI_API_KEY=...`) — **never commit it.** Expect ~17% top-50.

### 3g. Export YOUR model into the same format (new script)

```bash
source /home/promptgfm-bio/.venv/bin/activate && cd /home/promptgfm-bio
python scripts/export_promptgfm_rankings.py \
    --config configs/ablations/ablation_4_full_model.yaml \
    --checkpoint results/ablation_4_full_model_seed42/best_model.pt \
    --zero_shot data/splits/zero_shot_rare_diseases.json \
    --out data/baselines/rankings/promptgfm.json
```

(Use a real seed-42 full-model checkpoint — upload one from your Phase 2 results,
or from the workstation, into `results/ablation_4_full_model_seed42/`.) This
reuses `evaluate.py`'s exact model + dataset load, so the ranking matches your
evaluation runs.

### 3h. Produce the comparison table

```bash
python scripts/score_baselines.py \
  --truth  data/baselines/disease_true_genes.json \
  --method promptgfm=data/baselines/rankings/promptgfm.json \
  --method shepherd=data/baselines/rankings/shepherd_symbols.json \
  --method phrank=data/baselines/rankings/phrank.json \
  --method lirical=data/baselines/rankings/lirical.json \
  --method pubmedbert=data/baselines/rankings/pubmedbert_cosine.json \
  --method llm_direct=data/baselines/rankings/llm_direct.json \
  --out data/baselines/comparison_table.md
```

Reports **Hit@10 / Hit@50 / MRR + coverage**, identical definitions across all
methods (the fair cross-method currency — keep AUROC for your internal ablation
table only). `comparison_table.md` is a centerpiece of the paper.

### 3i. Pause + pull results back

```bash
# on the VM: pause it from the JarvisLabs console when idle
```
```powershell
# on your laptop:
scp -P <PORT> -r <user>@<HOST>:/home/promptgfm-bio/data/baselines ".\data\"
```

**Phase 1 cost:** L4 active time, paused between sessions ≈ **~₹900** + a few $ of
LLM API.

---

## 4. What you must NOT change (both phases)

- `configs/ablations/*.yaml` — **frozen**. `batch_size: 768`, FP16 AMP
  (`mixed_precision: true`), 100 epochs, `data.random_seed: 42`,
  `deterministic:false`, `benchmark:true`. Phase 0's smoke must run the same
  config the 40 runs will.
- Do **not** enable TF32 or switch FP16→BF16, and do **not** apply the TopVenue
  plan's speed-ups (`torch.compile`, batch 1536, FlashAttention-2) — they change
  numerics; all 40 runs must be identical except the seed.
- Phase 1 trains nothing of yours — the only "optimization" is the cheap L4 and
  aggressive pausing.

---

## 5. Cost & time summary

| Phase | Product / GPU | Pricing | ~Active hrs | ~Cost |
|---|---|---|---|---|
| 0 — bring-up + smoke | A100 40GB Template | On-Demand | ~0.5 | **< ₹100** |
| 1 — baselines | L4 24GB VM (paused between sessions) | On-Demand | ~22 | **~₹900** |
| **Phase 0 + 1 total** | | | | **~₹1,000** |
| *(Phase 2 — 40 runs, separate)* | A100 40GB Template | Spot if available | ~53 | *~₹2,300 spot / ~₹4,500 on-demand* |

Your ₹1,000 covers Phase 0 + the start of Phase 1. **Recharge ~₹5,000–7,000
before Phase 2.**

---

## 6. Security checklist (because the repo is public)

- [ ] `.env` is gitignored and **not** on GitHub (`git check-ignore .env` echoes it).
- [ ] No API keys in any committed file; LLM keys live only as VM env vars.
- [ ] `*.pt`, `data/raw/`, `checkpoints/` are absent from the repo.
- [ ] Ideally, repo set to **Private** (Settings → Danger Zone).

---

## 7. Two spots that need on-VM confirmation

1. **`phenotype.hpoa` column names** — `prepare_baseline_inputs.py` auto-detects
   them, but HPO releases vary. `prep_report.txt`'s coverage number is your check.
2. **SHEPHERD's exact run command** — copy it from the current README's
   causal-gene-discovery section (entry-point names change between releases).

When you're on the VM, send `head -5 data/raw/hpo/phenotype.hpoa` and the
SHEPHERD README's run section, and I'll pin those two exactly.

---

## 8. Cheat-sheet

```bash
# ===== PHASE 0 (A100 Template, On-Demand) =====
cd /home && git clone https://github.com/pes1ug23am910/PromptGFM-Bio-JarvisLabs.git promptgfm-bio
cd /home/promptgfm-bio && python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
CUDA=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
export WANDB_MODE=offline
python scripts/test_gpu.py && python scripts/verify_setup.py
#  -> upload data/processed/{biomedical_graph.pt,hpo_gene_disease_edges.csv}
chmod +x scripts/*jarvis*.sh
bash scripts/smoke_test_path2_v2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log
#  require ">>> SMOKE PASS"; then PAUSE (do NOT run parity_check)

# ===== PHASE 1 (L4 VM, root) =====
cd /home/promptgfm-bio && source .venv/bin/activate && pip install mygene
python scripts/prepare_baseline_inputs.py && cat data/baselines/prep_report.txt   # ~117 with HPO
python scripts/make_shepherd_input.py
#  ... conda SHEPHERD env + Dataverse data + causal-gene-discovery -> rankings/shepherd_symbols.json
#  ... phrank / lirical / pubmedbert_cosine / llm_direct -> rankings/*.json
python scripts/export_promptgfm_rankings.py \
  --config configs/ablations/ablation_4_full_model.yaml \
  --checkpoint results/ablation_4_full_model_seed42/best_model.pt \
  --out data/baselines/rankings/promptgfm.json
python scripts/score_baselines.py --truth data/baselines/disease_true_genes.json \
  --method promptgfm=data/baselines/rankings/promptgfm.json \
  --method shepherd=data/baselines/rankings/shepherd_symbols.json \
  --method phrank=data/baselines/rankings/phrank.json \
  --method pubmedbert=data/baselines/rankings/pubmedbert_cosine.json \
  --out data/baselines/comparison_table.md
#  PAUSE VM; scp data/baselines back to laptop
```

*GPU prices reflect your JarvisLabs console (A100 40GB IN2 ₹84.24/hr; L4 IN2
₹41.31/hr). The parity check from the pre-Path-2 guide is intentionally omitted.*
