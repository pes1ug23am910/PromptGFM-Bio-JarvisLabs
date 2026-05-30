# PromptGFM-Bio — Phase 0 & Phase 1: Standalone Execution Guide

**You are on Windows.** Project at `E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio`.
**This guide covers only:** Phase 0 (A100 parity check) and Phase 1 (external baselines on an L4 VM).
**Prereq:** the 12 ablations are done; 10-seed work (Phase 2) comes later.

---

## 0. The three decisions, answered

**GitHub?** Yes — push **code only** to a **private** repo. Never push `.env`, the large data binaries (`biomedical_graph.pt`, BIOGRID/STRING archives, `docs/*.mp4`/`*.pptx`), or `checkpoints/`. Git is how code reaches the instance; heavy data goes separately (JupyterLab upload or `scp`).

**What to upload — Phase 0 (parity, A100 Template):**
| Item | Path | How |
|---|---|---|
| All code | `src/`, `scripts/`, `configs/`, `requirements.txt` | git clone |
| Processed graph | `data/processed/biomedical_graph.pt` | scp / Jupyter upload |
| Edge file | `data/processed/hpo_gene_disease_edges.csv` | scp / Jupyter upload |
| Zero-shot set | `data/splits/zero_shot_rare_diseases.json` | git (small) |
| Reference result | `results/ablation_4_full_model_seed42/evaluation_results.json` | scp / Jupyter upload |
| 4 jarvis scripts | `scripts/*-jarvis.sh`, `scripts/_jarvis_env.sh` | git (add them first) |

**What to upload — Phase 1 (baselines, L4 VM):** everything from Phase 0 **plus** the raw phenotype files (used to build HPO inputs): `data/raw/hpo/phenotype.hpoa`, `data/raw/hpo/phenotype_to_genes.txt`, `data/raw/hpo/genes_to_phenotype.txt`, `data/raw/orphanet/en_product4.xml`. SHEPHERD's own code + model are downloaded *on the VM*, not uploaded.

**What to update/optimize?** For these two phases, deliberately almost nothing performance-related:
- **Update** `.gitignore` (merge `gitignore_additions.txt`).
- **Add** the four `*-jarvis.sh` scripts and the three baseline scripts to `scripts/`.
- **Never touch** `configs/ablations/*.yaml` — Phase 0's entire purpose is bit-identical reproduction; Phase 1 doesn't train your model. There is no safe accuracy-preserving "speedup" to make here, and that's correct.
- The only environment tweak you may need on the instance is installing the PyG companion wheels (`torch-scatter/sparse/cluster`) matching the instance CUDA — that's a setup step, not a code edit.

---

## 1. One-time prep on your Windows machine

Open **PowerShell** in the project folder:

```powershell
cd E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio
```

### 1a. Add the new scripts

Copy these into `scripts\` (from the files provided alongside this guide):
`_jarvis_env.sh`, `parity_check-jarvis.sh`, `run_ablations_extra_seeds-jarvis.sh`, `run_evaluations_extra_seeds-jarvis.sh`, `prepare_baseline_inputs.py`, `make_shepherd_input.py`, `score_baselines.py`.

### 1b. Fix `.gitignore`, then verify secrets are excluded

Merge the lines from `gitignore_additions.txt` into your existing `.gitignore`. Then **prove `.env` and the graph are not staged** before any push:

```powershell
git status            # .env, *.pt, data/raw must NOT appear under "to be committed"
git check-ignore .env data\processed\biomedical_graph.pt
# both paths should be echoed back = they are ignored. If .env is NOT ignored, STOP.
```

### 1c. Push code to a private GitHub repo

```powershell
# if not already a git repo:
git init
git branch -M main

# create an EMPTY private repo on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/PromptGFM-Bio.git   # skip if remote exists
git add .
git commit -m "Cloud run prep: jarvis scripts, baseline scripts, gitignore"
git push -u origin main
```

Confirm on GitHub that the repo is **Private** and that `.env`, `data/raw/`, and `*.pt` are absent.

---

## 2. PHASE 0 — Parity check (A100, ~1.5 hr, ~₹126)

**Purpose:** prove the A100 reproduces your workstation seed-42 numbers within tolerance, so seeds 45–51 (Phase 2) can be pooled with 42/43/44. Cheapest insurance in the project.

### 2a. Launch the instance

JarvisLabs console → **Templates → PyTorch**. Step 1 pick GPU: **A100 (40GB) — ₹84.24/hr, IN2**. Step 3 pricing: **On-Demand** (you do NOT want this interrupted). Step 4 storage: **100 GB** default. Add an instance name, **Launch**.

> Before launching anything, JarvisLabs → Settings → **add your SSH public key** (the VM screen errors without it; do it now so Phase 1 is ready too).

### 2b. Connect and bootstrap (in the instance terminal / JupyterLab terminal)

```bash
cd /home
git clone https://github.com/<your-username>/PromptGFM-Bio.git promptgfm-bio
cd /home/promptgfm-bio          # the jarvis scripts default to exactly this path

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If torch-scatter / torch-sparse / torch-cluster fail to build, install the
# prebuilt wheels matching torch 2.1.0 + the instance CUDA (check: python -c "import torch;print(torch.version.cuda)")
# e.g. for CUDA 12.1:
# pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

python scripts/test_gpu.py        # expect: CUDA available, GPU = A100
python scripts/verify_setup.py    # expect torch 2.1.0 / PyG 2.4.0 / transformers 4.35.0
```

### 2c. Upload the graph + reference result

The graph and reference JSON are gitignored, so transfer them. **Easiest on Windows: JupyterLab.** In the instance's JupyterLab file browser, navigate to `/home/promptgfm-bio/data/processed/` and drag-drop `biomedical_graph.pt` and `hpo_gene_disease_edges.csv`; create `results/ablation_4_full_model_seed42/` and upload `evaluation_results.json` there.

**Or via scp from a PowerShell window on your laptop** (HOST/PORT from the JarvisLabs instance "SSH" panel):

```powershell
scp -P <PORT> "E:\...\PromptGFM-Bio\data\processed\biomedical_graph.pt" `
    <user>@<HOST>:/home/promptgfm-bio/data/processed/
scp -P <PORT> "E:\...\PromptGFM-Bio\data\processed\hpo_gene_disease_edges.csv" `
    <user>@<HOST>:/home/promptgfm-bio/data/processed/
# reference result for the parity comparison:
scp -P <PORT> "E:\...\PromptGFM-Bio\results\ablation_4_full_model_seed42\evaluation_results.json" `
    <user>@<HOST>:/home/promptgfm-bio/results/ablation_4_full_model_seed42/
```

### 2d. Run the parity check

```bash
cd /home/promptgfm-bio && source .venv/bin/activate
chmod +x scripts/*jarvis*.sh
bash scripts/parity_check-jarvis.sh 2>&1 | tee logs/parity_$(date +%Y%m%d).log
```

It retrains `ablation_4_full_model` seed 42 into a **separate** `checkpoints/parity_…` dir (your real seed-42 is untouched), evaluates, and prints a PASS/FAIL table at 0.005 tolerance.

### 2e. Interpret + clean up

- **PASS** (worst delta ≤ 0.005, normally 0.001–0.003): the A100 matches the workstation. You're cleared to pool all 10 seeds in Phase 2. Delete the parity artifacts: `rm -rf checkpoints/parity_* results/parity_*`.
- **FAIL** (delta ≥ 0.01): work the checklist the script prints — torch/PyG/transformers versions vs `requirements.txt`, TF32 disabled, AMP still FP16, `deterministic:false`/`benchmark:true`. Re-run. If it still won't match, report the 7 new seeds as a separate line and keep 42/43/44 as-is.

**Then PAUSE the instance** (don't destroy) if you're not going straight into Phase 2 — `/home` persists and you pay only ~₹1.13/hr storage.

**Phase 0 cost:** ~1.5 hr × ₹84.24 ≈ **₹126**.

---

## 3. PHASE 1 — External baselines (L4 VM, ~₹900 active)

This is the gap every Q1 reviewer checks first. Goal: SHEPHERD, Phrank, LIRICAL, PubMedBERT-cosine, and an LLM-direct baseline, all scored on your 117 zero-shot diseases with one uniform table. Runs on a **VM + L4** (root access for conda/Java; inference workload — cheap GPU).

### 3.0 The fairness design (state this in the paper)

Your `PromptEncoder.create_prompt()` already builds prompts from a disease's **HPO phenotype list**. So the phenotype terms your model sees are derivable from the *same* Orphanet/HPO files the baselines need. That means you can run the **fair Option-A comparison** — every method receives the identical HPO term list — with no LLM phenotype-extraction step. Cleaner and stronger than the original plan assumed.

### 3.1 Launch the VM

JarvisLabs → **VM**. Select Resource: GPU → **L4 — ₹41.31/hr, IN2**. 100 GB storage. Pricing: On-Demand. Ensure your SSH key is added (Settings). **Launch VM.**

### 3.2 Bootstrap two isolated environments

SHEPHERD targets an older PyTorch/PyG stack and **will conflict with your torch 2.1 venv**. Keep them separate; both live under persistent `/home`.

```bash
# Your code env (for the prep scripts + PubMedBERT baseline)
cd /home
git clone https://github.com/<your-username>/PromptGFM-Bio.git promgfm-bio || true
cd /home/promgfm-bio    # (any path is fine on the VM; prep scripts use relative paths)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install mygene      # for symbol -> Ensembl mapping (SHEPHERD needs Ensembl)
```

Upload the raw phenotype files the prep script reads (JupyterLab drag-drop or scp): `data/raw/hpo/phenotype.hpoa`, `phenotype_to_genes.txt`, `genes_to_phenotype.txt`, `data/raw/orphanet/en_product4.xml`, plus `data/processed/biomedical_graph.pt`, `data/processed/hpo_gene_disease_edges.csv`, and `data/splits/zero_shot_rare_diseases.json`.

### 3.3 Build the shared baseline inputs (once)

```bash
python scripts/prepare_baseline_inputs.py
cat data/baselines/prep_report.txt          # <-- READ THIS
```

**Stop and check the report.** "with >=1 HPO term" must be ~117. If many OMIM diseases come back empty, the `phenotype.hpoa` column names differ from what the script auto-detected — open the file header and adjust, then re-run. Do **not** proceed with empty phenotype lists; that would unfairly cripple the baselines and invalidate the comparison.

### 3.4 SHEPHERD (the headline baseline)

SHEPHERD uses a **pretrained model on its own KG — you do NOT retrain it.** You feed each disease as a "patient" and read its causal-gene ranking.

```bash
# install miniconda (VM has root)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /home/miniconda3
source /home/miniconda3/bin/activate

cd /home
git clone https://github.com/mims-harvard/SHEPHERD.git
cd SHEPHERD
conda env create -f environment.yml
conda activate shepherd
bash install_pyg.sh
```

**Download their data + pretrained model** from Harvard Dataverse (`doi:10.7910/DVN/TZTPFL`, linked in their README) into the path the README specifies, preserving directory structure (the rare-disease KG + checkpoints).

**Build the SHEPHERD input** (in your PromptGFM venv, not the shepherd env):

```bash
# back in your code env
source /home/promgfm-bio/.venv/bin/activate
cd /home/promgfm-bio
python scripts/make_shepherd_input.py    # -> data/baselines/shepherd_patients.jsonl
```

**Run SHEPHERD's causal-gene-discovery** (in the shepherd env) using its predict/run entry point with `shepherd_patients.jsonl` + the pretrained checkpoint as arguments — **the exact command is in their README's "causal gene discovery" section; copy it from there** (entry-point names change between releases, so I won't hard-code a wrong one). Capture its per-disease ranked Ensembl gene list to JSON, then map Ensembl→symbol (invert `symbol_to_ensembl.json`) into `data/baselines/rankings/shepherd_symbols.json` shaped `{disease_id: [ranked SYMBOLS]}`.

> Realistic expectation: budget one full focused day for the conda env + `install_pyg.sh` (the fragile step). If it won't build, open a GitHub issue — the maintainers respond. **Pause the VM overnight** between sessions (you keep `/home`, pay storage only).

### 3.5 Phrank (fast, ~1 day, no GPU)

Phrank (`bitbucket.org/bejerano/phrank`) is a pure-Python HPO phenotype-similarity ranker. Install it, give it the HPO DAG + gene-phenotype annotations (`phenotype_to_genes.txt`), and for each disease rank genes by phenotypic match to its HPO list. Write `data/baselines/rankings/phrank.json` in the same `{disease: [symbols]}` shape.

### 3.6 LIRICAL (Java, ~1 day, CPU)

```bash
sudo apt-get update && sudo apt-get install -y default-jdk
# download LIRICAL release + its data per its docs; run phenotype-only mode per disease
```

Capture ranked genes → `data/baselines/rankings/lirical.json`.

### 3.7 PubMedBERT-cosine (the "static fusion" strawman, ~hours)

Reuse your own `PromptEncoder` (frozen): embed each disease prompt, embed each candidate gene's description, rank by cosine similarity. Pure inference on the L4 (or CPU). Write `data/baselines/rankings/pubmedbert_cosine.json`. This is the static baseline your dynamic FiLM architecture is designed to beat.

### 3.8 LLM-direct (Kim et al. protocol, no GPU, ~$ few)

For each disease, prompt an LLM (GPT-4 / Claude) via API to list likely causal genes; parse to a ranked list → `data/baselines/rankings/llm_direct.json`. Set the API key as an environment variable on the VM — **never commit it.** Expect ~17% top-50 (Kim et al. AJHG 2024); this quantifies the LLM-only gap your method closes.

### 3.9 Export YOUR model's rankings into the same format

So PromptGFM-Bio sits in the same table, export its per-disease ranked symbols (your full model already produces full-vocab rankings via `model.get_gene_rankings(...)`, used in `case_study.py`). Write `data/baselines/rankings/promptgfm.json` as `{disease: [ranked SYMBOLS]}`. (A ~30-line script around `get_gene_rankings` over the 117 diseases.)

### 3.10 Produce the comparison table

```bash
python scripts/score_baselines.py \
  --truth data/baselines/disease_true_genes.json \
  --method promptgfm=data/baselines/rankings/promptgfm.json \
  --method shepherd=data/baselines/rankings/shepherd_symbols.json \
  --method phrank=data/baselines/rankings/phrank.json \
  --method lirical=data/baselines/rankings/lirical.json \
  --method pubmedbert=data/baselines/rankings/pubmedbert_cosine.json \
  --method llm_direct=data/baselines/rankings/llm_direct.json \
  --out data/baselines/comparison_table.md
```

This reports **Hit@10, Hit@50, MRR** with definitions identical across all methods (the fair cross-method currency; keep AUROC for your internal ablation table). The resulting `comparison_table.md` is a centerpiece of the paper.

**Then PAUSE the VM.** Download results to your laptop:

```powershell
scp -P <PORT> -r <user>@<HOST>:/home/promgfm-bio/data/baselines ".\data\"
```

**Phase 1 cost:** mostly idle-GPU active time on L4 at ₹41.31/hr, spread over Month 1 with pauses between sessions ≈ **~₹900**, plus a few dollars of LLM API.

---

## 4. What you must NOT change (both phases)

- `configs/ablations/*.yaml` — frozen. `batch_size: 768`, FP16 AMP (`mixed_precision: true`), `num_negatives: 5`, 100 epochs, LR 5e-4, loss weights 1.0/0.5/0.3, `data.random_seed: 42`, `deterministic:false`, `benchmark:true`.
- Do **not** enable TF32 or switch FP16→BF16 for any run that feeds the 10-seed study (Phase 0's parity run is held to this too).
- Phase 1 trains nothing of yours, so there is no model config to optimize — the only "optimization" is choosing the cheap L4 and pausing aggressively.

---

## 5. Phase 0 + 1 cost & time summary

| Phase | Product / GPU | Active GPU-hrs | Cost (₹) |
|---|---|---|---|
| 0 — Parity | A100 40GB Template (on-demand) | ~1.5 | ~126 |
| 1 — Baselines | L4 24GB VM (on-demand, paused between sessions) | ~22 | ~900 |
| **Total** | | **~24** | **~₹1,030** |

Your ₹1,000 balance covers essentially all of this. Recharge ~₹11k before Phase 2 (the 28-seed batch).

---

## 6. Cheat-sheet (Phase 0 + 1)

```bash
# ---- PHASE 0 (A100 Template, on-demand) ----
cd /home && git clone https://github.com/<you>/PromptGFM-Bio.git promptgfm-bio
cd /home/promptgfm-bio && python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python scripts/test_gpu.py && python scripts/verify_setup.py
# upload graph + edges + results/ablation_4_full_model_seed42/evaluation_results.json
chmod +x scripts/*jarvis*.sh
bash scripts/parity_check-jarvis.sh 2>&1 | tee logs/parity_$(date +%Y%m%d).log
# PASS -> rm -rf checkpoints/parity_* results/parity_* ; PAUSE instance

# ---- PHASE 1 (L4 VM, root) ----
cd /home/promgfm-bio && source .venv/bin/activate && pip install mygene
python scripts/prepare_baseline_inputs.py && cat data/baselines/prep_report.txt   # verify ~117 have HPO
python scripts/make_shepherd_input.py
#   ... conda SHEPHERD env, download Dataverse data, run causal gene discovery
#   ... Phrank / LIRICAL / PubMedBERT-cosine / LLM-direct -> data/baselines/rankings/*.json
#   ... export your model: data/baselines/rankings/promptgfm.json
python scripts/score_baselines.py --truth data/baselines/disease_true_genes.json \
  --method promptgfm=data/baselines/rankings/promptgfm.json \
  --method shepherd=data/baselines/rankings/shepherd_symbols.json \
  --out data/baselines/comparison_table.md
# PAUSE VM; scp results back to laptop
```

---

## 7. The two verification points I could not pre-resolve

1. **`phenotype.hpoa` column names** — the prep script auto-detects `database_id`/`hpo_id` but HPO releases vary. The `prep_report.txt` coverage number is your check; if low, fix the header names and re-run.
2. **SHEPHERD's exact run command** — entry-point names change between releases, so copy the invocation from the current README's causal-gene-discovery section rather than trusting a hard-coded one.

When you're on the VM, send me a `head -5 data/raw/hpo/phenotype.hpoa` and the SHEPHERD README's run section, and I'll finalize those two spots exactly.

---

*GPU prices reflect your JarvisLabs console on 28 May 2026. SHEPHERD input contract (JSON-lines; `positive_phenotypes` as HPO terms; `true_genes`/`all_candidate_genes` as Ensembl IDs; conda + install_pyg.sh; data on Harvard Dataverse doi:10.7910/DVN/TZTPFL) confirmed from the mims-harvard/SHEPHERD repository.*
