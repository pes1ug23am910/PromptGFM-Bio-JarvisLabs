# PromptGFM-Bio — Conversation Handover (lossless)

**Purpose:** continue this work in a fresh Claude chat without information loss.
**Generated:** end of the long working session, 28 May 2026.
**Target:** Bioinformatics (Oxford) journal + RECOMB 2027 submission.

> Start the new chat by saying: *"Continue PromptGFM-Bio work from this handover document"* and attach this file plus the items in Section 12.

---

## Table of contents

1. Who you are and what the project is
2. Current concrete state of the work
3. The decision that drives everything: Path 2
4. The critical finding (the seed-flow bug)
5. JarvisLabs environment (verified GPU pricing, products)
6. The phase plan, end to end
7. Deliverables already produced — exact paths and status
8. The single most important sequence of commands to execute next
9. Two unresolved verification points (Phase 1 only)
10. The two pre-existing artifacts that DO carry over to Path 2
11. The artifacts that are SUPERSEDED and should not be used
12. What to upload at the start of the new chat
13. The aggregator that still needs to be written (after the 40 runs complete)
14. Risk register / troubleshooting

---

## 1. Who you are and what the project is

**You (the user):** Yash. Windows 11 Pro plan user. Project lives at
`E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio`.
Workstation has RTX 4090 / RTX 4500 Ada used for the original 12 ablations.
JarvisLabs.ai balance ≈ ₹1,000 at handover time (recharge needed before Phase 2).

**PromptGFM-Bio:** a prompt-conditioned graph foundation model for **rare disease
gene prioritization**. Frozen PubMedBERT encodes disease descriptions / HPO
phenotype lists; FiLM-conditioned GraphSAGE message-passes over a heterogeneous
biomedical graph; an MLP predictor scores gene-disease pairs.

**Graph:** 19,576 genes, 16,841 diseases, 11,794 phenotypes, 1.85M STRING PPI
edges (≥700 confidence). Built from BioGRID + STRING + HPO + Orphanet +
DisGeNET in `data/processed/biomedical_graph.pt`.

**Headline numbers from existing 12 ablations (4 variants × 3 seeds):**
full-model standard test AUROC ≈ 0.96, Hit@50 ≈ 0.55; zero-shot AUROC 0.9413,
Hit@50 0.219 on 117 rare diseases (59 OMIM + 58 ORPHA). These numbers are
**informative but methodologically tainted** — see Section 4.

**Pinned deps** (`requirements.txt`): `torch==2.1.0`, `torch-geometric==2.4.0`,
`transformers==4.35.0`, PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`).

**Target venues:** **Bioinformatics (Oxford), rolling submission**, with
**RECOMB 2027** abstract ~7 Nov / paper ~20 Nov 2026 as parallel target.
Fallback ladder: Genome Medicine → Cell Reports Methods → Bioinformatics
Advances → PLOS Comp Bio.

---

## 2. Current concrete state

**Done:**
- 12 ablation runs (workstation). All four variants × seeds 42/43/44. Each
  has `evaluation_results.json` (test + stratified) and `zero_shot_results.json`
  in `results/ablation_*_seed{42,43,44}/`.
- 117 zero-shot rare diseases identified in `data/splits/zero_shot_rare_diseases.json`
  (built by `find_rare_diseases.py --seed 42`).
- Ablation paragraph drafted with paired-t-test statistics
  (`ablation_paragraph.md`). Ordering Full > Prompt ≈ GNN > MLP across all 7
  metrics; several improvements at p<0.10 but not p<0.05.

**Critically incomplete (this is what blocks Q1 acceptance):**
- External baselines (SHEPHERD, Phrank, LIRICAL, Exomiser, PubMedBERT-cosine,
  GPT/Claude-direct) — **none done**. Mandatory for Q1.
- Statistical rigor — only 3 seeds, no bootstrap CIs, no multiple-testing
  correction, no power analysis.
- FiLM interpretability (γ/β extraction, layer-wise ablation) — **none done**.
- Biological case studies (extend to 5 diseases, pathway analysis) — **none done**.
- Robustness (paraphrased descriptions) + (optional) second dataset — **none done**.
- Writing — **none started**.

**Critically wrong (just discovered, see Section 4):**
- The 12 existing ablation runs were trained under a broken seed flow. They
  are **superseded** by the Path 2 plan, not pooled with it.

---

## 3. The decision that drives everything: Path 2

You chose **Path 2** — the highest-probability path to Q1 acceptance — because
the existing seed mechanism has methodology issues a Q1 reviewer would flag in
ten minutes. Path 2 is:

1. Apply a small two-file code patch (one helper + one one-line split-seed
   change + one seeding call, in each of `scripts/train.py` and
   `scripts/evaluate.py`).
2. Discard the 12 old ablation runs (don't pool them).
3. Run **40 fresh ablation runs** (10 seeds × 4 variants, seeds 42–51) on
   JarvisLabs A100 under the patched code.
4. Run **Phase 1 (external baselines)** in parallel on a separate L4 VM.

Under Path 2, the train/val/test split is **fixed** at `data.random_seed: 42`
(which equals the original seed-42 split), so the existing 117-disease zero-shot
set is valid for all 10 seeds with no leakage audit needed. Top-level
`config['seed']` deterministically seeds torch / numpy / Python `random` —
model init, negative sampling, DataLoader shuffling all vary reproducibly per
seed. "Mean ± std over 10 seeds" now means what reviewers think it means:
model-initialization variance on a fixed benchmark.

**Cost of Path 2 (through Phase 2):**
- 1 smoke test + 40 training + 40 evaluations on A100 40GB
- Phase 1 baselines on L4 24GB
- On-demand: ~₹6,500. With spot for training: ~₹3,900.
- **Recharge ~₹7,000 before Phase 2** (you have ~₹1,000).

---

## 4. The critical finding (the seed-flow bug)

I verified the LATEST codebase (the three uploaded `*_codebase.md` files from
28 May 2026) and confirmed:

- `scripts/train.py` line 4632 and `scripts/evaluate.py` line 340 both call
  `dataset.create_train_val_test_split(random_seed=config.get('seed', 42), ...)`.
- The top-level `config['seed']` therefore drives **only the data split**.
- `config['data']['random_seed']` (present in every config YAML) is **vestigial**
  — no code anywhere reads it. Grep-confirmed across all .py files.
- `torch.manual_seed` is only called in **hardcoded fallback paths** (lines 387,
  4480, 4927) for synthetic node features (taken only when graph lacks `.x`).
  It is **never called with the config seed**.
- Negative sampling uses `random.sample()` from Python stdlib — **never seeded**.
- DataLoader shuffling uses PyTorch's default RNG — **never seeded** from config.

**Therefore, the existing seeds 42/43/44 varied in two confounded ways:**
1. Different deterministic data splits (via the top-level seed).
2. Different non-reproducible model init / negative sampling / DataLoader order
   (process-time randomness, not config-driven).

A reviewer reading the methods would expect "n=3 seeds" to mean model-init
variance on a fixed benchmark; under the existing code it means something
different and less defensible. The 117-disease zero-shot set was built from
seed-42's split and is **not strictly leakage-free for seeds 43/44**.

Path 2 fixes both issues with the two-file patch in Section 7.

---

## 5. JarvisLabs environment (verified 28 May 2026)

From your console screenshots:

| GPU | VRAM | ₹/hr on-demand | Use |
|---|---|---|---|
| L4 24GB | 24 GB | **41.31** (IN2) | Phase 1 baselines (CPU/IO-heavy inference) |
| A100 40GB | 40 GB | **84.24** (IN2) | **Primary training GPU** (Phase 2) |
| A100 80GB | 80 GB | 140.94 (IN2) | Only if OOM (it won't at batch 768) |
| H100 80GB | 80 GB | 255.15 (IN2) / 283.50 (EU1) | Skip — premium not worth it |
| RTX PRO 6000 Blackwell | 96 GB | 179.01 (IN1) | Skip |
| H200 141GB | 141 GB | 360.45 (EU1) | Skip |

**Pricing options:** On-Demand, Spot (~-48%), 30/90/180-day commit (-21% to -32%).
For Path 2 training the 40 runs: spot is fine because per-epoch checkpointing
makes interruptions cheap (re-run; skip-if-exists resumes). For the smoke test
and Phase 1 work: on-demand.

**Two product types you'll use:**
- **Template (PyTorch)** — pre-built image with JupyterLab, VS Code, SSH. Use
  for all training and evaluation (smoke + 40 runs + evaluations).
- **VM** — bare instance with root SSH. Use for Phase 1 (SHEPHERD/conda/Docker).

**Persistence rule** (saves significant money): `/home` is persistent across
pause/resume; `/root` is wiped. Always clone the repo to `/home/promptgfm-bio`
and build the venv there. **Pause every instance the moment you stop active
work** — storage rate is ~₹1.13/hr vs. full compute rate.

**SSH key:** add your public key in JarvisLabs → Settings before launching any
VM (the VM launch UI errors without one).

---

## 6. The phase plan, end to end

| Phase | Goal | Where | GPU | Active hours | Cost (on-demand) |
|---|---|---|---|---|---|
| **0** (Path 2) | Smoke test the patches | A100 Template | A100 40GB | 0.2 | ~₹17 |
| **1** | External baselines (SHEPHERD, Phrank, LIRICAL, PubMedBERT-cos, LLM-direct) | L4 VM | L4 24GB | ~22 | ~₹900 |
| **2** | 40 fresh training runs + evals + aggregator | A100 Template | A100 40GB | ~60 | ~₹5,060 (₹2,640 spot) |
| **3** | FiLM interpretability (γ/β heatmap, 9 layer-wise runs) | A100/L4 | mixed | ~15 | ~₹1,265 |
| **4** | 5 biological case studies + pathway analysis | L4 | L4 24GB | ~10 | ~₹415 |
| **5** (optional) | Robustness + second dataset | A100 | A100 40GB | ~19 | ~₹1,600 |
| **6** | Writing (Methods → Results → Related → Intro → Abstract) | Claude only | — | 0 | — |
| **7** | Adversarial review + bioRxiv preprint + Bioinformatics submission + RECOMB | Claude only | — | 0 | — |

**Timeline:** May (now) — Phase 0 smoke + Phase 1 start; June — Phase 1 finish +
Phase 2; July — Phases 3 and 4; August — Phase 5 + figures; September — writing
sprint; October — Bioinformatics submission + preprint; November — RECOMB
abstract ~Nov 7, paper ~Nov 20.

---

## 7. Deliverables already produced — exact paths and status

All files live under `/mnt/user-data/outputs/` (download / save locally
before the chat closes). Three groupings:

### 7a. **CURRENT — Path 2 deliverables** (use these for Phase 2)

`/mnt/user-data/outputs/path2/`

- **`PATH2_PATCHES.md`** — Copy-paste edit instructions for `scripts/train.py`
  and `scripts/evaluate.py`. Five small blocks total: a `_set_all_seeds` helper
  inserted in each file, a one-line change in `create_dataloaders` (train.py)
  and `_load_dataset` (evaluate.py) to use `data.random_seed` for the split,
  and a `_set_all_seeds(config.get('seed', 42))` call at the top of
  `run_finetuning`, `run_pretraining`, and `main()`.
- **`PATH2_DEPLOYMENT_GUIDE.md`** — ordered operational guide from "apply
  patches on Windows" to "40 runs done and committed." Has Section "What about
  the old 12 ablation runs?" → drop from headline; optionally cite as pilot.
- **`scripts/_jarvis_env.sh`** — shared env header for all Path 2 scripts.
  Sets `ALL_SEEDS="42 43 44 45 46 47 48 49 50 51"`. **Refuses to run** if the
  patches are not present in train.py / evaluate.py (grep gate).
- **`scripts/smoke_test_path2-jarvis.sh`** — ~10 min on A100. Trains
  ablation_4 for 3 epochs at seeds 42 and 43, then auto-verifies four invariants
  from the logs (split edge counts identical across seeds, `split seed = 42`
  logged in both, `_set_all_seeds` fired with the right seed, training losses
  differ between seeds). Run this BEFORE the 40-run batch.
- **`scripts/run_all_seeds-jarvis.sh`** — the 40-run trainer. 10 seeds × 4
  variants. Skip-if-exists. Supports `ALL_SEEDS="…"` env override to split
  across two GPUs.
- **`scripts/run_evaluations_all_seeds-jarvis.sh`** — the 40-run evaluator
  (test + stratified + zero-shot per (variant, seed)). Skip-if-exists.

All four shell scripts are `bash -n` and `shellcheck -S warning` clean.
The smoke test's invariant-verification logic was hand-traced against the
patches.

### 7b. **STILL VALID — Phase 0+1 deliverables that carry over**

`/mnt/user-data/outputs/phase0_1/`

- **`PromptGFM-Bio_Phase0_Phase1_Guide.md`** — the standalone guide. **Phase 0
  section (parity check) is OBSOLETE under Path 2** — `smoke_test_path2-jarvis.sh`
  replaces it. **Phase 1 section (external baselines) is FULLY VALID and is
  your operational manual** for the L4 VM work. Read Sections 3.0 through 3.10
  of that guide for the SHEPHERD walkthrough, conda env setup, fairness design,
  etc.
- **`gitignore_additions.txt`** — merge into your `.gitignore` before pushing
  to private GitHub. Excludes `.env`, large binaries, `*.pt`, raw data, logs,
  `__pycache__`.
- **`scripts/prepare_baseline_inputs.py`** — Phase 1: builds
  `data/baselines/{disease_hpo_terms.json, disease_true_genes.json,
  symbol_to_ensembl.json, all_candidate_genes_*.json, prep_report.txt}`.
  Run on the L4 VM in your PromptGFM venv after `pip install mygene`.
- **`scripts/make_shepherd_input.py`** — Phase 1: converts the prep output into
  `shepherd_patients.jsonl` for SHEPHERD's causal-gene-discovery input.
- **`scripts/score_baselines.py`** — Phase 1: uniform rank-based scorer
  producing the head-to-head comparison table (Hit@10, Hit@50, MRR — identical
  definitions across all methods, including PromptGFM-Bio). Smoke-tested.

### 7c. **THE BIG-PICTURE GUIDE** (still useful for Phases 3+)

`/mnt/user-data/outputs/PromptGFM-Bio_JarvisLabs_Execution_Guide.md` — the
~5,500-word execution guide covering all phases.

⚠️ **Two corrections to that document** (will fix when Path 2 is settled):
- Its claim that "data.random_seed is held fixed across all seeds" is correct
  **only after the Path 2 patches are applied**. Before the patches, it was wrong.
- Cost estimates need ~₹1,400 added vs. the original Phase 2 estimate
  (because we now do 40 runs instead of 28).

### 7d. **SUPERSEDED — do not use**

`/mnt/user-data/outputs/jarvis_scripts/` — replaced by Path 2 versions:
- `parity_check-jarvis.sh` → replaced by `smoke_test_path2-jarvis.sh`
- `run_ablations_extra_seeds-jarvis.sh` → replaced by `run_all_seeds-jarvis.sh`
- `run_evaluations_extra_seeds-jarvis.sh` → replaced by `run_evaluations_all_seeds-jarvis.sh`
- The old `_jarvis_env.sh` (had `EXTRA_SEEDS`) → replaced by Path 2 version
  (has `ALL_SEEDS` + patch-presence gate)

---

## 8. The single most important sequence of commands to execute next

### Step A — On your Windows machine

```powershell
cd E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio

# 1. Apply PATH2_PATCHES.md to scripts\train.py and scripts\evaluate.py.
#    (Use VS Code's find/replace. Indentation must match.)

# 2. Verify (each should print 3+ hits):
Select-String -Path "scripts\train.py" -Pattern "_set_all_seeds|split_seed = config"
Select-String -Path "scripts\evaluate.py" -Pattern "_set_all_seeds|split_seed = config"

# 3. Merge gitignore_additions.txt into .gitignore.
#    Verify .env / *.pt / data/raw are NOT staged:
git status
git check-ignore .env data\processed\biomedical_graph.pt

# 4. Copy the four Path 2 scripts (+ the three Phase 1 .py files) into scripts\:
#    _jarvis_env.sh, smoke_test_path2-jarvis.sh,
#    run_all_seeds-jarvis.sh, run_evaluations_all_seeds-jarvis.sh,
#    prepare_baseline_inputs.py, make_shepherd_input.py, score_baselines.py

# 5. CRITICAL: move the old 12 ablation results aside so skip-if-exists
#    does NOT preserve the broken-seed runs:
git mv results "results_pilot_OLD_broken_seed_flow" 2>$null
git mv checkpoints "checkpoints_pilot_OLD_broken_seed_flow" 2>$null
# (Or just rename — they don't need to be in git.)

# 6. Push code to a PRIVATE GitHub repo:
git add scripts\ .gitignore
git commit -m "Path 2: fixed split via data.random_seed; deterministic per-seed model RNG"
git push
```

### Step B — Launch JarvisLabs A100 Template (on-demand)

JarvisLabs console → Templates → PyTorch → A100 40GB IN2 → On-Demand → 100 GB → Launch.

### Step C — Bootstrap the instance

```bash
cd /home && git clone https://github.com/<you>/PromptGFM-Bio.git promptgfm-bio
cd /home/promptgfm-bio
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python scripts/test_gpu.py        # confirm A100 detected
python scripts/verify_setup.py
# Upload via JupyterLab drag-drop OR scp:
#   data/processed/biomedical_graph.pt
#   data/processed/hpo_gene_disease_edges.csv
chmod +x scripts/*jarvis*.sh
```

### Step D — Smoke test (~10 min, ~₹17)

```bash
bash scripts/smoke_test_path2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log
```

Expected end: `>>> SMOKE PASS — Path 2 is wired correctly.` If FAIL, fix the
patch and re-run before proceeding.

### Step E — The 40 training runs (~53 GPU-hours)

```bash
tmux new -s train
bash scripts/run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%Y%m%d).log
# Ctrl+B, D to detach. tmux attach -t train to return.
```

### Step F — The 40 evaluations (~7 GPU-hours)

```bash
bash scripts/run_evaluations_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_eval_$(date +%Y%m%d).log
```

### Step G — Pause the A100, launch the L4 VM, do Phase 1

Continue with Phase 1 baselines on a separate L4 VM as documented in
`PromptGFM-Bio_Phase0_Phase1_Guide.md` Section 3 (SHEPHERD setup, prep script,
SHEPHERD/Phrank/LIRICAL/PubMedBERT-cosine/LLM-direct, `score_baselines.py`).
Phases 1 and 2 are independent and can run in parallel on separate instances.

---

## 9. Two unresolved verification points (Phase 1 only)

Both deliberately left unresolved because they need on-VM verification rather
than guessing:

1. **`phenotype.hpoa` column names.** `prepare_baseline_inputs.py` auto-detects
   `database_id`/`hpo_id` but HPO releases vary. After running the prep script,
   read `data/baselines/prep_report.txt`; "with >=1 HPO term" should be ~117.
   If many OMIM diseases come back empty, open the .hpoa header and adjust the
   column-name lookup.

2. **SHEPHERD's exact run command.** The mims-harvard/SHEPHERD repo's
   entry-point names change between releases. Copy the invocation from the
   current README's "causal gene discovery" section. Confirmed contract:
   JSON-lines input with `positive_phenotypes` (HPO term IDs), `true_genes`
   and `all_candidate_genes` (Ensembl IDs); conda env + `install_pyg.sh`; data
   on Harvard Dataverse `doi:10.7910/DVN/TZTPFL`.

When you're on the VM, paste `head -5 data/raw/hpo/phenotype.hpoa` and the
SHEPHERD README's "causal gene discovery" section into the new chat to lock
both down.

---

## 10. The two pre-existing artifacts that DO carry over to Path 2

1. **`data/splits/zero_shot_rare_diseases.json`** — built with
   `find_rare_diseases.py --seed 42`, which used `np.random.seed(42)` to drive
   the split. Under Path 2 the split is fixed at `data.random_seed: 42`, which
   uses **the identical `np.random.seed(42)` call**. So the 117 diseases are
   produced by the same shuffled edge order → **the existing zero-shot set is
   still valid under Path 2. No regeneration needed.**

2. **All four `configs/ablations/*.yaml`** — every one already contains
   `data.random_seed: 42` (currently vestigial) AND top-level `seed: 42`.
   The Path 2 code reads both correctly. The sed override in the runner
   touches only top-level `seed:`. **No config edits needed.**

---

## 11. The artifacts that are SUPERSEDED and should not be used

1. **The 12 existing ablation runs** in `results/ablation_*_seed{42,43,44}/`
   and `checkpoints/ablation_*_seed{42,43,44}/`. **Move them aside before
   running Path 2** (rename to `*_pilot_OLD_broken_seed_flow/`). Otherwise
   `run_all_seeds-jarvis.sh`'s skip-if-exists logic will preserve the broken
   runs.

2. **The "underscore" result directories** that the project structure shows
   (`results/ablation_X_seed_42/`, `seed_43`, `seed_44`) — these are
   pre-fix-era residue documented in `SESSION_HANDOVER_2026-04-20.md`. Delete
   or move to the same `*_pilot_OLD/` folder.

3. **The four old jarvis scripts** in `/mnt/user-data/outputs/jarvis_scripts/`
   (parity_check, run_ablations_extra_seeds, run_evaluations_extra_seeds, and
   the EXTRA_SEEDS-style `_jarvis_env.sh`). The Path 2 versions in
   `/mnt/user-data/outputs/path2/scripts/` replace them.

4. **The "Phase 0 parity check"** described in
   `PromptGFM-Bio_Phase0_Phase1_Guide.md` Section 2. Under Path 2 there is
   nothing to be parity-equivalent with (we're not pooling old + new seeds).
   The smoke test takes its place.

---

## 12. What to upload at the start of the new chat

To pick up cleanly, upload (or ensure the chat has access to) the following.
Project Knowledge: prefer attaching files in the new chat rather than relying
on the older copies in `/mnt/project/` (those are out of date — the codebase
markdowns below are authoritative).

**Essential:**
- `PromptGFM-Bio_HANDOVER.md` (this document)
- `configs_codebase.md`, `src_codebase.md`, `scripts_codebase.md` (the
  three latest-code bundles you uploaded on 28 May 2026 — these are the
  source of truth for current code)
- `PromptGFM-Bio_Structure.txt` (the directory tree)
- `data/splits/zero_shot_rare_diseases.json` (the 117 IDs)
- The six **Path 2 deliverable files** from `/mnt/user-data/outputs/path2/`
  (so the new chat can reference them when troubleshooting)

**Helpful:**
- The three Phase 1 Python scripts from `/mnt/user-data/outputs/phase0_1/scripts/`
  (`prepare_baseline_inputs.py`, `make_shepherd_input.py`, `score_baselines.py`)
- `PromptGFM-Bio_Phase0_Phase1_Guide.md` (Phase 1 sections still valid)
- `PromptGFM-Bio_JarvisLabs_Execution_Guide.md` (big-picture, for Phases 3+)
- `New_PromptGFM_TopVenue_Plan.md`, `PromptGFM_Publication_Strategy_UPDATED.md`,
  `PromptGFM_Paper_Plan.md` (publication strategy, related work positioning)
- `PromptGFM_Bio_Final_Project_Report.md`, `ablation_paragraph.md`
  (current results to be replaced by Path 2 outputs)
- `Few shot learning for phenotype-driven.pdf` (the SHEPHERD paper — already
  in your repo at `docs/Papers/`)

**Not needed (large, regenerable, or already in /mnt/project):**
- The large `*.pt` graph file — it lives on the workstation and uploads to
  JarvisLabs instances; the chat doesn't need it.
- The 12 existing `evaluation_results.json` files — superseded.
- Raw BIOGRID / STRING archives.
- Notebooks (the v6 notebook is informational only at this point).

---

## 13. The aggregator that still needs to be written

After the 40 training runs and 40 evaluations finish, the next deliverable
is an aggregator that produces the paper's **Table 1**. Should:

- Pool the 40 `evaluation_results.json` files (10 seeds × 4 variants).
- Pool the 40 `zero_shot_results.json` files (same shape).
- Compute mean ± std per metric per variant.
- **Bootstrap 95% CIs** (10k iterations, resampling diseases within the 117-disease zero-shot pool) for AUROC, Hit@10, Hit@50, MRR.
- **Paired tests** (Full vs each ablation) with **Holm–Bonferroni** correction
  across the metric × comparison family.
- **Power analysis**: with n=10 paired observations, state the minimum
  detectable effect at α=0.05, power=0.8.
- Emit a **LaTeX-ready** main table for the paper.

Ask the new chat to write this once you have the 40 result JSONs. Tell it to
mirror the AUROC / Hit@K / MRR definitions from `src/evaluation/metrics.py` so
the headline table is consistent with your internal numbers.

---

## 14. Risk register / troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Smoke test FAIL: split edge counts differ across seeds | Path 2 patch not applied to `create_dataloaders` | Re-read `PATH2_PATCHES.md` block 1b; confirm `split_seed = config.get('data', {}).get('random_seed', ...)` is present |
| Smoke test FAIL: `split seed = X` line not in log | `logger.info` line in the patch was dropped | Add it back; this also confirms the new code path is executing |
| Smoke test FAIL: losses identical across seeds 42 and 43 | `_set_all_seeds(config.get('seed', 42))` not added at top of `run_finetuning` | Re-apply block 1c |
| `_jarvis_env.sh` aborts with "Path 2 patch missing" | Gate detected the un-patched code | Apply the patches first; never bypass the gate |
| `torch-scatter`/`-sparse`/`-cluster` won't install | PyG companion wheels are CUDA-version specific | `pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html` (match your instance CUDA) |
| Spot instance interrupted mid-batch | Expected with spot | Re-run the same `run_all_seeds-jarvis.sh`; skip-if-exists resumes from the next missing (variant, seed) |
| Old seed-42 results overwritten | You forgot Section 11 step 1 (move old results aside) | Recover from `results_pilot_OLD_broken_seed_flow/` you renamed earlier. If you didn't rename, the old results are gone — they were going to be discarded anyway. |
| Bus error / `/dev/shm` exhaustion in DataLoader | Known issue from earlier handovers | Already handled in `train.py` (graph tensors stored once, fork context). If recurs, lower `num_workers` in the config (this is a DataLoader plumbing setting, not a numerics change). |
| SHEPHERD env build (`install_pyg.sh`) fails | SHEPHERD targets old torch/PyG | Use a fully isolated conda env (never reuse the PromptGFM venv); if stuck >1 day, open a GitHub issue on mims-harvard/SHEPHERD — they respond |
| Phase 1 prep coverage low (many OMIM diseases have no HPO terms) | `phenotype.hpoa` column names differ | Open the file header, adjust `prepare_baseline_inputs.py` column lookup, re-run — do NOT proceed with empty phenotype lists |
| Running low on JarvisLabs balance mid-batch | — | Pause everything (you keep `/home`), recharge, resume |
| Bioinformatics rejects with major revision asking for SHEPHERD | Phase 1 not yet done | This is exactly why Phase 1 is mandatory; do not submit before it is complete |

---

## End-of-handover checklist

When you start the new chat, paste this minimal kickoff:

> "Continue PromptGFM-Bio work from the attached `PromptGFM-Bio_HANDOVER.md`.
> I'm on Windows 11 Pro, project at `E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio`. The next concrete action is **Step A** in Section 8 of the handover — applying `PATH2_PATCHES.md` to `scripts/train.py` and `scripts/evaluate.py`. After I apply them I'll paste the modified train.py and evaluate.py sections back so you can verify the patch took correctly, then we'll proceed to Steps B–G."

That gives a fresh chat enough to:
- See Path 2 is locked in (no re-deliberation).
- Know your environment.
- Know exactly where in the sequence you are.
- Know what you'll send next (patched code) and what it should do (verify).

---

*Handover complete. The 40-run study under Path 2 is the entire Q1 risk
reduction for this paper. Nothing about the old 12 ablations is wasted —
they informed the design and revealed the seeding bug — but they are not
part of the headline result.*
