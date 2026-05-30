# PromptGFM-Bio — Complete Execution Guide (JarvisLabs → Q1 Submission)

**Prepared:** 28 May 2026
**Scope:** Everything from "12 ablations are done" to "manuscript submitted to Bioinformatics / RECOMB 2027."
**Compute provider:** JarvisLabs.ai (pricing verified from your 28 May 2026 console screenshots, in ₹)
**Target venues:** Bioinformatics (Oxford) journal + RECOMB 2027 conference (Path B)

---

## How to use this document

Read Sections 1–4 once to understand the strategy and the cost/GPU logic. Then work phase by phase (Sections 6–13), each of which tells you: which JarvisLabs product to launch, which GPU, what to upload, what commands to run, how long it takes, and what it costs. Section 14 is the month-by-month timeline; Section 15 is the consolidated budget; Section 16 is troubleshooting; Section 17 is a one-page command cheat-sheet you can keep open while working.

A single principle governs the whole plan and is repeated wherever it matters: **do not change anything that alters the numerics of the already-trained seeds 42, 43, 44.** Q1 reviewers penalize inconsistency harder than they reward speed. Every efficiency gain below is chosen to be either provably numerics-neutral or quarantined to a separate reporting line.

---

## 1. Current state and what remains

### What you already have (done)

- A working PromptGFM-Bio system: frozen PubMedBERT → FiLM conditioning → GraphSAGE over 1.85M STRING PPI edges → MLP predictor.
- Heterogeneous graph: 19,576 genes, 16,841 diseases, 11,794 phenotypes.
- **12 ablation runs complete** (4 variants × 3 seeds 42/43/44), each with `evaluation_results.json` (standard test + stratified) and `zero_shot_results.json` (117 zero-shot rare diseases).
- Headline numbers: full-model test AUROC ≈ 0.96, Hit@50 ≈ 0.55; zero-shot AUROC 0.9413, Hit@50 0.219, with the full model leading every metric and the ordering Full > Prompt ≈ GNN > MLP consistent across all 7 metrics.
- A clean zero-shot disease set (`data/splits/zero_shot_rare_diseases.json`): 59 OMIM + 58 ORPHA = 117 diseases, none in the training split.
- A drafted ablation paragraph with paired-t-test statistics.

### What remains (the four Q1 gaps + writing)

| Gap | Why it blocks a Q1 venue | Phase |
|---|---|---|
| **External baselines** on your zero-shot split (SHEPHERD, Phrank, LIRICAL, Exomiser, PubMedBERT-cosine, GPT/Claude-direct) | Internal ablations alone never clear a top venue. Reviewers demand head-to-head against published methods. | Phase 1 |
| **Statistical rigor**: 3 → 10 seeds, bootstrap CIs, multiple-testing correction, power analysis | n=3 gives several results only at p<0.10; reviewers want p<0.05 with corrected multiplicity. | Phase 2 |
| **FiLM interpretability**: γ/β analysis, layer-wise ablation | Your central claim ("dynamic conditioning of message passing") is currently empirical, not mechanistic. | Phase 3 |
| **Biological validation**: 5 case studies + pathway coherence | A computational genomics paper needs biological grounding, not just metrics. | Phase 4 |
| Robustness + (optional) second dataset | Strengthens but does not block. | Phase 5 |
| Writing, adversarial review, submission | The deliverable. | Phases 6–7 |

---

## 2. The accuracy-preservation principle (read this before touching anything)

Your 10-seed statistical study only works if all 10 seeds are produced under **bit-for-bit identical configuration**. Seeds 42/43/44 were trained on the workstation (RTX 4090 / RTX 4500 Ada, FP16 AMP, batch 768). When you add seeds 45–51 on a JarvisLabs A100, the *hardware* changes but the *configuration must not*.

**Frozen across all 10 seeds (never edit for the new runs):**

- `batch_size: 768`
- `learning_rate: 0.0005`, AdamW, cosine schedule, 5 warmup epochs
- `num_negatives: 5`, random negative sampling
- `num_epochs: 100`, early-stopping patience 15
- loss weights BCE 1.0 / ranking 0.5 / listnet 0.3
- `mixed_precision: true` meaning **FP16** (the original seeds used FP16, not BF16)
- `data.random_seed: 42` (the split and the zero-shot set are held fixed; only the top-level `seed:` varies)
- `deterministic: false`, `benchmark: true`

**Allowed to differ (numerics-neutral, safe efficiency):**

- GPU model (4090/Ada → A100): unavoidable and fine, *if the parity check passes*.
- `num_workers`, `prefetch_factor`, `persistent_workers`: DataLoader plumbing, deterministic given the seed; does not change results.
- `cudnn.benchmark` autotuning: picks faster kernels with identical math.

**Forbidden for the new seeds (silently shifts accuracy):**

- TF32 matmul (`torch.backends.cuda.matmul.allow_tf32 = True`) — A100 default in some stacks; would change mantissa precision.
- FP16 → BF16 AMP switch.
- Any change to batch size, negatives, epochs, LR, loss weights, or model dims.
- `torch.compile` **unless** separately parity-verified (kernel fusion can reorder FP reductions).

This is why Phase 0 (parity check) exists. It costs ~₹126 and protects the entire 10-seed result.

---

## 3. JarvisLabs fundamentals: Template vs VM, persistence, pricing

### Template vs VM — the decision

JarvisLabs offers two products you'll use:

- **Template (e.g. the PyTorch template)** — a pre-built image with PyTorch, CUDA, JupyterLab, VS Code, and SSH already configured. You launch, pull your repo, and run. **Use this for all PromptGFM-Bio training and evaluation** (parity check, the 28 seed runs, layer-wise ablations, case-study inference). It saves 30–60 minutes of environment setup each session and your code already targets a standard PyTorch stack.

- **VM** — bare instance with full root access and an SSH key. More setup, more control. **Use this for the external baselines** (SHEPHERD, Exomiser, Phrank, LIRICAL) because they ship as conda environments / Docker / Java with dependency sets that conflict with your torch 2.1 stack. Root access and isolated conda envs are exactly what they need.

### Persistence — the rule that saves you money and grief

On both products:

- **`/home` is persistent** across pause/resume. Clone your repo here, build your venvs/conda envs here, store checkpoints and results here.
- **`/root` and other paths are wiped** on pause or destroy. Never put anything you care about there.
- **Pausing** keeps `/home` and charges only the storage rate (~₹1.13/hr for 100 GB) instead of the full compute rate. **Pause whenever you step away.** SHEPHERD setup spans days of intermittent work, not hours of continuous work — pausing between sessions is the single biggest cost saver.
- Default disk is **100 GB SSD**, which is plenty: your code + the graph `.pt` + checkpoints + the 117-disease JSON fit comfortably. Don't pay for more.

### Current JarvisLabs pricing (your 28 May 2026 console)

| GPU | VRAM | On-demand ₹/hr | Region | Best use in this project |
|---|---|---|---|---|
| **L4** | 24 GB | **41.31** | IN2 | External baselines (inference, CPU/IO-heavy) |
| **A30** | 24 GB | 38.88 | IN2 | Cheapest alternative for baselines/inference |
| **A100 40GB** | 40 GB | **84.24** | IN2 | **Primary training workhorse** (all seed runs) |
| A100 80GB | 80 GB | 140.94 | IN2 | Only if 40 GB OOMs (it won't at batch 768) |
| RTX PRO 6000 (Blackwell) | 96 GB | 179.01 | IN1 | Newer than plan; not needed |
| H100 80GB | 80 GB | 255.15 (IN2) / 283.50 (EU1) | IN2/EU1 | Skip — premium not worth it at your scale |
| H200 141GB | 141 GB | 360.45 | EU1 | Overkill |

**Pricing options on each card:** On-Demand (pay as you go), **Spot (~−48%, interruptible)**, and 30/90/180-day commits (−21% to −32%). For this project:

- **On-demand** for the parity check and any interactive work (SHEPHERD setup, interpretability notebooks).
- **Spot** is worth it for the 28-run seed batch *only*, because your training already checkpoints every epoch (`save_interval: 1`), so an interruption costs at most one partial run. Spot A100 40GB ≈ ₹44/hr would roughly halve the Phase 2 GPU bill. Do **not** use spot for the parity check or interactive sessions.
- **Long-term commits** are irrelevant unless you expand to Path C (Nature Machine Intelligence), which you are not.

### Your balance

The console shows **₹1,000**. Path B needs roughly **₹11,000–12,000** total. Recharge before Phase 2 (the first expensive phase). Phase 1 alone fits inside ~₹1,000 if you're disciplined about pausing.

---

## 4. GPU selection logic (why these choices)

The plan's core finding still holds at current prices: **A100 40GB is the cost-optimal training GPU.** Faster cards (H100, H200) cost more *in total* because their per-hour premium outpaces the wall-clock they save at your run length (~1.3 hr/run). A100 40GB at ₹84.24/hr beats them all on total spend.

For **baselines and inference** (Phase 1, parts of Phase 4), you are not training — GPU utilization is low — so the **L4 at ₹41.31/hr** (or A30 at ₹38.88) is the right call. Paying A100 rates to run Java-based phenotype rankers or a frozen-BERT cosine baseline wastes money.

Quick mapping:

| Work type | GPU | Product |
|---|---|---|
| Training (parity, 28 seeds, layer-wise) | A100 40GB | Template |
| Baselines (SHEPHERD/Phrank/LIRICAL/Exomiser) | L4 24GB | VM |
| Interpretability + case-study inference | L4 24GB (or A100 if you're already on one) | Template or VM |

---

## 5. Getting your code and data onto JarvisLabs

You'll do this once per fresh instance (or never again if you only pause). Three things must reach `/home`:

1. **The repository.** Easiest is `git clone` from your private GitHub. If the repo isn't pushed, use `scp` / `rsync` from your laptop, or JarvisLabs' File Storage upload.
2. **The processed graph** `data/processed/biomedical_graph.pt` (large binary; not in git). Transfer via `scp`/`rsync` or File Storage.
3. **The edge file** `data/processed/hpo_gene_disease_edges.csv`, the **raw Orphanet/HPO files** (`data/raw/orphanet/en_product4.xml`, `data/raw/hpo/phenotype.hpoa`, `phenotype_to_genes.txt`), and the **zero-shot JSON** (`data/splits/zero_shot_rare_diseases.json`).
4. For the parity check and pooled reporting, also bring **one reference result**: `results/ablation_4_full_model_seed42/evaluation_results.json`.

**Transfer pattern (from your laptop, with the instance's SSH details from the JarvisLabs console):**

```bash
# Example — replace HOST/PORT with the values JarvisLabs shows for your instance
RSYNC_OPTS="-avzP -e 'ssh -p <PORT>'"

# Code (or use git clone on the instance instead)
rsync -avzP -e "ssh -p <PORT>" ./PromptGFM-Bio/ user@<HOST>:/home/promptgfm-bio/

# Just the heavy/needed data if cloning code separately
rsync -avzP -e "ssh -p <PORT>" \
  data/processed/biomedical_graph.pt \
  data/processed/hpo_gene_disease_edges.csv \
  user@<HOST>:/home/promptgfm-bio/data/processed/
```

**Set up SSH key first** (the VM launch screen warns about this): JarvisLabs → Settings → add your public key before launching, or you can't connect.

**One-time env build on the instance (Template already has PyTorch, but pin your stack):**

```bash
cd /home/promptgfm-bio
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Try requirements.txt first; if the PyG companion wheels (torch-scatter/sparse/cluster)
# fail against the instance CUDA build, install them from the matching wheel index:
pip install -r requirements.txt
python scripts/test_gpu.py          # confirm CUDA + GPU name
python scripts/verify_setup.py      # confirm package versions
```

If `torch-scatter`/`torch-sparse`/`torch-cluster` fail to build, install the prebuilt wheels matching torch 2.1.0 + your CUDA (e.g. `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html`). This is the single most common setup snag; budget 30 minutes for it.

---

## 6. Phase 0 — Parity check (the insurance step)

**Goal:** prove the A100 reproduces your workstation seed-42 numbers within tolerance, so seeds 45–51 can be pooled with 42/43/44.

**Product / GPU:** Template + A100 40GB, **on-demand** (you want it not to be interrupted).

**Upload:** repo + graph + `results/ablation_4_full_model_seed42/evaluation_results.json` (the reference).

**Commands:**

```bash
cd /home/promptgfm-bio
source .venv/bin/activate
# Put the four jarvis scripts in scripts/ (they resolve _jarvis_env.sh via $(dirname "$0"))
cp /path/to/{_jarvis_env.sh,parity_check-jarvis.sh,run_ablations_extra_seeds-jarvis.sh,run_evaluations_extra_seeds-jarvis.sh} scripts/
chmod +x scripts/*jarvis*.sh

bash scripts/parity_check-jarvis.sh 2>&1 | tee logs/parity_$(date +%Y%m%d).log
```

**What it does:** retrains `ablation_4_full_model` at seed 42 into a *separate* `checkpoints/parity_…` dir (your real seed-42 is never touched), evaluates, and prints a PASS/FAIL table at 0.005 tolerance.

**Interpreting the result:**

- **PASS** (worst delta ≤ 0.005, expected ~0.001–0.003): proceed to Phase 2 and pool all 10 seeds.
- **FAIL** (delta ≥ 0.01): stop. Work the checklist the script prints — torch/PyG/transformers versions vs `requirements.txt`, TF32 disabled, AMP still FP16, deterministic/benchmark flags. Re-run until it passes, or accept reporting the 7 new seeds as a separate line and keeping 42/43/44 as-is.

**Time/cost:** ~1.5 hr → **~₹126**. Pause the instance the moment it finishes if you're not going straight into Phase 2.

---

## 7. Phase 1 — External baselines (the most important remaining work)

This is the gap every reviewer will check first. The plan's own "single most important thing this week" is: get SHEPHERD running on your 117-disease zero-shot split. Everything in this phase runs on a **VM + L4 24GB** (root access for conda/Docker/Java; inference workload).

### 7.0 The fairness design (why this comparison is clean)

Your `PromptEncoder.create_prompt()` already builds prompts from a disease's **HPO phenotype list** (`Phenotypes: <terms>`). That means the phenotype terms your model consumes are derivable from the *same* Orphanet/HPO source that SHEPHERD, Phrank, and LIRICAL require as input. So you can run the **fair Option-A comparison** (everyone gets the same HPO term list) without needing an LLM to extract phenotypes. That's a stronger methodological story than the original plan assumed — state it explicitly in the paper.

Two comparison modes to report:

- **Option A (fair input):** every method receives the disease's curated HPO term list. Tests the ranking model given equivalent input.
- **Option B (end-to-end, optional):** only your model gets free text; baselines get HPO extracted from that text. Tests real-world utility when only a text description exists. Run A first; B is a bonus.

### 7.1 Prepare the shared inputs (once, reused by all baselines)

All HPO-based baselines need, per disease: a list of **HPO term IDs** and a list of **candidate genes**. SHEPHERD specifically needs genes as **Ensembl IDs** (your graph uses gene symbols), so a symbol→Ensembl map is required.

Sources already in your repo:

- ORPHA diseases → HPO terms: `data/raw/orphanet/en_product4.xml`
- OMIM diseases → HPO terms: `data/raw/hpo/phenotype.hpoa` (has OMIM and ORPHA rows)
- HPO term → genes: `data/raw/hpo/phenotype_to_genes.txt`
- Your existing mapping logic: `src/data/hpo_bridge.py`, `src/data/orphadata.py`

Write a small prep script (run it on the VM, in your PromptGFM venv — it's pure pandas, no GPU):

```python
# scripts/prepare_baseline_inputs.py
# Produces, for each of the 117 zero-shot diseases:
#   data/baselines/disease_hpo_terms.json   {disease_id: [HP:xxxxxxx, ...]}
#   data/baselines/disease_true_genes.json  {disease_id: [SYMBOL, ...]}
#   data/baselines/symbol_to_ensembl.json   {SYMBOL: ENSG..., ...}
#   data/baselines/all_candidate_genes.json [ENSG..., ...]   (full 19,576-gene vocab)
import json, pandas as pd, xml.etree.ElementTree as ET
from pathlib import Path

ZS = json.load(open("data/splits/zero_shot_rare_diseases.json"))["disease_ids"]

# 1) OMIM/ORPHA -> HPO from phenotype.hpoa (columns: database_id, hpo_id, ...)
hpoa = pd.read_csv("data/raw/hpo/phenotype.hpoa", sep="\t", comment="#")
# column name is typically 'database_id' and 'hpo_id' (verify header on your file)
dis2hpo = (hpoa.groupby("database_id")["hpo_id"]
                .apply(lambda s: sorted(set(s))).to_dict())

# 2) ORPHA -> HPO from en_product4.xml (fallback / enrichment for ORPHA codes)
#    Parse <Disorder><OrphaCode>..<HPODisorderAssociationList><HPOId>HP:.. 
tree = ET.parse("data/raw/orphanet/en_product4.xml")
orpha2hpo = {}
for dis in tree.iter("Disorder"):
    code = dis.findtext("OrphaCode")
    terms = [h.findtext("HPOId") for h in dis.iter("HPO") if h.findtext("HPOId")]
    if code and terms:
        orpha2hpo[f"ORPHA:{code}"] = sorted(set(terms))

def hpo_for(did):
    if did in dis2hpo: return dis2hpo[did]
    if did in orpha2hpo: return orpha2hpo[did]
    return []

disease_hpo = {d: hpo_for(d) for d in ZS}

# 3) true genes per disease — reuse your existing edge file
edges = pd.read_csv("data/processed/hpo_gene_disease_edges.csv")
edges = edges[edges["score"] >= 0.3]
dis2genes = edges.groupby("disease")["gene"].apply(lambda s: sorted(set(s))).to_dict()
disease_true = {d: dis2genes.get(d, []) for d in ZS}

# 4) symbol -> Ensembl. Use STRING protein.info (has preferred name) or an HGNC map.
#    Simplest robust path: mygene or a static HGNC table. Here, placeholder note:
#    Build symbol_to_ensembl from data/raw/string/9606.protein.info.v12.0.txt if it
#    carries Ensembl, else download HGNC complete set once.
# (left as an explicit step — see 7.1 note below)

Path("data/baselines").mkdir(parents=True, exist_ok=True)
json.dump(disease_hpo,  open("data/baselines/disease_hpo_terms.json","w"), indent=2)
json.dump(disease_true, open("data/baselines/disease_true_genes.json","w"), indent=2)
print("diseases with >=1 HPO term:", sum(1 for v in disease_hpo.values() if v), "/", len(ZS))
```

**7.1 note on Ensembl mapping:** SHEPHERD needs Ensembl gene IDs. The cleanest one-time source is the HGNC complete set (symbol ↔ Ensembl gene ID) or `mygene`. Download once on the VM (`pip install mygene`), map your 19,576 symbols, and cache `symbol_to_ensembl.json`. Genes that don't map (a small fraction) are dropped from SHEPHERD's candidate list and that exclusion is documented in the paper. Phrank/LIRICAL/Exomiser work in HPO+gene-symbol space and don't need this.

**Sanity check before proceeding:** confirm most of the 117 diseases have ≥1 HPO term. If a chunk of OMIM IDs come back empty, the `phenotype.hpoa` header/column names differ from the assumed `database_id`/`hpo_id` — open the file, fix the column names, re-run. Don't proceed with empty phenotype lists; that would unfairly cripple the baselines.

### 7.2 SHEPHERD (the headline baseline) — VM + L4

SHEPHERD uses a **pretrained model on its own rare-disease KG**; you do **not** retrain it. You feed each of your 117 diseases as a "patient" (its HPO terms + candidate Ensembl genes) and read its causal-gene ranking. This is a legitimate zero-shot use of their published model.

**Setup (VM, root, separate conda env — do NOT reuse your torch-2.1 venv):**

```bash
# On the L4 VM
cd /home
git clone https://github.com/mims-harvard/SHEPHERD.git
cd SHEPHERD
# SHEPHERD ships a conda env (its README; verified):
#   conda env create -f environment.yml
#   conda activate shepherd
#   bash install_pyg.sh
# Install miniconda first if absent:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /home/miniconda3
source /home/miniconda3/bin/activate
conda env create -f environment.yml
conda activate shepherd
bash install_pyg.sh
```

**Get their data + pretrained model** from Harvard Dataverse (link in their README; the KG + checkpoints). Place under the path their README specifies, preserving directory structure.

**Build the SHEPHERD input file** — JSON-lines, one disease-as-patient per line, each with the keys their README mandates:

```python
# scripts/make_shepherd_input.py  (run in the PromptGFM venv; writes a .jsonl)
import json
hpo  = json.load(open("data/baselines/disease_hpo_terms.json"))
true = json.load(open("data/baselines/disease_true_genes.json"))
s2e  = json.load(open("data/baselines/symbol_to_ensembl.json"))
cand = json.load(open("data/baselines/all_candidate_genes.json"))  # Ensembl, full vocab

with open("data/baselines/shepherd_patients.jsonl","w") as f:
    for did, terms in hpo.items():
        if not terms:        # skip diseases with no phenotypes (document the count)
            continue
        rec = {
            "id": did,
            "positive_phenotypes": terms,                 # HPO term IDs
            "true_genes": [s2e[g] for g in true.get(did,[]) if g in s2e],
            "all_candidate_genes": cand,                  # full-vocab ranking, fair to your model
        }
        f.write(json.dumps(rec) + "\n")
print("wrote shepherd_patients.jsonl")
```

**Run SHEPHERD's causal-gene-discovery script** (exact entry point per their README — typically a `predict`/`run` script with the patient file and pretrained checkpoint as args). Capture its per-disease ranked gene list (Ensembl) to a JSON. Then map Ensembl ranks back to symbols and feed into your common harness (7.6).

**Realistic expectation:** budget a full focused day for SHEPHERD setup. It targets an older PyTorch/PyG stack, so the conda env + `install_pyg.sh` is the fragile step. If it won't build, the maintainers are responsive on GitHub issues. Pause the VM overnight between sessions.

### 7.3 Phrank — VM + L4 (fast, 1 day)

Phrank (bitbucket.org/bejerano/phrank) is a lightweight pure-Python phenotype-similarity ranker. Install, point it at the HPO DAG + gene-phenotype annotations, and for each disease rank genes by phenotypic match to its HPO term list. No GPU needed; can run in your PromptGFM venv. Output the same `{disease: [ranked gene symbols]}` JSON.

### 7.4 LIRICAL — VM (Java, ~1 day)

LIRICAL is a public Java tool consuming HPO terms. Install the JDK on the VM, download LIRICAL's data, run in phenotype-only mode per disease, capture the ranked gene/disease output. CPU-only.

### 7.5 Trivial LM + LLM baselines (fast, ~2–3 days total)

- **PubMedBERT + cosine:** embed each disease's text/phenotype prompt with frozen PubMedBERT (you already have the encoder), embed each gene's description, rank genes by cosine. Pure inference; run on L4 or even CPU. This is the "static fusion" strawman your architecture beats.
- **GPT/Claude-direct:** for each disease, prompt the LLM (via API) to name likely causal genes (Kim et al. AJHG protocol). Parse to a ranked list. ~$ a few dollars of API; no JarvisLabs GPU.

### 7.6 The common comparison harness (write once)

Every baseline produces `{disease_id: [ranked_gene_symbols]}`. Your model already produces `zero_shot_results.json`. Write one scorer that reads any ranked list + the ground-truth `disease_true_genes.json` and computes **AUROC, Hit@10, Hit@50, MRR** identically across methods, then emits a single comparison table.

```python
# scripts/score_baselines.py — uniform scoring for all methods on the 117 diseases
import json, numpy as np

def metrics(ranked, true_set, n_genes=19576):
    # ranked: list of gene symbols best-first; true_set: set of correct symbols
    ranks = [i+1 for i,g in enumerate(ranked) if g in true_set]
    if not ranks: return dict(auroc=np.nan, hit10=0, hit50=0, mrr=0.0)
    hit10 = int(any(r<=10 for r in ranks)); hit50 = int(any(r<=50 for r in ranks))
    mrr = 1.0/min(ranks)
    # AUROC over full vocab: P(true ranked above random negative)
    best = min(ranks); auroc = 1 - (best-1)/(n_genes-1)
    return dict(auroc=auroc, hit10=hit10, hit50=hit50, mrr=mrr)

# load each method's {disease: ranked_list}, average metrics across diseases,
# print a method x metric table (mean over the 117 diseases).
```

(Use the AUROC definition consistent with your own `evaluate.py` so the comparison is apples-to-apples — read `src/evaluation/metrics.py` and mirror it rather than the sketch above if they differ.)

**Phase 1 time/cost:** ~22 GPU-hours on L4 (mostly idle GPU; you're paying for the instance while you work and pausing between sessions). At ₹41.31/hr that's **~₹900** of active time. Spread over Month 1.

---

## 8. Phase 2 — Statistical rigor (the 28 new runs)

**Product / GPU:** Template + A100 40GB. **Spot** is worth it here (epoch checkpointing makes interruptions cheap); use on-demand if spot capacity is unavailable.

**Precondition:** Phase 0 parity PASS.

**Upload:** repo + graph + the three jarvis scripts (already there from Phase 0 if you didn't destroy the instance) + your existing 12 results for pooling.

### 8.1 Train seeds 45–51 × 4 variants (28 runs)

```bash
cd /home/promptgfm-bio && source .venv/bin/activate
tmux new -s train
bash scripts/run_ablations_extra_seeds-jarvis.sh 2>&1 | tee -a logs/extra_train.log
# Ctrl+B, D to detach. Reattach: tmux attach -t train
```

Skip-if-exists makes this safe to resume after a spot interruption — just re-run the same command. To parallelize across two spot GPUs, run a subset on each: `EXTRA_SEEDS="45 46 47 48" bash …` on one, `EXTRA_SEEDS="49 50 51" bash …` on the other.

**Time:** 28 runs × ~1.33 hr ≈ **37 GPU-hours**.

### 8.2 Evaluate the 28 runs

```bash
bash scripts/run_evaluations_extra_seeds-jarvis.sh 2>&1 | tee -a logs/extra_eval.log
```

Produces `evaluation_results.json` + `zero_shot_results.json` for each new (variant, seed). **Time:** ~6 GPU-hours.

### 8.3 Aggregate to 10 seeds + statistics (CPU, can run anywhere)

Write `scripts/aggregate_10seeds.py` that:

- Pools seeds {42,43,44,45,46,47,48,49,50,51} per variant.
- Computes mean ± std for every metric (standard + zero-shot).
- **Bootstrap 95% CIs** (resample diseases within the 117-disease zero-shot pool, 10k iterations) on AUROC, Hit@K, MRR.
- **Paired tests** (Full vs each variant) with **Holm–Bonferroni** correction across the metric × comparison family.
- **Power analysis**: with n=10 paired observations, state the minimum detectable effect at α=0.05, power=0.8.
- Emits a **LaTeX-ready** Table 1 (the paper's main result).

This needs no GPU — run it on the instance before pausing, or download the JSONs and run locally.

**Phase 2 time/cost:** ~43 GPU-hours. On-demand A100: **~₹3,620**. On spot (~₹44/hr): **~₹1,900**.

---

## 9. Phase 3 — FiLM interpretability

**Product / GPU:** Template + L4 (or stay on A100 if already up). Two sub-parts.

### 9.1 γ/β extraction (free — uses existing code)

`FiLMConditioning.get_film_params()` already exists in `conditioning.py`. Load the full-model checkpoint, pick ~12 diseases spanning categories (neurological / metabolic / skeletal), extract γ and β per FiLM layer, and produce:

- A **pairwise cosine-similarity heatmap** of disease-level γ vectors, clustered by category. If neuro diseases cluster together and apart from metabolic, your "dynamic conditioning" claim is mechanistically supported. If γ/β are disease-generic, you learn that early (and reframe honestly — still publishable, weaker).
- A **top-gene-dimension** readout: which embedding dims get the largest γ-scaling per disease, checked for biological relevance.

Pure inference; a notebook on top of one checkpoint. **~3 GPU-hours.**

### 9.2 Layer-wise ablation (9 training runs)

Train variants with FiLM active at **layer 1 only / layer 2 only / layer 3 only** (3 configs × 3 seeds = 9 runs) to localize where conditioning matters. These are **new configs**, so they don't touch the 10-seed study — make `configs/ablations/film_layer{1,2,3}_only.yaml` derived from `ablation_4_full_model.yaml` with a layer flag, and run them with the same seed-override pattern.

**~12 GPU-hours** on A100 (₹1,010 on-demand).

**Phase 3 total:** ~15 GPU-hours → **~₹1,265** (A100) or less if 9.2 runs on a cheaper card.

---

## 10. Phase 4 — Biological case studies

**Product / GPU:** L4 (inference) + Claude for literature.

`case_study.py` already scaffolds Angelman (UBE3A), Rett (MECP2), Fragile X (FMR1). Extend to **5 diseases** (add 2 ultra-rare from your zero-shot set). For each: top-10 predicted genes with scores, then per gene look up GO annotations + KEGG/Reactome pathway membership (UniProt API). Save one JSON per disease.

Then, in a **separate Claude chat with Research mode on** (Sonnet 4.6 is fine for this), for each disease's top-10 genes, search literature for: prior association with the disease or related conditions, biological function, recent (2023–2026) supporting papers. Produce a structured, cited table. **Pathway coherence** of the top-K predictions (do they cluster in known pathways?) is direct evidence that the GNN walks the PPI graph sensibly — this becomes a paper figure.

**Phase 4 time/cost:** ~10 GPU-hours on L4 → **~₹415**, plus Claude usage.

---

## 11. Phase 5 — Robustness + second dataset (polish, optional)

Strengthens the submission; not a blocker. Do it only if Months 1–4 stay on schedule.

- **Robustness:** paraphrase each disease description (synonyms, reorderings) and re-evaluate to show the model isn't keying on surface lexical features. ~8 runs, ~11 GPU-hours on A100.
- **Second dataset:** add a second evaluation source (DDG2P or ClinVar gene-disease assertions), re-evaluate the best model. ~6 runs, ~8 GPU-hours.

**Phase 5 time/cost:** ~19 GPU-hours → **~₹1,600** (A100). Skippable for the first submission; add in revision if a reviewer asks.

---

## 12. Phase 6 — Writing

Minimal GPU. Use Claude in your Project. Suggested model split: **Sonnet 4.6** for first-pass drafting (fast, good writer), **Opus 4.7** for revision and cross-file consistency.

**Section order** (each reuses the previous):

1. **Methods** first — most mechanical, grounded in `promptgfm.py` / `gnn_backbone.py` / `conditioning.py`. 1.5–2 pages for a journal: full training-loop pseudocode, hyperparameter table.
2. **Results** — the numbers you now have: main 10-seed table, stratified-by-rarity, zero-shot, the 6-method baseline comparison, interpretability, case studies.
3. **Related Work** (new Research-mode chat) — 1.5–2 pages positioning against SHEPHERD, Mantis-ML 2.0, FuseLinker, Kim et al., TEA-GLM, HiGPT, FiLM, GraphSAGE.
4. **Introduction** — written once you know what the paper proves.
5. **Abstract** — last, a compression of the finished paper.
6. **Case-study paragraph** alongside Results; **Conclusion** trivially at the end.

---

## 13. Phase 7 — Adversarial review + submission

**Adversarial review (fresh chat, not your Project, Opus 4.7, Extended Thinking on):** paste the full manuscript and prompt for a harsh Bioinformatics-style review focusing on (1) baseline comparability, (2) statistical justification at your sample sizes, (3) whether case studies are cherry-picked, (4) whether FiLM interpretability supports the claims. Ask for the minimum change to flip a reject to borderline-accept. Fix accordingly.

**Submission:**

- Post a **bioRxiv/arXiv preprint** in October to date-stamp priority (free insurance against being scooped, and it satisfies RECOMB's parallel-submission rule).
- Submit to **Bioinformatics** (rolling) in October.
- Reformat to RECOMB's page limit and submit the abstract by **~7 Nov 2026**, full paper by **~20 Nov 2026**.
- Keep the fallback ladder ready: Genome Medicine → Cell Reports Methods → Bioinformatics Advances → PLOS Computational Biology.

---

## 14. Master timeline (6 months)

| Month | Focus | JarvisLabs | Approx GPU-hrs |
|---|---|---|---|
| **May (now)** | Phase 0 parity (A100), Phase 1 start: SHEPHERD setup + run (L4 VM) | A100 Template (1 day) + L4 VM (rest) | ~25 |
| **June** | Phase 1 finish (Phrank/LIRICAL/Exomiser/LM baselines), Phase 2 (28 seeds + stats) | L4 VM + A100 Template | ~55 |
| **July** | Phase 3 interpretability, Phase 4 case studies | L4 / A100 | ~25 |
| **August** | Phase 5 (robustness + 2nd dataset) + first figures | A100 | ~38 |
| **September** | Writing sprint (Methods→Results→Related→Intro→Abstract) | none (CPU/Claude) | ~2 |
| **October** | Dr. Das review, revisions, bioRxiv preprint, **Bioinformatics submission** | none | 0 |
| **November** | RECOMB reformat + **abstract ~Nov 7, paper ~Nov 20** | none | 0 |
| Dec–Feb | Revision cycle; ISMB 2027 (Jan) as RECOMB fallback | as needed | buffer |

---

## 15. Consolidated budget (current JarvisLabs prices)

| Phase | GPU | Hrs | On-demand cost |
|---|---|---|---|
| 0 — Parity check | A100 40GB | 1.5 | ₹126 |
| 1 — External baselines | L4 24GB | ~22 | ₹909 |
| 2 — 28 seeds + eval | A100 40GB | ~43 | ₹3,622 (₹1,900 on spot) |
| 3 — Interpretability | A100/L4 | ~15 | ₹1,265 |
| 4 — Case studies | L4 24GB | ~10 | ₹413 |
| 5 — Robustness + 2nd dataset (optional) | A100 40GB | ~19 | ₹1,601 |
| Inference / misc / interpretability extras | mixed | ~10 | ₹600 |
| Buffer (failures, reruns, parity FAIL retry) | mixed | ~20 | ₹1,400 |
| **Total (Path B, on-demand)** | | **~140 hrs** | **~₹11,900** |
| **Total with spot for Phase 2** | | | **~₹10,200** |
| **Total skipping Phase 5** | | **~121 hrs** | **~₹10,300** (₹8,600 with spot) |

**Recharge plan:** you have ₹1,000. Phase 0 + the start of Phase 1 fit inside that. **Recharge ~₹11,000–12,000 before Phase 2.** If budget is tight, run Phase 2 on spot (saves ~₹1,700) and defer Phase 5 to revision.

**Discipline that protects the budget:** pause every instance the moment you stop actively using it (`/home` persists; you pay ~₹1.13/hr instead of full rate). The plan's hour estimates assume active GPU time only — wall-clock spans months with the instance paused most of the time.

---

## 16. Risk register & troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Parity check FAILs (delta ≥ 0.01) | torch/PyG/transformers version drift, TF32 on, or BF16 instead of FP16 | Pin to `requirements.txt`; ensure no `allow_tf32=True`; confirm `mixed_precision: true` = FP16. Re-run. If still off, report 7 new seeds separately and keep 42/43/44. |
| `torch-scatter`/`-sparse`/`-cluster` won't install | PyG companion wheels are CUDA-version specific | `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html` (match your CUDA). |
| SHEPHERD `environment.yml` / `install_pyg.sh` fails | Old PyTorch/PyG targets vs instance CUDA | Use a fully isolated conda env (not your venv); if stuck >1 day, open a GitHub issue — maintainers respond. |
| Many OMIM diseases have empty HPO lists | Wrong column names in `phenotype.hpoa` | Open the file, correct `database_id`/`hpo_id` names, re-run prep. Don't proceed with empty lists. |
| SHEPHERD can't rank some genes | Gene not in SHEPHERD's KG / no Ensembl mapping | Document the excluded fraction; report metrics over the intersectable gene set. |
| Spot instance interrupted mid-training | Expected with spot | Re-run the same `run_ablations_extra_seeds-jarvis.sh`; skip-if-exists resumes. Per-epoch checkpoints mean ≤1 lost run. |
| Bus error / `/dev/shm` exhaustion in DataLoader | Known issue from your handovers | Already handled in `train.py` (graph tensors stored once, fork context). If it recurs, lower `num_workers`. |
| Baseline beats you on one metric | Possible and fine | Report honestly; position your strength (text-only zero-shot where others need curated HPO). An honest mixed result is stronger than a cherry-picked sweep. |
| Interpretability shows γ/β are disease-generic | Conditioning may be acting as a prediction-head, not message-pass modulator | Reframe the claim honestly; still publishable as "learned disease-aware prediction." Better to know before submission. |
| Running low on balance mid-phase | — | Pause everything; recharge; resume. Nothing is lost from `/home`. |

---

## 17. One-page command cheat-sheet

```bash
# ───────── INSTANCE BOOTSTRAP (once per fresh instance) ─────────
cd /home && git clone <your-repo> promptgfm-bio   # or rsync from laptop
cd /home/promptgfm-bio
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python scripts/test_gpu.py && python scripts/verify_setup.py
# copy the 4 jarvis scripts into scripts/, chmod +x scripts/*jarvis*.sh

# ───────── PHASE 0: PARITY (A100 Template, on-demand) ─────────
bash scripts/parity_check-jarvis.sh 2>&1 | tee logs/parity_$(date +%Y%m%d).log
#   -> PASS: proceed.  FAIL: fix versions/TF32/AMP, re-run.

# ───────── PHASE 1: BASELINES (L4 VM, root) ─────────
python scripts/prepare_baseline_inputs.py     # HPO + true genes + Ensembl map
python scripts/make_shepherd_input.py         # -> shepherd_patients.jsonl
#   ... clone+conda SHEPHERD, run its causal-gene-discovery, capture rankings
python scripts/score_baselines.py             # uniform comparison table

# ───────── PHASE 2: 28 SEEDS (A100 Template, spot OK) ─────────
tmux new -s train
bash scripts/run_ablations_extra_seeds-jarvis.sh   2>&1 | tee -a logs/extra_train.log
bash scripts/run_evaluations_extra_seeds-jarvis.sh 2>&1 | tee -a logs/extra_eval.log
python scripts/aggregate_10seeds.py            # mean±std, bootstrap CI, Holm-Bonferroni, LaTeX table
#   parallel subset on a 2nd GPU:  EXTRA_SEEDS="49 50 51" bash scripts/run_ablations_extra_seeds-jarvis.sh

# ───────── PHASE 3: INTERPRETABILITY ─────────
#   notebook: load full-model ckpt -> get_film_params() -> gamma/beta heatmap
#   layer-wise: make film_layer{1,2,3}_only.yaml, run with seed-override pattern

# ───────── PHASE 4: CASE STUDIES (L4) ─────────
python scripts/run_case_studies.py            # extend case_study.py to 5 diseases
#   literature: separate Claude chat, Research mode

# ───────── HOUSEKEEPING ─────────
# PAUSE the instance whenever you stop. /home persists; you pay ~storage only.
# Download results to laptop:  rsync -avzP -e "ssh -p <PORT>" user@<HOST>:/home/promptgfm-bio/results ./
```

---

## 18. The single most important next action

Launch an **L4 VM**, set up an **isolated conda env for SHEPHERD**, build the 117-disease input file from your existing Orphanet/HPO files, and produce the first SHEPHERD-vs-PromptGFM-Bio comparison numbers. That one table is what every Q1 reviewer checks first, and it's what turns your next conversation with Dr. Das from "I have ablations" into "I beat (or honestly compare against) the published state of the art." Budget 2–3 focused days and ~₹350 of L4 time, pausing between sessions.

Run the **parity check first** though (A100, ~₹126, ~1.5 hr) — it's cheap insurance that unlocks the entire 10-seed study, and you want it green before you invest the Phase 2 compute.

---

*Prepared for the PromptGFM-Bio project. GPU prices reflect the JarvisLabs console as of 28 May 2026 and may change — verify before recharging. SHEPHERD input contract (JSON-lines with `positive_phenotypes` as HPO terms and `true_genes`/`all_candidate_genes` as Ensembl IDs; conda + install_pyg.sh setup) confirmed from the mims-harvard/SHEPHERD repository.*
