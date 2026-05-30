# Path 2 Deployment — Operational Guide

This supersedes the seed-related sections of the Phase 0+1 guide. The Phase 1
(baseline) work is unchanged. The change is in **Phase 2**, which now produces
a **single internally-consistent 10-seed study under fixed split + properly
seeded model RNG.**

---

## What changed (one paragraph)

The existing seeds 42/43/44 trained under a broken seed flow: the top-level
`seed:` only controlled the data split, while `torch`, `numpy`, and Python
`random` (used in negative sampling) were never seeded — so model init and
sampling were process-time-random and non-reproducible. The new Path 2 patch
(see `PATH2_PATCHES.md`) seeds **all** RNGs from `config['seed']` and moves
the *split* seed to `config['data']['random_seed']`, which stays at 42 for
every run. The old 12 ablation results are **superseded**, not used.

---

## Cost (Path 2, full)

| Step | GPU | Hours | Cost (on-demand) | Cost (spot, ~-48%) |
|---|---|---|---|---|
| Smoke test (one-off) | A100 40GB | ~0.2 | ~₹17 | — |
| 40 training runs (10 seeds × 4 variants × ~1.33 h) | A100 40GB | ~53 | **~₹4,470** | ~₹2,330 |
| 40 evaluations (test + stratified + zero-shot) | A100 40GB | ~7 | ~₹590 | ~₹310 |
| Phase 1 baselines (unchanged) | L4 24GB | ~22 | ~₹900 | — |
| Buffer (failures / 1 spot interruption) | mixed | ~6 | ~₹500 | ~₹300 |
| **Total through Phase 2** | | **~88** | **~₹6,477** | **~₹3,860** |

Recharge target: **₹7,000 before Phase 2** (or ~₹4,500 if committing to spot for training).

---

## The exact command order

### 0. ONCE on Windows (before launching anything)

Apply `PATH2_PATCHES.md` to `scripts\train.py` and `scripts\evaluate.py`, then:

```powershell
cd E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio
Select-String -Path "scripts\train.py" -Pattern "_set_all_seeds|split_seed = config"     # expect 3+ hits
Select-String -Path "scripts\evaluate.py" -Pattern "_set_all_seeds|split_seed = config"  # expect 3+ hits
# Copy the four new shell scripts into scripts\:
#   _jarvis_env.sh, smoke_test_path2-jarvis.sh,
#   run_all_seeds-jarvis.sh, run_evaluations_all_seeds-jarvis.sh
git add scripts\train.py scripts\evaluate.py scripts\*.sh
git commit -m "Path 2: fixed split via data.random_seed; deterministic per-seed model RNG"
git push
```

### 1. Launch the A100 Template (on-demand)

JarvisLabs → Templates → PyTorch → **A100 40GB IN2** → On-Demand → 100 GB → Launch.
SSH key already added (from earlier).

### 2. Bootstrap on the instance

```bash
cd /home
git clone https://github.com/<you>/PromptGFM-Bio.git promptgfm-bio
cd /home/promptgfm-bio
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python scripts/test_gpu.py        # confirm A100 detected
python scripts/verify_setup.py    # confirm torch 2.1 / pyg 2.4 / transformers 4.35
# Upload via JupyterLab drag-drop OR scp (from PowerShell):
#   data/processed/biomedical_graph.pt
#   data/processed/hpo_gene_disease_edges.csv
chmod +x scripts/*jarvis*.sh
```

### 3. Smoke test (~10 min, ~₹17)

```bash
bash scripts/smoke_test_path2-jarvis.sh 2>&1 | tee logs/smoke_path2_$(date +%Y%m%d).log
```

Expected end of output:

```
 >>> SMOKE PASS — Path 2 is wired correctly. Safe to launch run_all_seeds-jarvis.sh.
```

If it says **SMOKE FAIL**, the patch wasn't applied correctly — fix and re-run. Do NOT proceed to step 4 on a fail.

### 4. The 40 training runs (~53 GPU-hours)

```bash
tmux new -s train
bash scripts/run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%Y%m%d).log
# Ctrl+B, D to detach.  tmux attach -t train  to return.
```

Skip-if-exists is safe to resume after any interruption. To split across two GPUs:

```bash
# GPU 1:
ALL_SEEDS="42 43 44 45 46" bash scripts/run_all_seeds-jarvis.sh
# GPU 2:
ALL_SEEDS="47 48 49 50 51" bash scripts/run_all_seeds-jarvis.sh
```

### 5. The 40 evaluations (~7 GPU-hours)

```bash
bash scripts/run_evaluations_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_eval_$(date +%Y%m%d).log
```

Produces both `evaluation_results.json` and `zero_shot_results.json` per (variant, seed).
The 117-disease zero-shot set is valid for all 10 seeds because the split is now fixed.

### 6. Aggregate (CPU, can run on instance or downloaded locally)

Write the 10-seed aggregator that pools all 40 result JSONs, produces mean ± std + bootstrap CI, applies Holm–Bonferroni across the metric × comparison family, and emits the LaTeX-ready main table. (We can write this together once the 40 runs are done — the analysis is downstream of these scripts.)

### 7. Pause / destroy the A100

When step 6 is complete, **pause** the instance (you keep `/home`, pay ~₹1.13/hr storage) if you'll come back, or destroy after downloading results.

### 8. Phase 1 baselines (L4 VM, unchanged from earlier guide)

Continue as in the Phase 0+1 guide (SHEPHERD on a separate L4 VM with its own conda env, then Phrank / LIRICAL / PubMedBERT-cosine / LLM-direct, then `score_baselines.py`). The 117 zero-shot diseases and the disease → true-genes map produced by `prepare_baseline_inputs.py` are unchanged under Path 2.

---

## What about the old 12 ablation runs?

**Drop from headline results.** They were trained under the broken seed flow.
Two acceptable mentions in the paper:

1. **Don't mention them at all** — cleanest. The 10-seed study under Path 2 is your only ablation table.
2. **Cite as preliminary results in supplementary** — "An earlier 3-seed pilot under a non-fixed split informed the present 10-seed protocol." Honest, low-stakes.

Either is fine. Most reviewers prefer (1).

---

## What gets written / commits should include

After the run completes, your repository should contain:

```
scripts/train.py        (patched)
scripts/evaluate.py     (patched)
scripts/_jarvis_env.sh  (new)
scripts/run_all_seeds-jarvis.sh           (new)
scripts/run_evaluations_all_seeds-jarvis.sh (new)
scripts/smoke_test_path2-jarvis.sh        (new)
PATH2_PATCHES.md        (this patch record, in repo root or docs/)
checkpoints/<variant>_seed{42..51}/best_model.pt      (40 checkpoints)
results/<variant>_seed{42..51}/evaluation_results.json (40 std + 40 zero-shot)
logs/path2_*.log
```

Commit + tag this state (e.g. `git tag path2-frozen` after the 40 runs finish), so the methodology and the artifacts are versioned together — reviewers may ask.

---

## Why this maximizes acceptance probability

Two methodological objections that **would** be raised by a careful Q1 reviewer of the previous mechanism are eliminated:

1. **"Why does each seed train on a different data split?"** — gone, the split is fixed.
2. **"How was the zero-shot set verified leak-free across all seeds?"** — gone, one split means one valid zero-shot set.

And one objection that the new code prevents from being raised at all:

3. **"Is your training deterministic given the reported seed?"** — yes, demonstrated by the smoke test.

The cost over the original plan is ~₹1,400 (₹100 smoke + 12 extra training runs to re-do 42/43/44 under the patched code) — negligible compared to the cost of a major-revision cycle over either of those objections.
