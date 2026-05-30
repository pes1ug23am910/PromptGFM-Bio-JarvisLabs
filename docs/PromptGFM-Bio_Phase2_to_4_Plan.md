# PromptGFM-Bio — Phase 2 → 4 Plan (everything after Phase 0/1)

**Standalone plan.** Windows 11. Project `E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio`.
Cloud repo `https://github.com/pes1ug23am910/PromptGFM-Bio-JarvisLabs.git`.

Assumes Phase 0 (smoke gate passed) and Phase 1 (baselines) are done per the
earlier runbook. This covers:

| Phase | What | Where | ~Cost |
|---|---|---|---|
| **2** | The 40-run study (10 seeds × 4 ablations) + 40 evals + aggregation | A100 40GB Template | ~₹2,640 spot / ~₹5,060 on-demand |
| **3** | Analysis, case studies, figures, reproducibility bundle | CPU (instance or laptop) | ~₹0–100 |
| **4** | Paper assembly + submission | local | ₹0 + LLM API for proofing |

> **Three rules that protect this whole phase.** (1) **Recharge to ~₹7,000
> (on-demand) or ~₹4,500 (spot) before launching Phase 2** — you cannot run 40
> trains on ₹1,000. (2) **Move the old results/checkpoints aside BEFORE launch**
> (the skip-if-exists trap, §2a) or the runners will silently keep the broken
> pilot runs. (3) **Freeze the configs** — no `torch.compile`, batch 1536, BF16,
> or FlashAttention-2; all 40 runs must be identical except the seed.

---

## PHASE 2 — Main experiments (the 40-run Path 2 study)

This produces your only headline ablation table: 10 seeds (42–51) × 4 variants
(`ablation_1_mlp_only`, `ablation_2_prompt_only`, `ablation_3_gnn_only`,
`ablation_4_full_model`), all on one A100 under the fixed split + seeded RNG.

### 2a. Pre-flight (on the A100 Template, after Phase 0's SMOKE PASS)

The instance is already up from Phase 0 with the env built and data uploaded. If
it was paused, resume it. Then **quarantine the old pilot artifacts** so
skip-if-exists can't preserve broken-seed-flow runs:

```bash
cd /home/promptgfm-bio && source .venv/bin/activate && export WANDB_MODE=offline

# Move the OLD 12-run pilot aside (do NOT delete — keep for the supplementary note)
[ -d results ]     && mv results     results_pilot_OLD_broken_seed_flow_$(date +%Y%m%d)
[ -d checkpoints ] && mv checkpoints checkpoints_pilot_OLD_broken_seed_flow_$(date +%Y%m%d)
mkdir -p results checkpoints logs

# Confirm the Path 2 patches are in the cloned code (must be, since you pushed them)
grep -c "_set_all_seeds" scripts/train.py scripts/evaluate.py     # >=1 each
grep -c "_seed_worker"   scripts/train.py                          # 3  (Patch A)
grep -c "_dump_per_disease\|per_disease_out" scripts/evaluate.py   # >=5 (Patch B)
```

If any grep is 0, the patch isn't in the repo — fix locally, push, `git pull` on
the instance, and re-run the smoke test before spending on the batch.

> **Spot vs on-demand:** use **Spot** for the 40 trains + 40 evals (saves
> ~₹2,000–2,400; `save_interval: 1` + skip-if-exists make interruptions cheap).
> If A100 40GB Spot shows *no capacity* at launch, on-demand is numerically
> identical — don't hunt spot. To switch an instance's pricing you relaunch;
> keep `/home` data via pause, not destroy.

### 2b. The 40 training runs (~53 GPU-hours)

Always run inside **tmux** so a disconnect doesn't kill it:

```bash
tmux new -s train
cd /home/promptgfm-bio && source .venv/bin/activate && export WANDB_MODE=offline
bash scripts/run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%Y%m%d).log
# detach: Ctrl+B then D     reattach: tmux attach -t train
```

Each run writes `checkpoints/<variant>_seed<NN>/best_model.pt`. Skip-if-exists
means a resumed/relaunched instance picks up exactly where it stopped.

**Single A100 for all 40 is the cleanest reproducibility story.** Only split for
speed if you must, and only across **two identical A100 40GB IN2** instances:

```bash
# instance 1:
ALL_SEEDS="42 43 44 45 46" bash scripts/run_all_seeds-jarvis.sh
# instance 2:
ALL_SEEDS="47 48 49 50 51" bash scripts/run_all_seeds-jarvis.sh
```

Monitor: `nvidia-smi` (GPU busy), `tail -f logs/path2_train_*.log`,
`ls checkpoints/*/best_model.pt | wc -l` (counts completed runs toward 40).

### 2c. The 40 evaluations (~7 GPU-hours)

```bash
tmux new -s eval
cd /home/promptgfm-bio && source .venv/bin/activate
bash scripts/run_evaluations_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_eval_$(date +%Y%m%d).log
```

Per (variant, seed) this writes into `results/<variant>_seed<NN>/`:
- `evaluation_results.json` (test split + 4 rarity strata),
- `zero_shot_results.json` (the 117 zero-shot diseases),
- `zero_shot_results_per_disease.json` ← **created automatically by Patch B**,
  which is what unlocks the disease-level bootstrap in the aggregator.

Confirm completeness: `ls results/*/zero_shot_results_per_disease.json | wc -l`
should be **40**. If it's 0, Patch B isn't applied — re-pull and re-run the evals
(cheap) before aggregating, so you don't lose the disease-resampled CIs.

### 2d. (Optional) Few-shot table (paper's Table 3)

Your roadmap lists a few-shot block. `evaluate.py` supports it. **Test on one
config first** (the few-shot path is less exercised than zero-shot):

```bash
python scripts/evaluate.py \
  --config configs/ablations/ablation_4_full_model.yaml \
  --checkpoint checkpoints/ablation_4_full_model_seed42/best_model.pt \
  --few-shot 1 3 5 10 \
  --output results/ablation_4_full_model_seed42/fewshot_results.json
cat results/ablation_4_full_model_seed42/fewshot_results.json   # sanity-check shape
```

If it produces sensible per-K numbers, repeat across the 10 seeds (loop the seed
in the checkpoint/config path). If it errors or looks off, skip few-shot for the
first submission — it's a "required block" in your roadmap but not load-bearing
for the core ablation claim.

### 2e. Aggregate → tables + stats (CPU; ~1 min)

```bash
python scripts/aggregate_results.py \
  --results-root results \
  --out-dir results/aggregate \
  --bootstrap 10000
```

Outputs in `results/aggregate/`:
- `stats_report.txt` — mean±std, paired t + Wilcoxon w/ Holm–Bonferroni, power/MDE,
  seed-level CIs, and **disease-level bootstrap CIs** (Hit@10/50, MRR, macro-AUROC).
- `table_zero_shot.tex`, `table_test.tex` — paste-ready LaTeX with significance markers.
- `aggregate_summary.json` — every number, for figures/scripts.

Read the MANIFEST it prints first — it lists every pooled file so you can confirm
no `*_pilot_OLD_*` directory sneaked in.

### 2f. Freeze + download + pause

```bash
# version the methodology + artifacts together (reviewers may ask)
cd /home/promptgfm-bio
git add -A && git commit -m "Path 2: 40-run results + aggregate" && git tag path2-frozen
git push jarvis main --tags    # NOTE: results/checkpoints are gitignored; this commits code+logs+tables only
```
```powershell
# pull the artifacts to your laptop (from PowerShell):
$H="<HOST>"; $P="<PORT>"; $U="<user>"
$dest="E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio"
scp -P $P -r ${U}@${H}:/home/promptgfm-bio/results/aggregate "$dest\results\"
scp -P $P -r ${U}@${H}:/home/promptgfm-bio/results "$dest\results_cloud_backup\"   # all 40 JSONs + checkpoints if you want them
```

Then **Pause** the A100 (keep `/home`, ~₹1.13/hr) if you'll return, or **Destroy**
after the scp completes.

**Phase 2 cost:** ~₹2,640 spot / ~₹5,060 on-demand (40 train + 40 eval + buffer).

> **What to do with the old 12 runs:** drop them from headline results. Either
> don't mention them, or one supplementary line: "an earlier 3-seed pilot under a
> non-fixed split informed the present 10-seed protocol." Do **not** pool them.

---

## PHASE 3 — Analysis & paper artifacts (CPU / local)

Goal: turn the 40-run outputs + Phase 1 baselines into the paper's tables and
figures, plus a reproducibility bundle.

### 3a. The ablation tables (already produced in §2e)

`table_zero_shot.tex` is your headline. **Lead it with the ranking metrics**
(Hit@10 / Hit@50 / MRR, with disease-bootstrap CIs) and report **macro per-disease
AUROC** from the disease bootstrap rather than only the pooled AUROC — the pooled
number can be dominated by easy diseases, and reviewers in this space expect the
macro view. `table_test.tex` is the standard-split table (Table 1: AUROC, AUPR,
Hit@K, MRR, MAP). The stratified Hit@50 block in `stats_report.txt` is Table 2.

### 3b. Cross-method comparison table (combine with Phase 1)

From Phase 1 you have `data/baselines/comparison_table.md` (PromptGFM-Bio vs
SHEPHERD / Phrank / LIRICAL / PubMedBERT-cosine / LLM-direct, on Hit@10/50/MRR).
If you re-export your model from a Path 2 checkpoint for consistency:

```bash
python scripts/export_promptgfm_rankings.py \
  --config configs/ablations/ablation_4_full_model.yaml \
  --checkpoint results/ablation_4_full_model_seed42/best_model.pt \
  --out data/baselines/rankings/promptgfm.json
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

This is the table every reviewer checks first. It must show your method on top of
the external baselines on the same 117 diseases with identical metric definitions.

### 3c. Case studies (qualitative depth)

```bash
python scripts/case_study.py    # uses model.get_gene_rankings over chosen diseases
```

Pick 2–3 rare diseases where the full model surfaces the true causal gene in the
top-k and a baseline doesn't; show the ranked list. One concrete "this disease,
this gene, rank 3 vs rank 200" example does more for acceptance than another
decimal. (If `case_study.py` needs disease IDs/args, pass diseases from your
zero-shot set.)

### 3d. Figures

Table/figure plan from your roadmap: **Figure 1** = Precision@K / NDCG@K profile;
**Figure 2** = reproducibility matrix (per-seed metric heatmap showing tight
spread). A compact, self-contained plotter is in the **Appendix** below — it reads
the 40 `evaluation_results.json` files directly, so it needs no extra wiring.

### 3e. Reproducibility bundle (the publication checklist)

Assemble, per your checklist.

**Run manifest** — one row per (variant, seed): config, checkpoint, metrics.
Generate it from the result JSONs (this block is at column 0 — paste it as-is):

```bash
python - <<'PY'
import json, glob, csv, re
rows = []
for p in sorted(glob.glob("results/*/evaluation_results.json")):
    m = re.search(r"(ablation_[1-4]_[a-z_]+?)_seed(\d+)", p)
    if not m:
        continue
    t = json.load(open(p)).get("test", {})
    rows.append([m.group(1), m.group(2),
                 f"configs/ablations/{m.group(1)}.yaml",
                 f"checkpoints/{m.group(1)}_seed{m.group(2)}/best_model.pt",
                 round(t.get("auroc", float("nan")), 4),
                 round(t.get("hit_rate@50", float("nan")), 4),
                 round(t.get("mrr", float("nan")), 4)])
with open("results/aggregate/run_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "seed", "config", "checkpoint", "auroc", "hit@50", "mrr"])
    w.writerows(rows)
print(f"wrote run_manifest.csv ({len(rows)} runs)")
PY
```

Also include: the **config files** (already in `configs/ablations/`); the
**reproducibility commands** (the smoke + `run_all_seeds` + `run_evaluations` +
`aggregate_results` invocations from this plan, in the repo README); and a
**code/data availability statement** (repo URL + tag `path2-frozen`; a Zenodo
deposit if the venue wants a DOI).

---

## PHASE 4 — Paper assembly & submission

### 4a. Map results → claims (the four experimental blocks)

1. **Core model comparison** — the ablation table (Full vs MLP/Prompt/GNN), with
   significance. Claim: FiLM conditioning + GNN each contribute, and the full
   model is best with corrected p-values.
2. **Cross-method comparison** — vs SHEPHERD et al. Claim: state-of-the-art on the
   117 zero-shot rare diseases on Hit@K/MRR.
3. **Rarity-aware analysis** — stratified Hit@50. Claim: gains hold (or are largest)
   for the rarest diseases.
4. **Few-shot** (if §2d worked) — graceful improvement with K.

### 4b. Venue & timeline

- **Bioinformatics (Oxford)** — rolling submission, no fixed deadline; submit when
  the four blocks + reproducibility bundle are ready. Good primary target.
- **RECOMB 2027** — conference track; as of mid-2026 the 2027 CFP/dates are not yet
  posted. Watch `recomb.org` and confirm the abstract/full-paper deadlines when the
  CFP appears, then back-plan from there. Don't anchor on a guessed date.
- Practical order: get the manuscript submission-ready for Bioinformatics first;
  the same artifacts feed a RECOMB submission when its CFP opens.

### 4c. Pre-submission rigor pass

- [ ] All headline numbers come from the **Path 2 40-run** aggregate (not the old pilot).
- [ ] Phase 1 baselines (esp. **SHEPHERD**) are in the comparison table — this is the
      single biggest acceptance risk if missing.
- [ ] Methods section states: fixed split (`data.random_seed=42`), 10 seeds 42–51,
      deterministic-given-seed (smoke invariant E), FP16 AMP, batch 768, 100 epochs.
- [ ] CIs reported: seed std on every cell; disease-bootstrap CIs on the zero-shot
      headline; multiple-testing correction (Holm) named.
- [ ] Macro per-disease AUROC reported (not only pooled).
- [ ] Repo public + tagged `path2-frozen`; configs + run manifest included.
- [ ] No data leakage claim is defensible (one split → one valid zero-shot set).

### 4d. What NOT to do at submission time

- Don't re-tune the model "to push the number" — it invalidates the frozen 40-run
  consistency. Performance is fixed; the win is rigor.
- Don't pool old + new seeds. Don't switch numerics (TF32/BF16/compile) for any
  reported run.

---

## Master cheat-sheet

```bash
# ===== PHASE 2 (A100 Template; spot for the batch) =====
cd /home/promptgfm-bio && source .venv/bin/activate && export WANDB_MODE=offline
mv results results_pilot_OLD_$(date +%Y%m%d) 2>/dev/null; mv checkpoints checkpoints_pilot_OLD_$(date +%Y%m%d) 2>/dev/null
mkdir -p results checkpoints logs
grep -c "_seed_worker" scripts/train.py            # 3
grep -c "per_disease_out" scripts/evaluate.py      # >=5
tmux new -s train;  bash scripts/run_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_train_$(date +%F).log   # Ctrl+B D
ls checkpoints/*/best_model.pt | wc -l             # -> 40
tmux new -s eval;   bash scripts/run_evaluations_all_seeds-jarvis.sh 2>&1 | tee -a logs/path2_eval_$(date +%F).log
ls results/*/zero_shot_results_per_disease.json | wc -l   # -> 40 (Patch B)
python scripts/aggregate_results.py --results-root results --out-dir results/aggregate --bootstrap 10000
git add -A && git commit -m "Path 2 results" && git tag path2-frozen && git push jarvis main --tags
# scp results/aggregate to laptop; PAUSE/DESTROY A100

# ===== PHASE 3 (CPU/local) =====
python scripts/export_promptgfm_rankings.py --config configs/ablations/ablation_4_full_model.yaml \
  --checkpoint results/ablation_4_full_model_seed42/best_model.pt --out data/baselines/rankings/promptgfm.json
python scripts/score_baselines.py --truth data/baselines/disease_true_genes.json \
  --method promptgfm=data/baselines/rankings/promptgfm.json \
  --method shepherd=data/baselines/rankings/shepherd_symbols.json --out data/baselines/comparison_table.md
python scripts/case_study.py
python scripts/make_figures.py --results-root results --out-dir results/figures   # appendix script
```

---

## Appendix — `scripts/make_figures.py` (Figure 1 + reproducibility matrix)

Save this as `scripts/make_figures.py`. It reads the 40 `evaluation_results.json`
directly (shapes confirmed) and needs only matplotlib + numpy (both in
`requirements.txt`).

```python
#!/usr/bin/env python3
"""make_figures.py — Figure 1 (Precision@K / NDCG@K profile, full model, mean over
seeds) and a per-seed reproducibility heatmap. Reads results/<variant>_seed<NN>/evaluation_results.json."""
import argparse, glob, json, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [10, 20, 50, 100]
RE = re.compile(r"(ablation_[1-4]_[a-z_]+?)_seed(\d+)")

def load(results_root):
    by = defaultdict(dict)  # variant -> seed -> test-metrics
    for p in glob.glob(f"{results_root}/*/evaluation_results.json"):
        m = RE.search(p)
        if not m:
            continue
        by[m.group(1)][int(m.group(2))] = json.load(open(p)).get("test", {})
    return by

def fig1(by, out):
    full = by.get("ablation_4_full_model", {})
    if not full:
        print("[fig1] no full-model results"); return
    seeds = sorted(full)
    prec = np.array([[full[s].get(f"precision@{k}", np.nan) for k in KS] for s in seeds])
    ndcg = np.array([[full[s].get(f"ndcg@{k}", np.nan) for k in KS] for s in seeds])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(KS, np.nanmean(prec, 0), yerr=np.nanstd(prec, 0), marker="o", label="Precision@K")
    ax.errorbar(KS, np.nanmean(ndcg, 0), yerr=np.nanstd(ndcg, 0), marker="s", label="NDCG@K")
    ax.set_xlabel("K"); ax.set_ylabel("score"); ax.set_xscale("log"); ax.set_xticks(KS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(); ax.set_title("Full model: Precision@K / NDCG@K (mean ± std, 10 seeds)")
    fig.tight_layout(); fig.savefig(f"{out}/fig1_precision_ndcg.pdf"); print("wrote fig1")

def fig2(by, out, metric="auroc"):
    variants = [v for v in ["ablation_1_mlp_only","ablation_2_prompt_only",
                            "ablation_3_gnn_only","ablation_4_full_model"] if v in by]
    seeds = sorted({s for v in variants for s in by[v]})
    M = np.array([[by[v].get(s, {}).get(metric, np.nan) for s in seeds] for v in variants])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([v.replace("ablation_","").replace("_"," ") for v in variants])
    ax.set_xlabel("seed"); ax.set_title(f"Reproducibility matrix ({metric})")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(f"{out}/fig2_repro_matrix.pdf"); print("wrote fig2")

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/figures")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    by = load(a.results_root); fig1(by, a.out_dir); fig2(by, a.out_dir)
```

*Prices reflect your JarvisLabs console (A100 40GB IN2 ₹84.24/hr on-demand;
~₹44/hr spot). The old 12-run pilot is excluded from all headline results.*
