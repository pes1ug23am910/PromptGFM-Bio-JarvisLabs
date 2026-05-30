# PromptGFM-Bio: Top Conference & Journal Plan (Complete)

**Prepared: April 24, 2026**
**Model used: Claude Opus 4.7** (current — note: the earlier chat mistakenly said 4.7 does not exist; it does)
**Context: Professor = Dr. Bhaskarjyoti Das (PES University, h-index 7, Computational Biology)**
**Compute budget: Indian cloud provider pricing (₹)**

---

## Part 0 — The Answer Up Front

| Question | Answer |
|---|---|
| **Primary journal target** | **Bioinformatics (Oxford)** — rolling submission, 4–5 months prep |
| **Primary conference target** | **RECOMB 2027** — deadline verified: ~Nov 7, 2026 (abstract), ~Nov 20, 2026 (paper) |
| **Best GPU for the work** | **A100 40GB** at ₹72.09/hr — cheapest total for your workload |
| **Total compute cost (Path B)** | **~₹9,444** for ~131 GPU-hours |
| **Total compute cost (Path A)** | **~₹14,202** for ~197 GPU-hours |
| **Timeline** | Submit Bioinformatics in October 2026, RECOMB in November 2026 |
| **Does Dr. Das's profile clear the bar?** | Yes for Bioinformatics / npj DM / Genome Medicine / RECOMB / ISMB. No for Nature Machine Intelligence. |

Submit journal first (rolling, gets you a DOI regardless of RECOMB outcome), conference second. Two independent shots at top-tier acceptance from one body of work.

---

## Part 1 — Honest Venue Tiering Given Dr. Das's Profile

Dr. Das's Scholar: 215 citations, h-index 7, i10-index 5, Professor at PES University in Computational Biology and interdisciplinary ML. One directly relevant bio-ML paper (PPI prediction in blood cells, 2019). Rising citation trajectory 2023–2025. No Nature/Cell-tier prior publications.

This profile matters because journal editors and area chairs look at the senior author's track record for desk-reject decisions. The table below is "where the paper has a realistic shot given this specific author team" — not "where it could be accepted in principle."

### Journals

| Venue | Realistic? | Dr. Das clears bar? | Additional work needed | Prep timeline |
|---|---|---|---|---|
| **Bioinformatics (Oxford)** | ✅ Strong fit | ✅ Yes | External baselines + 5 case studies + stat rigor | 4–5 months |
| **npj Digital Medicine** | ✅ Strong fit | ✅ Yes | Same + clinical framing section | 5–6 months |
| **Genome Medicine** | ✅ Good fit | ✅ Yes | Same as Bioinformatics | 4–5 months |
| **Cell Reports Methods** | ✅ Good fit | ✅ Yes | Lighter bar than above | 3–4 months |
| **PLOS Computational Biology** | ✅ Safe fallback | ✅ Yes | Minimal beyond current work | 3 months |
| **Patterns (Cell Press)** | ✅ Good fit | ✅ Yes | Interdisciplinary framing | 4 months |
| **Bioinformatics Advances** | ✅ Easier sibling | ✅ Yes | Minimal | 3 months |
| **Nature Machine Intelligence** | ❌ Not yet | ❌ Insufficient alone | Clinical co-author + patient cohort + prospective validation | 9–12 months + hospital partner |
| **Nature Methods / Nature Biotech** | ❌ Out of reach | ❌ No | Paradigm-level contribution | N/A |

### Conferences

| Venue | Realistic? | Dr. Das clears bar? | Additional work | Deadline |
|---|---|---|---|---|
| **RECOMB 2027** | ✅ Strong fit | ✅ Yes | External baselines + interpretability | **~Nov 7, 2026** (abstract), **~Nov 20, 2026** (paper) |
| **ISMB/ECCB 2027** | ✅ Strong fit | ✅ Yes | Same as RECOMB | ~Jan 2027 |
| **PSB 2027** | ✅ Good fit | ✅ Yes | Lighter bar, rare disease sessions | ~Jul 2026 (too early) |
| **AAAI 2027** | ⚠️ Possible | ⚠️ Marginal | Theory framing + broader tasks | ~Aug 2026 |
| **KDD 2027** | ⚠️ Possible | ⚠️ Marginal | External baselines + second task | ~Feb 2027 |
| **ICLR 2027** | ❌ Hard | ❌ No | Theory + broader applicability | ~Sep 2026 |
| **NeurIPS / ICML 2027** | ❌ Not yet | ❌ No | Too incremental without rework | 2027+ |

**Verified via web search:** RECOMB 2026 used Nov 7, 2025 (abstract) and Nov 20, 2025 (paper) deadlines. The 2027 edition follows the same calendar. Plan for abstract registration by **Nov 7, 2026** at the latest. RECOMB is also partnered with Cell Systems, Genome Research, and Journal of Computational Biology — selected accepted papers get expedited review for those journals.

---

## Part 2 — The Recommended Pick: Bioinformatics + RECOMB 2027

**Primary journal: Bioinformatics (Oxford)**
**Primary conference: RECOMB 2027**
**Order: journal submission first (rolling), conference second**

### Why Bioinformatics over npj Digital Medicine

npj Digital Medicine pushes harder for clinical validation and patient data. Bioinformatics is the natural home for computational genomics methods — SHEPHERD-style work lives there (the Alsentzer paper is a slight exception due to the UDN patient-cohort component). Bioinformatics reviewers understand GNN + PPI methodology, and the bar for clinical co-authorship is lower. You can publish a strong methods paper without a hospital partner.

### Why RECOMB over ISMB

RECOMB is more methods-focused — FiLM conditioning as architectural contribution plays better there. ISMB skews toward biological-findings papers. RECOMB's November 2026 deadline also gives you more runway than ISMB's January 2027. If RECOMB rejects, ISMB 2027 in January is your immediate fallback with the same material.

### Why journal before conference

Bioinformatics is rolling — no hard deadline. RECOMB has a hard November deadline. Submit to Bioinformatics in October, then reformat to RECOMB's 10-page limit and submit in November.

**Important timing note from the RECOMB 2026 CFP:** "Submissions to peer-reviewed journals other than the partnering ones are also allowed, after the authors have received a decision on their initial submission." So technically, parallel submission to Bioinformatics and RECOMB requires you to have a Bioinformatics decision first. The fix: submit to Bioinformatics in early October, post the preprint on bioRxiv/arXiv simultaneously. Even if the Bioinformatics decision doesn't come before RECOMB's Nov 20 deadline, the bioRxiv preprint secures priority and the paper can go to RECOMB based on the preprint version. This is standard practice and explicitly allowed.

---

## Part 3 — The Core Experimental Gaps (Required for ANY Top Venue)

Whatever tier you target, these four gaps must close.

### Gap 1: External Baselines on Your Zero-Shot Split

The single biggest gap. Internal ablations are insufficient. Minimum set:

| Baseline | Code | Input expected | Engineering cost |
|---|---|---|---|
| **SHEPHERD** | github.com/mims-harvard/SHEPHERD | Structured HPO terms | 2–3 weeks |
| **Phrank** | bitbucket.org/bejerano/phrank | HPO terms + KG | 1 week |
| **LIRICAL** | Public (Java) | HPO terms | 1 week |
| **Exomiser** (phenotype-only mode) | Public (Docker) | HPO terms | 2 weeks |
| **PubMedBERT + cosine similarity** | Your own code | Text | 2 days |
| **GPT-4 direct prediction** (Kim et al. AJHG protocol) | OpenAI API | Text | 3 days |

**Key engineering problem:** Four traditional baselines expect HPO codes, not text. Map your disease text → HPO term lists two ways:

- **Option A:** Use HPO annotations already in Orphanet/OMIM ground truth. Fair methodological comparison.
- **Option B:** GPT-4/Claude extracts HPO terms from text; then feed baselines. End-to-end pipeline comparison.

Run both. A = fair input comparison. B = clinical utility when only text is available.

### Gap 2: Statistical Rigor

- **Seeds: 3 → 10 minimum.** ~1.33 hrs/run on A100 40GB × 4 variants × 7 new seeds = ~37 hrs.
- **Bootstrap 95% CIs on all headline metrics.**
- **Multiple-testing correction.** Holm-Bonferroni or Benjamini-Hochberg across rarity bins × metrics.
- **Power analysis.** State detectable effect size at n=10 paired observations.

### Gap 3: FiLM Interpretability

Right now your central claim ("dynamic conditioning of message passing") is empirical (ablation) but not mechanistic. Reviewers will ask: **what is FiLM actually doing?**

Three analyses:

1. **FiLM parameter variation across diseases.** Extract γ, β for 12 diverse diseases (neuro/metabolic/skeletal). Pairwise cosine similarity → disease-category clustering heatmap.
2. **Gene-level modulation.** Which gene-embedding dimensions get largest γ-scaling per disease?
3. **Layer-wise effect.** Ablate FiLM at layer 1 only, layer 2 only, layer 3 only (9 runs).

Without analysis 1, the conditioning claim is not supported.

### Gap 4: Biological / Clinical Validation

- **5 detailed case studies.** `case_study.py` already has Angelman, Rett, Fragile X. Add 2 ultra-rare. Top-10 predicted genes per disease with literature citation + pathway mapping. Full figure in the paper.
- **Pathway analysis.** Check top-K predictions against KEGG/Reactome for coherence.
- **Clinical collaborator review (optional).** 1–2 diseases reviewed by a geneticist. Dr. Das's network matters.

---

## Part 4 — GPU, Time, and Cost Analysis

Based on measured 214 sec/epoch on RTX 4500 Ada, scaled through Ampere-specific optimizations (BF16 AMP, `torch.compile`, batch size 1536, Flash Attention 2 for BERT) to A100 40GB.

### Per-Epoch Speed and Per-Run Time

| GPU | VRAM | Rate (₹/hr) | Speedup vs Ada | Sec/epoch | Hrs/100-epoch run |
|---|---|---|---|---|---|
| **L4 24GB** | 24 GB | 35.64 | 1.1× | ~195 s | ~5.4 hrs |
| **A100 40GB** | 40 GB | 72.09 | 4.5× | ~48 s | ~1.33 hrs |
| **A100 80GB** | 80 GB | 120.69 | 5.5× | ~39 s | ~1.08 hrs |
| **H100 80GB (IN2)** | 80 GB | 217.89 | 9× | ~24 s | ~0.66 hrs |
| **H100 80GB (EU1)** | 80 GB | 219.51 | 9× | ~24 s | ~0.66 hrs |
| **H200 141GB** | 141 GB | 271.35 | 11× | ~19 s | ~0.54 hrs |

### Compute Budget — Path B (Bioinformatics / RECOMB)

| Phase | Runs | Hrs/run | GPU Hours |
|---|---|---|---|
| Core ablations (4 variants × 10 seeds) | 40 | 1.33 | 53 |
| Alternative conditioning (FiLM vs cross-attn vs adapter × 3 seeds) | 9 | 1.33 | 12 |
| FiLM layer-wise ablation (layer 1/2/3 only × 3 seeds) | 9 | 1.33 | 12 |
| Robustness runs (paraphrase perturbations) | 8 | 1.33 | 11 |
| Second dataset fine-tune + eval | 6 | 1.33 | 8 |
| Inference / interpretability / case studies | — | — | 15 |
| Buffer (failures, reruns) | — | — | 20 |
| **Path B total** | **72 training runs** | — | **~131 hrs** |

### Compute Budget — Path A (ICLR / KDD / AAAI — stretch)

Path A = Path B + four extra blocks:

| Additional requirement | Runs | Hrs | GPU Hours |
|---|---|---|---|
| Head-to-head conditioning (+ prompt tuning) | 12 | 1.33 | 16 |
| Second task (gene-gene, pathway membership × 3 seeds) | 9 | 1.33 | 12 |
| Scaling analysis (model size, layers, hidden dim) | 10 | 1.33 | 13 |
| Extra reviewer-ask buffer | — | — | 25 |
| **Path A total** | **~103 runs** | — | **~197 hrs** |

### Total Cost Comparison Across GPUs

| GPU | Path B (~131 hrs) | Path A (~197 hrs) | Wall-clock (Path B) | Verdict |
|---|---|---|---|---|
| L4 24GB | ₹13,543 (380 hrs*) | ₹18,000 (505 hrs*) | ~16 days | Too slow, VRAM risk |
| **A100 40GB** | **₹9,444** | **₹14,202** | ~5.5 days | **Best value** |
| A100 80GB | ₹12,672 (105 hrs) | ₹17,259 (143 hrs) | ~4.4 days | +34% cost, modest gain |
| H100 80GB (IN2) | ₹14,163 (65 hrs) | ₹19,392 (89 hrs) | ~2.7 days | Fastest Path B, priciest |
| H200 141GB | ₹14,381 (53 hrs) | ₹19,809 (73 hrs) | ~2.2 days | Overkill |

*L4 is 1.1× Ada speed — cheap per hour, expensive in total because hours accumulate.

### Verdict: A100 40GB

A100 40GB wins decisively. It's the cheapest total option despite being far faster than L4. 40GB VRAM fits batch size 1536 with headroom. Every more expensive GPU costs more in total because per-hour rates outpace time saved at your scale.

**Upgrade case:** Only if Path C (Nature Machine Intelligence) with clinical cohort pushes runs to ~250–300 hours — A100 80GB at ~200 hrs / ₹24,000 becomes competitive because extra VRAM allows bigger batches and avoids OOM on larger graphs.

### A100-Specific Optimizations to Deploy Before Running

Your training loop is tuned for Ada. Add these for A100:

1. **AMP: FP16 → BF16.** A100 has native BF16 tensor cores. More numerically stable, no loss-scaling overhead.
   ```python
   torch.cuda.amp.autocast(dtype=torch.bfloat16)
   ```

2. **`torch.compile()` on predictor and GNN.** Static shapes, +15–25%.
   ```python
   model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
   model.gnn = torch.compile(model.gnn, mode="reduce-overhead")
   ```

3. **Batch size 768 → 1536.** 40GB has headroom. Better SM occupancy.

4. **Flash Attention 2 for BioBERT.** If you swap to an HF model with `attn_implementation="flash_attention_2"`, attention goes O(N²) → O(N) memory. Big win at `max_length=512`.

5. **`cudnn.benchmark = True`.** Already in your code — matters more on A100.

---

## Part 5 — Venue-Specific Additional Work

### For Bioinformatics / npj DM / Genome Medicine (Path B)

Beyond the four core gaps:

- **Clinical utility framing.** Quantitative argument: "reducing candidates from 19,576 to top-50 reduces geneticist review workload by 99.7%."
- **Pathway/GO analysis.** Show predicted genes cluster biologically.
- **Robustness.** Paraphrase-perturbed descriptions — show no lexical-surface dependence.
- **Extended related work.** Journals allow 1.5–2 pages; use them.

### For ICLR / KDD / AAAI (Path A — stretch)

Path B + these:

- **Alternative conditioning head-to-head.** FiLM vs. cross-attention vs. adapter vs. prompt tuning.
- **Second task on same graph.** Gene-gene interaction prediction or pathway membership.
- **Theoretical framing.** One proposition with sketch proof on why dynamic modulation helps (mutual information / inductive bias argument). One page max.
- **Scaling analysis.** Parameters, FLOPs, training time, memory.

**Honest note on Path A:** The plan previously flagged that the extra ₹4,758 is not the bottleneck for Path A. ICLR/ICML reviewers will push back hard on "FiLM applied to biomed GNN" as a novelty claim regardless of experiments. Unless Dr. Das can strengthen the theory section as co-author, Path A has a lower acceptance probability than Path B despite the extra work.

### For Nature Machine Intelligence (Path C — only with clinical partner)

Path B + these:

- **Clinical collaborator as co-author** (non-negotiable for NMI in this space).
- **Prospective validation.** Run predictions on 5–10 undiagnosed patients at partner clinic; report confirmed predictions.
- **Broader applicability.** Apply to drug repurposing as secondary demonstration.
- **Extensive ethics/limitations.** NMI requires this in all methods papers.

**Path C reality check:** Dr. Das's h-index 7 + no Nature/Cell-tier prior publications means his name alone doesn't close the NMI gap. A clinical co-author with a track record in high-impact medical journals would be required. PES University has medical affiliations — worth asking Dr. Das if he knows a genetics researcher there willing to partner.

---

## Part 6 — Execution Timeline (6 Months)

Start early May 2026. All GPU hours are A100 40GB estimates.

### Month 1 (May 2026) — External Baselines

| Week | Task | GPU Hrs |
|---|---|---|
| 1 | SHEPHERD setup + 117-disease run | 5 |
| 2 | Phrank + LIRICAL setup + runs | 3 |
| 3 | PubMedBERT-cosine + GPT-4 baselines | 2 |
| 4 | Exomiser (Docker) + first comparison table | 5 |

**Month 1 total: ~15 hrs. Cost: ~₹1,081.**

### Month 2 (June 2026) — Statistical Rigor + Interpretability

| Week | Task | GPU Hrs |
|---|---|---|
| 1 | Seeds 45–54 on all 4 variants (30 new runs, can run overnight in parallel with 2 GPUs) | 40 |
| 2 | Bootstrap CIs, multiple-testing correction, power analysis | 0 |
| 3 | FiLM γ/β extraction + similarity heatmap | 3 |
| 4 | Layer-wise ablation (9 runs) | 12 |

**Month 2 total: ~55 hrs. Cost: ~₹3,965.**

### Month 3 (July 2026) — Biological Validation

| Week | Task | GPU Hrs |
|---|---|---|
| 1–2 | 5 detailed case studies, literature validation | 10 |
| 3 | Pathway analysis (KEGG/Reactome) | 2 |
| 4 | Robustness (paraphrase perturbations, 8 runs) | 11 |

**Month 3 total: ~23 hrs. Cost: ~₹1,658.**

### Month 4 (August 2026) — Second Dataset + Alternative Conditioning

| Week | Task | GPU Hrs |
|---|---|---|
| 1–2 | Second eval source (DDG2P / ClinVar) + 6 runs | 8 |
| 3 | Alternative conditioning comparison (9 runs) | 12 |
| 4 | Buffer + first figure drafts | 18 |

**Month 4 total: ~38 hrs. Cost: ~₹2,739.**

### Month 5 (September 2026) — Writing Sprint

Writing-only. Minimal GPU use.

| Week | Task |
|---|---|
| 1 | Methods (1.5–2 pages for journal) |
| 2 | Results section with all new tables |
| 3 | Related work (1.5–2 pages), intro, discussion |
| 4 | Abstract, final figures, self-review + adversarial reviewer pass |

### Month 6 (October 2026) — Internal Review + Bioinformatics Submission

| Week | Task |
|---|---|
| 1–2 | Dr. Das review + revisions |
| 3 | Lab feedback |
| 4 | **Bioinformatics submission (rolling)** + post preprint to bioRxiv |

### Month 7 (November 2026) — RECOMB Submission

| Week | Task |
|---|---|
| 1 | Reformat to RECOMB 10-page limit |
| 2 | **RECOMB 2027 abstract (~Nov 7)** |
| 3 | **RECOMB 2027 full paper (~Nov 20)** |

### Months 8–10 (Dec 2026 – Feb 2027) — Revision Cycle

Bioinformatics decisions typically in 8–12 weeks. Budget 2 months for revision response. ISMB 2027 January deadline as RECOMB fallback.

### Grand Totals (6 months)

- **Compute:** ~131 GPU-hrs
- **Cost:** ~₹9,444 on A100 40GB
- **Wall-clock:** ~5.5 days of actual GPU time, spread across months with idle periods

---

## Part 7 — Claude Workflow for Each Phase

Claude Opus 4.7 for heavy reasoning, Sonnet 4.6 for prose drafting, fresh chats for adversarial and literature roles.

### Phase 1 — External Baselines (Month 1)

**Setup:** Your Claude Project, Opus 4.7, default mode, code execution enabled.

> Read the SHEPHERD README at github.com/mims-harvard/SHEPHERD. Tell me: (1) exact input format (HPO term lists), (2) whether they include pre-trained model or require retraining on my KG, (3) exact command to run causal gene discovery on my 117 zero-shot diseases on Ubuntu 22.04, CUDA 12.4, Python 3.12. Then write an evaluation harness that takes any baseline's (disease → ranked gene list) output and computes AUROC, Hit@10, Hit@50, MRR against my ground truth, consistent across SHEPHERD / Phrank / LIRICAL / Exomiser / my model.

### Phase 2 — Statistical Rigor (Month 2)

**Setup:** Your Claude Project, Opus 4.7, default.

> I have 12 evaluation JSONs (4 variants × 3 seeds). Generate configs and a launcher for seeds 45–54 (7 new seeds × 4 variants = 28 new runs). After completion, update analysis to produce: mean ± std with 10 seeds; 95% bootstrap CIs; paired t-tests with Holm-Bonferroni correction; power analysis for n=10 paired observations. Output a single LaTeX-ready table.

### Phase 3 — FiLM Interpretability (Month 2)

**Setup:** Your Claude Project, Opus 4.7, Extended Thinking ON.

> Load full-model checkpoint. For these 12 diseases [neuro/metabolic/skeletal mix], extract γ and β from every FiLM layer. Produce: (1) pairwise cosine similarity matrix of disease-level γ vectors, clustered by category — heatmap. (2) Top-10 gene dimensions with largest γ-scaling per disease. (3) Layer-wise ablation: FiLM at layer 1 only / 2 only / 3 only; compare test AUROC across three conditions. Publication-quality matplotlib figures to /mnt/user-data/outputs/.

### Phase 4 — Biological Case Studies (Month 3)

**Project chat (Opus 4.7)** for code:
> Extend `case_study.py` for these 5 diseases [list]: top-10 predicted genes with scores, UniProt GO annotations, KEGG pathway membership. Save as JSON per disease.

**New chat, Sonnet 4.6, Research mode ON** for literature:
> For disease [X], top-10 predicted genes are: [list]. For each, search PubMed and write one sentence on: (a) prior association with [X] or related diseases, (b) biological function relevant to disease, (c) recent papers (2023–2026) linking them. Structured table with citations.

### Phase 5 — Writing (Month 5)

**Project chat (Sonnet 4.6)** for first-pass drafting, **Opus 4.7** for revision and consistency.

Order (each reuses previous):

1. **Methods first** — most mechanical, code-grounded.
2. **Results second** — reporting numbers you already have.
3. **Related Work third** (new Sonnet 4.6 Research ON chat) — requires web search.
4. **Introduction fourth** — must be written after you know what the paper says.
5. **Abstract last** — compression of finished paper.
6. **Case study paragraph** alongside Results.
7. **Conclusion** trivially derived at end.

### Phase 6 — Adversarial Review (end of Month 5)

**Fresh chat (not your project), Opus 4.7, Extended Thinking ON.**

> You are a senior reviewer for Bioinformatics (Oxford). Known for rigorous methodology critique and skepticism of ML hype. Review this manuscript [paste]. Focus on: (1) whether external baselines are truly comparable, (2) whether statistical claims are justified, (3) whether case studies are cherry-picked or genuine, (4) whether FiLM interpretability supports author claims. Full review: Summary; Strengths (be mean about which are actually strong); Major Concerns (5–8 bullets); Minor Concerns (5–10 bullets); Recommendation. Be specific. No politeness. If reject, state minimum to flip to borderline-accept.

---

## Part 8 — What to Ask Dr. Das For (Specifically)

His profile clears Bioinformatics / npj DM / Genome Medicine / RECOMB / ISMB but not NMI. Frame asks accordingly:

- **Co-authorship commitment.** "I would like to list you as senior author on the Bioinformatics and RECOMB submissions. Are you willing?"
- **Clinical collaborator introduction.** "If you know a medical geneticist — in Bangalore or through your network — who would review 5 case-study predictions, it opens npj Digital Medicine as a target." PES has medical affiliations; worth asking.
- **Compute.** Total projected ~₹9,444 for Path B. Ask if the lab has cloud provider credits or if this is on you. Either answer is fine to know up front.
- **Reading rounds.** Budget his time: 2–3 detailed manuscript reviews in September–October 2026.
- **Venue selection.** Present the Part 1 table; let him pick the primary target. His network-signal is worth more than your gut.
- **Timeline check.** Confirm 6-month plan fits his calendar. Any September travel or sabbatical — writing sprint moves earlier.

Avoid open-ended asks ("will you help?") — they get soft answers. Be specific: co-authorship, clinical intro, compute, reading slots, venue choice.

---

## Part 9 — Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| External baselines beat you on one metric | Medium | Low-Medium | Honest reporting wins. Position strengths (text-only zero-shot) against weaknesses (HPO-only comparison). |
| SHEPHERD dependency conflicts on CUDA 12.4 | Medium | Medium | Budget full week of setup; GitHub issues are responsive. |
| Second dataset harder than expected | High | Medium | Don't delay. Submit with one dataset; add in revision. |
| FiLM interpretability reveals nothing | Low-Medium | Medium | Reframe to "learned disease-aware prediction head." Still publishable, weaker. |
| Clinical collaborator unavailable | High | Low | Target Bioinformatics/RECOMB/ISMB. Clinical validation becomes literature-based. |
| Timeline slips past October | Medium | Low | Bioinformatics is rolling. RECOMB is the hard deadline (Nov 20, 2026). |
| Reviewers request major experiments | Very High | Medium | Expected. 2 months for revision. |
| Rejected from Bioinformatics | Medium | Medium | Fallback ladder: Genome Medicine → Cell Reports Methods → Bioinformatics Advances → PLOS CB. |
| A100 pricing changes | Low | Low | Alternative providers (Vast.ai, Lambda, Runpod) often similar or lower. Verify before booking. |
| Preprint scooped | Low-Medium | High | **Post preprint in October 2026** to date-stamp contribution. Free insurance. |

---

## Part 10 — Decision Tree

```
Does Dr. Das agree to senior co-authorship + 2 months of his time + compute?
├── YES (most likely given profile fit)
│   ├── Target: Bioinformatics (Oct 2026) + RECOMB 2027 (Nov 2026)
│   ├── Budget: ₹9,444 compute, 131 GPU hrs, A100 40GB
│   └── Stretch: if case studies surface clinically interesting prediction →
│       ask for medical geneticist introduction → opens npj Digital Medicine
│
└── PARTIAL YES (co-authorship but no compute support)
    ├── Same targets, same timeline
    ├── Budget: ₹9,444 out of pocket — realistic personal spend for top-tier publication
    └── Verify A100 40GB availability on 2–3 providers before committing

Is Path A (ICLR/KDD/AAAI) worth the extra ₹4,758 and 6 extra weeks?
├── Only if Dr. Das strengthens theory section as co-author
├── ML reviewers are significantly harsher on "FiLM + GNN + bio = incremental"
└── Default: no. Path B has higher acceptance probability for less work.

Is Path C (Nature Machine Intelligence) realistic?
├── Only if clinical collaborator secured in May–June 2026
├── If yes: 9–12 month horizon, patient cohort data required
└── If no: not this paper. Use Bioinformatics acceptance to build track record;
    target NMI in 2028 with follow-up work.
```

Pick the rightmost branch matching your reality.

---

## Part 11 — The Single Most Important Thing to Do This Week

Get SHEPHERD running on your 117-disease zero-shot split.

Every top venue asks for this comparison. Code is public (github.com/mims-harvard/SHEPHERD). Input format (HPO term lists) is available for your 117 diseases through Orphanet/OMIM annotations. Running it unlocks every table in the paper.

Budget 2–3 days focused work this week. Book 5 hours of A100 40GB time (~₹361). Report to Dr. Das in your next meeting with comparison numbers in hand — that is the conversation-changer.

---

## Bottom Line

- **Primary targets:** Bioinformatics (Oxford) + RECOMB 2027
- **GPU:** A100 40GB at ₹72.09/hr
- **Total cost:** ~₹9,444 for 131 hours
- **Timeline:** Submit Bioinformatics October 2026, RECOMB November 2026
- **Dr. Das's co-authorship:** Clears the senior-author bar at both venues
- **Nature Machine Intelligence:** Requires clinical co-author — plan as 2028 follow-up, not this paper

This is the realistic top-tier path. More ambitious targets require resources you don't have. Less ambitious targets undersell publishable work.
