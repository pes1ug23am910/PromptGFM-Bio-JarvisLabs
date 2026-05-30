# PromptGFM-Bio: Plan for Top Conferences & Journals

**Prepared: April 24, 2026**
**Goal:** Target venues in the top tier of ML and/or biomedical informatics — not workshops.

---

## Part 0 — Honest Tiering of Venues

Before committing to a target, you need a clear view of the landscape. "Top conference" is not one thing; the fit of your current work varies by an order of magnitude across these venues.

### Tier 1: Top ML Main Conferences

| Venue | Deadline (est.) | Format | Fit for your work |
|---|---|---|---|
| NeurIPS main track | May 2026 (too late for 2026), May 2027 | 9 pages + refs | **Hard.** Reviewers will say FiLM + GNN + bio = incremental. Needs theoretical analysis, broader applicability, or multiple tasks. |
| ICML main track | Jan 2027 | 8 pages + refs | **Hard.** Same as above. |
| ICLR main track | Sep 2026 | 9 pages + refs | **Hard — but slightly easier.** ICLR leans more applied; a strong bio story plays better here than at ICML. |
| KDD | Feb 2027 | 9 pages | **Medium.** Applied ML is welcome. Needs external baselines. |
| AAAI | Aug 2026 | 7 pages | **Medium.** Applied bio-ML fits. Less prestigious than NeurIPS/ICML but still top-tier. |

### Tier 1: Top Biomedical Informatics Journals

| Venue | Timeline (sub → decision) | Fit |
|---|---|---|
| **Nature Machine Intelligence** | 4–8 months | **Stretch goal.** Requires clinical impact story + senior co-authors with track record. |
| **npj Digital Medicine** | 3–6 months | **Strong fit.** SHEPHERD was published here — natural peer comparison. |
| **Bioinformatics (Oxford)** | 3–5 months | **Strong fit.** Established home for computational genomics. |
| **Genome Medicine** | 3–5 months | **Strong fit.** Rare disease + genomics focus. |
| **Cell Reports Methods** | 3–5 months | **Good fit.** Newer, growing venue. |
| **Patterns (Cell Press)** | 3–5 months | **Good fit.** Interdisciplinary data-science. |
| **PLOS Computational Biology** | 3–5 months | Good fit. Lower impact factor but strong community. |

### Tier 1: Top Bio-ML Conferences

| Venue | Deadline | Fit |
|---|---|---|
| **RECOMB 2027** | ~Nov 2026 | **Strong fit.** Top bio-ML conference. |
| **ISMB 2027** | ~Jan 2027 | **Strong fit.** Largest bio-ML conference. |
| **PSB 2027** | ~Jul 2026 | **Good fit.** Smaller but prestigious; rare disease sessions. |

### Recommended Primary Targets

Based on current work + realistic timelines + fit:

1. **Primary journal target: npj Digital Medicine or Bioinformatics.** Directly matches the work; SHEPHERD as peer paper; 6-month realistic timeline.
2. **Primary conference target: RECOMB 2027 (deadline ~Nov 2026) or ISMB 2027 (~Jan 2027).** These are where your peer community actually is.
3. **Stretch goal: ICLR 2027 or Nature Machine Intelligence.** Only if you can close all gaps AND secure a clinical collaborator through your professor.

Do NOT spread yourself across all tiers. Pick one journal + one conference target, with submissions staggered. Journal first (if accepted/rejected you can resubmit); conference as a secondary submission from the same work later.

---

## Part 1 — The Core Experimental Gaps (Required for ALL Top Venues)

Whatever tier you target, these gaps must close. No top venue will accept the work without them.

### Gap 1: External Baselines on Your Zero-Shot Split

This is the single biggest gap. Internal ablations are not enough. Minimum set:

| Baseline | Code | Input expected | Engineering cost |
|---|---|---|---|
| **SHEPHERD** | github.com/mims-harvard/SHEPHERD | Structured HPO terms | 2–3 weeks (requires mapping your text queries to HPO term lists) |
| **Phrank** | bitbucket.org/bejerano/phrank | HPO terms + KG | 1 week |
| **LIRICAL** | Public (Java) | HPO terms | 1 week |
| **Exomiser** (with phenotype-only mode) | Public (Docker) | HPO terms | 2 weeks |
| **PubMedBERT + cosine similarity** (trivial LM baseline) | Your own code | Text | 2 days |
| **GPT-4 direct prediction** (from Kim et al. AJHG 2024 protocol) | OpenAI API | Text | 3 days |

**Key engineering problem:** SHEPHERD, Phrank, LIRICAL, and Exomiser all expect structured HPO phenotype codes as input. Your model accepts free text. For a fair comparison, you need to map your disease text descriptions → HPO term lists. Two options:

- **Option A:** Use the HPO annotations already in your ground truth — each disease in Orphanet/OMIM has associated HPO terms. This is the honest comparison.
- **Option B:** Have GPT-4 or Claude extract HPO terms from your disease text descriptions, then feed those to the baselines. This evaluates the full pipeline (text → HPO → baseline ranking) against your (text → PromptGFM) pipeline.

Run both. Option A shows how you compare given equivalent input; Option B shows end-to-end utility when only text is available.

### Gap 2: Statistical Rigor

- **Seeds: 3 → 5 minimum, 10 ideal.** At 2.6 hours per run × 4 variants × 7 additional seeds = ~73 additional compute hours. One weekend of overnight runs.
- **Bootstrap confidence intervals on all headline metrics.** Your current paired t-tests are good but reviewers will want CIs too.
- **Proper multiple-testing correction.** You have 4 variants × multiple metrics × multiple rarity bins. Apply Holm-Bonferroni or Benjamini-Hochberg on the p-values in Table 9.
- **Power analysis.** State explicitly how large an effect you can detect with your current seed count.

### Gap 3: FiLM Interpretability

Your core novelty claim is "dynamic conditioning of message passing on disease text." Right now you demonstrate this empirically (ablation) but not mechanistically. Reviewers will ask: **what is FiLM actually doing?**

Three analyses to run:

1. **FiLM parameter variation across diseases.** Extract γ, β for a dozen diverse diseases. Compute pairwise cosine similarity. Are neurological diseases similar to each other and different from metabolic? A similarity heatmap is the figure.
2. **Gene-level modulation.** For a given disease, which genes have the largest γ-scaled feature magnitudes? Are those genes biologically relevant to the disease?
3. **Layer-wise effect.** Is the FiLM effect concentrated at layer 1 (input-near) or layer 3 (output-near)? Ablate conditioning at each layer independently.

Without at least analysis 1, the conditioning claim is not scientifically supported.

### Gap 4: Biological / Clinical Validation

- **3–5 detailed case studies.** Your `case_study.py` already scaffolds Angelman, Rett, Fragile X. Extend to include at least two ultra-rare diseases. For each: top-10 predicted genes, literature citation for each plausible prediction, biological pathway mapping. This should be a full figure in the paper.
- **Pathway-level analysis.** Take a disease, look at the top-K predicted genes, and check against known pathways (KEGG, Reactome). The paper claim is that the GNN learns to walk the PPI graph based on the disease — pathway coherence of predictions is direct evidence.
- **Clinical collaborator review.** One or two diseases should have their top-K predictions reviewed by a clinical geneticist for plausibility. Your professor's network matters here.

---

## Part 2 — Venue-Specific Additional Work

### Path A — Target: ICLR / KDD / AAAI Main Track

Additional requirements beyond the core gaps:

- **Alternative conditioning mechanisms compared head-to-head.** FiLM vs. cross-attention vs. adapter layers vs. prompt tuning. Even 3 runs per alternative is enough to establish FiLM as the empirically-strongest choice.
- **Generalization beyond gene prioritization.** A second task on the same graph — for example, gene-gene interaction prediction, disease-disease similarity, or pathway membership. Shows the method is not narrow.
- **Some theoretical framing.** Not a full theorem, but a clean framing of "why dynamic modulation helps" in terms of mutual information, inductive bias, or expressive power. A single proposition with a sketch proof is enough to signal depth.
- **Scaling analysis.** Parameters, FLOPS, training time, memory. Reviewers at ML venues will ask.

Additional compute: ~30–50 more training runs. Budget 6–8 weeks.

### Path B — Target: npj Digital Medicine / Bioinformatics / Genome Medicine

Additional requirements beyond the core gaps:

- **Clinical utility framing.** The paper must argue the case that a diagnostic geneticist would actually use this. Include a quantitative argument — e.g., reducing the candidate list from 19,576 genes to top-50 means a geneticist reviews 50 genes instead of manually sifting through a genome-wide variant call set.
- **Real-world cohort (if accessible).** If your professor has access to a rare-disease patient cohort (even pseudo-anonymized), running PromptGFM-Bio on their cases is a very strong addition. This is the single highest-leverage addition for a clinical journal.
- **Pathway and gene ontology analysis.** Show that predicted genes cluster in biologically coherent ways.
- **Robustness analysis.** Vary the disease description wording (paraphrases, translations, synonyms). Show the model is not relying on surface-level lexical features.
- **Extended related work.** Journals allow longer related work than conferences. Use 1.5–2 pages to position against every relevant method.

Additional compute: ~20–30 more training runs. Budget 4–6 weeks experimentally + 4–6 weeks writing and revision.

### Path C — Target: Nature Machine Intelligence

Everything in Path B, plus:

- **Clinical collaborator as co-author.** NMI rarely publishes pure methodology without clinical grounding. You need a geneticist or genetic counselor reviewing and ideally contributing.
- **Prospective validation.** Even a small one — run predictions on 5–10 previously-undiagnosed patients at a partner clinic and report which predictions were confirmed. This is often the bar for NMI in this space.
- **A broader applicability story.** NMI likes methods that are shown to work across multiple data modalities or tasks. Consider applying to drug repurposing (same graph structure, different labels) as a secondary demonstration.
- **Extensive ethics and limitations discussion.** NMI now requires this in all methods papers.

This path requires 9–12 months and almost certainly requires your professor's active sponsorship to reach feasibility.

---

## Part 3 — Execution Timeline

Assuming you start in May 2026. The timeline is aggressive for npj Digital Medicine / Bioinformatics submission by October/November 2026.

### Month 1 (May): External Baselines Round 1

- Week 1: Get SHEPHERD running. Download code, set up environment on your workstation, map your 117 zero-shot diseases to HPO term lists, run zero-shot evaluation.
- Week 2: Get Phrank and LIRICAL running. Same zero-shot split.
- Week 3: PubMedBERT-only and GPT-4 baselines (cheap).
- Week 4: Exomiser (hardest — use Docker; allocate buffer time). First baseline comparison table.

### Month 2 (June): Statistical Rigor + Interpretability

- Week 1: Run seeds 45–49 on all 4 ablation variants (20 additional runs, ~50 hours of compute). Aim for 5 seeds minimum.
- Week 2: Bootstrap CIs on all metrics. Multiple-testing correction. Power analysis.
- Week 3: FiLM interpretability — γ/β extraction for 12 diseases, similarity heatmap, layer-wise ablation.
- Week 4: Gene-level modulation analysis for 3 case-study diseases.

### Month 3 (July): Biological Validation

- Weeks 1–2: Extended case studies — 5 diseases, top-10 predictions per disease, literature validation for each.
- Week 3: Pathway analysis (KEGG / Reactome) for top-K predictions.
- Week 4: Robustness analysis — paraphrase perturbations on disease descriptions.

### Month 4 (August): Second-Dataset Generalization + Alternative Conditioning

- Weeks 1–2: Integrate a second evaluation source (candidates: DDG2P, ClinVar gene-disease assertions, or Decipher gene list). Re-evaluate best model.
- Week 3: Alternative conditioning comparison — FiLM vs. cross-attention vs. adapter. 3 seeds each for 3 alternatives = 9 runs (~24 hours).
- Week 4: Buffer for unexpected issues + first draft of figures.

### Month 5 (September): Writing Sprint

- Week 1: Full methods section (1.5–2 pages for journal)
- Week 2: Full results section with all new tables
- Week 3: Extended related work, introduction, discussion
- Week 4: Abstract, figures finalized, self-review + adversarial reviewer pass

### Month 6 (October): Internal Review + Submission

- Weeks 1–2: Professor review + revisions.
- Week 3: Internal lab feedback.
- Week 4: Journal submission.

### Months 7–9: Revision Cycle

Journals typically return 2–4 months later with major revisions. Budget 2 months to respond.

---

## Part 4 — Claude Workflow for Each Phase

This is the practical operational layer. For each phase, the right Claude product, model, and mode matters.

### Phase 1: External Baselines

**Setup:** Your Claude Project, Opus 4.7, Default mode. Code execution enabled (for the Docker/Java environment setup).

Prompts to use:

> Download the SHEPHERD repository from github.com/mims-harvard/SHEPHERD. Read the README and setup instructions. Then tell me: (1) what their input format is (I expect HPO term lists), (2) whether they include a pre-trained model or require retraining, (3) the exact command to run their causal gene discovery on my 117 zero-shot diseases. I need to set this up on my workstation (Ubuntu, CUDA 12.4, Python 3.12).

> I have disease text descriptions for my 117 zero-shot rare diseases. I need to map each to a list of HPO phenotype term IDs for the SHEPHERD/Phrank/LIRICAL baselines. Two approaches: (A) use the HPO annotations already present in my Orphanet/OMIM ground truth for these diseases, (B) have an LLM extract HPO terms from the text descriptions. I want to do both for a fair comparison. Write the code for (A) first, using my existing data in `data/processed/`.

> Build a unified evaluation harness that takes any baseline's (disease → ranked gene list) output and computes AUROC, Hit@10, Hit@50, MRR against my ground truth. This needs to work for SHEPHERD, Phrank, LIRICAL, Exomiser, and my model in a consistent way.

### Phase 2: Statistical Rigor

**Setup:** Your Claude Project, Opus 4.7, Default mode.

> I have 12 evaluation JSONs (4 variants × 3 seeds). I need to add seeds 45, 46, 47, 48, 49 — producing 32 total runs. Generate the config files and a bash launcher script. After runs complete, update the existing results analysis code in `scripts/` to produce: (1) mean ± std tables with 5 seeds, (2) 95% bootstrap CIs for each metric, (3) paired t-tests with Holm-Bonferroni correction, (4) a power analysis showing what effect size I can detect with n=5 paired observations.

### Phase 3: FiLM Interpretability

**Setup:** Your Claude Project, Opus 4.7, Extended Thinking ON.

> I want to analyze what FiLM is learning. Load the full model checkpoint. For each of these 12 diseases [list], extract the γ and β vectors from every FiLM layer. Produce: (1) pairwise cosine similarity matrix between disease-level γ vectors, visualized as a heatmap clustered by disease category (neurological, metabolic, skeletal, etc.). (2) For disease-layer pairs, show which gene embedding dimensions get largest γ-scaling. (3) Ablate FiLM at layer 1 only, layer 2 only, layer 3 only; compare test AUROC across these three conditions. Write the code and produce matplotlib figures.

### Phase 4: Biological Case Studies

**Setup:** Your Claude Project, Opus 4.7, Default mode for code, Sonnet 4.6 in a new Research-mode chat for literature lookup on predicted genes.

Project prompt:

> Extend `case_study.py` to produce, for each of these 5 diseases: [list], the top-10 predicted genes with their scores. For each predicted gene, look up (via UniProt API or equivalent) its GO annotations and known pathway memberships. Save as a JSON per disease.

Research-mode prompt (new chat, Sonnet 4.6, Research ON):

> I am writing a case study for a paper on rare disease gene prediction. For the disease [X], the model's top-10 predicted genes are: [list]. For each gene, search PubMed / literature and write one sentence on: (a) whether it has been previously associated with [X] or closely related diseases, (b) its biological function relevant to the disease, (c) any recent papers (2023–2026) linking it. Return a structured table.

### Phase 5: Writing

**Setup:** Your Claude Project, Sonnet 4.6 for first draft, Opus 4.7 for revision.

Use the section-specific prompts from the earlier plan (`PromptGFM_Paper_Plan.md`), expanded for journal length:

- Methods: 1 page → 1.5–2 pages (include full pseudo-code of training loop, detailed hyperparameter table in methods not appendix)
- Results: 1 page → 3–4 pages (one subsection per experiment: main benchmark, stratified, zero-shot, baselines, interpretability, case studies)
- Related Work: 0.5 page → 1.5–2 pages (comprehensive)
- Introduction: 0.5 page → 1 page

### Phase 6: Adversarial Review

**Setup:** Fresh chat (not your project), Opus 4.7, Extended Thinking ON.

> You are a senior reviewer for Bioinformatics / npj Digital Medicine assigned to this manuscript. You are specifically known for rigorous methodology critique and skepticism of ML hype. Review the full paper [paste]. Focus on: (1) whether external baselines are truly comparable, (2) whether statistical claims are justified given sample sizes, (3) whether the biological case studies are post-hoc cherry-picked or genuine evidence, (4) whether the FiLM interpretability actually shows what the authors claim. Write a full journal review: Summary, Strengths (be mean about which are actually strong), Major Concerns (5–8 bullets, each a specific issue that must be addressed), Minor Concerns (5–10 bullets), Recommendation (reject / major revision / minor revision / accept). Do not be polite. Be specific.

---

## Part 5 — What to Ask Your Professor For (Specifically)

Targeting a top venue is hard to do alone. Be explicit about what you need:

- **Co-authorship commitment.** If he is not a co-author, who is? A journal submission without a senior author's backing is weaker.
- **Clinical collaborator introduction.** Minimum: a geneticist or genetic counselor willing to review the top-K predictions on 5 case-study diseases and sign off. Ideal: a collaborator willing to appear as co-author.
- **Compute.** External baselines + extra seeds + interpretability analysis + ablations needs ~200 additional compute hours. Your workstation can do it if you are patient, but lab cluster access would accelerate by 5–10x.
- **Data access.** If any patient cohort data (even summary statistics) is available through the lab or a collaborator, that is the highest-leverage addition for a top-tier journal.
- **Reading and critique.** 2–3 rounds of detailed manuscript feedback over 4 months.
- **Venue selection buy-in.** Ask him to pick between npj Digital Medicine, Bioinformatics, Genome Medicine, and (if ambitious) Nature Machine Intelligence. His network and experience in each matters more than your gut feeling.

---

## Part 6 — Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| External baselines outperform you on some metric | Medium | Low-Medium | Honest reporting; position strengths (text-only zero-shot, where others need HPO) and weaknesses openly. A paper that honestly loses on one metric and wins on another is stronger than one that cherry-picks. |
| SHEPHERD not runnable on your setup (dependency conflicts) | Medium | Medium | Budget 1 full week for setup; if still stuck, contact the authors (they published the code, have maintainer duty). |
| Second dataset harder to acquire than expected | High | Medium | Don't delay submission for this. Submit to Bioinformatics with one dataset; add second in revision if requested. |
| FiLM interpretability reveals nothing interesting | Low-Medium | Medium | If γ/β are disease-generic, your conditioning is not actually learning what you claim. This is worth knowing — it would change the paper's framing from "conditioned message passing" to "learned disease-aware prediction head" (still publishable, weaker story). |
| Clinical collaborator unavailable | High | Medium | Target Bioinformatics / RECOMB / ISMB instead of NMI. Clinical validation becomes "literature-based" rather than "clinician-reviewed". |
| Timeline slips past October | High | Low | Submission windows for npj DM and Bioinformatics are rolling. Aim for October but December is fine. |
| Reviewers request major additional experiments | Very High | Medium | Expected. Budget 2 months for revision. |
| Rejected from first-choice journal | Medium | Medium | Have a ranked fallback list ready: (1) npj DM, (2) Bioinformatics, (3) Genome Medicine, (4) Cell Reports Methods, (5) PLOS CB. Each is a top-tier peer-reviewed venue. |

---

## Part 7 — Decision Tree

```
Does your professor give you compute + clinical collaborator + 8 months?
├── YES → Target Nature Machine Intelligence with Bioinformatics as fallback. Execute all 6 months + clinical validation.
└── NO
    ├── Does he give you compute + 6 months (no clinical collab)?
    │   ├── YES → Target Bioinformatics or npj Digital Medicine. Execute 6-month plan.
    │   └── NO
    │       ├── Does he give you 3 months of his time (reading + compute access)?
    │       │   ├── YES → Target RECOMB/ISMB conference deadline. Compressed plan: external baselines (1 mo) + stat rigor (2 wk) + interpretability (2 wk) + writing (1 mo).
    │       │   └── NO → Workshop submission (fallback from earlier plan). Still a publication and a credential.
```

Pick the rightmost branch that matches your reality, not the leftmost you wish were true. Your professor's answer to the resources question will set the ceiling.

---

## Bottom Line

Target: **Bioinformatics or npj Digital Medicine in ~6 months** is the realistic top-venue path.

The three non-negotiables are:

1. **External baselines on your zero-shot split** (SHEPHERD + Phrank + LIRICAL + a trivial LM baseline).
2. **Interpretability analysis of FiLM** that either supports or refutes your central claim.
3. **5+ biological case studies with literature validation.**

Everything else — extra seeds, alternative conditioning, theoretical framing, second dataset — is polish. It strengthens the submission but is not a blocker.

**Reach goal:** Nature Machine Intelligence, with clinical collaborator added through your professor. 9–12 months. Only realistic if he commits seriously.

**Floor:** RECOMB 2027 or ISMB 2027. Both are top bio-ML conferences; both are realistic in 5–6 months from today.

The single most important thing you can do this month is secure an external baseline comparison against SHEPHERD. Everything else flows from there, and every top venue will ask for it.
