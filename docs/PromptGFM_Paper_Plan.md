# PromptGFM-Bio: Detailed Plan — Paper Creation & Professor Engagement

**Prepared: April 24, 2026**
**Goal:** Convert current results into a submittable paper AND secure professor sponsorship for the next phase of work.

---

## Part 0 — Reality Check: Which "Top Conference"?

Your strategy documents already targeted workshops (LoG, Graph Learning for Drug Discovery, AI for Science) and your final report explicitly flags the main-track gap. Before committing a target, it is worth being honest about what each tier actually requires, because your professor will ask.

| Tier | Example venues | Realistic with current work? | What is missing |
|---|---|---|---|
| **Top-ML main track** | NeurIPS / ICML / ICLR main | No | External baselines (SHEPHERD, Phrank, LIRICAL, Exomiser); larger seed count (≥5); broader ablations (alternate conditioning variants, GAT vs SAGE); possibly a second dataset |
| **Top domain journal** | Nature Machine Intelligence, npj Digital Medicine, Bioinformatics, Cell Reports Methods | **Yes — with baselines added** | External baselines on the same zero-shot split; a clinical-utility framing; 1–2 biological case studies with literature backing |
| **Top bio-ML conference** | RECOMB, ISMB, PSB | **Yes — with baselines added** | External baselines; cleaner reproducibility story |
| **Workshop at top ML venue** | NeurIPS LoG, Graph Learning for Drug Discovery, ICML Comp Bio | **Yes — submittable today** | Nothing blocking; polish only |

Your honest answer to "target venue" is **one of two paths**:

- **Path A (fast, low risk):** Workshop submission at NeurIPS 2026 (deadline ~Aug-Sep 2026) — gets a line on your CV and real reviewer feedback while you build the main-track version.
- **Path B (slow, high reward):** Bioinformatics / npj Digital Medicine — same venue as SHEPHERD, natural fit, but requires external baselines. Timeline: 3–5 months of additional work.

**Recommended framing for the professor meeting:** present Path A as submitted/imminent and Path B as the goal he can help you reach. This lets you ask for his help without appearing to be asking him to rescue a weak submission.

---

## Part 1 — Professor Meeting Prep (This Week)

The meeting is the most important short-term output. Everything you produce this week should serve the meeting, not the eventual paper. Three deliverables, in priority order.

### Deliverable 1 — A 2-page Progress Brief

A single document your professor can read in 10 minutes that answers:

1. What did you build?
2. Does it work?
3. What are the open problems?
4. What specifically do you need from him?

Do not hand him the 12-page final report first — he will not read it. The brief is the door-opener; the full report is the appendix he reads if interested.

Structure:

- **Paragraph 1 (what):** One sentence on the problem (rare disease gene prioritisation from text + PPI graph), one sentence on the architectural contribution (FiLM conditioning of GraphSAGE message passing on frozen PubMedBERT), one sentence on why it matters vs. prior work (SHEPHERD uses structured HPO codes; yours uses free text).
- **Paragraph 2 (results):** The headline numbers from the report — AUROC 0.9606 standard, 0.9413 zero-shot, Hit@50 54.9%, 117-disease zero-shot split. Include one sentence on the popularity-prior finding (MLP reaches 0.94 via popularity but collapses on rare diseases; Full model leads by 16–18% on ultra-rare diseases — this is the real contribution).
- **Paragraph 3 (ablation):** 4 variants × 3 seeds = 12 runs completed. Full > Prompt ≈ GNN > MLP on every metric. Both branches statistically significant at p < 0.05.
- **Paragraph 4 (what is missing):** External baselines (SHEPHERD, Phrank, LIRICAL). Currently comparing against internal ablations only.
- **Paragraph 5 (decision point):** Two paths — workshop submission now or journal-quality submission after external baselines. State which you recommend and why.
- **Paragraph 6 (specific asks):** See §1.3 below.

### Deliverable 2 — The 4-Page Workshop Draft (v0)

Even a rough draft signals progress in a way no memo can. Aim for a full 4-page version by the meeting, even if the related work is thin and the figures are placeholders. Two reasons:

1. It proves the work is real and communicable, not just metrics in a notebook.
2. It anchors the conversation on concrete sections ("the related work needs X", "the figure should show Y") instead of abstract planning.

Sections (see §3 for how to draft each with Claude):
- Introduction + motivation (½ page)
- Related work (½ page)
- Methods (1 page)
- Experiments (1 page: main table + stratified table + zero-shot)
- Discussion + case study (½ page)
- Conclusion (¼ page)

### Deliverable 3 — An "Asks" Document (1 page)

The most important artifact. Be specific. Vague asks get vague help.

Sample asks (adapt to your situation):

- **Co-authorship.** Are you willing to be a co-author on the workshop submission and/or the journal follow-up? If yes, in what order?
- **Compute.** I need to run SHEPHERD and Phrank as external baselines on my zero-shot split. SHEPHERD has publicly released code. Do I have access to lab compute for this, or should I use my current workstation?
- **Clinical validation.** For the journal paper, a 1–2 disease case study validated against a clinician would meaningfully strengthen it. Do you have a collaborator you could introduce me to?
- **Letter of recommendation / positioning.** If I submit to LoG / Graph Learning for Drug Discovery, would you read the draft and provide reviewer-style feedback?
- **Timeline.** If we target Bioinformatics or npj Digital Medicine as the primary venue, what is a realistic timeline given my other commitments?

---

## Part 2 — Paper-Writing Workflow with Claude

The workflow is a function of which Claude product, which model, and which mode you use. Here is the mapping that works for this project.

### Tool & Mode Map

| Task | Interface | Model | Mode | Why |
|---|---|---|---|---|
| Code-grounded methods section | Your Claude Project (has all your `.py` files) | Opus 4.7 | Default | Needs to read `promptgfm.py`, `conditioning.py`, `gnn_backbone.py` for accurate equations and architecture details |
| Related work drafting | New chat (not project) | Sonnet 4.6 | Research ON | Needs web search over recent papers; project knowledge is stale for this |
| Results narrative | Your Claude Project | Opus 4.7 | Default, Extended Thinking ON | Numerical reasoning over your ablation table benefits from extended thinking |
| Abstract + intro | Your Claude Project | Sonnet 4.6 | Default | Sonnet writes tighter prose; Opus tends to over-elaborate for short-form |
| Figures (matplotlib) | Your Claude Project | Opus 4.7 | Default (with code execution) | Can actually run the plotting code in artifacts |
| Adversarial reviewer sim | New chat | Opus 4.7 | Extended Thinking ON | You want the harshest possible reviewer; extended thinking helps it reason about weaknesses |
| Consistency check across sections | Your Claude Project | Opus 4.7 | Default | Cross-reference checks are Opus-favouring |

### Why a Project (Not Normal Chat) for Most of This

Your project already has `promptgfm.py`, `conditioning.py`, `gnn_backbone.py`, the final report, the ablation JSONs, and your handover docs. When Claude writes your methods section, it should cite your actual equations and architecture — not guess. The `project_knowledge_search` tool retrieves these, and the system prompt already tells Claude to prioritise it. This is why the project is worth using.

### Why New Chats for Literature and Adversarial Review

For the related work, you want fresh web search results, not stale cached knowledge. For the adversarial reviewer, you want Claude to forget it is helping you and pretend to be a hostile referee. Project context biases it toward being supportive.

---

## Part 3 — Section-by-Section Drafting Plan

Order matters. Write the sections in the order that lets each one reuse earlier work. This is not the order they appear in the paper.

### Order of Drafting

1. **Methods** first — it is the most mechanical, least ambiguous section. Your code defines it.
2. **Experiments / Results** second — you already have the numbers; this is reporting, not interpretation.
3. **Related work** third — requires literature search, and the framing depends on what you claim in Methods.
4. **Introduction** fourth — must be written after you know what the paper actually says.
5. **Abstract** last — it is a compression of the finished paper.
6. **Discussion / Case study** fits between Results and Conclusion; draft it alongside Results.
7. **Conclusion** last, trivially derived from the rest.

### Specific Prompts for Each Section

Use these in your **Claude Project** unless noted. Replace `[METRIC]` etc. with actual numbers from your JSON files.

#### Methods Section (~1 page)

> Read `promptgfm.py`, `gnn_backbone.py`, `conditioning.py`, and `prompt_encoder.py` in full. Then write a methods section for a 4-page workshop paper covering (1) problem formulation with formal notation for the heterogeneous graph and the prediction task, (2) prompt encoder (frozen PubMedBERT, CLS pooling, cached embedding dimension 768), (3) GraphSAGE backbone (3 layers, 128→512→512, mean aggregation, with the SAGE update equation), (4) FiLM conditioning (show γ and β derivation from prompt, and the `h' = γ ⊙ h + β` equation, noting γ≈1, β≈0 initialisation), (5) predictor head, and (6) training objective (BCE + pairwise ranking + ListNet, with the exact loss weights from `base_config.yaml`). Use LaTeX-ready math. Cite Hamilton 2017 for GraphSAGE and Perez 2018 for FiLM. Do not describe anything the code does not actually do — I have caught errors before when the description drifts from the implementation.

#### Results Section (~1 page)

> I have three evaluation result sets: standard test (`ablation_X_seedY-evaluation_results.json`), stratified-by-rarity (in the final report §8), and zero-shot (`ablation_X_seedY-zero_shot_results.json`). Using the tables and paired t-tests from `PromptGFM_Bio_Final_Project_Report.md`, write a results section organised as: (a) Table 1: overall performance on standard test, mean ± std across 3 seeds. (b) Table 2: stratified Hit@50 by rarity bin with delta column. (c) Table 3: zero-shot results. (d) Narrative paragraph on the popularity-prior phenomenon — MLP reaches 0.94 AUROC on aggregate but collapses on ultra-rare diseases, and the full model's advantage is concentrated precisely where the popularity prior fails. Frame this as the paper's central empirical finding. Report the paired t-test p-values honestly, distinguishing p<0.05 from p<0.10. Do not overclaim.

#### Related Work (~½ page) — Run this in a **new chat with Research mode ON, Sonnet 4.6**

> I am writing a 4-page workshop paper on prompt-conditioned GNNs for rare disease gene prioritisation. My architecture is: frozen PubMedBERT → FiLM → GraphSAGE → MLP. My novelty claim is dynamic conditioning of message passing on free-text disease descriptions. I need a ½-page related work section that positions against: (1) SHEPHERD (Alsentzer et al., npj Digital Medicine 2025) — uses GAT with structured HPO codes, not text. (2) FuseLinker (Xiao et al., JBI 2024) — fuses static LLM embeddings at input layer only. (3) Kim et al. (AJHG 2024) — LLMs as direct gene predictors, no graph. (4) Mantis-ML 2.0 (Middleton et al., Science Advances 2024) — NLP for feature selection, not message-passing conditioning. (5) TEA-GLM (NeurIPS 2024) and HiGPT (KDD 2024) — general-domain LLM+GNN work. Search Semantic Scholar and arXiv for any 2025–2026 papers I may have missed, especially on biorxiv. Draft the related work in four paragraphs following the structure in `novelty_assessment_promptgfm_bio_UPDATED.md` §4. Use my own voice — avoid marketing language like "novel" or "unprecedented".

#### Introduction (~½ page)

> Drafting now because methods, results, and related work are settled. Write a ½-page introduction for a workshop paper. Structure: (¶1) The rare disease diagnostic odyssey — 300M+ patients, 5–7 year average diagnosis time, ~19,576 protein-coding genes to search. (¶2) Why existing methods fall short — popularity bias, no text understanding, no graph reasoning, no zero-shot (from §2.1 of the report). (¶3) Our contribution in three bullets: (i) FiLM-conditioned message passing — the same PPI graph produces disease-specific gene embeddings, (ii) full-vocabulary ranking against 19,576 genes (not small candidate sets), (iii) 117 zero-shot rare diseases, larger than SHEPHERD's set. (¶4) A one-sentence results preview: AUROC 0.9413 on zero-shot, +57% relative Hit@10 over MLP. Keep under 300 words. No fluff.

#### Abstract (~150 words)

> Write a 150-word abstract based on the finished Introduction, Methods, and Results. Structure: (1 sentence) problem, (1 sentence) approach, (1 sentence) what makes it different, (2–3 sentences) headline results with actual numbers, (1 sentence) implication. Do not use the words "novel", "unprecedented", "first-ever", or "revolutionary".

#### Case Study Paragraph (~⅓ page)

> Read `case_study.py`. It implements case studies for Angelman, Rett, and Fragile X. Using the machinery there, pick the disease where (a) the primary causal gene is ranked highest by the full model in our zero-shot set, (b) the top-5 predictions have plausible biological connections (e.g., pathway partners, interactors, or phenotypically related genes), and (c) the result is interesting rather than trivial. Write a 1-paragraph case study with the top-5 predicted genes, a one-sentence biological justification for each, and a one-sentence note on why this validates the model's reasoning over raw memorisation.

---

## Part 4 — Gap Analysis: Workshop vs. Main-Track vs. Journal

What you actually have versus what each path needs:

### ✅ Already Done (Workshop-Ready)

- 4-variant ablation × 3 seeds (12 runs) with statistical tests
- Full-vocabulary ranking (19,576 genes) on 10,267 test queries
- 117-disease clean zero-shot split, zero-shot evaluation complete
- Stratified-by-rarity breakdown (ultra-rare / very-rare / rare / common)
- Engineering fixes documented (6 bugs resolved — shows rigour)
- Reproducible training pipeline, fixed seeds, config files versioned

### ⚠️ Gaps — Required for Workshop Polish

- Related work section written (~½ day with Claude)
- Three case study paragraphs with biology (~1 day)
- Figures: stratified Hit@50 bar chart, zero-shot bar chart, architecture diagram (~1–2 days)
- Adversarial reviewer pass (~½ day)
- Reproducibility appendix (~½ day)

**Workshop path total: ~1 week of focused work.**

### ⚠️ Gaps — Required for Journal / Main-Track

Everything above, plus:

- **External baselines on same zero-shot split.** SHEPHERD: code public at github.com/mims-harvard/SHEPHERD. Phrank and LIRICAL: public code. Exomiser: Docker image available. Time estimate: 2–3 weeks to get all three running on your zero-shot set with proper input adaptation (SHEPHERD expects HPO codes, not text — you will need to map).
- **More seeds.** 3 → 5 (or 10 for a main-track). Adds ~6–18 training runs. Your training takes ~2.6 hrs per run — so ~15–45 hours of compute. Fits on a workstation overnight batch.
- **A second independent dataset.** If Orphanet is your only rare-disease ground truth, reviewers will ask for OMIM-only or Decipher. This is a real data-engineering task — budget 2 weeks.
- **Interpretability analysis.** For a top venue the "FiLM learns something meaningful" claim needs support. Extract γ, β for a few diseases and show they vary in interpretable ways (e.g., different scales on neural vs. metabolic genes for neurological vs. metabolic diseases).
- **Clinical validation case study.** Ideally reviewed by a geneticist. This is where your professor's network matters.

**Journal/main-track total: ~3–5 months of focused work.**

---

## Part 5 — Timeline

### Two-Week Sprint (Pre-Professor Meeting)

| Week | Day | Task | Tool |
|---|---|---|---|
| 1 | Mon | Draft progress brief (2 pages) | Claude Project, Sonnet 4.6 |
| 1 | Tue | Draft Methods section | Claude Project, Opus 4.7 |
| 1 | Wed | Draft Results section + tables | Claude Project, Opus 4.7 |
| 1 | Thu | Generate figures (matplotlib) | Claude Project, Opus 4.7 with code exec |
| 1 | Fri | Related work (new chat, Research mode) | Sonnet 4.6, Research ON |
| 2 | Mon | Introduction + Abstract | Claude Project, Sonnet 4.6 |
| 2 | Tue | Case study paragraph | Claude Project, Opus 4.7 |
| 2 | Wed | Assemble 4-page draft | Claude Project |
| 2 | Thu | Adversarial reviewer pass | New chat, Opus 4.7, Extended Thinking |
| 2 | Fri | Revise based on self-review; prepare "Asks" document | Claude Project |

### Month 1 After Professor Meeting

Contingent on what your professor says. Most likely path:

- Weeks 3–4: Get SHEPHERD running on your zero-shot split.
- Weeks 5–6: Get Phrank and LIRICAL running.
- Weeks 7–8: Re-run analysis with baselines; update tables; rewrite results with comparison.

### Months 2–4 (if pursuing journal path)

- Second dataset integration.
- Additional seeds (3 → 5 or 10).
- Interpretability analysis of FiLM parameters.
- Clinical case study (requires professor's collaborator).
- Full journal-length manuscript.

---

## Part 6 — Concrete Claude Prompts (Ready to Paste)

### Prompt A — Generate the Progress Brief (Day 1)

In your **Claude Project, Opus 4.7, default mode**:

> Using `PromptGFM_Bio_Final_Project_Report.md`, `SESSION_HANDOVER_2026-04-20.md`, and the ablation result JSONs in the project, write a 2-page progress brief for my professor. He is a busy researcher who will read it in 10 minutes. Structure: ¶1 problem and contribution (3 sentences), ¶2 headline results, ¶3 ablation findings with the popularity-prior narrative, ¶4 what is missing for journal-quality submission, ¶5 two paths forward (workshop now vs. journal in 3 months), ¶6 specific asks (compute, clinical collaborator, co-authorship). Write it in my voice — no marketing language, no "exciting results", no "paradigm shift". Plain, precise, slightly understated. End with a single-line proposed next step.

### Prompt B — Generate the Main Figure (Day 4)

In your **Claude Project, Opus 4.7, with code execution**:

> Load all 12 `ablation_*_seed*-evaluation_results.json` files plus the stratified results from the final report. Produce three matplotlib figures: (1) Grouped bar chart of Hit@50 across 4 variants × 4 rarity bins (ultra-rare / very-rare / rare / common). Error bars for ±1 std across 3 seeds. This is the headline figure — it should visually show that the Full model's advantage grows as diseases get rarer. (2) Zero-shot Hit@K curve for K=1,5,10,20,50,100 across 4 variants. (3) Architecture diagram as a TikZ-style SVG I can include in LaTeX. Use a clean, publication-ready style: no 3D, no gradients, Source Sans or similar sans-serif, white background, distinct colour palette (consider viridis or ColorBrewer Set2). Save to `/mnt/user-data/outputs/`.

### Prompt C — Adversarial Reviewer (Day 9)

In a **new chat, Opus 4.7, Extended Thinking ON**. Do *not* use your project — you want a hostile outsider.

> You are a senior reviewer for NeurIPS 2026 Learning on Graphs workshop. You are assigned the attached submission. You have 10 minutes. You are specifically known for being harsh on (1) weak baselines, (2) overclaimed contributions, (3) statistical power issues with small seed counts, (4) unclear novelty relative to prior work. Give me a reviewer form with: Summary (3 sentences), Strengths (3 bullets, be mean about which ones are actually strong), Weaknesses (5 bullets, ranked by severity), Questions for authors (5 bullets), Overall rating 1–10 with confidence 1–5, and a one-paragraph meta-review. Do not be polite. Do not hedge. If you would reject, say so and explain exactly what would flip your vote to borderline-accept. [Paste the 4-page draft here.]

### Prompt D — Related Work (Day 5)

In a **new chat, Sonnet 4.6, Research mode ON**. See §3 above for the full prompt — it is long enough to stand on its own.

### Prompt E — Cross-Section Consistency Check (Day 10)

In your **Claude Project, Opus 4.7, default mode**:

> I have a 4-page draft. Check for consistency issues across sections: (1) Any number (AUROC, Hit@K, seed count, disease count, gene count) that appears in multiple places — verify it matches. Flag any mismatch. (2) Any claim in the abstract that is not supported by a result elsewhere. (3) Any reference to a figure or table by number that does not match the actual figure/table. (4) Any claim of statistical significance that does not match the t-test p-values from the report. (5) The phrase "zero-shot" — every use should refer to the 117-disease split, not the general test set. (6) The word "novel" — flag every occurrence and suggest a more specific phrase. Produce a checklist with line numbers, not a rewrite. [Paste draft.]

---

## Part 7 — Risk Register

What can go wrong and how to mitigate.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Professor wants main-track or nothing | Medium | High | Present both paths in the brief; let him pick. The workshop path is not "lesser" — it builds the main-track version with real reviewer signal. |
| Professor suggests a pivot (different disease focus, different architecture) | Medium | Medium | Come prepared with the full final report. Most pivot requests come from information gaps. |
| External baseline results deflate your numbers | Low | High | Your zero-shot set (117 diseases, text-only) is non-standard. SHEPHERD cannot run directly — it requires HPO codes. You will need to either (a) map text → HPO and run both, or (b) justify why the settings are not directly comparable. Plan this now. |
| 3 seeds is flagged as too few | High | Low | You already acknowledge this in the report. Run 2 more seeds (44, 45 → 3 more per variant = 6 more runs, ~16 hours of compute) before submission. |
| AUPR ≈ 0 flagged as broken | Medium | Low | You already explain this in the report (§7 note). Put the same explanation in the paper's limitations paragraph. |
| Adversarial reviewer finds a real problem | High | Variable | This is the point of running one. Build a response buffer of 2–3 days before submission. |

---

## Bottom Line

Your fastest path to professor buy-in is:

1. Write the 2-page brief this weekend (Prompt A).
2. Draft the 4-page workshop paper in the following 10 days (Prompts B–E).
3. Walk into the meeting with both artifacts plus a focused list of asks.
4. Let him choose the destination — workshop, journal, or both.

You are further along than the uncertainty in your framing suggests. The work is real, the numbers are honest, the story is clean. The missing piece is external baselines, and that is exactly the kind of thing a professor's sponsorship and compute access makes easier. Ask for it directly.
