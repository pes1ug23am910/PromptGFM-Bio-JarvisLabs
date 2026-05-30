# PromptGFM-Bio: Final Publication Strategy
## Target: NeurIPS/ICML Bio-ML Workshop | Timeline: 3–6 Months

---

## 🔴 WHAT YOU ACTUALLY HAVE (Reality Check)

From your training logs, three critical facts:

| Finding | Detail | Impact |
|---|---|---|
| **No message passing happening** | Graph has only gene↔disease edges; no gene-gene PPI edges → GraphSAGE falls back to MLP | Your "GNN" is currently a fancy lookup table |
| **Evaluation is broken** | Precision@10=1.0, Recall@10=0.0001 simultaneously → physically impossible | Will be instant reject if submitted as-is |
| **Strong baseline despite the above** | AUROC=0.813, AUPR=0.462 with NO graph structure | Suggests the model architecture is sound — adding PPI edges should meaningfully improve it |

**The good news:** You have STRING and BioGRID already downloaded. The story writes itself:  
`No PPI edges (current) → Add PPI edges → Message passing activates → Performance improves → That delta IS your paper.`

---

## PHASE 0 — Fix the Evaluation Bug (Do This First, 3–5 Days)

**This blocks everything. Do not run any new experiments until this is fixed.**

The symptom: Precision@10=1.000 with Recall@10=0.0001 means for each disease query,
you are ranking ~1 positive against a tiny candidate pool (probably just 5 negatives).
Perfect precision is trivial when your pool is that small. Real evaluations rank 1 positive
against ALL genes (~5,251), making Hits@K and Recall@K meaningful.

### Prompt 1 — Paste into your Claude Project (Opus 4.6, Normal mode):

```
Read evaluate.py and metrics.py carefully. Focus on how evaluation batches are constructed.

I have this suspicious result:
- Precision@10: 1.0000
- Recall@10:    0.0001  
- NDCG@10:      1.0000

These are mathematically inconsistent for a real ranking evaluation. 
Precision@10=1.0 with Recall@10=0.0001 means almost no true positives 
are being retrieved, yet precision is perfect — this only happens when 
each query has exactly 1 positive ranked against a very small negative pool.

Specifically investigate:
1. How many candidates are ranked per query? (Should be all ~5,251 genes)
2. How many positives exist per query in the test set?
3. Is there data leakage — are positive edges from train appearing in test?
4. Is the negative sampling at eval time too small (e.g., 5 negatives per positive)?
5. Show me the exact lines in evaluate.py where candidate sets are constructed.

Provide the specific code fix needed to make this a proper ranking evaluation
where each disease is ranked against ALL genes in the vocabulary.
```

---

## PHASE 1 — Enable Real Message Passing (Week 1–2)

This is your paper's central experiment and main contribution.

Your graph currently has:
```
gene ↔ disease edges: 9,741,610   ← used for training labels
gene ↔ gene edges:    0            ← THIS IS THE PROBLEM
```

You have BioGRID and STRING already downloaded at:
- `data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.zip`
- `data/raw/string/9606.protein.links.v12.0.txt.gz`

### Prompt 2 — Paste into your Claude Project (Opus 4.6, Normal mode):

```
Read preprocess.py, preprocess_all.py, and dataset.py carefully.

My graph is built at data/processed/biomedical_graph.pt and currently 
contains ONLY gene-disease bipartite edges. The training logs show:
"No gene-gene edges found in graph. Training without message passing."

I have these files already downloaded:
- data/raw/biogrid/BIOGRID-ALL-4.4.224.tab3.zip  (BioGRID PPI)
- data/raw/string/9606.protein.links.v12.0.txt.gz (STRING PPI, human)

Task: Show me exactly how to add gene-gene PPI edges from STRING 
(use confidence score ≥ 700 as threshold) into the existing 
biomedical_graph.pt so that GraphSAGE can actually perform 
message passing between gene nodes.

Specifically:
1. How are gene nodes currently identified/indexed in the graph?
2. How do STRING protein IDs map to gene nodes in the graph?
3. Write the preprocessing code to add STRING edges with score ≥ 700
4. After adding edges, confirm the graph has ('gene', 'interacts_with', 'gene') edges
5. Flag any ID mapping issues between STRING Ensembl IDs and the current gene vocabulary

Reference the exact variable names and data structures already used in preprocess.py.
```

### What to expect after this fix:
- GraphSAGE will now aggregate over gene-gene neighborhoods
- AUROC should improve from 0.813 — if it improves to 0.84+ that's a publishable delta
- If it does NOT improve, that is also an interesting finding worth reporting

---

## PHASE 2 — Build the Ablation Table (Week 2–3)

This is the minimum experiment table for a workshop paper.

| Model Variant | Gene-Gene Edges | FiLM Conditioning | Expected AUROC |
|---|---|---|---|
| No-MP, No-Prompt (pure MLP baseline) | ❌ | ❌ | ~0.75? |
| No-MP + Prompt (current model) | ❌ | ✅ | **0.813** ✅ have this |
| MP, No-Prompt (GraphSAGE only) | ✅ | ❌ | TBD |
| **Full PromptGFM-Bio** | ✅ | ✅ | TBD |

### Prompt 3 — Paste into your Claude Project (Opus 4.6, Normal mode):

```
Read train.py, finetune.py, promptgfm.py, gnn_backbone.py, and conditioning.py.

I need to run 4 ablation variants. For each, describe exactly what 
config changes or code flags I need to set:

1. Pure MLP baseline: No GNN, no prompt conditioning. Just gene/disease 
   embeddings → MLP → prediction.

2. Prompt-only (current): GNN initialized but no PPI edges → falls back 
   to MLP. FiLM conditioning active. (Already have this: AUROC=0.813)

3. GraphSAGE-only: PPI edges added, GNN active, but conditioning disabled 
   (FiLM scales set to 1, shifts to 0 — no prompt influence).

4. Full model: PPI edges + GraphSAGE + FiLM conditioning active.

For each variant, show me the minimal change to workstation_config.yaml 
or a command-line flag I can pass to train.py. I want to run all 4 with 
the same seed=42, same train/val/test split, 3 random seeds each.
```

---

## PHASE 3 — Zero-Shot Rare Disease Evaluation (Week 3–4)

This is your most publishable claim and what differentiates you from standard GNN papers.

### Prompt 4 — Paste into your Claude Project (Opus 4.6, Normal mode):

```
Read dataset.py and evaluate.py.

I want to add a zero-shot rare disease evaluation. The setup:
1. From the HPO/DisGeNET data, identify diseases with ≤ 5 known gene associations
   (these are "rare diseases" with sparse labels)
2. Hold these diseases OUT of training entirely (they should not appear in 
   train or val splits)
3. At test time, given only the disease name/description as a prompt, 
   rank all genes and compute Hits@10, Hits@50, MRR

Questions:
- How many diseases in our dataset have ≤ 5 known gene associations?
- Are these diseases currently leaking into training?
- Write the code to create a zero_shot_rare_diseases split
- What metric should I report for zero-shot? (MRR, Hits@K?)

Also identify 3 specific well-known rare diseases in our dataset 
(e.g., Progeria, Marfan syndrome, a specific orphan disease) 
that I can use as a qualitative case study — show top-5 predicted 
genes and verify against known biology.
```

---

## PHASE 4 — Literature Positioning (Week 4, Parallel to Above)

Run this in a **NEW chat** with **Research mode ON**, **Sonnet 4.6**:

### Prompt 5 — New Chat, Research Mode:

```
Search Semantic Scholar and arXiv for papers that combine ALL of:
- Graph Neural Networks for gene-disease prediction
- Prompt conditioning or language model integration with GNNs
- Rare disease gene discovery
- Few-shot or zero-shot biological prediction

For each relevant paper found, extract:
1. Venue and year
2. Dataset used (STRING? BioGRID? HPO? DisGeNET?)
3. Best reported AUROC or Hits@K
4. Whether they use dynamic conditioning (cross-attention/FiLM) 
   or just static feature concatenation
5. Whether they handle rare/orphan diseases

Key papers to check and compare against:
- SHEPHERD (rare disease, NeurIPS 2023)
- GIANT (genome-scale GNN)
- OHMNET (multi-layer PPI)
- BioKGBERT or similar LLM+KG hybrid
- Any "graph foundation model" biology papers from 2024-2025

Conclude: What is the closest existing paper to PromptGFM-Bio, 
and how does PromptGFM-Bio differ?
```

Then separately use **OpenRouter → o1 or GPT-4o** with this adversarial prompt:

```
You are a strict NeurIPS 2025 workshop reviewer for the "Graph Learning 
for Drug Discovery and Biology" workshop.

Here is the abstract of a submission:

"We present PromptGFM-Bio, a prompt-conditioned Graph Foundation Model 
for rare-disease gene-phenotype mapping. Unlike prior approaches that 
concatenate text features with graph embeddings, PromptGFM-Bio dynamically 
injects natural-language disease descriptions into GraphSAGE message passing 
via FiLM (Feature-wise Linear Modulation) conditioning on PubMedBERT embeddings. 
Trained on a heterogeneous biomedical graph (STRING PPI + DisGeNET + HPO), 
our model achieves AUROC=0.84 on gene-disease link prediction and demonstrates 
zero-shot generalization to rare diseases with ≤ 5 known gene associations, 
outperforming static-feature baselines by X%."

Rate this 1–5 on: novelty, experimental rigor, biological relevance, clarity.
List your top 3 reasons to accept and top 3 reasons to reject. Be harsh and specific.
What one experiment would change your borderline-reject to accept?
```

(Note: Update the abstract with your actual numbers once Phase 1-2 are done)

---

## PHASE 5 — Paper Writing (Month 3–4)

Target format: **4 pages + references** (standard NeurIPS/ICML workshop)

### Section-by-Section Prompts (Claude Project, Sonnet 4.6):

**Methods section:**
```
Based on the architecture in promptgfm.py, gnn_backbone.py, and conditioning.py,
write a 1.5-page Methods section for a NeurIPS workshop paper.

Include:
1. Problem formulation: heterogeneous graph G = (V_gene, V_disease, V_phenotype, E)
2. Prompt encoder: PubMedBERT → CLS pooling → prompt embedding p ∈ R^768
3. GNN backbone: GraphSAGE, L=3 layers, dim 128→512→512
4. FiLM conditioning: γ, β = MLP(p); h' = γ ⊙ h + β (write the equation)
5. Predictor: concatenate conditioned gene embedding + disease embedding → MLP → score
6. Training objective: binary cross-entropy with 5 negative samples per positive

Use LaTeX-style equations. Be precise. Cite Hamilton et al. 2017 for GraphSAGE 
and the original FiLM paper for the conditioning mechanism.
```

**Results section:**
```
Write a 1-page Results section based on this ablation table:
[paste your actual numbers from Phase 2]

Structure:
1. Main comparison table (4 ablation variants × 3 metrics: AUROC, AUPR, Hits@50)
2. Zero-shot rare disease results (separate table)
3. One qualitative case study paragraph about [specific rare disease]

Highlight: the delta between No-MP+Prompt and Full-Model shows that 
PPI-aware message passing provides X% AUROC gain; the delta between 
Full-Model and No-Prompt shows FiLM conditioning provides Y% gain.
```

---

## PHASE 6 — Target Workshops & Submit (Month 5–6)

### Primary Targets (NeurIPS 2026, deadlines ~August-September 2026):

| Workshop | Fit | Why |
|---|---|---|
| **Learning on Graphs (LoG)** | ⭐⭐⭐ Best fit | Directly covers GNNs + your architecture |
| **AI for Science** | ⭐⭐⭐ Strong | Broad bio-ML, well-attended |
| **Graph Learning for Drug Discovery** | ⭐⭐⭐ Strong | Most directly relevant to gene-disease |
| **New Frontiers in Graph Learning** | ⭐⭐ Good | If you have strong ablations |

### Backup (ICML 2026 workshops, deadlines ~April-May 2026):
- **Computational Biology** workshop at ICML
- **Machine Learning for Genomics** 

---

## YOUR TOOL STACK (Final)

| Task | Tool | Mode | Why |
|---|---|---|---|
| Fix eval bug | Opus 4.6 in Project | Normal | Cross-file code reasoning |
| Add PPI edges to graph | Opus 4.6 in Project | Normal | Complex data pipeline |
| Design ablation variants | Opus 4.6 in Project | Normal | Architecture-aware |
| Literature search | Sonnet 4.6 new chat | Research ON | Web search needed |
| Adversarial reviewer sim | OpenRouter → o1 | Normal | Harshest critic |
| Write paper sections | Sonnet 4.6 in Project | Normal | Faster, good writer |
| Polish & check consistency | Opus 4.6 in Project | Normal | Final quality bar |

---

## THE PAPER'S STORY IN ONE PARAGRAPH

> "Gene-disease prediction with GNNs typically ignores the semantic content of 
> disease phenotypes during message passing. We show that naively concatenating 
> LLM embeddings to graph features (the standard approach) leaves significant 
> performance on the table. PromptGFM-Bio instead dynamically conditions GraphSAGE 
> message passing via FiLM modulation of PubMedBERT disease embeddings, enabling 
> the same biological interaction graph to produce disease-specific gene 
> representations. On a heterogeneous graph of 5,363 genes, 16,841 diseases, and 
> 11,794 phenotypes, PromptGFM-Bio achieves AUROC=X vs. Y for a static-feature 
> baseline, and crucially demonstrates zero-shot generalization to rare diseases 
> with ≤ 5 known gene associations — a setting where traditional supervised 
> methods fail entirely."

---

## IMMEDIATE NEXT 3 ACTIONS (Do Today)

1. **Run Prompt 1** (eval bug fix) in your Claude Project → fix evaluate.py
2. **Run Prompt 2** (add STRING PPI edges) in your Claude Project → rebuild graph
3. **Retrain** the full model with PPI edges and record new AUROC

Everything else follows from those three numbers.
