# PromptGFM-Bio: Final Publication Strategy (UPDATED)
## Target: NeurIPS/ICML Bio-ML Workshop | Timeline: 3–6 Months

**Last Updated:** April 2026
**Status:** SHEPHERD resolved. All critical papers reviewed. Ablation runs in progress.

---

## 🟢 WHAT YOU ACTUALLY HAVE (Current Status)

| Finding | Detail | Status |
|---|---|---|
| **Message passing active** | 1,854,012 STRING PPI edges added (≥700 confidence) | ✅ Fixed |
| **Evaluation fixed** | Full-vocabulary ranking: all 19,576 genes per query | ✅ Fixed |
| **Current best results** | AUROC=0.9626, Hit Rate@50=55.7% on 10,267 test queries | ✅ Strong |
| **FiLM conditioning working** | Dynamic γ,β modulation confirmed in training | ✅ Verified |
| **Zero-shot diseases identified** | 117 clean zero-shot rare diseases | ✅ Ready |
| **Ablation design** | 4 variants × 3 seeds = 12 runs | 🔄 Running (~April 6 evening) |
| **Novelty assessment** | All 6 critical papers reviewed and confirmed safe | ✅ Complete |
| **SHEPHERD resolved** | Alsentzer et al., npj Digital Medicine 2025 — no FiLM, no text conditioning | ✅ Safe |

### Previous Issues (All Resolved)
- ~~No PPI edges → GraphSAGE fell back to MLP~~ → Added 1,854,012 STRING edges
- ~~Evaluation broken (Precision@10=1.0 with Recall@10=0.0001)~~ → Full-vocabulary ranking implemented
- ~~SHEPHERD unknown~~ → Read in full, confirmed no dynamic text conditioning

---

## PHASE 0 — Ablation Runs (In Progress, Completes ~April 6)

### 4 Ablation Variants

| Config | use_gnn | use_conditioning | Gene-Gene Edges | What It Tests |
|---|---|---|---|---|
| **ablation_1_mlp_only** | ❌ | ❌ | ❌ | Pure MLP baseline |
| **ablation_2_prompt_only** | ❌ | ✅ | ❌ | FiLM without message passing |
| **ablation_3_gnn_only** | ✅ | ❌ | ✅ | GraphSAGE without FiLM |
| **ablation_4_full_model** | ✅ | ✅ | ✅ | Full PromptGFM-Bio |

Each variant: 3 random seeds, 100 epochs, RTX 4090, ~93.5 sec/epoch.
Total: 12 runs ≈ 31 hours.

### Key Numbers to Watch
- **MLP baseline AUROC:** Early results suggest ~0.955. If this is close to the full model, the paper's story shifts from "FiLM dramatically improves performance" to "FiLM enables zero-shot generalization that MLP cannot achieve"
- **GNN-only vs Full Model delta:** This isolates FiLM's contribution
- **Prompt-only vs Full Model delta:** This isolates PPI message passing's contribution

---

## PHASE 1 — Evaluate All Ablation Models (April 7)

### Prompt for Claude Project (Opus 4.6):

```
Read evaluate.py and the ablation config files in configs/.

I have 12 trained models (4 configs × 3 seeds) with checkpoints in:
- checkpoints/ablation_1_mlp_only/seed_{42,123,456}/
- checkpoints/ablation_2_prompt_only/seed_{42,123,456}/
- checkpoints/ablation_3_gnn_only/seed_{42,123,456}/
- checkpoints/ablation_4_full_model/seed_{42,123,456}/

For each model, run full evaluation computing:
1. AUROC, AUPR on standard test set
2. Hit Rate@10, @50, @100
3. MRR (Mean Reciprocal Rank)
4. NDCG@100

Also run zero-shot evaluation on the 117 rare diseases in 
data/splits/zero_shot_rare_diseases.json using the --zero_shot flag.

Output a single comparison table with mean ± std across 3 seeds.
```

---

## PHASE 2 — Zero-Shot Evaluation (April 7)

### Prompt for Claude Project (Opus 4.6):

```
Read evaluate.py and find_rare_diseases.py.

I have 117 zero-shot rare diseases identified. For each ablation variant,
evaluate zero-shot performance:
1. These diseases had ZERO training edges
2. The model receives only the disease name/description as text input
3. Rank all 19,576 genes and compute Hit Rate@10, @50, MRR

Critical question: Can the MLP baseline (ablation_1) perform zero-shot at all?
It should NOT be able to — it has no mechanism to handle unseen disease IDs.
This is where FiLM conditioning should shine.

Also identify 3 specific rare diseases from the 117 for qualitative case studies.
For each, show top-5 predicted genes and verify against Orphanet/OMIM.
```

---

## PHASE 3 — Literature Positioning (April 8–10, Parallel)

### Updated Related Work Structure (Based on Completed Paper Reviews)

**Paragraph 1 — Static graph methods for gene prioritization:**
- Mantis-ML 2.0 (Middleton et al., Science Advances 2024): SGC + NLP feature selection on BIKG, median AUC 0.90 across 5,220 diseases. NLP selects features; doesn't condition message passing.
- KDGene, MGREL, KGE-based approaches: static, disease-agnostic gene embeddings
- Yuan et al. (2022): evaluation benchmark documenting that most methods use small candidate sets

**Paragraph 2 — LLMs for gene prioritization:**
- Kim et al. (AJHG 2024): GPT-4 at 17% top-50 accuracy, still lags behind Phen2Gene (55.3%). Strong bias toward highly-cited genes. Explicitly documents the gap PromptGFM-Bio fills.
- FuseLinker (Xiao et al., JBI 2024): Static LLM embeddings fused at input layer via weighted average → disease-agnostic MP. This is the "feature concatenation" baseline we argue against.

**Paragraph 3 — LLM-GNN integration (general domain):**
- TEA-GLM (Wang et al., NeurIPS 2024): GNN→LLM direction (opposite from us)
- HiGPT (Tang et al., KDD 2024): Instruction tuning for heterogeneous graphs, not bio
- FiLM (Perez et al., 2018): Applied in visual reasoning, not gene prioritization

**Paragraph 4 — SHEPHERD (closest predecessor):**
- Alsentzer et al. (npj Digital Medicine 2025): GAT over biomedical KG for rare disease diagnosis
- Uses structured HPO codes as node lookups, NOT free-text conditioning
- Evaluates on small candidate lists (13–244 genes), not full vocabulary
- PromptGFM-Bio extends by: (1) free-text input, (2) FiLM conditioning of every MP layer, (3) full-vocabulary ranking, (4) zero-shot on text-described diseases

### Key References Table

| Paper | Venue | Year | Role in Your Paper |
|---|---|---|---|
| Alsentzer et al. (SHEPHERD) | npj Digital Medicine | 2025 | Closest predecessor — differentiate carefully |
| Kim et al. | AJHG | 2024 | Motivates need for graph structure with text |
| Middleton et al. (Mantis-ML 2.0) | Science Advances | 2024 | Strongest graph baseline — position as complementary |
| Xiao et al. (FuseLinker) | JBI | 2024 | Static fusion baseline your architecture improves upon |
| Wang et al. (TEA-GLM) | NeurIPS | 2024 | LLM-GNN integration (opposite direction) |
| Tang et al. (HiGPT) | KDD | 2024 | Heterogeneous graph LM (general domain) |
| Yuan et al. | Brief. Bioinform. | 2022 | Evaluation protocol reference |
| Perez et al. (FiLM) | AAAI | 2018 | Original FiLM conditioning paper |
| Hamilton et al. (GraphSAGE) | NeurIPS | 2017 | GNN backbone |

---

## PHASE 4 — Paper Writing (April 8–14)

Target format: **4 pages + references** (standard NeurIPS/ICML workshop)

### The Paper's Story (Updated)

> "Gene-disease prediction with GNNs typically treats every disease query identically during 
> message passing — the language signal enters only as static input features (FuseLinker) or 
> at the prediction head, not during graph reasoning itself. Meanwhile, LLMs alone achieve 
> only 17% accuracy at gene prioritization (Kim et al., AJHG 2024), and even the closest 
> graph-based approach, SHEPHERD, requires structured HPO phenotype codes rather than 
> free-text descriptions. PromptGFM-Bio bridges this gap by dynamically conditioning 
> GraphSAGE message passing via FiLM modulation of frozen PubMedBERT disease embeddings, 
> enabling the same PPI network to produce disease-specific gene representations. On a 
> heterogeneous graph of 19,576 genes and 16,841 diseases, PromptGFM-Bio achieves 
> AUROC=0.9626 and demonstrates zero-shot generalization to 117 rare diseases with no 
> training associations — a setting where static methods fail entirely."

### Methods Section Prompt:

```
Based on the architecture in promptgfm.py, gnn_backbone.py, and conditioning.py,
write a 1.5-page Methods section for a NeurIPS workshop paper.

Include:
1. Problem formulation: heterogeneous graph G = (V_gene, V_disease, V_phenotype, E)
   with 19,576 genes, 16,841 diseases, 11,794 phenotypes, 1.85M PPI edges
2. Prompt encoder: frozen PubMedBERT → CLS pooling → prompt embedding p ∈ R^768
   (pre-computed and cached; BERT never called per-batch)
3. GNN backbone: GraphSAGE, L=3 layers, dim 128→512→512, on STRING PPI edges
4. FiLM conditioning: γ, β = MLP(p); h' = γ ⊙ h + β at each layer
   Note: initialized so γ ≈ 1, β ≈ 0 (identity at start)
5. Predictor: concat(conditioned_gene_emb, disease_emb) → MLP(256) → score
6. Training: L = 1.0×BCE + 0.5×ranking + 0.3×listnet; 5 negatives per positive;
   AdamW (lr=5e-4, cosine schedule, 5 warmup epochs); AMP (FP16)

Differentiate from: SHEPHERD (structured HPO codes, no FiLM),
FuseLinker (static embedding fusion at input), Kim et al. (LLM-only, no graph).

Use LaTeX-style equations. Cite Hamilton et al. 2017 for GraphSAGE,
Perez et al. 2018 for FiLM, Gu et al. 2021 for PubMedBERT.
```

### Results Section Prompt:

```
Write a 1-page Results section based on this ablation table:
[paste actual numbers from Phase 1]

Structure:
1. Main comparison table (4 ablation variants × metrics: AUROC, AUPR, HR@50, MRR)
   Report mean ± std across 3 seeds
2. Zero-shot rare disease results (separate table, 117 diseases)
   KEY CLAIM: MLP baseline cannot handle zero-shot; FiLM-conditioned model can
3. One qualitative case study: specific rare disease, top-5 predicted genes,
   verification against Orphanet/OMIM

Highlight two key deltas:
- GraphSAGE-only vs Full Model → FiLM conditioning contribution
- Full Model vs MLP baseline on zero-shot → text conditioning enables generalization
```

---

## PHASE 5 — Adversarial Review Simulation (April 14)

### Updated Abstract for Review Simulation:

```
You are a strict NeurIPS 2026 workshop reviewer for "Learning on Graphs."

Abstract:
"We present PromptGFM-Bio, a prompt-conditioned graph model for rare-disease 
gene prioritization. Unlike SHEPHERD (Alsentzer et al., 2025), which uses 
structured HPO phenotype codes as graph node lookups, and FuseLinker (Xiao et al., 
2024), which fuses static LLM embeddings at the GNN input layer, PromptGFM-Bio 
dynamically injects natural-language disease descriptions into GraphSAGE message 
passing via FiLM conditioning on frozen PubMedBERT embeddings. On a heterogeneous 
biomedical graph (19,576 genes, 1.85M STRING PPI edges), we achieve AUROC=0.9626 
on full-vocabulary gene ranking (all genes per query) and demonstrate zero-shot 
generalization to 117 rare diseases with no training associations. Ablation studies 
across 4 model variants × 3 seeds isolate the contributions of PPI message passing 
and FiLM conditioning independently."

Rate 1-5 on: novelty, experimental rigor, biological relevance, clarity.
Top 3 reasons to accept. Top 3 reasons to reject. Be harsh.
What one experiment would change borderline-reject to accept?
```

---

## PHASE 6 — Target Workshops & Submit (Month 5–6)

### Primary Targets (NeurIPS 2026, deadlines ~August-September 2026):

| Workshop | Fit | Why |
|---|---|---|
| **Learning on Graphs (LoG)** | ⭐⭐⭐ | Direct fit: GNN architecture + novel conditioning |
| **Graph Learning for Drug Discovery** | ⭐⭐⭐ | Most relevant: gene-disease, biological graphs |
| **AI for Science** | ⭐⭐⭐ | Broad bio-ML, well-attended |
| **New Frontiers in Graph Learning** | ⭐⭐ | If ablation delta is strong |

### Backup (ICML 2026 workshops, deadlines ~April-May 2026):
- Computational Biology workshop
- Machine Learning for Genomics

---

## YOUR TOOL STACK

| Task | Tool | Mode | Why |
|---|---|---|---|
| Evaluate ablation models | Opus 4.6 in Project | Normal | Cross-file code reasoning |
| Zero-shot evaluation | Opus 4.6 in Project | Normal | Complex evaluation pipeline |
| Literature search (updates) | Sonnet 4.6 new chat | Research ON | Check BioRxiv for 2025-2026 preprints |
| Adversarial reviewer sim | OpenRouter → o1 | Normal | Harshest critic |
| Write paper sections | Sonnet 4.6 in Project | Normal | Faster, good writer |
| Polish & check consistency | Opus 4.6 in Project | Normal | Final quality bar |

---

## IMMEDIATE NEXT 3 ACTIONS

1. **Wait for ablation runs to complete** (~April 6 evening) → Record all 12 checkpoints
2. **Run full evaluation on all 12 models** (Phase 1 prompt) → Build comparison table
3. **Run zero-shot evaluation** (Phase 2 prompt) → This is the strongest claim

The zero-shot result is your paper's headline. If the full model achieves meaningful Hit Rate@50 on zero-shot diseases while the MLP baseline gets ~0, that's the main result.

---

## RISK REGISTER

| Risk | Severity | Mitigation |
|---|---|---|
| MLP baseline AUROC very close to full model (~0.955 vs ~0.963) | HIGH | Shift narrative to zero-shot generalization; MLP cannot do zero-shot |
| FiLM delta is small on standard test set | MEDIUM | Zero-shot + case studies demonstrate qualitative value |
| Very recent BioRxiv preprint does FiLM+GNN for bio | LOW | Search before submission; your full-vocabulary evaluation is still rare |
| Scale comparison to Mantis-ML 2.0 (5,220 diseases) | MEDIUM | Position as architecturally novel + complementary, not competing on scale |
| Reviewer asks for real patient validation (like SHEPHERD's UDN) | MEDIUM | Acknowledge as future work; your zero-shot protocol is a different contribution |
