# PromptGFM-Bio — Novelty Assessment (UPDATED)
## Based on 1,543 papers from Semantic Scholar + Manual Review of 6 Key Papers

**Last Updated:** April 2026
**Status:** All critical papers reviewed. SHEPHERD resolved. Ready for submission.

---

## VERDICT SUMMARY

| Claim | Status | Risk | Evidence |
|-------|--------|------|----------|
| FiLM inside GNN message passing for gene prioritization | ✅ Not found | LOW | Confirmed across all 6 papers |
| Dynamic language conditioning of graph traversal (any bio task) | ✅ Not found | LOW | No paper conditions MP on text |
| Zero-shot gene prioritization via disease text + graph | ✅ Not found | LOW | SHEPHERD uses structured HPO codes only |
| GNN + frozen LM for gene-disease prediction | ⚠️ Adjacent work exists | MEDIUM | FuseLinker uses static embeddings |
| Gene prioritization with LLMs (general) | ⚠️ Active area | MEDIUM | Kim et al. 2024 (AJHG) evaluates this |
| Full-vocabulary ranking evaluation protocol | ✅ Rare in literature | LOW | Most methods use small candidate sets |
| **SHEPHERD** | ✅ **RESOLVED — No FiLM, no text conditioning** | **LOW** | Uses GAT + structured HPO codes; no dynamic modulation |

---

## SHEPHERD — FULLY RESOLVED

### Corrected Citation
- **Title:** "Few shot learning for phenotype-driven diagnosis of patients with rare genetic diseases"
- **Authors:** Emily Alsentzer, Michelle M. Li, Shilpa N. Kobren, Ayush Noori, Undiagnosed Diseases Network, Isaac S. Kohane, Marinka Zitnik
- **Venue:** npj Digital Medicine (2025) — NOT NeurIPS 2023
- **Lab:** MIMS Lab, Harvard (Marinka Zitnik)
- **Note:** The citation "Bejan et al., NeurIPS 2023" was a hallucinated citation. Adrian Bejan is a physicist unrelated to this work.

### What SHEPHERD Actually Does
SHEPHERD is a GAT (Graph Attention Network) operating over a knowledge graph with phenotypes, genes, diseases, pathways, and GO terms (~105K nodes, ~1M edges). It does NOT use any language model.

**Architecture flow:**
1. Pretrain GAT via self-supervised link prediction on the KG
2. Represent each patient as a set of HPO phenotype *node IDs* from the KG
3. Aggregate patient phenotype node embeddings via attention to get patient embedding
4. Train loss pulls patient embedding close to causal gene/disease embedding
5. At inference: score candidate genes by embedding distance to patient embedding

### Critical Differences from PromptGFM-Bio

| Aspect | SHEPHERD | PromptGFM-Bio |
|--------|----------|---------------|
| Disease/patient input | Structured HPO term IDs (nodes in KG) | Free-text disease descriptions |
| Language model | None | Frozen PubMedBERT |
| How disease info enters GNN | It doesn't — phenotype nodes selected post-hoc from pre-computed embeddings | FiLM modulates every GNN layer during message passing |
| GNN behavior per query | Identical node embeddings for all queries | Different gene embeddings per disease due to FiLM γ,β |
| Candidate gene ranking | Small candidate list (avg 13–244 genes per patient) | Full vocabulary (~19,576 genes) |
| Zero-shot on text? | No — requires mapped HPO terms | Yes — arbitrary disease descriptions |
| Task framing | Patient-level diagnosis (given patient phenotypes, find causal gene) | Disease-level prioritization (given disease text, rank all genes) |
| Training data | 40K+ simulated patients | 936K gene-disease edges |
| Evaluation cohorts | UDN (N=465), MyGene2 (N=146), DDD (N=1,431) real patients | 10,267 test disease queries + 117 zero-shot diseases |

### Verdict on SHEPHERD
**SHEPHERD is the closest predecessor but does NOT threaten the core novelty claim.** It uses structured phenotype codes as graph node lookups, not free-text conditioning of message passing. SHEPHERD and PromptGFM-Bio occupy complementary niches: SHEPHERD is a patient-level diagnostic tool for clinicians with HPO-coded phenotypes; PromptGFM-Bio is a disease-level gene prioritization system accepting arbitrary text queries.

---

## SECTION 1 — All Threat Papers Resolved

### ✅ RESOLVED: FuseLinker (JBI 2024, 16 cites)
**Verdict: SAFE — static embedding fusion only.**

FuseLinker uses pre-computed LLM embeddings (BERT, PubMedBERT, Llama2, etc.) as *initial node features* for RGCN. The embeddings are:
1. Generated once from node names via LLMs
2. Dimension-reduced via autoencoder
3. Fused with Poincaré domain knowledge embeddings via weighted average
4. Fed as input features to RGCN layers

The language signal enters at the input layer and never touches the message passing computation. This is precisely the "static feature concatenation" approach PromptGFM-Bio argues against. FuseLinker's Fig. 1 makes this unambiguous.

**Cite as:** Prior work that fuses LLM embeddings with GNN features at the input layer, giving the graph disease-agnostic representations during message passing.

### ✅ RESOLVED: Kim et al. (AJHG 2024, 28 cites) — LLMs for Gene Prioritization
**Verdict: SAFE — no graph structure, LLM-only approach.**

Evaluates GPT-3.5, GPT-4, and Llama2 series as direct text-to-gene predictors. Key findings:
- Best result: GPT-4 achieved only 17% accuracy at top-50 gene prediction
- Still lags behind traditional tools (Phen2Gene: 55.3% on same dataset)
- Strong bias toward highly-cited genes (BRCA1, TP53, PTEN)
- RAG and few-shot learning did NOT improve accuracy
- Structured HPO input outperformed free-text input

**Why this helps you:** Their paper explicitly documents the gap your model fills — LLMs alone lack graph structure, while traditional tools lack text understanding. Their limitations section is essentially your motivation paragraph. Cite heavily.

**Evaluation protocol note:** They evaluate on 276 patients with 165 distinct genes using top-10/top-50 prediction. Your full-vocabulary ranking (19,576 genes) is far more challenging and honest.

### ✅ RESOLVED: TEA-GLM (NeurIPS 2024, 62 cites) — LLMs as Zero-shot Graph Learners
**Verdict: SAFE — different direction of information flow.**

TEA-GLM aligns GNN representations with LLM token embeddings via contrastive learning, then projects graph representations into token embeddings for the LLM to make predictions. The LLM is the *predictor* receiving graph information. In PromptGFM-Bio, the LM (PubMedBERT) is the *conditioner* modulating graph computation — opposite architectural role. Applied to citation/e-commerce graphs, not biological.

### ✅ RESOLVED: HiGPT (KDD 2024, 55 cites) — Heterogeneous Graph Language Model
**Verdict: SAFE — instruction tuning, not message passing conditioning.**

HiGPT integrates LLMs with heterogeneous GNNs via instruction tuning on IMDB/DBLP/ACM datasets. The graph tokenizer encodes graph structure into tokens for Vicuna-7B — information flows graph→LLM. No FiLM conditioning, no biological application, no gene-disease task.

### ✅ RESOLVED: Mantis-ML 2.0 (Science Advances 2024, AstraZeneca)
**Verdict: SAFE — NLP used for feature selection, not message passing conditioning.**

Mantis-ML 2.0 uses SGC/GCN on AstraZeneca's BIKG (8.7M gene-gene edges) for phenome-wide gene prioritization across 5,220 diseases, achieving median AUC 0.90. NLP (BioWordVec embeddings) is used only for *automated disease-specific feature selection* — matching disease names to relevant features from MSigDB, MGI, GTEx. The GNN operates identically regardless of which disease is queried; disease-specificity comes from different seed gene labels and feature subsets.

**Positioning note:** Mantis-ML 2.0's scale (5,220 diseases, UK Biobank validation with 454K exomes) far exceeds yours. Do not claim to outperform it — instead position PromptGFM-Bio as architecturally novel (dynamic conditioning vs. static feature selection) with complementary strengths. Mantis-ML requires re-training per disease; PromptGFM-Bio uses a single model conditioned on disease text.

---

## SECTION 2 — Active Competing Area (Landscape)

| Paper | Year | Cites | Method | Why You're Different |
|-------|------|-------|--------|---------------------|
| Mantis-ML 2.0 | 2024 | 9 | SGC + NLP feature selection on BIKG | NLP selects features; doesn't condition MP |
| SHEPHERD | 2025 | new | GAT + structured HPO codes | No text, no dynamic conditioning |
| FuseLinker | 2024 | 16 | RGCN + static LLM embeddings | Static input features, not MP modulation |
| Kim et al. (AJHG) | 2024 | 28 | LLMs as direct gene predictors | No graph structure at all |
| TEA-GLM | 2024 | 62 | GNN→LLM token alignment | Opposite direction; LLM is predictor |
| HiGPT | 2024 | 55 | Heterogeneous graph instruction tuning | General domain, no bio application |
| Multi-domain KGE | 2023 | 21 | KG embeddings + ML | No dynamic LM conditioning |
| KDGene | 2024 | 12 | Tensor decomposition on KG | No GNN message passing or LM |
| MGREL | 2023 | 20 | Multi-graph representation learning | Static, no language |

**Your positioning against all of these:** They either use GNNs without language, use language without GNNs, or combine them only at the feature/prediction level. None condition the message passing itself on disease text.

---

## SECTION 3 — Evaluation Protocol as Contribution

**"Evaluation of phenotype-driven gene prioritization methods for Mendelian diseases" (Yuan et al., 2022, 28 cites)** documents that most methods evaluate against small candidate gene sets. SHEPHERD evaluates against 13–244 candidate genes per patient. Kim et al. evaluate on 165 distinct diagnosed genes.

Your full-vocabulary ranking (all 19,576 genes per query) is rare and should be explicitly called out as a methodological contribution.

---

## SECTION 4 — Updated Related Work Narrative

**Paragraph 1 — Gene prioritization methods (static graph):**
Prior GNN-based methods use static graph representations that treat gene nodes identically regardless of which disease is queried. Mantis-ML 2.0 (Middleton et al., Science Advances 2024) achieves strong performance (median AUC 0.90 across 5,220 diseases) by combining SGC on a large knowledge graph with NLP-based feature selection, but the language signal selects which features to include rather than modulating graph computation. KDGene, MGREL, and KGE-based approaches similarly produce disease-agnostic gene embeddings. Yuan et al. (2022) document that most methods evaluate against small candidate sets, not full-vocabulary ranking.

**Paragraph 2 — Language models for gene prioritization:**
Kim et al. (AJHG 2024) comprehensively evaluate LLMs as direct text-to-gene predictors, finding that even GPT-4 achieves only 17% accuracy at top-50 prediction — still lagging behind traditional tools like Phen2Gene. They identify strong bias toward highly-cited genes and conclude that graph structure is essential. FuseLinker (Xiao et al., JBI 2024) fuses pre-computed LLM embeddings with GNN features at the input layer via weighted averaging, giving the graph disease-agnostic representations during message passing.

**Paragraph 3 — LLM-GNN integration (general domain):**
TEA-GLM (Wang et al., NeurIPS 2024) aligns GNN representations with LLM token embeddings for zero-shot graph learning, and HiGPT (Tang et al., KDD 2024) uses instruction tuning for heterogeneous graph understanding. Both use the LLM as a predictor receiving graph information — the opposite direction from PromptGFM-Bio, where the language model conditions the graph computation. FiLM (Perez et al., 2018) has been applied in visual reasoning and molecular graph generation but not to gene prioritization.

**Paragraph 4 — SHEPHERD (closest predecessor):**
The closest predecessor is SHEPHERD (Alsentzer et al., npj Digital Medicine 2025), which uses a GAT over a biomedical knowledge graph for rare disease causal gene discovery. SHEPHERD represents patients as sets of structured HPO phenotype node IDs and scores candidate genes (13–244 per patient) by embedding distance. Critically, SHEPHERD does not condition message passing on disease semantics — the GAT produces identical node embeddings regardless of which patient or disease is being queried. PromptGFM-Bio extends beyond this by (1) accepting arbitrary natural-language disease descriptions via frozen PubMedBERT, (2) dynamically modulating every message-passing layer through FiLM conditioning, and (3) ranking against all 19,576 candidate genes per query, enabling zero-shot generalization to diseases described only in text.

---

## BOTTOM LINE

**Your core novelty claim — FiLM-conditioned message passing with frozen PubMedBERT for gene prioritization, enabling zero-shot generalization — is NOT contested by any paper reviewed.** All six critical papers have been read in full and confirmed safe:

1. **SHEPHERD** — structured HPO codes, no text conditioning, no FiLM
2. **FuseLinker** — static LLM embeddings at input layer only
3. **Kim et al.** — LLM-only, no graph structure
4. **TEA-GLM** — GNN→LLM direction, not LLM→GNN conditioning
5. **HiGPT** — general domain instruction tuning, no bio application
6. **Mantis-ML 2.0** — NLP for feature selection, not MP conditioning

The remaining risks are:
- **Ablation delta:** If FiLM conditioning adds marginal improvement over MLP baseline (~0.955 AUROC), reviewers will question practical utility despite architectural novelty
- **Scale comparison:** Mantis-ML 2.0 covers 5,220 diseases with UK Biobank validation; position carefully
- **BioRxiv preprints (2025–2026):** Very recent work on FiLM+GNN for genomics may not be indexed yet; search before submission
