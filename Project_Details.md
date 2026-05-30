# PromptGFM–Bio

**Prompt-conditioned Graph Foundation Model for Rare-Disease Gene–Phenotype Mapping**

---

# 1. Project Overview

Rare diseases suffer from an extreme **data scarcity problem**: many diseases have very few known gene associations, making traditional machine-learning approaches unreliable. At the same time, modern biomedical knowledge graphs (protein–protein interactions, gene–disease links, phenotype ontologies) contain rich relational information that can be leveraged using **Graph Neural Networks (GNNs)**. However, standard GNN approaches rely primarily on structural signals and fixed node features, which limits their ability to incorporate **semantic biomedical knowledge encoded in textual descriptions** of diseases and phenotypes.

This project proposes **PromptGFM–Bio**, a **prompt-conditioned graph foundation model** that integrates natural-language biomedical prompts with graph representation learning. Instead of treating textual disease descriptions as static features, the model dynamically conditions graph message passing on disease-specific prompts, enabling the same biological graph to generate different gene-ranking predictions depending on the disease query. The system is designed specifically to improve **rare-disease gene discovery**, including few-shot and zero-shot scenarios.

---

# 2. Core Research Problem

The project addresses the following research question:

**How can natural-language biomedical knowledge be integrated into graph foundation models so that graph reasoning becomes task-adaptive and effective in long-tail (rare-disease) prediction settings?**

Existing approaches generally fall into two categories:

* **Graph-only learning:** uses network structure but ignores disease semantics.
* **Text-feature concatenation methods:** append textual embeddings as node features but do not dynamically influence reasoning.

Both approaches fail to enable **task-adaptive graph reasoning**, which is necessary for predicting gene associations for diseases with limited labeled data.

---

# 3. Key Idea: Prompt-Conditioned Graph Reasoning

PromptGFM–Bio introduces **prompt conditioning during graph message passing**:

1. A biological knowledge graph is constructed from protein interaction networks, gene–disease associations, and phenotype relations.
2. Disease descriptions and phenotype information are converted into **prompt embeddings** using biomedical language encoders.
3. These prompt embeddings dynamically **modulate GNN layers** using conditioning mechanisms such as:

   * Feature-wise Linear Modulation (FiLM)
   * Cross-attention fusion
4. The conditioned graph embeddings are used to **rank candidate genes** associated with the queried disease.

Thus, the model learns to answer queries of the form:

> “Given this disease description and phenotype information, which genes are most likely involved?”

---

# 4. Datasets and Knowledge Sources

The project integrates several widely used biomedical data resources:

**Biological interaction networks**

* STRING protein–protein interaction network
* BioGRID interaction datasets

**Gene–disease associations**

* DisGeNET
* Orphanet / Orphadata (rare disease metadata)

**Gene and protein annotations**

* UniProt
* HGNC gene symbol database

**Phenotype ontology**

* Human Phenotype Ontology (HPO)

These datasets collectively form a **heterogeneous biomedical knowledge graph** linking genes, proteins, diseases, and phenotypes.

---

# 5. Model Architecture

The proposed architecture consists of four main components:

### (1) Graph Backbone

A GNN (GraphSAGE, GAT, or GIN) learns structural representations of the biological network.

### (2) Prompt Encoder

Disease descriptions and phenotype lists are encoded using biomedical sentence embedding models (e.g., BioBERT, SBERT).

### (3) Prompt Conditioning Module

Prompt embeddings influence the GNN through conditioning mechanisms such as:

* FiLM modulation (scaling and shifting node embeddings)
* Cross-attention layers integrating prompt information into message passing

### (4) Prediction Head

A ranking head predicts gene–disease association scores for candidate genes.

---

# 6. Training Strategy

The training process consists of two stages:

### Stage 1 — Graph Foundation Pretraining

Self-supervised objectives are used to learn transferable graph representations:

* Masked node prediction
* Edge contrastive learning
* Context prediction across graph neighborhoods

### Stage 2 — Prompt-conditioned Finetuning

The model is trained on gene–disease association prediction tasks using:

* Link prediction losses
* Ranking losses
* Few-shot disease splits for rare-disease evaluation

---

# 7. Evaluation Plan

The model is evaluated using multiple perspectives:

**Prediction performance**

* AUROC
* AUPR
* Precision@K
* Mean Average Precision (MAP)

**Rare-disease performance**

* Few-shot evaluation (1–5 labeled genes)
* Leave-one-disease-out testing

**Ablation studies**

* Without prompt conditioning
* Static prompt features
* Different conditioning mechanisms
* With vs without graph pretraining

**Qualitative validation**

* Case studies comparing predicted genes with biomedical literature

---

# 8. Expected Contributions

The project aims to contribute the following:

1. **Prompt-conditioned graph foundation model framework**
   A new paradigm where graph reasoning is dynamically conditioned on semantic prompts.

2. **Improved rare-disease gene discovery**
   Demonstration of improved ranking performance in long-tail biomedical prediction tasks.

3. **Integration of LLM semantics with GNN structure**
   A practical method for combining language models and graph learning in biomedical domains.

4. **Reproducible open research pipeline**
   Public implementation including datasets, preprocessing scripts, training pipelines, and evaluation benchmarks.

---

# 9. Novelty of the Project

This work is novel in three primary ways:

* Moves beyond static text-feature fusion by **actively conditioning message passing** on prompts.
* Applies **graph foundation model pretraining** to heterogeneous biomedical knowledge graphs.
* Targets **few-shot and zero-shot rare-disease prediction**, a setting underexplored in current graph-learning research.

---

# 10. Practical Impact

Potential long-term impacts include:

* Accelerated discovery of candidate genes for rare diseases
* Improved integration of biomedical knowledge graphs and language models
* Foundation for task-adaptive biomedical graph reasoning systems

Importantly, the system produces **hypothesis-generation outputs**, not clinical recommendations.

---

# 11. Implementation Scope (Semester-Level)

Within a 10–12 week semester project, the team will:

* Construct the biomedical graph dataset
* Implement the graph backbone and prompt conditioning
* Train foundation pretraining tasks on subgraphs
* Perform prompt-conditioned finetuning
* Conduct ablations and rare-disease evaluations
* Produce a workshop-level research paper

---

# 12. Summary (Short Form)

PromptGFM–Bio develops a **prompt-aware graph foundation model** that integrates biomedical text knowledge with graph learning to improve rare-disease gene discovery. By conditioning graph reasoning on disease-specific prompts, the model enables adaptive predictions even in low-data settings and provides a reproducible framework for combining language models with biomedical knowledge graphs.

---

# Extra Context

Great! Let me help you make strategic decisions for your project. Based on the research and your project scope, here are my recommendations:

## 1. Conditioning Mechanism: **Start with FiLM, then add Cross-Attention**

**Recommended approach: Hybrid strategy**

### Phase 1: FiLM (Weeks 5-6)
- **Why start here:** Simpler to implement, faster iteration, fewer parameters
- Provides a strong baseline for prompt conditioning
- FiLM is effective across diverse tasks and requires predicting far fewer parameters than more complex mechanisms, though with less modulation potential
- Good for debugging your pipeline

### Phase 2: Cross-Attention (Weeks 7-8, if time permits)
- **Why add this:** Cross-attention conditioning outperforms FiLM especially when conditioning signal dimensionality increases or when adaptability to diverse spatial/semantic contexts is critical, achieving higher performance in reconstruction quality
- Better for your goal since disease descriptions are high-dimensional and semantically rich
- Cross-attention mechanisms in GNNs can process both node representations and extracted features dynamically, improving expressiveness

**Implementation tip:** Use FiLM in early GNN layers, cross-attention in later layers for best of both worlds.

---

## 2. Baseline Strategy: **Multiple Baselines**

You need **3 baselines** to demonstrate value:

### Baseline 1: Standard GNN (No Text)
- Pure graph structure learning (GraphSAGE/GAT)
- Only uses PPI network topology
- **Shows:** Value of incorporating any textual information

### Baseline 2: Static Text Concatenation
- Encode disease description with BioBERT → concat to node features
- **Shows:** Value of *dynamic* conditioning vs static features
- This is your key comparison!

### Baseline 3: Text-Only Baseline
- Disease description → gene ranking (no graph)
- **Shows:** Value of graph structure

**Ablation studies:**
- Prompt conditioning OFF vs ON
- FiLM vs Cross-attention vs Hybrid
- With vs without pretraining

---

## 3. Rare Disease Definition: **<5 Known Gene Associations**

Based on research findings:

### Proposed Criteria
**Ultra-rare diseases:** 1-2 known genes
**Very rare diseases:** 3-5 known genes  
**Moderately rare diseases:** 6-10 known genes

### Rationale
- Orphanet contains 6,172 clinically unique rare diseases, with 71.9% classified as genetic
- The European definition defines rare diseases as affecting fewer than 1 in 2,000 people, with about 80% having identified genetic origins
- A dataset of 4,166 rare monogenic diseases linked to 3,163 causative genes shows many diseases have very few associated genes

### Evaluation Splits
```
Train: Diseases with 10+ known genes (common rare diseases)
Val: Diseases with 6-10 known genes  
Test-Few-Shot: Diseases with 3-5 known genes (support: 1-3, query: rest)
Test-Zero-Shot: Diseases with 1-2 known genes (use only phenotype description)
```

---

## 4. Case Study: **Angelman Syndrome**

**Perfect choice** for your validation! Here's why:

### Disease Characteristics
- Angelman syndrome is a rare genetic disorder affecting approximately 1 in 15,000 individuals, caused by loss of function of the UBE3A gene
- **Primary gene:** UBE3A (chromosome 15q11-q13)
- 70% caused by maternal deletion, 11% by UBE3A mutation, 6% by imprinting defects

### Why It's Ideal for Your Project

**1. Well-characterized phenotype:**
- Severe developmental delay
- Speech impairment
- Ataxia (movement problems)
- Seizures
- Characteristic happy demeanor
- Rich phenotypic description for prompt generation

**2. Single primary gene with mechanism:**
- Clear gene-disease association
- Well-understood molecular pathway (ubiquitin-proteasome)
- Your model should rank UBE3A highly

**3. Related genes for validation:**
- UBE3A targets MAPK1, PRMT5, CDK1, CDK4, β-catenin, and UBXD8 for ubiquitination
- These interacting genes should also rank reasonably high
- Tests if your GNN captures biological pathways

**4. Differential diagnoses to test specificity:**
- Differential diagnoses include Rett syndrome (MECP2 gene), Prader-Willi syndrome, Mowat-Wilson syndrome (ZEB2 gene), and Pitt-Hopkins syndrome (TCF4 gene)
- Your model should rank UBE3A higher than these other genes

### Case Study Validation Plan

**Positive controls (should rank high):**
1. UBE3A (primary gene)
2. Genes in ubiquitin pathway that interact with UBE3A
3. Genes at 15q11-q13 locus

**Negative controls (should rank low):**
4. MECP2 (Rett syndrome)
5. ZEB2 (Mowat-Wilson)
6. Random genes unrelated to neurological function

**Qualitative validation:**
- Compare top 20 predicted genes to literature
- Check if predicted genes are in:
  - Same pathway as UBE3A
  - Same cellular process
  - Same brain regions
  - Known to interact in PPI networks

---

## Summary Recommendations

| Decision | Recommendation | Timeline |
|----------|---------------|----------|
| **Conditioning** | Start FiLM (Week 5-6), add Cross-Attention if time (Week 7-8) | Progressive |
| **Baselines** | 3 baselines: GNN-only, Static concat, Text-only | Weeks 3-4 |
| **Rare Definition** | <5 genes (ultra-rare 1-2, very rare 3-5) | Test splits |
| **Case Study** | Angelman Syndrome (UBE3A) | Week 11-12 |

This strategy gives you:
- ✅ Clear technical contribution (FiLM → Cross-attention progression)
- ✅ Strong baselines showing your approach's value
- ✅ Rigorous rare disease evaluation
- ✅ Biologically meaningful validation
---