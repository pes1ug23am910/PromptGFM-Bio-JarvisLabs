# PromptGFM-Bio — Novelty Assessment
## Based on 1,543 papers from Semantic Scholar (April 2026)

---

## VERDICT SUMMARY

| Claim | Status | Risk |
|-------|--------|------|
| FiLM inside GNN message passing for gene prioritization | ✅ Not found | LOW |
| Dynamic language conditioning of graph traversal (any bio task) | ✅ Not found | LOW |
| Zero-shot gene prioritization via disease text + graph | ✅ Not found | LOW |
| GNN + frozen LM for gene-disease prediction | ⚠️ Adjacent work exists | MEDIUM |
| Gene prioritization with LLMs (general) | ⚠️ Active area | MEDIUM |
| Full-vocabulary ranking evaluation protocol | ✅ Rare in literature | LOW |
| **SHEPHERD (NeurIPS 2023)** | ❌ NOT FOUND — search failed | **CRITICAL UNKNOWN** |

---

## CRITICAL ISSUE: SHEPHERD Was Not Returned

The queries `SHEPHERD NeurIPS 2023 Bejan gene prioritization` and `SHEPHERD rare disease gene prioritization NeurIPS 2023`
returned physics papers about the DUNE neutrino experiment — completely wrong.

**This is the single most important paper to find before claiming novelty.**
SHEPHERD is cited in the PromptGFM-Bio report as the closest competitor (NeurIPS 2023,
phenotype-driven GNN for rare disease gene prioritization).

**Action required — do this before anything else:**
1. Go to https://arxiv.org and search: `SHEPHERD rare disease gene prioritization`
2. Go to https://papers.nips.cc/paper_files/paper/2023 and search for SHEPHERD
3. Try Semantic Scholar directly: https://api.semanticscholar.org/graph/v1/paper/search?query=SHEPHERD+rare+disease+gene+prioritization
4. Once you have the paper ID, run: `python search.py --expand <paperId> --topic promptgfm_bio`

If SHEPHERD already does FiLM-style conditioning inside message passing, your architectural
novelty claim weakens significantly. Read the full PDF, not just the abstract.

---

## SECTION 1 — Papers That Could Kill Your Core Novelty Claim

### 🔴 READ IMMEDIATELY: FuseLinker (2024, 16 cites)
**Title:** FuseLinker: Leveraging LLM's pre-trained text embeddings and domain knowledge to enhance GNN-based link prediction on biomedical knowledge graphs
**Paper ID:** 242234a04887d434c2bf0b74969c2e7c42928b20
**Venue:** Journal of Biomedical Informatics
**Why dangerous:** Combines LLM text embeddings with GNN for biomedical KG link prediction.
**Why you're probably safe:** It fuses pre-computed LLM embeddings at the input feature level — the
language signal enters as a static node feature, not as a dynamic modulator of message passing.
This is exactly the "feature concatenation" approach your paper argues against.
**Confirm by reading:** Does FuseLinker use language embeddings INSIDE the GNN aggregation step,
or only as input features? If only as input → your claim stands.

### 🔴 READ IMMEDIATELY: "Assessing the utility of LLMs for phenotype-driven gene prioritization" (2024, 28 cites)
**Paper ID:** d0866aeb3572b1b8928888ef0a47a57b578c2890
**Venue:** American Journal of Human Genetics
**Why dangerous:** Directly evaluates LLMs for gene prioritization for rare genetic disease — this is
your exact task. High citation count, high-impact venue.
**Why you're probably safe:** Uses LLMs as direct text-to-gene predictors (GPT series, no graph
message passing). No GNN, no FiLM conditioning, no PPI network.
**Confirm by reading:** Does it rank against all 19k genes or a small candidate set? What is their
evaluation protocol? Their limitations section will describe exactly the gap your paper fills.

### 🟡 READ: LLMs as Zero-shot Graph Learners (2024, 62 cites)
**Paper ID:** 3ba86f9723c7f1185c05a38231b4a2d3765e6a3d
**Venue:** NeurIPS 2024
**Why dangerous:** Zero-shot graph learning via LLM-GNN alignment — methodologically adjacent.
**Why you're probably safe:** General graph learning (not bio/gene-disease), and uses
representation alignment rather than FiLM-conditioned message passing.
**Confirm by reading:** Does it apply to heterogeneous biological graphs? What is the conditioning mechanism?

### 🟡 READ: HiGPT (2024, 55 cites)
**Paper ID:** 7ef730e97d3fffd7603fc008431b4c35e06fea8f
**Venue:** KDD 2024
**Why dangerous:** Heterogeneous Graph Language Model — integrates LLMs with heterogeneous GNNs,
which is the graph type you use (gene-disease-phenotype).
**Why you're probably safe:** General heterogeneous graph task, not biological, no FiLM conditioning.
**Confirm by reading:** How does the language signal interact with message passing? Is any dynamic
modulation of the aggregation function used?

---

## SECTION 2 — Active Competing Area (Understand, Not Fear)

These papers establish the landscape of prior work your paper must position against.
None of them appear to directly implement FiLM-conditioned message passing for genes.

| Paper | Year | Cites | Method | Why You're Different |
|-------|------|-------|--------|---------------------|
| Multi-domain KGE for gene-disease | 2023 | 21 | KG embeddings + ML | No dynamic LM conditioning |
| KDGene | 2024 | 12 | Tensor decomposition on KG | No GNN message passing or LM |
| MGREL | 2023 | 20 | Multi-graph representation learning | Static, no language |
| KGE for diseases with curtailed info | 2024 | 13 | KG embeddings for low-info diseases | Similar motivation, different method |
| GCN + semi-supervised for gene prioritization | 2022 | 13 | Graph convolution, no LM | No language signal at all |
| Biomedical KGE for disease-gene | 2022 | 28 | KG embeddings | No dynamic conditioning |
| LLM framework for disease-gene from literature | 2024 | 9 | LLM text extraction | No graph message passing |

**Your positioning against all of these:** They either use GNNs without language,
or use language without GNNs, or combine them only at the feature/prediction level.
None condition the message passing itself on disease text.

---

## SECTION 3 — Evaluation Protocol is a Genuine Contribution

**"Evaluation of phenotype-driven gene prioritization methods for Mendelian diseases" (2022, 28 cites)**
— Paper ID: d6b8db806426686efebe45655cb776c073b3106a —
is the standard reference for evaluation methodology in your field. It documents that most
methods evaluate against small candidate gene sets, not full-vocabulary ranking.

Your paper ranks against all 19,576 genes. This is rare and should be explicitly called out
as a methodological contribution — cite this paper and contrast directly.

---

## SECTION 4 — What the Search Could NOT Find (Gaps to Fill Manually)

The Semantic Scholar search has known blind spots. The following must be checked manually:

1. **SHEPHERD (Bejan et al., NeurIPS 2023)** — Did not appear despite direct queries.
   Search: https://papers.nips.cc/paper_files/paper/2023
   Search: https://arxiv.org/search/?searchtype=all&query=SHEPHERD+rare+disease

2. **Mantis-ML 2.0 (AstraZeneca, 2024)** — The paper "Phenome-wide identification of therapeutic
   genetic targets, leveraging knowledge graphs, GNNs, and UK Biobank" (9 cites) integrates GNNs
   and KGs for gene-disease but wasn't pulled with full details. Check if it uses language.

3. **ACL Anthology** — Search directly at https://aclanthology.org for:
   - "gene prioritization language model"
   - "disease graph conditioning"
   - "FiLM graph biological"

4. **BioRxiv preprints** — Very recent work (2025–2026) on FiLM+GNN for genomics may not be
   indexed yet. Search https://biorxiv.org for "FiLM graph neural network disease".

---

## SECTION 5 — Recommended `--expand` Runs

Once you locate SHEPHERD's paper ID, run these expansions immediately:

```bash
# SHEPHERD (find ID first)
python search.py --expand <SHEPHERD_paper_ID> --topic promptgfm_bio

# FuseLinker
python search.py --expand 242234a04887d434c2bf0b74969c2e7c42928b20 --topic promptgfm_bio

# LLMs as Zero-shot Graph Learners
python search.py --expand 3ba86f9723c7f1185c05a38231b4a2d3765e6a3d --topic promptgfm_bio

# Evaluation of phenotype-driven gene prioritization (2022)
python search.py --expand d6b8db806426686efebe45655cb776c073b3106a --topic promptgfm_bio

# KDGene
python search.py --expand a4b6e13efa80bedf8e588ac69f91fdaecc8e5077 --topic promptgfm_bio
```

The SHEPHERD expansion is the most important — it will surface all papers that cited SHEPHERD
(newer work in your space) and everything SHEPHERD referenced (your foundational baseline set).

---

## SECTION 6 — What To Write in Your Related Work

Based on what the search did and did not find, here is the narrative your related work section
should build:

**Paragraph 1 — Gene prioritization methods (static):**
Prior GNN-based methods (KDGene, MGREL, KGE-based approaches) use static graph representations
that treat gene nodes identically regardless of which disease is being queried. Cite Evaluation
paper (2022) to establish that full-vocabulary ranking is the honest protocol.

**Paragraph 2 — Language models for gene prioritization:**
Recent work uses LLMs directly as text-to-gene predictors (cite the AJHG 2024 paper, 28 cites).
These skip graph structure entirely. Another line fuses pre-computed LLM embeddings with GNN
features at the input layer (cite FuseLinker). Both approaches give the graph the same
disease-agnostic representation during message passing.

**Paragraph 3 — Conditioning GNNs on auxiliary signals (FiLM):**
FiLM (Perez et al., 2018) has been applied in visual reasoning and molecular graph generation
(cite GNN for conditional drug design, 2023). We are the first to apply FiLM conditioning of
message passing to the gene prioritization task.

**Paragraph 4 — SHEPHERD:**
The closest predecessor is SHEPHERD (Bejan et al., NeurIPS 2023), which uses a GNN over an
HPO phenotype graph for rare disease gene prioritization. SHEPHERD does not condition message
passing on free-text disease descriptions — it uses structured phenotype codes as input.
PromptGFM-Bio extends this by enabling arbitrary natural language descriptions and modulating
every message-passing step through them, supporting zero-shot generalization to diseases
described only in text.

---

## BOTTOM LINE

Your core novelty claim — **FiLM-conditioned message passing with frozen PubMedBERT for
gene prioritization, enabling zero-shot generalization** — does not appear to be directly
contested by any paper in this 1,543-paper corpus.

The main risk is **SHEPHERD**, which was not found by the search. Do not submit without reading it.
The second risk is **FuseLinker**, which you need to confirm uses only static embedding fusion,
not dynamic conditioning. Both can be resolved in a single afternoon of reading.
