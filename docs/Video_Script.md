Here's a 4-minute script (~520 words at comfortable speaking pace), timed per slide:

---

## Speaking Script — PromptGFM-Bio (4 minutes)

---

**[Slide 1 — Title | ~15 sec]**

Hi everyone, I'm Yash Verma, and today I'll be presenting PromptGFM-Bio — a prompt-conditioned graph foundation model for rare disease gene prioritisation, built as part of the Deep Learning on Graphs course project.

---

**[Slide 2 — Project Overview | ~35 sec]**

The motivating problem: rare diseases affect over 300 million people worldwide, yet identifying the causal gene is like finding a needle in a haystack of 19,576 protein-coding genes. For most rare diseases, fewer than five causal genes are even known — a classic long-tail problem where standard supervised approaches completely break down.

PromptGFM-Bio frames this as a link prediction task on a large heterogeneous biomedical graph. Given only a natural language disease description, the model ranks all candidate genes by predicted association strength.

---

**[Slide 3 & 4 — What's Unique | ~40 sec]**

The core novelty is *where* language enters the model. Most prior systems — like SHEPHERD — either require structured HPO phenotype codes, or tack on text only at the final prediction head. We do something different: a frozen PubMedBERT encoder produces a disease embedding that dynamically modulates the GNN's message passing at *every layer* via FiLM conditioning. This means the same protein-protein interaction network produces different gene representations depending on which disease you're asking about.

We also evaluate against the full vocabulary — all 19,576 genes per query — rather than a small candidate pool, which gives an honest picture of clinical utility.

---

**[Slide 5 — Dataset | ~25 sec]**

The knowledge graph integrates five public sources: STRING PPI, BioGRID, DisGeNET, HPO, and Orphanet. The resulting graph has 19,576 gene nodes, 16,841 disease nodes, over 1.85 million PPI edges, and nearly 10 million gene-disease edges. Training uses roughly 936,000 supervised edges with an 80/10/10 split.

---

**[Slide 6 — Architecture | ~20 sec]**

The architecture has four components: a frozen PubMedBERT prompt encoder, a three-layer GraphSAGE backbone, FiLM conditioning layers that inject the disease prompt as scale and shift parameters into each message-passing step, and a final MLP prediction head. Training uses a combined BCE plus pairwise ranking plus ListNet loss.

---

**[Slide 7 — Ablation Results | ~50 sec]**

To isolate each component's contribution, we trained four variants under three random seeds on a held-out set of 117 zero-shot rare diseases — diseases the model never saw during training.

The results are in this table. The key finding: the full model beats every ablation on every metric. Relative to the MLP baseline, AUROC improves by 2.6 points and Hit@50 jumps from 16.8% to 21.9% — a 30% relative gain. Importantly, the two single-component variants achieve nearly *identical* AUROC to each other, but both lag the full model significantly. Paired t-tests confirm the improvements are statistically significant — the text branch and the graph branch are complementary, not redundant.

---

**[Slide 8 — Progress Summary | ~25 sec]**

On the standard test set of 10,000+ disease queries, the full model achieves AUROC 0.9606 and places the true causal gene in the top 50 out of nearly 20,000 candidates for 54.9% of queries. The entire pipeline — graph construction, training, and evaluation — runs in under three hours on a single RTX 4090.

---

**[Slide 9 — Thank You | ~10 sec]**

That's the project. Thank you — happy to take questions.

---

**Total: ~4 minutes.** The ablation slide is the longest beat — give it room. Slides 3–4 can be spoken across continuously without pausing between them.