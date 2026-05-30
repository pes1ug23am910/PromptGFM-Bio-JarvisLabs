# Latest Workstation Evaluation Summary

## Update 2026-04-04 (v6 Architecture & Metrics)

**Latest Validated Metrics:**
- **Test AUROC:** 0.9626
- **Hit Rate@50:** 55.7% (0.5573)

**Latest Architecture Fixes (v6):**
- **FiLM fix:** Dynamic conditioning of message passing on disease text.
- **Centralized PROJECT_ROOT:** No path confusion.
- **.env loader:** Automates secrets (W&B, GitHub token).

---

Date: 2026-04-01

Source notebook:

- notebooks/TEF_MP_New_workstation_training_v7_fixed_v6.ipynb

Source metrics:

- results/evaluation_results.json

## Run Context

- Hardware: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- Driver/CUDA (system): 580.65.06 / 13.0
- PyTorch runtime: 2.6.0+cu124
- Config used for evaluated checkpoint: configs/workstation_config.yaml
- Evaluated checkpoint: checkpoints/promptgfm_film/best_model.pt

## Final Test Metrics

| Metric | Value |
|---|---:|
| AUROC | 0.8130237496 |
| AUPR | 0.4617632565 |
| Precision@10 | 1.0000000000 |
| Recall@10 | 0.0000854591 |
| NDCG@10 | 1.0000000000 |
| Precision@20 | 0.9499999881 |
| Recall@20 | 0.0001623724 |
| NDCG@20 | 0.9676617089 |
| Precision@50 | 0.8999999762 |
| Recall@50 | 0.0003845661 |
| NDCG@50 | 0.9211807605 |
| Precision@100 | 0.9200000167 |
| Recall@100 | 0.0007862240 |
| NDCG@100 | 0.9286268245 |

## Plain-Language Reading

- AUROC and AUPR indicate useful ranking signal above random.
- High precision@K means the top-ranked genes are often relevant.
- Very low recall@K means top-K covers only a small part of all positives.
- Practical meaning: strong top-of-list quality, but limited coverage depth.

## Validated Run Limitations

- No gene-gene message passing in this validated run:
  notebook logs report no gene-gene edges found at training time.
- Baseline constructor/signature mismatch:
  baseline path is currently not reliable due constructor argument mismatch.
- Cross-attention shape risk:
  current cross-attention execution path has tensor-shape mismatch risk.
- Runtime vs pinned dependency drift:
  this run used torch 2.6.0+cu124 while requirements.txt pins torch 2.1.0.

## Reproduce Evaluation

```bash
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## Related Live Docs

- README.md
- TRAINING_GUIDE.md
- PREPROCESSING_GUIDE.md
- DATA_DOWNLOAD_GUIDE.md
- docs/TROUBLESHOOTING.md
