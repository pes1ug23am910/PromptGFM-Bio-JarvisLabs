# Troubleshooting Guide

This guide is aligned with the current codebase and validated workstation run.

## 1. Environment and Version Drift

Symptom:

- Unexpected runtime behavior across machines.

Cause:

- requirements.txt pins torch==2.1.0.
- Validated workstation runtime used torch 2.6.0+cu124.

Check:

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch_geometric.__version__)"
```

Action:

- Pick one runtime profile and keep it consistent across setup, train, and evaluate.

## 2. No Gene-Gene Message Passing

Symptom:

- Training log says no gene-gene edges were found and training proceeds without message passing.

Cause:

- Processed graph edge types do not include a supported gene-gene relation at runtime.

Check:

```bash
python -c "import torch; g=torch.load('data/processed/biomedical_graph.pt'); print(g.edge_types)"
```

Action:

- Rebuild preprocessing inputs and verify edge types before retraining.

## 3. Baseline Config Failure

Symptom:

- TypeError when using baseline path, such as unexpected keyword arguments in model construction.

Cause:

- Current baseline constructor/signature mismatch between script parameters and model class arguments.

Action:

- Use configs/workstation_config.yaml for stable runs.
- Treat baseline path as experimental until fixed.

## 4. Cross-Attention Mode Failure

Symptom:

- Shape or unpack errors when running cross-attention conditioning.

Cause:

- Current forward path extracts 2D gene embeddings before conditioning, while cross-attention path expects 3D tensors.

Action:

- Use FiLM conditioning for stable runs (validated path).
- Treat cross-attention as experimental until tensor-shape path is fixed.

## 5. Few-Shot CLI Expectations

Symptom:

- --few-shot is accepted but no few-shot results are produced.

Cause:

- scripts/evaluate.py parses the argument but currently has no active few-shot branch in main execution.

Action:

- Use standard or --stratified evaluation for now.

## 6. OOM or Throughput Issues

Actions:

- Reduce batch size in configs/workstation_config.yaml.
- Increase gradient accumulation steps.
- Keep mixed precision enabled on CUDA.
- Resume from checkpoint for long runs.

## 7. Evaluation Metrics Look Strange (High Precision, Low Recall)

Interpretation:

- High precision@K means many top-ranked predictions are correct.
- Very low recall@K means top-K covers only a tiny slice of all positives in a large ranking space.

This is expected for the current baseline and does not mean evaluation is broken.

## 8. Minimal Recovery Command Set

```bash
python scripts/download_data.py --dataset all --force
python scripts/preprocess_all.py --force
python scripts/train.py --config configs/workstation_config.yaml
python scripts/evaluate.py --config configs/workstation_config.yaml --checkpoint checkpoints/promptgfm_film/best_model.pt
```

## 9. Reference

- ../LATEST_EVALUATION_SUMMARY.md
- ../TRAINING_GUIDE.md
