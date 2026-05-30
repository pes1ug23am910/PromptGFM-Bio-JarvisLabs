# PromptGFM-Bio Handover

## 1) Project overview

This project trains a PromptGFM-based gene–disease ranking system for rare-disease gene prioritization. In practice, it combines a biological graph, disease text/phenotype descriptions, and a GNN + frozen PubMedBERT prompt encoder to rank candidate genes for a disease. The evaluation pipeline then scores the model on full-vocabulary gene ranking metrics such as AUROC, AUPR, Precision@K, Recall@K, MAP, MRR, NDCG, and Hit Rate@K. fileciteturn14file0turn14file3

## 2) Notebook / workspace used

The work was carried out in the ARC Labs workstation notebook export `TEF_MP_New_workstation_training_v7_fixed_v6.md`, based on the PromptGFM-Bio workstation notebook state captured in the notebook export `TE_2_New_workstation_training_v7_fixed_v6.ipynb`. The project root is `/home/mluser/projects_yash/new_project/PromptGFM-Bio`. fileciteturn15file0

## 3) Core problem that had to be solved

The main operational problem was that training initially crashed or stalled because the data pipeline was returning graph-wide tensors through the DataLoader batch payload. That caused multiprocessing shared-memory pressure and bus errors (`No space left on device (28)` / shm failures). A second issue was that the trainer still expected `node_features` and `edge_index` inside each batch after the batch format had been changed, which caused a `KeyError: 'node_features'`. fileciteturn10file0turn9file0turn11file0

A separate evaluation bottleneck also existed: the evaluator ranks roughly 19,576 genes for each disease query, so test-time evaluation is intentionally expensive by design. fileciteturn14file0

## 4) Changes made and why

### 4.1 `scripts/train.py` patching
The training notebook was used to patch `scripts/train.py` so that:

- `config['data']['num_workers']` is respected instead of hard-coding worker count.
- `multiprocessing_context="fork"` is used for DataLoaders.
- PyG ABI-related warnings are suppressed in the training log. fileciteturn3file0turn15file0

These changes were made to stabilize multiprocessing on the workstation and reduce noise in the logs.

### 4.2 Graph tensor transport fix
The more important architectural fix was to stop sending `node_features` and `edge_index` through every DataLoader batch. Instead:

- the graph tensors are stored once on the trainer with `set_graph_tensors()`,
- `_forward_batch()` reads them from the trainer,
- `validate()` uses the cached graph tensors for the initial GNN embedding pass. fileciteturn11file0turn10file0

This was done because the graph tensors are constant for every batch, so re-sending them was wasting shared memory and causing worker crashes.

### 4.3 Checkpoint cadence change
Checkpointing was adjusted so that `best_model.pt` is always saved when validation improves, while epoch checkpoints are written only periodically rather than every epoch. The final operational choice was to save periodic checkpoints every 10 epochs to reduce disk usage. fileciteturn13file0

## 5) Effects of the changes

### Positive effects
- Training became stable on the workstation.
- DataLoader worker crashes from `/dev/shm` exhaustion stopped.
- The batch payload became much smaller and cheaper to serialize.
- The training loop now successfully runs with `num_workers=4`.
- Validation and checkpointing continue to work normally. fileciteturn10file0turn12file0turn13file0

### Negative / side effects encountered
- The first trainer patch broke because `_forward_batch()` still expected `batch['node_features']`.
- The earlier shell-based edit command failed once because it was run from the wrong directory and the pasted command was corrupted.
- Evaluation remains computationally heavy because the evaluator scores the full gene universe for each disease query. fileciteturn10file0turn11file0turn14file0

## 6) What was learned

1. **Transporting constant tensors through multiprocessing is the real bottleneck.**  
   `fork` alone is not enough if each batch still carries large tensors back through the queue.

2. **Batch contracts must stay in sync across the pipeline.**  
   Once the batch dictionary changed, both the trainer and validator had to be updated together.

3. **Best-model checkpoints are more valuable than dense epoch checkpoints.**  
   Saving the best checkpoint on improvement gives recovery coverage without excessive disk usage.

4. **Evaluation is a separate performance problem from training.**  
   The current evaluation design is correct but intentionally expensive because it does full-vocabulary ranking. fileciteturn9file0turn13file0turn14file0

## 7) Current status of the project

The project is now in a working state:

- preprocessing completed and produced `data/processed/biomedical_graph.pt`,
- the biological graph contains gene-gene, gene-disease, and disease-gene reverse edges,
- training runs stably on the RTX 4090 workstation,
- the graph tensors are cached on the trainer rather than sent through every batch,
- evaluation on the test split completed successfully. fileciteturn5file0turn10file0turn12file0

The latest test metrics recorded in `TEF_MP_evaluation_results.json` are:

- AUROC: 0.9626
- AUPR: 0.0188
- Precision@10: 0.0557
- Recall@10: 0.0695
- NDCG@10: 0.0795
- MRR: 0.1588
- Hit Rate@100: 0.6537 fileciteturn15file1

These results indicate that the ranking system is functioning end-to-end, with strong AUROC and usable ranking behavior, although precision-oriented ranking metrics remain modest as expected for a large candidate space. fileciteturn15file1turn14file3

## 8) Practical notes for the next maintainer

- Re-run the training launch cell after any edit to `src/training/finetune.py`; it is a separate Python process and picks up the code changes automatically.
- Keep `best_model.pt` as the primary recovery artifact.
- Keep periodic checkpoints sparse (10 epochs is a sensible default).
- Expect evaluation to take time because the current script ranks all genes for each disease query. fileciteturn11file0turn13file0turn14file0

## 9) One-line summary

This project is now a stable rare-disease gene-ranking pipeline: preprocessing works, training works, the shared-memory crash is fixed, and the current model has a strong test AUROC with full evaluation metrics already recorded. fileciteturn12file0turn15file1
