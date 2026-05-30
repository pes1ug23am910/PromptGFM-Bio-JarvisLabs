# Codebase Dump: `configs/`

This document consolidates all the code files within the `configs/` directory structure for LLM analysis.

## File: `configs/baseline_config.yaml`

```yaml
# Baseline Configuration - GNN-Only Model (No Prompt)
# For ablation studies comparing GNN-only vs PromptGFM

# Model Configuration
model:
  name: 'gnn_only_baseline'
  
  # GNN Architecture
  gnn_type: 'graphsage'  # Options: 'graphsage', 'gat', 'gin'
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3
  
  # No prompt encoder for baseline
  use_prompt: false
  
  # Node Features
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128

# Training Configuration
training:
  # Optimizer
  optimizer: 'adam'
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # Training Loop
  num_epochs: 100
  batch_size: 64
  gradient_clip: 1.0
  
  # Loss Function
  loss_type: 'bce'  # Binary Cross-Entropy for baseline
  
  # Early Stopping
  early_stopping_patience: 10
  min_delta: 0.0001
  
  # Learning Rate Scheduling
  lr_scheduler: 'cosine'  # Options: 'cosine', 'plateau', 'step'
  lr_warmup_epochs: 5
  
  # Checkpointing
  save_best: true
  save_last: true
  checkpoint_dir: 'checkpoints/baseline'

# Data Configuration
data:
  # Paths
  graph_file: 'data/processed/biomedical_graph.pt'
  edge_file: 'data/processed/hpo_gene_disease_edges.csv'
  min_score: 0.3
  
  # Splits
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  
  # Negative Sampling
  num_negatives: 5  # Negative samples per positive sample
  negative_sampling_strategy: 'random'  # Options: 'random', 'hard'
  
  # DataLoader
  num_workers: 4
  pin_memory: true

# Evaluation Configuration
evaluation:
  # Metrics to compute
  metrics:
    - 'auroc'
    - 'aupr'
    - 'precision_at_k'
    - 'recall_at_k'
    - 'map'
    - 'mrr'
    - 'ndcg_at_k'
    - 'hit_rate_at_k'
  
  # K values for ranking metrics
  k_values: [10, 20, 50, 100]
  
  # Stratified evaluation
  stratify_by_rarity: true

# Logging Configuration
logging:
  use_wandb: true
  wandb_project: 'promptgfm-bio'
  wandb_run_name: 'baseline_gnn_only'
  wandb_entity: null  # Set your W&B username/team
  
  log_interval: 10  # Log every N batches
  
# Hardware Configuration
hardware:
  device: 'cuda'  # Options: 'cuda', 'cpu'
  mixed_precision: true  # Use automatic mixed precision (AMP)
```

## File: `configs/base_config.yaml`

```yaml
# Base Configuration for PromptGFM-Bio

data:
  graph_path: 'data/processed/biomedical_graph.pt'
  splits_path: 'data/splits/'
  rare_disease_threshold: 5  # diseases with <5 genes
  few_shot_k: [1, 3, 5]
  negative_sample_ratio: 10  # 10 negatives per positive

model:
  gnn:
    type: 'graphsage'  # or 'gat', 'gin'
    num_layers: 3
    hidden_dim: 512
    dropout: 0.2
    use_residual: true
    
  prompt_encoder:
    model_name: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    pooling: 'cls'
    max_length: 128
    freeze_bert: false  # finetune BERT or freeze
    
  conditioning:
    type: 'film'  # or 'cross_attn', 'none'
    num_heads: 8  # for cross-attention
    
  output_dim: 256

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  num_epochs: 100
  early_stopping_patience: 10
  gradient_clip: 1.0
  
  optimizer: 'adamw'
  scheduler: 'cosine'
  warmup_epochs: 5
  
  loss_weights:
    ranking_loss: 1.0
    contrastive_loss: 0.5

evaluation:
  metrics: ['auroc', 'aupr', 'precision@k', 'map', 'mrr']
  k_values: [10, 20, 50]
  stratify_by_rarity: true

logging:
  use_wandb: true
  wandb_project: 'promptgfm-bio'
  log_interval: 10  # log every 10 batches
  save_checkpoint_every: 5  # epochs

hardware:
  device: 'cuda'  # RTX 4060 detected - 8GB VRAM
  num_workers: 4
  pin_memory: true
  # RTX 4060 optimal settings:
  mixed_precision: true  # Use AMP for faster training
  cudnn_benchmark: true  # Optimize for fixed input sizes
```

## File: `configs/cross_attention_config.yaml`

```yaml
# Cross-Attention Configuration - PromptGFM with Cross-Attention Conditioning
# Alternative conditioning mechanism for comparison with FiLM

# Model Configuration
model:
  name: 'promptgfm_cross_attention'
  
  # Use prompt conditioning
  use_prompt: true
  
  # Prompt Encoder (BioBERT)
  prompt_encoder:
    model_name: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    pooling_strategy: 'cls'
    max_length: 512
    freeze_encoder: false
  
  # GNN Architecture
  gnn_type: 'gat'  # GAT works well with cross-attention
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3
  num_heads: 4  # For GAT
  
  # Conditioning Mechanism
  conditioning_type: 'cross_attention'
  prompt_dim: 768
  attention_heads: 8
  attention_dropout: 0.1
  
  # Node Features
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  
  # Prediction Head
  prediction_hidden_dim: 128

# Training Configuration
training:
  # Optimizer
  optimizer: 'adamw'
  learning_rate: 0.0005
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
  # Training Loop
  num_epochs: 100
  batch_size: 24  # Smaller due to attention memory usage
  gradient_clip: 1.0
  accumulation_steps: 3  # Effective batch size = 72
  
  # Loss Function
  loss_type: 'combined'
  loss_weights:
    bce: 1.0
    ranking: 0.5
    contrastive: 0.3
  margin: 0.5
  
  # Early Stopping
  early_stopping_patience: 15
  min_delta: 0.0001
  
  # Learning Rate Scheduling
  lr_scheduler: 'cosine'
  lr_warmup_epochs: 5
  lr_min: 1e-6
  
  # Checkpointing
  save_best: true
  save_last: true
  save_interval: 5
  checkpoint_dir: 'checkpoints/promptgfm_cross_attn'
  pretrained_checkpoint: null

# Data Configuration
data:
  # Paths
  graph_file: 'data/processed/biomedical_graph.pt'
  edge_file: 'data/processed/hpo_gene_disease_edges.csv'
  min_score: 0.3
  
  # Splits
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  
  # Negative Sampling
  num_negatives: 5
  negative_sampling_strategy: 'random'
  
  # Prompt Configuration
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  
  # DataLoader
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Evaluation Configuration
evaluation:
  metrics:
    - 'auroc'
    - 'aupr'
    - 'precision_at_k'
    - 'recall_at_k'
    - 'map'
    - 'mrr'
    - 'ndcg_at_k'
    - 'hit_rate_at_k'
  
  k_values: [10, 20, 50, 100]
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  
  few_shot_k: [1, 3, 5, 10]
  evaluate_on:
    - 'val'
    - 'test'

# Case Studies Configuration
case_studies:
  enabled: true
  studies:
    - name: 'angelman'
      known_genes: ['UBE3A']
    - name: 'rett'
      known_genes: ['MECP2']
    - name: 'fragile_x'
      known_genes: ['FMR1']

# Logging Configuration
logging:
  use_wandb: true
  wandb_project: 'promptgfm-bio'
  wandb_run_name: 'promptgfm_cross_attn_finetune'
  wandb_entity: null
  wandb_notes: 'PromptGFM with Cross-Attention conditioning - supervised fine-tuning'
  wandb_tags:
    - 'promptgfm'
    - 'cross_attention'
    - 'finetune'
    - 'rare-disease'
  
  log_interval: 10
  log_gradients: true
  log_model_architecture: true

# Hardware Configuration
hardware:
  device: 'cuda'
  mixed_precision: true
  gpu_memory_fraction: 0.9

# Reproducibility
seed: 42
deterministic: true
benchmark: false
```

## File: `configs/finetune_config.yaml`

```yaml
# Fine-tuning Configuration - PromptGFM with FiLM Conditioning
# Main configuration for supervised training with prompt conditioning

# Model Configuration
model:
  name: 'promptgfm_film'
  
  # Use prompt conditioning
  use_prompt: true
  
  # Prompt Encoder (BioBERT)
  prompt_encoder:
    model_name: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    pooling_strategy: 'cls'  # Options: 'cls', 'mean', 'max'
    max_length: 512
    freeze_encoder: false  # Set to true to freeze BioBERT weights
  
  # GNN Architecture
  gnn_type: 'graphsage'  # Options: 'graphsage', 'gat', 'gin'
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3
  
  # Conditioning Mechanism
  conditioning_type: 'film'  # Options: 'film', 'cross_attention', 'hybrid'
  prompt_dim: 768  # BioBERT output dimension
  
  # Node Features
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  
  # Prediction Head
  prediction_hidden_dim: 128

# Training Configuration
training:
  # Optimizer
  optimizer: 'adamw'  # Use AdamW for better generalization
  learning_rate: 0.0005  # Lower LR for fine-tuning
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
  # Training Loop
  num_epochs: 100
  batch_size: 32  # Smaller batch size due to BioBERT memory usage
  gradient_clip: 1.0
  accumulation_steps: 2  # Gradient accumulation for effective batch size = 64
  
  # Loss Function
  loss_type: 'combined'  # Options: 'bce', 'ranking', 'listnet', 'contrastive', 'focal', 'combined'
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5  # For ranking loss
  
  # Early Stopping
  early_stopping_patience: 15
  min_delta: 0.0001
  
  # Learning Rate Scheduling
  lr_scheduler: 'cosine'  # Options: 'cosine', 'plateau', 'step'
  lr_warmup_epochs: 5
  lr_min: 1e-6
  
  # Checkpointing
  save_best: true
  save_last: true
  save_interval: 5  # Save checkpoint every N epochs
  checkpoint_dir: 'checkpoints/promptgfm_film'
  
  # Optional: Load pretrained weights
  pretrained_checkpoint: null  # Path to pretrained model (if using pretraining)

# Data Configuration
data:
  # Paths
  graph_file: 'data/processed/biomedical_graph.pt'
  edge_file: 'data/processed/hpo_gene_disease_edges.csv'
  min_score: 0.3
  
  # Splits
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  
  # Negative Sampling
  num_negatives: 5  # Negative samples per positive sample
  negative_sampling_strategy: 'random'  # Options: 'random', 'hard'
  
  # Prompt Configuration
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  
  # DataLoader
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Evaluation Configuration
evaluation:
  # Metrics to compute
  metrics:
    - 'auroc'
    - 'aupr'
    - 'precision_at_k'
    - 'recall_at_k'
    - 'map'
    - 'mrr'
    - 'ndcg_at_k'
    - 'hit_rate_at_k'
  
  # K values for ranking metrics
  k_values: [10, 20, 50, 100]
  
  # Stratified evaluation
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2    # 1-2 known genes
    very_rare: 5     # 3-5 known genes
    rare: 15         # 6-15 known genes
    common: 1000     # 16+ known genes
  
  # Few-shot evaluation
  few_shot_k: [1, 3, 5, 10]
  
  # Evaluation splits
  evaluate_on:
    - 'val'
    - 'test'

# Case Studies Configuration
case_studies:
  enabled: true
  studies:
    - name: 'angelman'
      known_genes: ['UBE3A']
    - name: 'rett'
      known_genes: ['MECP2']
    - name: 'fragile_x'
      known_genes: ['FMR1']

# Logging Configuration
logging:
  use_wandb: false
  wandb_project: 'promptgfm-bio'
  wandb_run_name: 'promptgfm_film_finetune'
  wandb_entity: null  # Set your W&B username/team
  wandb_notes: 'PromptGFM with FiLM conditioning - supervised fine-tuning'
  wandb_tags:
    - 'promptgfm'
    - 'film'
    - 'finetune'
    - 'rare-disease'
  
  log_interval: 10  # Log every N batches
  log_gradients: true
  log_model_architecture: true

# Hardware Configuration
hardware:
  device: 'cuda'  # Options: 'cuda', 'cpu'
  mixed_precision: true  # Use automatic mixed precision (AMP)
  gpu_memory_fraction: 0.9  # Reserve 90% of GPU memory

# Reproducibility
seed: 42
deterministic: true
benchmark: false  # Set to true for faster training if input sizes are fixed

```

## File: `configs/kaggle_config.yaml`

```yaml
# Kaggle Configuration for PromptGFM-Bio
# ─────────────────────────────────────────────────────────────────────────────
# Tuned for Kaggle T4 x2 (16 GB VRAM per GPU)
# Observed: only 3.4/15 GB VRAM used at batch_size=64 → bumped to 256
# At batch_size=256: ~4× speedup (5 it/s → ~20 it/s, ~12 min/epoch)
# freeze_encoder=true: BioBERT is frozen — saves ~2 GB VRAM + speeds up backprop
# ─────────────────────────────────────────────────────────────────────────────

# Model Configuration
model:
  name: 'promptgfm_film'

  use_prompt: true

  prompt_encoder:
    model_name: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    pooling_strategy: 'cls'
    max_length: 512
    freeze_encoder: true           # ← freeze BioBERT: saves ~2 GB VRAM, faster backprop

  # GNN Backbone — larger hidden_dim thanks to 16 GB VRAM
  gnn_type: 'graphsage'           # Options: 'graphsage', 'gat', 'gin'
  hidden_dim: 512                 # ↑ 256→512 (was 256 on laptop config)
  num_layers: 3
  dropout: 0.3

  # Conditioning
  conditioning_type: 'film'
  prompt_dim: 768

  # Node features
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128

  # Prediction head
  prediction_hidden_dim: 256      # ↑ 128→256

# Training Configuration
training:
  optimizer: 'adamw'
  learning_rate: 0.0005
  weight_decay: 0.01
  betas: [0.9, 0.999]

  num_epochs: 100
  batch_size: 256                 # ↑ 64→256 (only 3.4/15 GB VRAM used at 64 — 4× headroom)
  gradient_clip: 1.0
  accumulation_steps: 1           # effective batch size = 256

  loss_type: 'combined'
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5

  early_stopping_patience: 15
  min_delta: 0.0001

  lr_scheduler: 'cosine'
  lr_warmup_epochs: 5
  lr_min: 1e-6

  # Checkpoint every epoch (important for 9-h session limit)
  save_best: true
  save_last: true
  save_interval: 1               # Every epoch — recover from session timeout
  checkpoint_dir: 'checkpoints/promptgfm_film'

  pretrained_checkpoint: null

# Data Configuration
data:
  graph_file: 'data/processed/biomedical_graph.pt'
  edge_file: 'data/processed/hpo_gene_disease_edges.csv'
  min_score: 0.3

  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42

  num_negatives: 5
  negative_sampling_strategy: 'random'

  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20

  num_workers: 4                  # ↑ 2→4  (T4 x2 has more CPU cores available)
  pin_memory: true
  prefetch_factor: 2

# Evaluation Configuration
evaluation:
  metrics:
    - 'auroc'
    - 'aupr'
    - 'precision_at_k'
    - 'recall_at_k'
    - 'map'
    - 'mrr'
    - 'ndcg_at_k'
    - 'hit_rate_at_k'
  k_values: [10, 20, 50, 100]
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  few_shot_k: [1, 3, 5, 10]
  evaluate_on:
    - 'val'
    - 'test'

# Case Studies
case_studies:
  enabled: true
  studies:
    - name: 'angelman'
      known_genes: ['UBE3A']
    - name: 'rett'
      known_genes: ['MECP2']
    - name: 'fragile_x'
      known_genes: ['FMR1']

# Logging — disable W&B by default; set use_wandb: true + call wandb.login() in notebook
logging:
  use_wandb: false
  wandb_project: 'promptgfm-bio'
  wandb_run_name: 'kaggle_t4_run'
  wandb_entity: null
  wandb_tags:
    - 'promptgfm'
    - 'film'
    - 'kaggle'
    - 'rare-disease'
  log_interval: 20               # Less frequent logging (Kaggle stdout is slower)
  log_gradients: false           # Saves memory / time
  log_model_architecture: true

# Hardware
hardware:
  device: 'cuda'
  mixed_precision: true          # FP16 — works on T4 and P100
  gpu_memory_fraction: 0.92      # Leave ~8% for PyG scatter operations

# Reproducibility
seed: 42
deterministic: false             # deterministic=True is slow; disable on Kaggle
benchmark: true                  # Faster when graph sizes are fixed
```

## File: `configs/pretrain_config.yaml`

```yaml
# Pretraining Configuration - Self-Supervised Learning (Optional)
# For pre-training GNN on graph structure before supervised fine-tuning

# Model Configuration
model:
  name: 'promptgfm_pretrain'
  
  # GNN Architecture
  gnn_type: 'graphsage'  # Options: 'graphsage', 'gat', 'gin'
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3
  
  # Node Features
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128

# Pretraining Configuration
pretraining:
  # Pretraining Tasks
  tasks:
    masked_node_prediction:
      enabled: true
      weight: 1.0
      mask_ratio: 0.15
      mask_type: 'random'  # Options: 'random', 'degree_based'
    
    edge_contrastive:
      enabled: true
      weight: 0.5
      num_negatives: 5
      temperature: 0.07
    
    graph_contrastive:
      enabled: true
      weight: 0.3
      augmentation:
        - 'node_drop'
        - 'edge_drop'
        - 'feature_mask'
      drop_ratio: 0.2
      temperature: 0.07
  
  # Pretraining Loop
  num_epochs: 50
  batch_size: 64
  gradient_clip: 1.0

# Training Configuration
training:
  # Optimizer
  optimizer: 'adamw'
  learning_rate: 0.001  # Higher LR for pretraining
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
  # Learning Rate Scheduling
  lr_scheduler: 'cosine'
  lr_warmup_epochs: 3
  lr_min: 1e-6
  
  # Checkpointing
  save_best: true
  save_last: true
  save_interval: 5
  checkpoint_dir: 'checkpoints/pretrain'

# Data Configuration
data:
  # Paths
  graph_file: 'data/processed/biomedical_graph.pt'
  
  # DataLoader
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Logging Configuration
logging:
  use_wandb: true
  wandb_project: 'promptgfm-bio'
  wandb_run_name: 'pretrain_self_supervised'
  wandb_entity: null
  wandb_notes: 'Self-supervised pretraining on biomedical graph'
  wandb_tags:
    - 'pretrain'
    - 'self-supervised'
    - 'graph-contrastive'
  
  log_interval: 20

# Hardware Configuration
hardware:
  device: 'cuda'
  mixed_precision: true

# Reproducibility
seed: 42
deterministic: true
benchmark: false

```

## File: `configs/workstation_config.yaml`

```yaml
model:
  name: promptgfm_film
  use_prompt: true
  prompt_encoder:
    model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    pooling_strategy: cls
    max_length: 512
    freeze_encoder: true
  gnn_type: graphsage
  hidden_dim: 512
  num_layers: 3
  dropout: 0.3
  conditioning_type: film
  prompt_dim: 768
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  prediction_hidden_dim: 256
training:
  optimizer: adamw
  learning_rate: 0.0005
  weight_decay: 0.01
  betas:
  - 0.9
  - 0.999
  num_epochs: 100
  batch_size: 768
  gradient_clip: 1.0
  accumulation_steps: 1
  loss_type: combined
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5
  early_stopping_patience: 15
  min_delta: 0.0001
  lr_scheduler: cosine
  lr_warmup_epochs: 5
  lr_min: 1e-6
  save_best: true
  save_last: true
  save_interval: 1
  checkpoint_dir: checkpoints/promptgfm_film
  pretrained_checkpoint: null
  gradient_accumulation_steps: 1
data:
  graph_file: data/processed/biomedical_graph.pt
  edge_file: data/processed/hpo_gene_disease_edges.csv
  min_score: 0.3
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  num_negatives: 5
  negative_sampling_strategy: random
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
evaluation:
  metrics:
  - auroc
  - aupr
  - precision_at_k
  - recall_at_k
  - map
  - mrr
  - ndcg_at_k
  - hit_rate_at_k
  k_values:
  - 10
  - 20
  - 50
  - 100
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  few_shot_k:
  - 1
  - 3
  - 5
  - 10
  evaluate_on:
  - val
  - test
case_studies:
  enabled: true
  studies:
  - name: angelman
    known_genes:
    - UBE3A
  - name: rett
    known_genes:
    - MECP2
  - name: fragile_x
    known_genes:
    - FMR1
logging:
  use_wandb: false
  wandb_project: promptgfm-bio
  wandb_run_name: kaggle_t4_run
  wandb_entity: null
  wandb_tags:
  - promptgfm
  - film
  - kaggle
  - rare-disease
  log_interval: 20
  log_gradients: false
  log_model_architecture: true
hardware:
  device: cuda
  mixed_precision: true
  gpu_memory_fraction: 0.92
seed: 42
deterministic: false
benchmark: true
```

## File: `configs/ablations/ablation_1_mlp_only.yaml`

```yaml
experiment_name: "ablation_1_mlp_only"
model:
  name: promptgfm_film
  use_prompt: true
  prompt_encoder:
    model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    pooling_strategy: cls
    max_length: 512
    freeze_encoder: true
  gnn_type: graphsage
  hidden_dim: 512
  num_layers: 3
  dropout: 0.3
  conditioning_type: film
  prompt_dim: 768
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  prediction_hidden_dim: 256
  use_gnn: false         # ablation: no GraphSAGE message passing (pure MLP)
  use_conditioning: false # ablation: no FiLM conditioning (identity: gamma=1, beta=0)
training:
  optimizer: adamw
  learning_rate: 0.0005
  weight_decay: 0.01
  betas:
  - 0.9
  - 0.999
  num_epochs: 100
  batch_size: 768
  gradient_clip: 1.0
  accumulation_steps: 1
  loss_type: combined
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5
  early_stopping_patience: 15
  min_delta: 0.0001
  lr_scheduler: cosine
  lr_warmup_epochs: 5
  lr_min: 1e-6
  save_best: true
  save_last: true
  save_interval: 1
  checkpoint_dir: checkpoints/ablation_1_mlp_only
  pretrained_checkpoint: null
  gradient_accumulation_steps: 1
data:
  graph_file: data/processed/biomedical_graph.pt
  edge_file: data/processed/hpo_gene_disease_edges.csv
  min_score: 0.3
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  num_negatives: 5
  negative_sampling_strategy: random
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
evaluation:
  metrics:
  - auroc
  - aupr
  - precision_at_k
  - recall_at_k
  - map
  - mrr
  - ndcg_at_k
  - hit_rate_at_k
  k_values:
  - 10
  - 20
  - 50
  - 100
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  few_shot_k:
  - 1
  - 3
  - 5
  - 10
  evaluate_on:
  - val
  - test
case_studies:
  enabled: true
  studies:
  - name: angelman
    known_genes:
    - UBE3A
  - name: rett
    known_genes:
    - MECP2
  - name: fragile_x
    known_genes:
    - FMR1
logging:
  use_wandb: false
  wandb_project: promptgfm-bio
  wandb_run_name: kaggle_t4_run
  wandb_entity: null
  wandb_tags:
  - promptgfm
  - film
  - kaggle
  - rare-disease
  log_interval: 20
  log_gradients: false
  log_model_architecture: true
hardware:
  device: cuda
  mixed_precision: true
  gpu_memory_fraction: 0.92
seed: 42
deterministic: false
benchmark: true
```

## File: `configs/ablations/ablation_2_prompt_only.yaml`

```yaml
experiment_name: "ablation_2_prompt_only"
model:
  name: promptgfm_film
  use_prompt: true
  prompt_encoder:
    model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    pooling_strategy: cls
    max_length: 512
    freeze_encoder: true
  gnn_type: graphsage
  hidden_dim: 512
  num_layers: 3
  dropout: 0.3
  conditioning_type: film
  prompt_dim: 768
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  prediction_hidden_dim: 256
  use_gnn: false        # ablation: no GraphSAGE message passing (no PPI edges)
  use_conditioning: true # ablation: FiLM conditioning ON (prompt only, no MP)
training:
  optimizer: adamw
  learning_rate: 0.0005
  weight_decay: 0.01
  betas:
  - 0.9
  - 0.999
  num_epochs: 100
  batch_size: 768
  gradient_clip: 1.0
  accumulation_steps: 1
  loss_type: combined
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5
  early_stopping_patience: 15
  min_delta: 0.0001
  lr_scheduler: cosine
  lr_warmup_epochs: 5
  lr_min: 1e-6
  save_best: true
  save_last: true
  save_interval: 1
  checkpoint_dir: checkpoints/ablation_2_prompt_only
  pretrained_checkpoint: null
  gradient_accumulation_steps: 1
data:
  graph_file: data/processed/biomedical_graph.pt
  edge_file: data/processed/hpo_gene_disease_edges.csv
  min_score: 0.3
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  num_negatives: 5
  negative_sampling_strategy: random
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
evaluation:
  metrics:
  - auroc
  - aupr
  - precision_at_k
  - recall_at_k
  - map
  - mrr
  - ndcg_at_k
  - hit_rate_at_k
  k_values:
  - 10
  - 20
  - 50
  - 100
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  few_shot_k:
  - 1
  - 3
  - 5
  - 10
  evaluate_on:
  - val
  - test
case_studies:
  enabled: true
  studies:
  - name: angelman
    known_genes:
    - UBE3A
  - name: rett
    known_genes:
    - MECP2
  - name: fragile_x
    known_genes:
    - FMR1
logging:
  use_wandb: false
  wandb_project: promptgfm-bio
  wandb_run_name: kaggle_t4_run
  wandb_entity: null
  wandb_tags:
  - promptgfm
  - film
  - kaggle
  - rare-disease
  log_interval: 20
  log_gradients: false
  log_model_architecture: true
hardware:
  device: cuda
  mixed_precision: true
  gpu_memory_fraction: 0.92
seed: 42
deterministic: false
benchmark: true
```

## File: `configs/ablations/ablation_3_gnn_only.yaml`

```yaml
experiment_name: "ablation_3_gnn_only"
model:
  name: promptgfm_film
  use_prompt: true
  prompt_encoder:
    model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    pooling_strategy: cls
    max_length: 512
    freeze_encoder: true
  gnn_type: graphsage
  hidden_dim: 512
  num_layers: 3
  dropout: 0.3
  conditioning_type: film
  prompt_dim: 768
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  prediction_hidden_dim: 256
  use_gnn: true          # ablation: GraphSAGE message passing ON (PPI edges active)
  use_conditioning: false # ablation: FiLM conditioning OFF (identity: gamma=1, beta=0)
training:
  optimizer: adamw
  learning_rate: 0.0005
  weight_decay: 0.01
  betas:
  - 0.9
  - 0.999
  num_epochs: 100
  batch_size: 768
  gradient_clip: 1.0
  accumulation_steps: 1
  loss_type: combined
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5
  early_stopping_patience: 15
  min_delta: 0.0001
  lr_scheduler: cosine
  lr_warmup_epochs: 5
  lr_min: 1e-6
  save_best: true
  save_last: true
  save_interval: 1
  checkpoint_dir: checkpoints/ablation_3_gnn_only
  pretrained_checkpoint: null
  gradient_accumulation_steps: 1
data:
  graph_file: data/processed/biomedical_graph.pt
  edge_file: data/processed/hpo_gene_disease_edges.csv
  min_score: 0.3
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  num_negatives: 5
  negative_sampling_strategy: random
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
evaluation:
  metrics:
  - auroc
  - aupr
  - precision_at_k
  - recall_at_k
  - map
  - mrr
  - ndcg_at_k
  - hit_rate_at_k
  k_values:
  - 10
  - 20
  - 50
  - 100
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  few_shot_k:
  - 1
  - 3
  - 5
  - 10
  evaluate_on:
  - val
  - test
case_studies:
  enabled: true
  studies:
  - name: angelman
    known_genes:
    - UBE3A
  - name: rett
    known_genes:
    - MECP2
  - name: fragile_x
    known_genes:
    - FMR1
logging:
  use_wandb: false
  wandb_project: promptgfm-bio
  wandb_run_name: kaggle_t4_run
  wandb_entity: null
  wandb_tags:
  - promptgfm
  - film
  - kaggle
  - rare-disease
  log_interval: 20
  log_gradients: false
  log_model_architecture: true
hardware:
  device: cuda
  mixed_precision: true
  gpu_memory_fraction: 0.92
seed: 42
deterministic: false
benchmark: true
```

## File: `configs/ablations/ablation_4_full_model.yaml`

```yaml
experiment_name: "ablation_4_full_model"
model:
  name: promptgfm_film
  use_prompt: true
  prompt_encoder:
    model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    pooling_strategy: cls
    max_length: 512
    freeze_encoder: true
  gnn_type: graphsage
  hidden_dim: 512
  num_layers: 3
  dropout: 0.3
  conditioning_type: film
  prompt_dim: 768
  gene_feature_dim: 128
  disease_feature_dim: 128
  phenotype_feature_dim: 128
  prediction_hidden_dim: 256
  use_gnn: true         # ablation: GraphSAGE message passing ON (PPI edges active)
  use_conditioning: true # ablation: FiLM conditioning ON — full model
training:
  optimizer: adamw
  learning_rate: 0.0005
  weight_decay: 0.01
  betas:
  - 0.9
  - 0.999
  num_epochs: 100
  batch_size: 768
  gradient_clip: 1.0
  accumulation_steps: 1
  loss_type: combined
  loss_weights:
    bce: 1.0
    ranking: 0.5
    listnet: 0.3
  margin: 0.5
  early_stopping_patience: 15
  min_delta: 0.0001
  lr_scheduler: cosine
  lr_warmup_epochs: 5
  lr_min: 1e-6
  save_best: true
  save_last: true
  save_interval: 1
  checkpoint_dir: checkpoints/ablation_4_full_model
  pretrained_checkpoint: null
  gradient_accumulation_steps: 1
data:
  graph_file: data/processed/biomedical_graph.pt
  edge_file: data/processed/hpo_gene_disease_edges.csv
  min_score: 0.3
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  num_negatives: 5
  negative_sampling_strategy: random
  use_disease_descriptions: true
  use_phenotype_lists: true
  max_phenotypes_per_disease: 20
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
evaluation:
  metrics:
  - auroc
  - aupr
  - precision_at_k
  - recall_at_k
  - map
  - mrr
  - ndcg_at_k
  - hit_rate_at_k
  k_values:
  - 10
  - 20
  - 50
  - 100
  stratify_by_rarity: true
  rarity_thresholds:
    ultra_rare: 2
    very_rare: 5
    rare: 15
    common: 1000
  few_shot_k:
  - 1
  - 3
  - 5
  - 10
  evaluate_on:
  - val
  - test
case_studies:
  enabled: true
  studies:
  - name: angelman
    known_genes:
    - UBE3A
  - name: rett
    known_genes:
    - MECP2
  - name: fragile_x
    known_genes:
    - FMR1
logging:
  use_wandb: false
  wandb_project: promptgfm-bio
  wandb_run_name: kaggle_t4_run
  wandb_entity: null
  wandb_tags:
  - promptgfm
  - film
  - kaggle
  - rare-disease
  log_interval: 20
  log_gradients: false
  log_model_architecture: true
hardware:
  device: cuda
  mixed_precision: true
  gpu_memory_fraction: 0.92
seed: 42
deterministic: false
benchmark: true
```

