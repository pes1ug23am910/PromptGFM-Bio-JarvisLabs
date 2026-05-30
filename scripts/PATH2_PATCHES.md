# Path 2 patches — apply BEFORE the Phase 2 cloud runs

These two small edits make `PromptGFM-Bio` reproducible in the way Q1 reviewers
expect: a **single fixed train/val/test split** for the 10-seed study, and the
top-level `seed:` deterministically seeding torch / numpy / Python `random`
(model init, negative sampling, DataLoader shuffling).

After applying, your existing configs work unchanged — `data.random_seed: 42`
is already present in every ablation YAML, and the top-level `seed:` becomes
the per-run model-RNG seed.

> Apply with any text editor on Windows. For each change, search for the OLD
> block (verbatim) and replace with the NEW block. Indentation must match.

---

## 1) `scripts/train.py` — three edits

### 1a. Add the seeding helper (after the imports / logger setup)

Find this line (it's near the top of the file, just after `setup_logger` is imported):

```python
logger = logging.getLogger(__name__)
```

Immediately **AFTER** that line, insert:

```python


# ---------------------------------------------------------------------------
# Path 2 reproducibility: deterministic per-seed RNG for ALL stochastic sources.
# Called once from run_finetuning / run_pretraining with config['seed'].
# The data split is seeded SEPARATELY in create_dataloaders() using
# config['data']['random_seed'], which is held FIXED across all 10 seeds.
# ---------------------------------------------------------------------------
def _set_all_seeds(seed: int) -> None:
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"[reproducibility] all RNGs seeded with seed={seed} "
                f"(split seed comes from config['data']['random_seed'])")
```

### 1b. Fix the split seed in `create_dataloaders`

Find this block (inside `def create_dataloaders(config):`):

```python
    # Split data
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=config.get('seed', 42)
    )
```

Replace with:

```python
    # Split data — Path 2: split seed comes from data.random_seed (fixed across
    # all model-init seeds), NOT from top-level config['seed']. Backward-compat
    # fallback: if data.random_seed is missing, use the top-level seed.
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed
    )
    logger.info(f"[reproducibility] split seed = {split_seed} "
                f"(should be FIXED across all 10 model-init seeds)")
```

### 1c. Seed all RNGs at the top of training

Find this block (the start of `def run_finetuning(config):`):

```python
def run_finetuning(config):
    """Run supervised fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("Starting Supervised Fine-tuning")
    logger.info("="*60)
    
    # Create dataloaders (GeneDiseaseDataset will load the graph internally)
    train_loader, val_loader, dataset = create_dataloaders(config)
```

Replace with:

```python
def run_finetuning(config):
    """Run supervised fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("Starting Supervised Fine-tuning")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
    
    # Create dataloaders (GeneDiseaseDataset will load the graph internally)
    train_loader, val_loader, dataset = create_dataloaders(config)
```

And for safety also patch `run_pretraining` the same way — find:

```python
def run_pretraining(config):
    """Run self-supervised pretraining."""
    logger.info("\n" + "="*60)
    logger.info("Starting Self-Supervised Pretraining")
    logger.info("="*60)
```

Replace with:

```python
def run_pretraining(config):
    """Run self-supervised pretraining."""
    logger.info("\n" + "="*60)
    logger.info("Starting Self-Supervised Pretraining")
    logger.info("="*60)
    _set_all_seeds(config.get('seed', 42))  # Path 2: model-RNG seed
```

---

## 2) `scripts/evaluate.py` — two edits

### 2a. Add the seeding helper

Find this line in `evaluate.py` (near the top, after the imports block):

```python
logger = logging.getLogger(__name__)
```

Immediately **AFTER** that line, insert:

```python


def _set_all_seeds(seed: int) -> None:
    """Path 2 reproducibility: identical helper to train.py._set_all_seeds."""
    import random as _py_random
    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 2b. Fix the split seed in `_load_dataset`

Find this block (inside `_load_dataset` — it's the only `create_train_val_test_split` call in evaluate.py):

```python
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=config.get('seed', 42),
    )
```

Replace with:

```python
    # Path 2: split seed = data.random_seed (FIXED across all seeds).
    split_seed = config.get('data', {}).get('random_seed', config.get('seed', 42))
    train_edges, val_edges, test_edges = dataset.create_train_val_test_split(
        train_ratio=config['data'].get('train_ratio', 0.8),
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.1),
        random_seed=split_seed,
    )
```

### 2c. Seed RNGs at the top of `main()`

Find this block:

```python
def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptGFM-Bio')
```

Replace with:

```python
def main():
    parser = argparse.ArgumentParser(description='Evaluate PromptGFM-Bio')
    # Path 2 reproducibility — model-RNG seed comes from config['seed'].
    # We can't seed before parsing args (config path is an arg), so the seed
    # call is moved just after we load the config below.
```

Then further down in `main()`, find the line that loads the config (commonly `config = load_config(args.config)` or `with open(args.config) as f: config = yaml.safe_load(f)`). Immediately AFTER that line, add:

```python
    _set_all_seeds(config.get('seed', 42))
```

---

## 3) The configs need NO changes

Every ablation YAML already contains `data: { random_seed: 42, ... }` and `seed: 42` at the top level. The Path 2 code reads both correctly. The sed override in the runner script (which rewrites only the top-level `seed:` line) now correctly varies model RNG without touching the split.

---

## 4) Verify the patches applied

After saving both files, from `E:\Workstation-7-4500-Ada-Backup\projects_yash\new_project\PromptGFM-Bio` in PowerShell:

```powershell
# train.py — should print 3 hits
Select-String -Path "scripts\train.py" -Pattern "_set_all_seeds|split_seed = config"

# evaluate.py — should print 3 hits
Select-String -Path "scripts\evaluate.py" -Pattern "_set_all_seeds|split_seed = config"
```

If you see fewer hits, one block didn't take — re-check indentation (Python is sensitive). Then commit and push the patched files to your private GitHub repo.

---

## 5) Why this is the Q1-correct fix

Before this patch: top-level `seed:` controlled only the *data split*. Negative sampling (`random.sample`), model init (`torch.manual_seed` was never called from config), and DataLoader order were driven by **process-time randomness** — so even running the *same* config twice could give different numbers, and "seed variance" across 42/43/44 conflated split-variance with process-noise.

After this patch: the train/val/test split is **bit-identical across all 10 seeds** (seeded by `data.random_seed=42`), the 117-disease zero-shot set is therefore valid for *every* seed (no leakage to audit), and the per-seed std reflects exclusively **model-initialization variance** — the standard, defensible quantity a reviewer expects from "mean ± std over 10 seeds." Two methodological footnotes get crossed off the reviewer's list before they're written.
