# PATH2_PATCHES_v2.md — two pre-launch patches

These build on `PATH2_PATCHES.md` (already applied: `_set_all_seeds` + fixed
split seed). **Apply BOTH of these before you spend money on the cloud**, because:

* **Patch A** changes the numerics of negative sampling, so it must be in place
  before the smoke test and the 40 training runs (all of which are fresh under
  Path 2, so changing it now costs nothing and removes a reviewer attack
  surface).
* **Patch B** is what makes proper **disease-level bootstrap** possible. It must
  be in place before the 40 *evaluations*, or you'd have to re-run them (paying
  twice) to get the per-disease records.

Neither patch can change your headline metrics in a way that matters: Patch A
only affects *which* negatives are drawn (decorrelating them, strictly an
improvement), and Patch B is purely additive output wrapped in try/except so it
**cannot alter `metrics` or crash an eval**.

Order of operations for the whole launch:

```
apply Patch A + Patch B  ->  smoke_test_path2_v2-jarvis.sh  (must print SMOKE PASS)
  ->  run_all_seeds-jarvis.sh (40 trains)  ->  run_evaluations_all_seeds-jarvis.sh (40 evals)
  ->  aggregate_results.py
```

---

## Patch A — deterministic, decorrelated negative sampling (`scripts/train.py`)

**Why.** Negative sampling uses Python's `random` inside `collate_fn`, which runs
in the 4 DataLoader workers (`num_workers: 4`). PyTorch reseeds each worker's
torch + numpy RNG but **not** stdlib `random`; with
`multiprocessing_context="fork"` every worker inherits the *same* `random` state,
so the workers draw *correlated* negatives. Same-seed runs are still very likely
reproducible (fork inherits the seeded main-process state), but this makes it
**provably** reproducible *and* independent across workers, and robust even if the
start method ever changes to spawn. Invariant (E) in the v2 smoke test is the
empirical proof.

### A1. Add the `_seed_worker` helper

Find the end of the `_set_all_seeds` function (added by PATH2_PATCHES.md). Its
last line is:

```python
    logger.info(f"[reproducibility] all RNGs seeded with seed={seed} "
                f"(split seed comes from config['data']['random_seed'])")
```

**Immediately after that function**, add:

```python


def _seed_worker(worker_id):
    """Path 2: deterministic, decorrelated per-worker RNG.

    Negative sampling in the DataLoader collate_fn uses Python `random`
    (random.sample / random.choices). PyTorch reseeds each worker's torch and
    numpy RNG but NOT stdlib `random`; with multiprocessing_context="fork" all
    workers would otherwise inherit the SAME `random` state -> correlated
    negatives. This reseeds both `random` and numpy per worker from
    torch.initial_seed() (== main-process torch seed + worker_id, which
    _set_all_seeds fixed from config['seed']), so negatives are reproducible
    across runs AND independent across workers. Relies on
    persistent_workers=True (the reseed runs once per worker; negatives still
    vary across epochs because each worker's stream keeps advancing)."""
    import random as _py_random
    import numpy as _np
    worker_seed = torch.initial_seed() % (2 ** 32)
    _py_random.seed(worker_seed)
    _np.random.seed(worker_seed)
```

### A2. Wire it into the train DataLoader

**OLD:**
```python
    train_loader = DataLoader(
        TensorDataset(torch.arange(len(train_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),  # DistributedSampler handles shuffling
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
```
**NEW:** (one line added)
```python
    train_loader = DataLoader(
        TensorDataset(torch.arange(len(train_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),  # DistributedSampler handles shuffling
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=_seed_worker,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
```

### A3. Wire it into the val DataLoader

**OLD:**
```python
    val_loader = DataLoader(
        TensorDataset(torch.arange(len(val_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
```
**NEW:** (one line added)
```python
    val_loader = DataLoader(
        TensorDataset(torch.arange(len(val_edges))),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=_seed_worker,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
```

**Verify (PowerShell on Windows, before pushing):**
```powershell
Select-String -Path "scripts\train.py" -Pattern "_seed_worker"
# expect 3 hits: the def + the two DataLoader uses
```

> Note: this patch is *recommended*, not strictly required for determinism — the
> v2 smoke test's invariant (E) is the real gate. If (E) already passes without
> it, negatives were deterministic via fork inheritance; Patch A still improves
> sample quality (decorrelated negatives) and makes the determinism robust.

---

## Patch B — per-disease dump for disease-level bootstrap (`scripts/evaluate.py`)

**Why.** `zero_shot_results.json` is aggregate-only. To bootstrap-resample the
117 zero-shot diseases (the "would this hold on a different draw of rare
diseases?" CI a top venue expects), the aggregator needs per-disease records.
This dump is additive and guarded; it never touches `metrics`.

### B1. Add the `_dump_per_disease` helper

Find evaluate.py's copy of `_set_all_seeds` (added by PATH2_PATCHES.md). **Right
after it**, add:

```python


def _dump_per_disease(rankings, disease_ids, out_path, k_values):
    """Path 2 (additive, guarded): write per-disease ranking records so the
    aggregator can bootstrap-resample the 117 zero-shot diseases for Hit@K and
    MRR. Mirrors metrics.py exactly: 1-indexed rank of the FIRST positive;
    Hit@K == (best_rank <= K). Pooled AUROC is NOT a per-disease mean, so a
    *macro* per-disease AUROC is stored separately (clearly different from the
    headline pooled AUROC). Wrapped so it can never alter metrics or crash the
    eval — on any error it logs a warning and returns."""
    try:
        import numpy as _np
        import json as _json
        import os as _os
        from sklearn.metrics import roc_auc_score as _auroc
        records = {}
        for did, (labels, scores) in zip(disease_ids, rankings):
            labels = _np.asarray(labels)
            scores = _np.asarray(scores)
            order = _np.argsort(scores)[::-1]          # descending == metrics.py
            sorted_labels = labels[order]
            pos = _np.where(sorted_labels == 1)[0]
            best_rank = int(pos[0] + 1) if pos.size > 0 else None   # 1-indexed
            rec = {
                "num_true": int(labels.sum()),
                "best_rank": best_rank,
                "rr": (1.0 / best_rank) if best_rank is not None else 0.0,
            }
            for k in k_values:
                rec[f"hit@{k}"] = 1 if (best_rank is not None and best_rank <= k) else 0
            try:
                rec["auroc"] = (float(_auroc(labels, scores))
                                if 0 < labels.sum() < labels.size else None)
            except Exception:
                rec["auroc"] = None
            records[str(did)] = rec
        _os.makedirs(_os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as _f:
            _json.dump(records, _f, indent=2)
        logger.info(f"  [per-disease] wrote {len(records)} records -> {out_path}")
    except Exception as _e:
        logger.warning(f"  [per-disease] dump skipped ({_e!r}); metrics unaffected.")
```

### B2. Extend `evaluate_split` signature

**OLD:**
```python
def evaluate_split(model, dataset, edges_df, config, device='cuda', train_edges_df=None):
```
**NEW:**
```python
def evaluate_split(model, dataset, edges_df, config, device='cuda', train_edges_df=None,
                   per_disease_out=None):
```

### B3. Track disease ids parallel to `rankings`

**OLD:**
```python
    all_gene_indices = torch.arange(num_genes, dtype=torch.long)
    rankings = []
```
**NEW:**
```python
    all_gene_indices = torch.arange(num_genes, dtype=torch.long)
    rankings = []
    ranked_disease_ids = []  # Path 2: kept in lock-step with `rankings`
```

### B4. Append the disease id where a ranking is appended

**OLD:**
```python
            rankings.append((labels_query, query_scores_arr))
```
**NEW:**
```python
            rankings.append((labels_query, query_scores_arr))
            ranked_disease_ids.append(disease)  # Path 2: alignment with rankings
```

### B5. Dump before the (main) return

This is the return at the end of `evaluate_split` — the line is exactly
`    return metrics, flat_scores, flat_labels` (the earlier early-return returns
`{}` and must NOT be touched).

**OLD:**
```python
    return metrics, flat_scores, flat_labels
```
**NEW:**
```python
    if per_disease_out:
        _dump_per_disease(rankings, ranked_disease_ids, per_disease_out, k_values)

    return metrics, flat_scores, flat_labels
```

### B6. Thread it through `evaluate_zero_shot`

**OLD (signature):**
```python
def evaluate_zero_shot(model, dataset, all_edges_df, config, device='cuda',
                       train_edges_df=None, zero_shot_json='data/splits/zero_shot_rare_diseases.json'):
```
**NEW (signature):**
```python
def evaluate_zero_shot(model, dataset, all_edges_df, config, device='cuda',
                       train_edges_df=None, zero_shot_json='data/splits/zero_shot_rare_diseases.json',
                       per_disease_out=None):
```

**OLD (its evaluate_split call):**
```python
    metrics, flat_scores, flat_labels = evaluate_split(
        model,
        dataset,
        zs_edges,
        config,
        device,
        train_edges_df=train_edges_df,
    )
```
**NEW:**
```python
    metrics, flat_scores, flat_labels = evaluate_split(
        model,
        dataset,
        zs_edges,
        config,
        device,
        train_edges_df=train_edges_df,
        per_disease_out=per_disease_out,
    )
```

### B7. Pass a path from `main()`

**OLD:**
```python
    if args.zero_shot:
        zs_metrics = evaluate_zero_shot(
            model,
            dataset,
            dataset.edges,          # full edge set — ground truth for all diseases
            config,
            device,
            train_edges_df=train_edges,
            zero_shot_json=args.zero_shot_file,
        )
        if zs_metrics:
            save_results(zs_metrics, args.zero_shot_output)
```
**NEW:** (`Path` is already imported in evaluate.py)
```python
    if args.zero_shot:
        zs_per_disease = str(Path(args.zero_shot_output).with_suffix("")) + "_per_disease.json"
        zs_metrics = evaluate_zero_shot(
            model,
            dataset,
            dataset.edges,          # full edge set — ground truth for all diseases
            config,
            device,
            train_edges_df=train_edges,
            zero_shot_json=args.zero_shot_file,
            per_disease_out=zs_per_disease,
        )
        if zs_metrics:
            save_results(zs_metrics, args.zero_shot_output)
```

**Verify:**
```powershell
Select-String -Path "scripts\evaluate.py" -Pattern "_dump_per_disease|per_disease_out"
# expect 5+ hits
```

> The per-disease file lands next to whatever `--zero_shot_output` the eval
> runner sets per (variant, seed) — e.g.
> `results/ablation_4_full_model_seed42/zero_shot_results_per_disease.json`.
> `aggregate_results.py` discovers these automatically (it globs `*per_disease*`)
> and upgrades from seed-level to disease-level bootstrap with no extra flags.

---

## Optional, stronger determinism evidence (not required)

If a reviewer pushes hard on reproducibility, you can also assert *bitwise*
checkpoint identity for two same-seed runs (stronger than matching logged
losses). After the v2 smoke test's two seed-42 runs:

```python
import torch
a = torch.load("checkpoints/smoke_ablation_4_full_model_seed42/best_model.pt", map_location="cpu")
b = torch.load("checkpoints/smoke_ablation_4_full_model_seed42b/best_model.pt", map_location="cpu")
sa, sb = a.get("model_state_dict", a), b.get("model_state_dict", b)
print("identical:", all(torch.equal(sa[k], sb[k]) for k in sa))
```

This is over-kill for acceptance — matching per-epoch losses (invariant E) at
4-decimal reporting precision is the standard bar — but it's there if you want it.
