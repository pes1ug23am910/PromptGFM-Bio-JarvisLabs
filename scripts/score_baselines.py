#!/usr/bin/env python3
"""
score_baselines.py
Uniform, apples-to-apples scoring of every method on the 117 zero-shot diseases.

Each method must provide a rankings file:
    {disease_id: [ranked_gene_SYMBOLS, best_first]}
(SHEPHERD outputs Ensembl -> map back to symbols first; helper below.)

We report Hit@10, Hit@50, MRR -- definitions identical for ALL methods, including
PromptGFM-Bio, so the comparison is unambiguous. (AUROC is kept for your internal
ablation table; rank-based metrics are the fair cross-method currency.)

Usage:
    python scripts/score_baselines.py \
        --truth data/baselines/disease_true_genes.json \
        --method promptgfm=data/baselines/rankings/promptgfm.json \
        --method shepherd=data/baselines/rankings/shepherd_symbols.json \
        --method phrank=data/baselines/rankings/phrank.json \
        --method pubmedbert=data/baselines/rankings/pubmedbert_cosine.json \
        --out data/baselines/comparison_table.md
"""
import argparse
import json
from statistics import mean


def metrics_for_disease(ranked, true_set):
    ranks = [i + 1 for i, g in enumerate(ranked) if g in true_set]
    if not ranks:
        return dict(hit10=0.0, hit50=0.0, mrr=0.0, found=0)
    best = min(ranks)
    return dict(
        hit10=1.0 if best <= 10 else 0.0,
        hit50=1.0 if best <= 50 else 0.0,
        mrr=1.0 / best,
        found=1,
    )


def score_method(rankings, truth):
    rows = []
    for did, true_genes in truth.items():
        true_set = set(true_genes)
        if not true_set:
            continue
        ranked = rankings.get(did, [])
        rows.append(metrics_for_disease(ranked, true_set))
    n = len(rows)
    if n == 0:
        return None
    return dict(
        n=n,
        hit10=mean(r["hit10"] for r in rows),
        hit50=mean(r["hit50"] for r in rows),
        mrr=mean(r["mrr"] for r in rows),
        coverage=mean(r["found"] for r in rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--method", action="append", required=True,
                    help="name=path/to/rankings.json (repeatable)")
    ap.add_argument("--out", default="data/baselines/comparison_table.md")
    args = ap.parse_args()

    truth = json.load(open(args.truth))
    results = {}
    for spec in args.method:
        name, path = spec.split("=", 1)
        rankings = json.load(open(path))
        m = score_method(rankings, truth)
        if m is None:
            print(f"[warn] no scorable diseases for {name}")
            continue
        results[name] = m

    # Markdown table, sorted by Hit@50 desc (your method should top it)
    lines = [
        "| Method | n | Hit@10 | Hit@50 | MRR | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["hit50"]):
        lines.append(f"| {name} | {m['n']} | {m['hit10']:.3f} | "
                     f"{m['hit50']:.3f} | {m['mrr']:.3f} | {m['coverage']:.3f} |")
    table = "\n".join(lines)
    open(args.out, "w").write(table + "\n")
    print(table)
    print(f"\n[done] table -> {args.out}")


if __name__ == "__main__":
    main()
