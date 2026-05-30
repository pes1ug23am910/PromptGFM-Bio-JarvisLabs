#!/usr/bin/env python3
"""
aggregate_results.py — Path 2 ten-seed aggregator for PromptGFM-Bio.

Pools the per-(variant, seed) result JSONs produced by
run_evaluations_all_seeds-jarvis.sh and produces the paper's headline tables
with full statistical rigor:

  * mean +/- std (ddof=1) per metric per variant, for:
      - standard test split
      - 4 rarity-stratified bins (ultra_rare / very_rare / rare / common)
      - the 117-disease zero-shot set
  * paired tests (reference variant vs each ablation, paired by seed):
      Student paired-t AND Wilcoxon signed-rank, with Holm-Bonferroni
      correction across the (metric x comparison) family within each table.
  * power analysis: with the observed n paired seeds, the minimum detectable
      effect (MDE) at alpha=0.05, power=0.8, reported in standardized units
      and in metric units (via the observed std of the per-seed differences).
  * bootstrap 95% CIs:
      - ALWAYS: seed-level (resample the n seed means) -> CI on the across-seed
        mean. Works from the aggregate JSONs you already have.
      - IF per-disease sidecars exist (see the evaluate.py patch in
        PATH2_PATCHES_v2.md): disease-level (resample the 117 zero-shot
        diseases) for Hit@10 / Hit@50 / MRR (and a macro per-disease AUROC).
        This is the stronger "would it hold on a different draw of rare
        diseases?" claim the handover asked for.
  * LaTeX-ready tables (main zero-shot + standard test) and a JSON dump.

Metric conventions mirror src/evaluation/metrics.py exactly:
  Hit@K  -> JSON key 'hit_rate@K'  (>=1 true gene in top-K; == best_rank <= K)
  MRR    -> JSON key 'mrr'         (1-indexed rank of FIRST positive, averaged)
  AUROC  -> JSON key 'auroc'       (GLOBAL pooled ROC over all candidate pairs;
                                    hence NOT disease-resampled — see above)

Dependencies: numpy, scipy (both already in requirements.txt). No statsmodels.

Usage
-----
  # after the 40 evals land under results/<variant>_seed<NN>/
  python aggregate_results.py --results-root results --out-dir results/aggregate

  # dry-run on the OLD 3-seed pilot just to see the machinery (will warn n<10):
  python aggregate_results.py --results-root results_pilot_OLD_broken_seed_flow \
      --out-dir /tmp/agg_pilot

Notes
-----
* The script prints a MANIFEST of every (variant, seed, file) it pools so you
  can SEE nothing stale (e.g. broken-seed-flow 'seed_42' dirs) is included.
* It de-duplicates on (variant, seed); duplicates are warned and the first
  lexicographic path wins. Point --results-root only at your Path 2 results.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as sps
except Exception as exc:  # pragma: no cover
    print(f"FATAL: scipy is required ({exc!r}). pip install scipy", file=sys.stderr)
    raise

# --------------------------------------------------------------------------- #
# Configuration: variant identity, display names, which metrics go in tables.
# --------------------------------------------------------------------------- #

VARIANT_ORDER = [
    "ablation_1_mlp_only",
    "ablation_2_prompt_only",
    "ablation_3_gnn_only",
    "ablation_4_full_model",
]
DISPLAY = {
    "ablation_1_mlp_only": "MLP only",
    "ablation_2_prompt_only": "Prompt only",
    "ablation_3_gnn_only": "GNN only",
    "ablation_4_full_model": "Full model",
}
REFERENCE = "ablation_4_full_model"  # the model every ablation is compared to

# Metric key in JSON -> (pretty label, n_decimals, higher_is_better)
ZS_TABLE = ["auroc", "hit_rate@10", "hit_rate@50", "hit_rate@100", "mrr"]
TEST_TABLE = ["auroc", "aupr", "hit_rate@10", "hit_rate@50", "hit_rate@100", "mrr", "map"]
STRAT_BINS = ["ultra_rare", "very_rare", "rare", "common"]
STRAT_METRIC = "hit_rate@50"
BOOTSTRAP_METRICS_DISEASE = ["hit_rate@10", "hit_rate@50", "mrr"]  # per-disease-mean safe

PRETTY = {
    "auroc": "AUROC", "aupr": "AUPR", "map": "MAP", "mrr": "MRR",
    "hit_rate@10": "Hit@10", "hit_rate@20": "Hit@20",
    "hit_rate@50": "Hit@50", "hit_rate@100": "Hit@100",
    "ndcg@10": "NDCG@10",
}

# matches both 'ablation_4_full_model_seed42' and '..._seed_42' (dir or flat file)
_RE = re.compile(r"(ablation_[1-4]_[a-z_]+?)_seed_?(\d+)")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def _parse_variant_seed(path: str) -> Optional[Tuple[str, int]]:
    m = _RE.search(path.replace("\\", "/"))
    if not m:
        return None
    variant, seed = m.group(1), int(m.group(2))
    if variant not in DISPLAY:
        return None
    return variant, seed


def _discover(results_root: str, contains: str, exclude: Optional[List[str]] = None,
              must_not_contain: Optional[List[str]] = None
              ) -> Dict[Tuple[str, int], str]:
    """Return {(variant, seed): filepath} for JSONs whose basename contains
    `contains` and none of `must_not_contain`."""
    exclude = exclude or []
    must_not_contain = must_not_contain or []
    found: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for p in glob.glob(os.path.join(results_root, "**", "*.json"), recursive=True):
        base = os.path.basename(p).lower()
        if contains not in base:
            continue
        if any(m in base for m in must_not_contain):
            continue
        if any(x in p.replace("\\", "/") for x in exclude):
            continue
        vs = _parse_variant_seed(p)
        if vs is None:
            continue
        found[vs].append(p)
    out: Dict[Tuple[str, int], str] = {}
    for vs, paths in found.items():
        paths = sorted(paths)
        if len(paths) > 1:
            print(f"  [warn] {len(paths)} files for {vs}; using {paths[0]} "
                  f"(others: {paths[1:]})")
        out[vs] = paths[0]
    return out


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #

def _mean_std(vals: List[float]) -> Tuple[float, float, int]:
    a = np.asarray([v for v in vals if v is not None and not _isnan(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def _bootstrap_ci_mean(values: List[float], n_boot: int, rng: np.random.Generator,
                       ci: float = 0.95) -> Tuple[float, float]:
    a = np.asarray([v for v in values if v is not None and not _isnan(v)], dtype=float)
    if a.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def _holm_bonferroni(pvals: List[float]) -> List[float]:
    """Return Holm-adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj


def _paired_tests(ref: List[float], other: List[float]
                  ) -> Dict[str, float]:
    """ref/other are per-seed values in the SAME seed order. Returns deltas + p's."""
    r = np.asarray(ref, dtype=float)
    o = np.asarray(other, dtype=float)
    mask = ~(np.isnan(r) | np.isnan(o))
    r, o = r[mask], o[mask]
    n = r.size
    out = {"n_pairs": n, "delta": float("nan"), "t_p": float("nan"),
           "wilcoxon_p": float("nan"), "diff_std": float("nan")}
    if n < 2:
        return out
    diff = r - o
    out["delta"] = float(diff.mean())
    out["diff_std"] = float(diff.std(ddof=1)) if n > 1 else 0.0
    # paired t
    try:
        out["t_p"] = float(sps.ttest_rel(r, o).pvalue)
    except Exception:
        pass
    # Wilcoxon (nonparametric; needs some nonzero diffs)
    try:
        if np.any(diff != 0):
            out["wilcoxon_p"] = float(sps.wilcoxon(r, o).pvalue)
        else:
            out["wilcoxon_p"] = 1.0
    except Exception:
        pass
    return out


def _paired_mde(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable standardized effect size (Cohen's d) for a two-sided
    paired t-test with n pairs, via the noncentral-t power function (bisection)."""
    if n < 2:
        return float("nan")
    df = n - 1
    t_crit = sps.t.ppf(1 - alpha / 2, df)

    def power_at(d: float) -> float:
        ncp = d * np.sqrt(n)
        return float(sps.nct.sf(t_crit, df, ncp) + sps.nct.cdf(-t_crit, df, ncp))

    lo, hi = 0.0, 5.0
    if power_at(hi) < power:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if power_at(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Core aggregation for one "block" (test / a stratified bin / zero-shot)
# --------------------------------------------------------------------------- #

def _collect_block(per_variant_seed_metrics: Dict[str, Dict[int, dict]],
                   metric_keys: List[str]
                   ) -> Dict[str, Dict[str, List[float]]]:
    """variant -> metric -> [values ordered by sorted seed]."""
    block: Dict[str, Dict[str, List[float]]] = {}
    for variant in VARIANT_ORDER:
        seeds = sorted(per_variant_seed_metrics.get(variant, {}).keys())
        block[variant] = {}
        for mk in metric_keys:
            block[variant][mk] = [
                per_variant_seed_metrics[variant][s].get(mk, float("nan"))
                for s in seeds
            ]
        block[variant]["_seeds"] = seeds  # type: ignore
    return block


def _aligned_pair(block: Dict[str, Dict[str, List[float]]], variant: str, mk: str
                  ) -> Tuple[List[float], List[float]]:
    """Return (ref_vals, variant_vals) aligned on the intersection of seeds."""
    ref_seeds = block[REFERENCE]["_seeds"]  # type: ignore
    var_seeds = block[variant]["_seeds"]    # type: ignore
    common = [s for s in ref_seeds if s in var_seeds]
    ref_map = dict(zip(ref_seeds, block[REFERENCE][mk]))
    var_map = dict(zip(var_seeds, block[variant][mk]))
    return [ref_map[s] for s in common], [var_map[s] for s in common]


def _summarize_block(name: str,
                     per_variant_seed_metrics: Dict[str, Dict[int, dict]],
                     metric_keys: List[str],
                     n_boot: int, rng: np.random.Generator,
                     alpha: float, power: float) -> dict:
    block = _collect_block(per_variant_seed_metrics, metric_keys)
    summary = {"name": name, "variants": {}, "comparisons": {}, "mde": {}}

    # mean/std + seed-level bootstrap CI per variant per metric
    for variant in VARIANT_ORDER:
        if not block[variant]["_seeds"]:  # type: ignore
            continue
        summary["variants"][variant] = {"n_seeds": len(block[variant]["_seeds"])}  # type: ignore
        for mk in metric_keys:
            mean, std, n = _mean_std(block[variant][mk])
            lo, hi = _bootstrap_ci_mean(block[variant][mk], n_boot, rng)
            summary["variants"][variant][mk] = {
                "mean": mean, "std": std, "n": n, "ci95": [lo, hi],
            }

    # paired tests: reference vs each ablation, Holm across (metric x comparison)
    fam_keys: List[Tuple[str, str]] = []
    fam_pvals_t: List[float] = []
    fam_pvals_w: List[float] = []
    for variant in VARIANT_ORDER:
        if variant == REFERENCE or not block[variant]["_seeds"]:  # type: ignore
            continue
        summary["comparisons"][variant] = {}
        for mk in metric_keys:
            ref_vals, var_vals = _aligned_pair(block, variant, mk)
            res = _paired_tests(ref_vals, var_vals)
            summary["comparisons"][variant][mk] = res
            fam_keys.append((variant, mk))
            fam_pvals_t.append(res["t_p"] if not _isnan(res["t_p"]) else 1.0)
            fam_pvals_w.append(res["wilcoxon_p"] if not _isnan(res["wilcoxon_p"]) else 1.0)

    adj_t = _holm_bonferroni(fam_pvals_t) if fam_pvals_t else []
    adj_w = _holm_bonferroni(fam_pvals_w) if fam_pvals_w else []
    for (variant, mk), pt, pw in zip(fam_keys, adj_t, adj_w):
        summary["comparisons"][variant][mk]["t_p_holm"] = pt
        summary["comparisons"][variant][mk]["wilcoxon_p_holm"] = pw

    # power / MDE per metric for the headline Full-vs-MLP contrast
    mlp = "ablation_1_mlp_only"
    if mlp in block and block[mlp]["_seeds"]:  # type: ignore
        for mk in metric_keys:
            ref_vals, var_vals = _aligned_pair(block, mlp, mk)
            res = _paired_tests(ref_vals, var_vals)
            n = res["n_pairs"]
            mde_d = _paired_mde(n, alpha, power)
            summary["mde"][mk] = {
                "n_pairs": n,
                "observed_delta": res["delta"],
                "observed_diff_std": res["diff_std"],
                "mde_cohens_d": mde_d,
                "mde_metric_units": (mde_d * res["diff_std"]
                                     if not _isnan(mde_d) and not _isnan(res["diff_std"])
                                     else float("nan")),
            }
    return summary


# --------------------------------------------------------------------------- #
# Disease-level bootstrap (optional; needs per-disease sidecars)
# --------------------------------------------------------------------------- #

def _disease_level_bootstrap(per_disease_files: Dict[Tuple[str, int], str],
                             n_boot: int, rng: np.random.Generator) -> dict:
    """For each variant, average each disease's metric over seeds, then bootstrap
    the 117 diseases. Returns variant -> metric -> {mean, ci95}."""
    # variant -> metric -> disease_id -> [per-seed values]
    acc: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    metric_map = {"hit_rate@10": "hit@10", "hit_rate@50": "hit@50",
                  "mrr": "rr", "auroc_macro": "auroc"}
    for (variant, _seed), path in per_disease_files.items():
        try:
            rec = _load_json(path)
        except Exception:
            continue
        for did, d in rec.items():
            for mk, dk in metric_map.items():
                if dk in d and d[dk] is not None:
                    acc[variant][mk][did].append(float(d[dk]))

    out: Dict[str, Dict[str, dict]] = {}
    for variant, mdict in acc.items():
        out[variant] = {}
        for mk, dmap in mdict.items():
            per_disease_mean = [float(np.mean(v)) for v in dmap.values() if len(v) > 0]
            if len(per_disease_mean) < 2:
                continue
            a = np.asarray(per_disease_mean, dtype=float)
            a = a[~np.isnan(a)]
            if a.size < 2:
                continue
            idx = rng.integers(0, a.size, size=(n_boot, a.size))
            means = a[idx].mean(axis=1)
            out[variant][mk] = {
                "n_diseases": int(a.size),
                "mean": float(a.mean()),
                "ci95": [float(np.percentile(means, 2.5)),
                         float(np.percentile(means, 97.5))],
            }
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt(mean: float, std: float, dec: int = 4) -> str:
    if _isnan(mean):
        return "--"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def _sig_marker(comp: dict) -> str:
    """*** p_holm<0.001, ** <0.01, * <0.05, (.) <0.10 on the t-test (Holm)."""
    p = comp.get("t_p_holm", float("nan"))
    if _isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "(.)"
    return ""


def _print_block(summary: dict, metric_keys: List[str]) -> str:
    lines = [f"\n=== {summary['name']} ===  (mean ± std over seeds; * = vs Full, Holm-corrected)"]
    header = ["variant".ljust(14)] + [PRETTY.get(m, m).ljust(20) for m in metric_keys]
    lines.append("  ".join(header))
    for variant in VARIANT_ORDER:
        v = summary["variants"].get(variant)
        if not v:
            continue
        row = [DISPLAY[variant].ljust(14)]
        for m in metric_keys:
            cell = _fmt(v[m]["mean"], v[m]["std"])
            if variant != REFERENCE:
                cell += _sig_marker(summary["comparisons"].get(variant, {}).get(m, {}))
            row.append(cell.ljust(20))
        lines.append("  ".join(row))
    return "\n".join(lines)


def _latex_table(summary: dict, metric_keys: List[str], caption: str, label: str) -> str:
    cols = "l" + "c" * len(metric_keys)
    head = " & ".join(["Variant"] + [PRETTY.get(m, m) for m in metric_keys]) + r" \\"
    body = []
    for variant in VARIANT_ORDER:
        v = summary["variants"].get(variant)
        if not v:
            continue
        cells = [DISPLAY[variant]]
        for m in metric_keys:
            txt = "--" if _isnan(v[m]["mean"]) else f"{v[m]['mean']:.4f} $\\pm$ {v[m]['std']:.4f}"
            if variant != REFERENCE:
                txt += _sig_marker(summary["comparisons"].get(variant, {}).get(m, {}))
            if variant == REFERENCE:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + r"}", r"\label{" + label + r"}",
        r"\begin{tabular}{" + cols + r"}", r"\toprule",
        head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt]\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ "
        r"vs.\ Full model (paired $t$, Holm--Bonferroni). $\pm$ is std over seeds.}",
        r"\end{table}",
    ])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/aggregate")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="substrings; any path containing one is skipped "
                         "(e.g. pilot_OLD)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- discover ---------------------------------------------------------- #
    # Classify unambiguously by basename. This matters because the default
    # zero-shot output name 'zero_shot_evaluation_results.json' contains BOTH
    # 'zero_shot' AND 'evaluation_results':
    #   standard  = has 'evaluation_results', NOT 'zero_shot', NOT 'per_disease'
    #   zs_agg    = has 'zero_shot',          NOT 'per_disease'
    #   zs_pd     = has 'per_disease'
    eval_files = _discover(args.results_root, "evaluation_results", args.exclude,
                           must_not_contain=["zero_shot", "per_disease"])
    zs_agg = _discover(args.results_root, "zero_shot", args.exclude,
                       must_not_contain=["per_disease"])
    zs_pd = _discover(args.results_root, "per_disease", args.exclude)

    print("MANIFEST (every pooled file — verify nothing stale is here):")
    for label, d in [("evaluation_results", eval_files),
                     ("zero_shot (aggregate)", zs_agg),
                     ("zero_shot (per-disease)", zs_pd)]:
        print(f"  [{label}] {len(d)} files")
        for (variant, seed), p in sorted(d.items()):
            print(f"      {variant} seed{seed:<3} <- {p}")
    n_variants = len({v for v, _ in eval_files})
    n_seeds = len({s for _, s in eval_files})
    print(f"\n  -> {n_variants} variants x {n_seeds} seeds for standard/stratified; "
          f"{len({s for _, s in zs_agg})} seeds for zero-shot.")
    if n_seeds < 10:
        print(f"  [warn] only {n_seeds} seeds found — Path 2 target is 10. "
              f"Stats/power will reflect the smaller n.")

    # ---- load into per-variant-per-seed metric dicts ----------------------- #
    test_by: Dict[str, Dict[int, dict]] = defaultdict(dict)
    strat_by: Dict[str, Dict[str, Dict[int, dict]]] = {b: defaultdict(dict) for b in STRAT_BINS}
    for (variant, seed), path in eval_files.items():
        try:
            j = _load_json(path)
        except Exception as e:
            print(f"  [warn] could not read {path}: {e!r}")
            continue
        if "test" in j:
            test_by[variant][seed] = j["test"]
        if "stratified" in j:
            for b in STRAT_BINS:
                if b in j["stratified"]:
                    strat_by[b][variant][seed] = j["stratified"][b]

    zs_by: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for (variant, seed), path in zs_agg.items():
        try:
            zs_by[variant][seed] = _load_json(path)
        except Exception as e:
            print(f"  [warn] could not read {path}: {e!r}")

    # ---- summarize blocks -------------------------------------------------- #
    results = {"meta": {"results_root": args.results_root, "bootstrap": args.bootstrap,
                        "alpha": args.alpha, "power": args.power,
                        "reference": REFERENCE}}
    out_text: List[str] = []

    zs_sum = _summarize_block("Zero-shot (117 rare diseases)", zs_by, ZS_TABLE,
                              args.bootstrap, rng, args.alpha, args.power)
    test_sum = _summarize_block("Standard test split", test_by, TEST_TABLE,
                                args.bootstrap, rng, args.alpha, args.power)
    results["zero_shot"] = zs_sum
    results["test"] = test_sum
    out_text.append(_print_block(zs_sum, ZS_TABLE))
    out_text.append(_print_block(test_sum, TEST_TABLE))

    # stratified Hit@50 table
    strat_sum = {"name": f"Stratified {PRETTY[STRAT_METRIC]} by rarity", "bins": {}}
    strat_lines = [f"\n=== {strat_sum['name']} ==="]
    strat_lines.append("variant".ljust(14) + "  " +
                       "  ".join(b.ljust(14) for b in STRAT_BINS))
    bin_blocks = {b: _summarize_block(b, strat_by[b], [STRAT_METRIC],
                                      args.bootstrap, rng, args.alpha, args.power)
                  for b in STRAT_BINS}
    strat_sum["bins"] = bin_blocks
    for variant in VARIANT_ORDER:
        cells = [DISPLAY[variant].ljust(14)]
        ok = False
        for b in STRAT_BINS:
            v = bin_blocks[b]["variants"].get(variant)
            if v:
                ok = True
                cell = _fmt(v[STRAT_METRIC]["mean"], v[STRAT_METRIC]["std"])
                if variant != REFERENCE:
                    cell += _sig_marker(bin_blocks[b]["comparisons"].get(variant, {}).get(STRAT_METRIC, {}))
                cells.append(cell.ljust(14))
            else:
                cells.append("--".ljust(14))
        if ok:
            strat_lines.append("  ".join(cells))
    results["stratified"] = strat_sum
    out_text.append("\n".join(strat_lines))

    # ---- disease-level bootstrap (optional) -------------------------------- #
    if zs_pd:
        dl = _disease_level_bootstrap(zs_pd, args.bootstrap, rng)
        results["zero_shot_disease_bootstrap"] = dl
        dl_lines = ["\n=== Zero-shot disease-level bootstrap (resampling the 117 diseases) ==="]
        dl_lines.append("  (per-disease metric averaged over seeds, then 95% CI by disease bootstrap)")
        for variant in VARIANT_ORDER:
            if variant not in dl:
                continue
            dl_lines.append(f"  {DISPLAY[variant]}:")
            for mk in BOOTSTRAP_METRICS_DISEASE + ["auroc_macro"]:
                if mk in dl[variant]:
                    e = dl[variant][mk]
                    dl_lines.append(f"      {PRETTY.get(mk, mk):<9} "
                                    f"{e['mean']:.4f}  95% CI [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]  "
                                    f"(n={e['n_diseases']})")
        out_text.append("\n".join(dl_lines))
    else:
        out_text.append("\n[info] No per-disease sidecars found -> disease-level bootstrap "
                        "skipped. Apply the evaluate.py patch in PATH2_PATCHES_v2.md BEFORE "
                        "the 40 evals to enable it. Seed-level CIs are reported above regardless.")

    # ---- power / MDE summary ---------------------------------------------- #
    mde_lines = ["\n=== Power analysis (Full vs MLP, paired t, alpha=0.05, power=0.80) ==="]
    for mk in ZS_TABLE:
        e = zs_sum["mde"].get(mk)
        if not e:
            continue
        mde_lines.append(
            f"  zero-shot {PRETTY.get(mk, mk):<9} n={e['n_pairs']:<2} "
            f"observed Δ={e['observed_delta']:+.4f}  "
            f"MDE: d={e['mde_cohens_d']:.3f} (≈{e['mde_metric_units']:.4f} in metric units)"
        )
    out_text.append("\n".join(mde_lines))

    # ---- write outputs ----------------------------------------------------- #
    report = "\n".join(out_text) + "\n"
    print(report)

    with open(os.path.join(args.out_dir, "aggregate_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None if _isnan(o) else o)
    with open(os.path.join(args.out_dir, "stats_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out_dir, "table_zero_shot.tex"), "w") as f:
        f.write(_latex_table(zs_sum, ZS_TABLE,
                             "Zero-shot rare-disease gene prioritisation "
                             "(mean over 10 seeds).", "tab:zeroshot"))
    with open(os.path.join(args.out_dir, "table_test.tex"), "w") as f:
        f.write(_latex_table(test_sum, TEST_TABLE,
                             "Standard test-split performance (mean over 10 seeds).",
                             "tab:test"))

    print(f"\nWrote: {args.out_dir}/aggregate_summary.json, stats_report.txt, "
          f"table_zero_shot.tex, table_test.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
