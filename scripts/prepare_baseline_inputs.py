#!/usr/bin/env python3
"""
prepare_baseline_inputs.py
Build the shared inputs every external baseline (SHEPHERD, Phrank, LIRICAL,
PubMedBERT-cosine, LLM-direct) needs for the 117 zero-shot rare diseases.

Run in the PromptGFM venv (pure pandas + mygene; no GPU):
    python scripts/prepare_baseline_inputs.py

Outputs (data/baselines/):
    disease_hpo_terms.json    {disease_id: [HP:xxxxxxx, ...]}     (model-equivalent input)
    disease_true_genes.json   {disease_id: [SYMBOL, ...]}         (ground truth, vocab-restricted)
    symbol_to_ensembl.json    {SYMBOL: ENSG..., ...}              (SHEPHERD needs Ensembl)
    all_candidate_genes_symbols.json  [SYMBOL, ...]               (full ranking vocab)
    all_candidate_genes_ensembl.json  [ENSG..., ...]
    prep_report.txt           coverage diagnostics — READ THIS before proceeding
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZS_PATH      = ROOT / "data/splits/zero_shot_rare_diseases.json"
HPOA_PATH    = ROOT / "data/raw/hpo/phenotype.hpoa"
ORPHA4_PATH  = ROOT / "data/raw/orphanet/en_product4.xml"
EDGES_PATH   = ROOT / "data/processed/hpo_gene_disease_edges.csv"
GRAPH_PATH   = ROOT / "data/processed/biomedical_graph.pt"
OUT_DIR      = ROOT / "data/baselines"
MIN_SCORE    = 0.3   # must match training config

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_zero_shot_ids():
    ids = json.load(open(ZS_PATH))["disease_ids"]
    print(f"[zs] {len(ids)} zero-shot diseases "
          f"({sum(i.startswith('OMIM') for i in ids)} OMIM, "
          f"{sum(i.startswith('ORPHA') for i in ids)} ORPHA)")
    return ids


def full_vocab_symbols():
    """Authoritative gene vocabulary = exactly what the model ranks (gene_to_idx)."""
    from src.data.dataset import GeneDiseaseDataset
    ds = GeneDiseaseDataset(graph_path=str(GRAPH_PATH),
                            edges_path=str(EDGES_PATH),
                            min_score=MIN_SCORE)
    symbols = sorted(ds.gene_to_idx.keys())
    print(f"[vocab] {len(symbols)} candidate genes (from gene_to_idx)")
    return symbols


def disease_to_hpo(zs_ids):
    """OMIM+ORPHA -> HPO terms from phenotype.hpoa, with ORPHA fallback to product4."""
    hpoa = pd.read_csv(HPOA_PATH, sep="\t", comment="#")
    # Be tolerant of header naming across HPO releases.
    cols = {c.lower(): c for c in hpoa.columns}
    dis_col = cols.get("database_id") or cols.get("databaseid") or list(hpoa.columns)[0]
    hpo_col = cols.get("hpo_id") or cols.get("hpoid")
    if hpo_col is None:
        raise SystemExit(f"[FATAL] could not find HPO id column in {HPOA_PATH}; "
                         f"columns are: {list(hpoa.columns)}")
    print(f"[hpoa] using disease column '{dis_col}', hpo column '{hpo_col}'")
    dis2hpo = (hpoa.groupby(dis_col)[hpo_col]
                   .apply(lambda s: sorted(set(s.dropna()))).to_dict())

    orpha2hpo = {}
    if ORPHA4_PATH.exists():
        tree = ET.parse(ORPHA4_PATH)
        for dis in tree.iter("Disorder"):
            code = dis.findtext("OrphaCode")
            terms = sorted({h.findtext("HPOId") for h in dis.iter("HPO")
                            if h.findtext("HPOId")})
            if code and terms:
                orpha2hpo[f"ORPHA:{code}"] = terms

    out = {}
    for d in zs_ids:
        out[d] = dis2hpo.get(d) or orpha2hpo.get(d) or []
    return out


def disease_to_true_genes(zs_ids, vocab):
    edges = pd.read_csv(EDGES_PATH)
    edges = edges[edges["score"] >= MIN_SCORE]
    vocab_set = set(vocab)
    g = (edges.groupby("disease")["gene"]
              .apply(lambda s: sorted(set(s) & vocab_set)).to_dict())
    return {d: g.get(d, []) for d in zs_ids}


def symbol_to_ensembl(symbols):
    try:
        import mygene
    except ImportError:
        raise SystemExit("[FATAL] pip install mygene  (needed for SHEPHERD Ensembl IDs)")
    mg = mygene.MyGeneInfo()
    res = mg.querymany(symbols, scopes="symbol", fields="ensembl.gene",
                       species="human", returnall=True)
    mapping = {}
    for hit in res["out"]:
        sym = hit.get("query")
        ens = hit.get("ensembl")
        if isinstance(ens, list):
            ens = ens[0].get("gene") if ens and isinstance(ens[0], dict) else None
        elif isinstance(ens, dict):
            ens = ens.get("gene")
        if sym and ens:
            mapping[sym] = ens
    return mapping


def main():
    zs = load_zero_shot_ids()
    vocab = full_vocab_symbols()
    hpo = disease_to_hpo(zs)
    true = disease_to_true_genes(zs, vocab)
    s2e = symbol_to_ensembl(vocab)

    cand_sym = vocab
    cand_ens = [s2e[s] for s in vocab if s in s2e]

    json.dump(hpo,  open(OUT_DIR / "disease_hpo_terms.json", "w"), indent=2)
    json.dump(true, open(OUT_DIR / "disease_true_genes.json", "w"), indent=2)
    json.dump(s2e,  open(OUT_DIR / "symbol_to_ensembl.json", "w"), indent=2)
    json.dump(cand_sym, open(OUT_DIR / "all_candidate_genes_symbols.json", "w"), indent=2)
    json.dump(cand_ens, open(OUT_DIR / "all_candidate_genes_ensembl.json", "w"), indent=2)

    n_hpo  = sum(1 for v in hpo.values() if v)
    n_true = sum(1 for v in true.values() if v)
    report = [
        "PromptGFM-Bio baseline-input prep report",
        "=" * 48,
        f"zero-shot diseases            : {len(zs)}",
        f"  with >=1 HPO term           : {n_hpo}   <-- must be ~all; if low, fix hpoa columns",
        f"  with >=1 true gene in vocab : {n_true}",
        f"candidate vocabulary (symbol) : {len(cand_sym)}",
        f"  mapped to Ensembl           : {len(cand_ens)}  ({100*len(cand_ens)/len(cand_sym):.1f}%)",
        f"  unmapped (excluded from SHEPHERD candidate set, documented): {len(cand_sym)-len(cand_ens)}",
        "",
        "Diseases with NO HPO terms (investigate before running baselines):",
        *[f"  {d}" for d, v in hpo.items() if not v][:25],
    ]
    (OUT_DIR / "prep_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\n[done] wrote artifacts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
