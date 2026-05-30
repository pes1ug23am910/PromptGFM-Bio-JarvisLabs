#!/usr/bin/env python3
"""
make_shepherd_input.py
Convert the 117 zero-shot diseases into SHEPHERD's JSON-lines "patient" format.

SHEPHERD input contract (per the mims-harvard/SHEPHERD README):
  one JSON object per line, each with:
    "id"                  : patient/disease identifier
    "positive_phenotypes" : list of HPO term IDs   (e.g. "HP:0001250")
    "true_genes"          : list of causal genes as ENSEMBL IDs
    "all_candidate_genes" : list of candidate genes as ENSEMBL IDs (for causal gene discovery)

Run after prepare_baseline_inputs.py (PromptGFM venv):
    python scripts/make_shepherd_input.py

Output: data/baselines/shepherd_patients.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data/baselines"

hpo  = json.load(open(B / "disease_hpo_terms.json"))
true = json.load(open(B / "disease_true_genes.json"))
s2e  = json.load(open(B / "symbol_to_ensembl.json"))
cand = json.load(open(B / "all_candidate_genes_ensembl.json"))

out_path = B / "shepherd_patients.jsonl"
written = skipped_no_hpo = skipped_no_true = 0

with open(out_path, "w") as f:
    for did, terms in hpo.items():
        if not terms:
            skipped_no_hpo += 1
            continue
        true_ens = [s2e[g] for g in true.get(did, []) if g in s2e]
        if not true_ens:
            # No mappable ground-truth gene -> cannot score causal gene discovery.
            skipped_no_true += 1
            continue
        rec = {
            "id": did,
            "positive_phenotypes": terms,
            "true_genes": true_ens,
            "all_candidate_genes": cand,
        }
        f.write(json.dumps(rec) + "\n")
        written += 1

print(f"[shepherd] wrote {written} patients -> {out_path}")
print(f"[shepherd] skipped {skipped_no_hpo} (no HPO terms), "
      f"{skipped_no_true} (no Ensembl-mappable true gene)")
print("[shepherd] NOTE the written/skipped counts; report them in the paper's "
      "methods (comparison is over the diseases all methods can score).")
