#!/usr/bin/env bash
# Independent re-run (run 3) of every mutation result reported in the paper,
# from a clean state, into files ending in _run3. Run from the repository root.
# cosmic-ray edits the gate module in place while it runs, so nothing else may
# run against src/ at the same time.
set -e
python -m pytest tests/ -q -p no:cacheprovider
python scripts/cross_gate_mutation_study.py --out scripts/cross_gate_mutation_results_run3.json
for g in "" _security _compliance; do
  cfg=mutation_study/cr_config$g.toml
  db=mutation_study/session${g}_run3.sqlite
  rm -f "$db"
  python -m cosmic_ray.cli init "$cfg" "$db"
  python -m cosmic_ray.cli exec "$cfg" "$db"
  python -m cosmic_ray.cli dump "$db" > "mutation_study/cosmic_ray_dump${g:-_fairness}_run3.jsonl"
  echo "cosmic-ray ${g:-_fairness} done"
done
git status --short src/
