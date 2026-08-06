#!/usr/bin/env bash
# One-shot paper table reproduction (method dumps + validated conversions).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 0. Faithfulness / cache wiring audit ==="
python3 scripts/audit_faithfulness.py

echo "=== 1. Materialize mGENRE from original dumps / raw beams ==="
python3 scripts/convert_mgenre_raw.py

echo "=== 2. Cache-based Table 2 / 3 / 11 (scripts/reproduce_tables.py) ==="
python3 scripts/reproduce_tables.py

echo "=== 3. Cross-check Table 2 (from_scratch artifacts) ==="
python3 scripts/reproduce_paper_from_scratch.py

echo "=== 4. Tables 3 / 6 / 7 / 8 / 11 ==="
python3 scripts/reproduce_remaining_tables.py

echo "=== 5. All paper tables 1–11 ==="
python3 scripts/reproduce_all_paper_tables.py

echo "=== Done ==="
echo "See artifacts/all_paper_tables.json (Tables 1–11),"
echo "    artifacts/reproduced_tables.json, artifacts/from_scratch/table2_from_scratch.json,"
echo "    artifacts/remaining_tables.json, artifacts/from_scratch/eigen_cse_provenance.json"
echo "Heuristic from-scratch: PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both --with-embeddings"
echo "Eigenthemes from-scratch: python3 scripts/run_eigenthemes.py --reuse-raw"
echo "mGENRE live GPU (optional): source scripts/mgenre_env.sh && python scripts/run_mgenre.py ..."
