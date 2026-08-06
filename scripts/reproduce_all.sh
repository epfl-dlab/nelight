#!/usr/bin/env bash
# One-shot paper table reproduction (method dumps + validated conversions).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi

# Idempotent: create .venv from the committed lockfile (no resolver drift).
uv sync --frozen

run() { uv run --frozen "$@"; }

echo "=== 0. Faithfulness / cache wiring audit ==="
run python scripts/audit_faithfulness.py

echo "=== 1. Materialize mGENRE from original dumps / raw beams ==="
run python scripts/convert_mgenre_raw.py

echo "=== 2. Cache-based Table 2 / 3 / 11 (scripts/reproduce_tables.py) ==="
run python scripts/reproduce_tables.py

echo "=== 3. Cross-check Table 2 (from_scratch artifacts) ==="
run python scripts/reproduce_paper_from_scratch.py

echo "=== 4. All paper tables 1–11 ==="
run python scripts/reproduce_all_paper_tables.py

echo "=== Done ==="
echo "See artifacts/all_paper_tables.json (Tables 1–11),"
echo "    artifacts/reproduced_tables.json, artifacts/from_scratch/table2_from_scratch.json"
echo "Heuristic from-scratch: uv sync --extra from-scratch && uv run --extra from-scratch python scripts/run_heuristics.py --dataset both"
echo "mGENRE on GPU: bash scripts/setup_mgenre.sh && source scripts/mgenre_env.sh && python scripts/run_mgenre.py --dataset quotebank --context 128"
