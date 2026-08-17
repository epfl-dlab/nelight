#!/usr/bin/env bash
# Recompute paper tables from in-repo caches (no GPU, no Wikidata dump).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

echo "=== 0. Env ==="
uv sync --frozen --extra from-scratch
run() { uv run --frozen --extra from-scratch "$@"; }
run python -c \
  "import nltk; [nltk.download(p, quiet=True) for p in
   ('punkt','punkt_tab','wordnet','omw-1.4','stopwords')]"

echo "=== 1. Assets ==="
run python scripts/check_assets.py

echo "=== 2. Heuristics + embeddings → artifacts/from_scratch ==="
run python scripts/run_heuristics.py --dataset both --with-embeddings

echo "=== 3. IScore ablation (Table 6) ==="
run python scripts/run_iscore_ablation.py

echo "=== 4. Tables 1–11 (merges shipped Eigen/mGENRE) ==="
run python scripts/reproduce_all_paper_tables.py

echo "=== Done → artifacts/all_paper_tables.json ==="
