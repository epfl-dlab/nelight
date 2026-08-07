#!/usr/bin/env bash
# Recompute heuristics from shipped caches, then rebuild Tables 1–11.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi

echo "=== 0. Environment (includes NLTK / torch for from-scratch scorers) ==="
uv sync --frozen --extra from-scratch
run() { uv run --frozen --extra from-scratch "$@"; }

run python -c \
  "import nltk; [nltk.download(p, quiet=True) for p in
   ('punkt','punkt_tab','wordnet','omw-1.4','stopwords')]"

echo "=== 1. Faithfulness / cache wiring audit ==="
run python scripts/audit_faithfulness.py

echo "=== 2. Recompute heuristics from caches (both datasets + embeddings) ==="
run python scripts/run_heuristics.py --dataset both --with-embeddings

echo "=== 3. Recompute IScore ablation (Table 6) ==="
run python scripts/run_iscore_ablation.py

echo "=== 4. Materialize mGENRE score dicts from original beam dumps ==="
run python scripts/convert_mgenre_raw.py
# Ensure mGENRE lands in ranked_scores after convert
run python - <<'PY'
import pickle
from pathlib import Path
FS = Path("artifacts/from_scratch")
for ds, pref in [("quotebank", "mGENRE_t128.pkl"), ("aida", "mGENRE_t256.pkl")]:
    ranked_p = FS / ds / "ranked_scores.pkl"
    best_p = FS / ds / "mGENRE_best.pkl"
    pref_p = FS / ds / pref
    src = best_p if best_p.exists() else pref_p
    if not ranked_p.exists() or not src.exists():
        continue
    ranked = pickle.load(open(ranked_p, "rb"))
    ranked["mGENRE"] = pickle.load(open(src, "rb"))
    pickle.dump(ranked, open(ranked_p, "wb"))
    print(f"merged mGENRE into {ranked_p}")
PY

echo "=== 5. Table 2 / 3 / 11 from recomputed scores ==="
run python scripts/reproduce_tables.py

echo "=== 6. Table 2 cross-check (from_scratch artifacts) ==="
run python scripts/reproduce_paper_from_scratch.py

echo "=== 7. All paper tables 1–11 ==="
run python scripts/reproduce_all_paper_tables.py

echo "=== Done ==="
echo "See artifacts/all_paper_tables.json (Tables 1–11),"
echo "    artifacts/reproduced_tables.json, artifacts/from_scratch/table2_from_scratch.json"
echo "mGENRE on GPU (optional): bash scripts/setup_mgenre.sh && source scripts/mgenre_env.sh && python scripts/run_mgenre.py --dataset quotebank --context 128"
