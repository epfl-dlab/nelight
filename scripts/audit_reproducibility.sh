#!/usr/bin/env bash
# End-to-end check: tables + heuristics from shipped caches.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 1
fi

echo "=== A. Tables from saved scores ==="
rm -rf .venv
uv sync --frozen
bash scripts/reproduce_all.sh

echo "=== B. Heuristics from caches (no embeddings) ==="
uv sync --frozen --extra from-scratch
uv run --frozen --extra from-scratch python -c \
  "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','wordnet','omw-1.4','stopwords')]"
OUT=artifacts/audit_from_scratch
rm -rf "$OUT"
uv run --frozen --extra from-scratch python scripts/run_heuristics.py \
  --dataset both --out "$OUT"

echo "=== C. Heuristics with embeddings ==="
uv run --frozen --extra from-scratch python scripts/run_heuristics.py \
  --dataset both --with-embeddings --out "$OUT"

echo "=== D. Check overall accuracy vs Table 2 ==="
uv run --frozen python - <<'PY'
import json, sys
TARGETS = {
  "LQID": (0.727, 0.554), "NP": (0.788, 0.536), "NS": (0.829, 0.588),
  "PRWD": (0.673, 0.517), "PRWP": (0.824, 0.607), "IScore": (0.922, 0.632),
  "NIScore": (0.898, 0.589), "EEIScore": (0.906, 0.562), "UIScore": (0.943, 0.621),
  "CSE": (0.833, 0.290), "CSSVE": (0.784, 0.471), "UCSE": (0.882, 0.363),
}
qb = json.load(open("artifacts/audit_from_scratch/quotebank/metrics.json"))
aida = json.load(open("artifacts/audit_from_scratch/aida/metrics.json"))
ok = True
for m, (tq, ta) in TARGETS.items():
    dq = qb["overall"][m] - tq
    da = aida["overall"][m] - ta
    tol = 0.02 if m in ("CSSVE", "UCSE") else 0.002
    good = abs(dq) < 0.002 and abs(da) < tol
    print(f"{m:10s} qb={qb['overall'][m]:.3f} aida={aida['overall'][m]:.3f} {'OK' if good else 'FAIL'}")
    ok &= good
sys.exit(0 if ok else 1)
PY

echo "=== E. Optional tools fail clearly without their downloads ==="
uv run --frozen python scripts/run_eigenthemes.py --dataset quotebank >/tmp/eigen_msg.txt 2>&1 || true
grep -q "Eigenthemes tree not found" /tmp/eigen_msg.txt
# mGENRE without setup: missing model path (or genre package if models exist but env was not set up)
uv run --frozen python scripts/run_mgenre.py --dataset quotebank >/tmp/mgenre_msg.txt 2>&1 || true
grep -Eq "Missing|No module named 'genre'|No module named \"genre\"" /tmp/mgenre_msg.txt

echo "=== AUDIT PASS ==="
