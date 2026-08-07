#!/usr/bin/env bash
# End-to-end check: recompute from caches + match Tables 1–11.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/reproduce_all.sh

echo "=== Extra: optional tools fail clearly without their downloads ==="
uv run --frozen --extra from-scratch python scripts/run_eigenthemes.py --dataset quotebank >/tmp/eigen_msg.txt 2>&1 || true
grep -q "Eigenthemes tree not found" /tmp/eigen_msg.txt
uv run --frozen --extra from-scratch python scripts/run_mgenre.py --dataset quotebank >/tmp/mgenre_msg.txt 2>&1 || true
grep -Eq "Missing|No module named 'genre'|No module named \"genre\"" /tmp/mgenre_msg.txt

echo "=== AUDIT PASS ==="
