#!/usr/bin/env bash
# Full from-scratch table rebuild + match check.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/reproduce_all.sh
python3 - <<'PY'
import json
s = json.load(open("artifacts/all_paper_tables.json"))["summary"]
assert s["all_match"], s
print("AUDIT PASS", s)
PY
