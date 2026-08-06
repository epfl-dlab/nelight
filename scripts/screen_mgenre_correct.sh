#!/usr/bin/env bash
# Optional helper for live mGENRE smoke checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/mgenre_env.sh"
cd "$ROOT"
python scripts/run_mgenre.py --dataset quotebank --context 128 --help >/dev/null
echo "mGENRE env OK (ROOT=$ROOT)"
