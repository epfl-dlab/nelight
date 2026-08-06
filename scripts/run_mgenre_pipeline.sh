#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/mgenre_env.sh"
cd "$ROOT"
python scripts/run_mgenre.py --dataset quotebank --context 128
python scripts/run_mgenre.py --dataset aida --context 256
