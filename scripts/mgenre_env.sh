#!/usr/bin/env bash
# Activate the live-mGENRE environment created by scripts/setup_mgenre.sh.
#
# Usage:
#   bash scripts/setup_mgenre.sh          # once
#   source scripts/mgenre_env.sh
#   python scripts/run_mgenre.py --dataset quotebank --context 128 --device cuda:0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MGENRE_VENV:-$ROOT/.venv-mgenre}"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-$ROOT/third_party/fairseq}"
GENRE_ROOT="${GENRE_ROOT:-$ROOT/third_party/GENRE}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV — run: bash scripts/setup_mgenre.sh" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "$FAIRSEQ_ROOT/fairseq" ]]; then
  echo "missing fairseq at $FAIRSEQ_ROOT — run: bash scripts/setup_mgenre.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export FAIRSEQ_ROOT
export GENRE_ROOT
export PYTHONPATH="${FAIRSEQ_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "mGENRE env: $VENV"
echo "FAIRSEQ_ROOT=$FAIRSEQ_ROOT"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
