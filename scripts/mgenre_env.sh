#!/usr/bin/env bash
# Optional env for live mGENRE / Eigenthemes GPU runs.
# Set FAIRSEQ_ROOT / CONDA_ENV as needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-$ROOT/workspace/speaker-disambiguation-quotebank/notebooks/fairseq/fairseq}"
CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "$CONDA_ENV" && -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="${FAIRSEQ_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
