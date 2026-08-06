#!/usr/bin/env bash
# Download and install everything needed to run mGENRE on a GPU.
#
# Creates third_party/{fairseq,GENRE}, models/mgenre/, and .venv-mgenre (Python 3.10).
#
#   bash scripts/setup_mgenre.sh
#   source scripts/mgenre_env.sh
#   python scripts/run_mgenre.py --dataset quotebank --context 128 --device cuda:0
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1 || ! command -v wget >/dev/null 2>&1; then
  echo "git and wget are required" >&2
  exit 1
fi

ROOT="$(pwd)"
TP="$ROOT/third_party"
MD="$ROOT/models/mgenre"
VENV="$ROOT/.venv-mgenre"

mkdir -p "$TP" "$MD"

echo "=== 1/4 Clone fairseq + GENRE ==="
if [[ ! -d "$TP/fairseq/.git" ]]; then
  git clone --depth 1 --branch fixing_prefix_allowed_tokens_fn \
    https://github.com/nicola-decao/fairseq.git "$TP/fairseq"
fi
if [[ ! -d "$TP/GENRE/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/GENRE.git "$TP/GENRE"
fi

echo "=== 2/4 Download model files (~6 GB) ==="
cd "$MD"
[[ -f fairseq_multilingual_entity_disambiguation/model.pt ]] || {
  [[ -f fairseq_multilingual_entity_disambiguation.tar.gz ]] || \
    wget -c https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz
  tar -xzf fairseq_multilingual_entity_disambiguation.tar.gz
  rm -f fairseq_multilingual_entity_disambiguation.tar.gz
}
[[ -f titles_lang_all105_marisa_trie_with_redirect.pkl ]] || \
  wget -c https://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl
[[ -f lang_title2wikidataID-normalized_with_redirect.pkl ]] || \
  wget -c https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
cd "$ROOT"

echo "=== 3/4 Create Python 3.10 environment ==="
uv python install 3.10 >/dev/null
[[ -x "$VENV/bin/python" ]] || uv venv --python 3.10 "$VENV"
# fairseq needs an older NumPy; run_mgenre.py patches torch.load for recent PyTorch
uv pip install --python "$VENV" \
  'numpy<1.24' 'torch' 'scipy' 'tqdm' 'transformers' 'marisa-trie' \
  'unidecode' 'sentencepiece' 'requests' 'beautifulsoup4' 'cython'
uv pip install --python "$VENV" --no-build-isolation -e "$TP/fairseq"
uv pip install --python "$VENV" --no-deps -e "$TP/GENRE"

echo "=== 4/4 Check imports ==="
export PYTHONPATH="$TP/fairseq${PYTHONPATH:+:$PYTHONPATH}"
"$VENV/bin/python" - <<'PY'
import torch
import fairseq
from genre.fairseq_model import mGENRE
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("fairseq", fairseq.__version__)
print("ready")
PY

cat <<EOF

Done. Next:

  source scripts/mgenre_env.sh
  python scripts/run_mgenre.py --dataset quotebank --context 128 --device cuda:0
  python scripts/run_mgenre.py --dataset aida --context 256 --device cuda:0

Scores go to artifacts/from_scratch/.../mGENRE_live_t*.pkl
(the saved paper scores are left unchanged).
EOF
