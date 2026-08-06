#!/usr/bin/env bash
# End-to-end entity-KB / embedding cache rebuild.
# Requires a Wikidata JSON dump (+ optional Wikipedia first-paragraph assets).
#
#   DUMP=/path/to/wikidata-YYYYMMDD-all.json.gz bash cache_building/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi

uv sync --frozen
run() { uv run --frozen "$@"; }

DUMP="${DUMP:?Set DUMP=/path/to/wikidata-YYYYMMDD-all.json.gz}"
OUT="${OUT:-artifacts/cache_build}"
DATA_QB="${DATA_QB:-data/Quotebank/data.json}"
DATA_AIDA="${DATA_AIDA:-data/AIDA/data.json}"
mkdir -p "$OUT"

run python cache_building/collect_candidate_qids.py \
  --data "$DATA_QB" --data "$DATA_AIDA" \
  --out "$OUT/candidate_qids.pkl"

run python cache_building/extract_wikidata_subgraph.py \
  --dump "$DUMP" --qids "$OUT/candidate_qids.pkl" \
  --out "$OUT/wikidata_subgraph.json.gz"

run python cache_building/extract_entity_metadata.py \
  --dump "$DUMP" --qids "$OUT/candidate_qids.pkl" \
  --out-dir "$OUT/entity_metadata"

EXTRA=()
[[ -n "${QID_PID:-}" && -n "${FIRST_PARAGRAPHS:-}" ]] && \
  EXTRA+=(--qid-pid "$QID_PID" --first-paragraphs "$FIRST_PARAGRAPHS")
[[ -n "${WP_RANKS:-}" ]] && EXTRA+=(--wp-ranks "$WP_RANKS")
[[ -n "${WD_RANKS:-}" ]] && EXTRA+=(--wd-ranks "$WD_RANKS")

run python cache_building/build_entity_kb.py \
  --dump "$OUT/wikidata_subgraph.json.gz" \
  --labels-dir "$OUT/entity_metadata" \
  --qids "$OUT/candidate_qids.pkl" \
  --out "$OUT/entity_kb.pkl" \
  "${EXTRA[@]}"

run python cache_building/build_unambiguous_mentions.py \
  --data "$DATA_QB" --out "$OUT/unambiguous_mentions_quotebank.pkl"
run python cache_building/build_unambiguous_mentions.py \
  --data "$DATA_AIDA" --out "$OUT/unambiguous_mentions_aida.pkl"

echo "Text / KB caches written under $OUT"
echo "Optional GPU embeddings (needs: uv sync --extra from-scratch):"
echo "  uv run --extra from-scratch python cache_building/build_text_embeddings.py entity \\"
echo "      --entity-kb $OUT/entity_kb.pkl --out $OUT/entity_embeddings.pkl"
echo "  uv run --extra from-scratch python cache_building/build_text_embeddings.py document \\"
echo "      --data $DATA_QB --out $OUT/document_embeddings.pkl"
echo "  uv run --extra from-scratch python cache_building/build_text_embeddings.py mention \\"
echo "      --data $DATA_QB --out $OUT/mention_embeddings.pkl"
echo "See cache_building/README.md"
