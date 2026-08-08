#!/usr/bin/env bash
# Optional local wiring to ~/gdrive-download (does not replace LFS caches).
# Creates workspace/ symlinks for live Eigenthemes / research-tree inspection.
set -euo pipefail
cd "$(dirname "$0")/.."
GDRIVE_ROOT="${GDRIVE_ROOT:-$HOME/gdrive-download}"
QB="$GDRIVE_ROOT/downloads/quotebank_el"
ET="$GDRIVE_ROOT/downloads2/speaker-disambiguation-quotebank/eigenthemes"
SD="$GDRIVE_ROOT/downloads2/speaker-disambiguation-quotebank"

mkdir -p workspace
[[ -d "$QB" ]] && ln -sfn "$QB" workspace/quotebank_el && echo "linked workspace/quotebank_el"
[[ -d "$ET" ]] && ln -sfn "$ET" workspace/eigenthemes && echo "linked workspace/eigenthemes"
[[ -d "$SD" ]] && ln -sfn "$SD" workspace/speaker-disambiguation-quotebank && echo "linked workspace/speaker-disambiguation-quotebank"

# Checksum spot-checks (byte-identical provenance)
python3 - <<'PY'
import hashlib, os
from pathlib import Path
root = Path(".")
gdrive = Path(os.environ.get("GDRIVE_ROOT", Path.home() / "gdrive-download")) / "downloads/quotebank_el"
pairs = [
    ("caches/quotebank/entity_embeddings.pkl", "embedding_wikicache_qb_cpu.pkl"),
    ("caches/quotebank/document_embeddings.pkl", "embedding_contentcache_qb_cpu.pkl"),
    ("caches/quotebank/mention_embeddings.pkl", "embedding_sentencecache_qb_cpu.pkl"),
    ("caches/quotebank/entity_kb_aliases.pkl", "quotebank_cache_alias.pkl"),
    ("caches/quotebank/unambiguous_mentions.pkl", "unambiguous_cache.pkl"),
]
def sha(p, n=2_000_000):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest(), p.stat().st_size
for a, b in pairs:
    pa, pb = root / a, gdrive / b
    if not pa.exists() or not pb.exists():
        print(f"skip {a} (missing)")
        continue
    sa, sb = sha(pa), sha(pb)
    print(("MATCH" if sa == sb else "DIFF "), a, "←", b)
PY
