"""Canonical cache paths for NELight reproduction.

Canonical names under ``caches/{quotebank,aida}/`` are preferred. Legacy
filenames (historical research trees) remain as optional fallbacks.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHES = ROOT / "caches"

CANONICAL = {
    "quotebank": {
        "entity_kb": CACHES / "quotebank" / "entity_kb.pkl",
        "entity_kb_aliases": CACHES / "quotebank" / "entity_kb_aliases.pkl",
        "entity_embeddings": CACHES / "quotebank" / "entity_embeddings.pkl",
        "document_embeddings": CACHES / "quotebank" / "document_embeddings.pkl",
        "mention_embeddings": CACHES / "quotebank" / "mention_embeddings.pkl",
        "unambiguous_mentions": CACHES / "quotebank" / "unambiguous_mentions.pkl",
    },
    "aida": {
        "entity_kb": CACHES / "aida" / "entity_kb.pkl",
        "unambiguous_mentions": CACHES / "aida" / "unambiguous_mentions.pkl",
        "entity_embeddings": CACHES / "aida" / "entity_embeddings.pkl",
        "document_embeddings": CACHES / "aida" / "document_embeddings.pkl",
        "mention_embeddings": CACHES / "aida" / "mention_embeddings.pkl",
    },
}

# Historical filenames from the original research trees (optional).
LEGACY = {
    "quotebank": {
        "entity_kb": [
            CACHES / "Quotebank" / "wikicache.pkl",
            CACHES / "workspace_caches" / "ultimate_wikicache.pkl",
        ],
        "entity_kb_aliases": [CACHES / "Quotebank" / "wikicache_alias.pkl"],
        "entity_embeddings": [CACHES / "Quotebank" / "wikidata_embeddings.pkl"],
        "document_embeddings": [CACHES / "Quotebank" / "content_embeddings.pkl"],
        "mention_embeddings": [CACHES / "Quotebank" / "sentence_embeddings.pkl"],
        "unambiguous_mentions": [CACHES / "unambiguous_cache.pkl"],
    },
    "aida": {
        "entity_kb": [
            CACHES / "aida_raw" / "aida_wikicache_ultimate.pkl",
            CACHES / "workspace_caches" / "aida_cache2_p.pkl",
        ],
        "unambiguous_mentions": [CACHES / "aida_raw" / "aida_unamb.pkl"],
        "entity_embeddings": [
            ROOT / "workspace" / "quotebank_el" / "aida" / "embedding_wikicache.pkl",
            CACHES / "aida_raw" / "embedding_wikicache.pkl",
        ],
        "document_embeddings": [
            ROOT / "workspace" / "quotebank_el" / "aida" / "embedding_contentcache.pkl",
            CACHES / "aida_raw" / "embedding_contentcache.pkl",
        ],
        "mention_embeddings": [
            ROOT / "workspace" / "quotebank_el" / "aida" / "embedding_sentencecache.pkl",
            CACHES / "aida_raw" / "embedding_sentencecache.pkl",
        ],
    },
}


def resolve(dataset: str, kind: str, *, required: bool = True) -> Path | None:
    """Return the first existing path for a cache, preferring canonical names."""
    ds = "quotebank" if dataset.lower() in {"quotebank", "qb"} else "aida"
    cand = [CANONICAL[ds][kind], *LEGACY[ds].get(kind, [])]
    for p in cand:
        if p is not None and Path(p).exists():
            return Path(p)
    if required:
        searched = ", ".join(str(p) for p in cand)
        raise FileNotFoundError(f"Missing cache {ds}/{kind}. Searched: {searched}")
    return None


def ensure_canonical_symlinks() -> list[str]:
    """Create professional-name symlinks pointing at legacy cache files."""
    created = []
    for ds, kinds in CANONICAL.items():
        for kind, dest in kinds.items():
            if dest.exists() or dest.is_symlink():
                continue
            legacy = next((p for p in LEGACY[ds].get(kind, []) if p.exists()), None)
            if legacy is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(legacy.resolve())
            created.append(f"{dest} → {legacy}")
    return created
