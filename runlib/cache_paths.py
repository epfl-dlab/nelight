"""Canonical cache paths for NELight reproduction.

Caches live under ``caches/{quotebank,aida}/``.
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


def resolve(dataset: str, kind: str, *, required: bool = True) -> Path | None:
    """Return the canonical path for a cache if it exists."""
    ds = "quotebank" if dataset.lower() in {"quotebank", "qb"} else "aida"
    if kind not in CANONICAL[ds]:
        if required:
            raise KeyError(f"Unknown cache kind {ds}/{kind}")
        return None
    p = CANONICAL[ds][kind]
    if p.exists():
        return p
    if required:
        raise FileNotFoundError(f"Missing cache {ds}/{kind} at {p}")
    return None


def ensure_canonical_symlinks() -> list[str]:
    """No-op kept for call-site compatibility (canonical files are real paths)."""
    return []
