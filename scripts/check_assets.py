#!/usr/bin/env python3
"""Verify shipped caches needed for table reproduction are present."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from runlib.cache_paths import resolve


REQUIRED = [
    ("quotebank", "entity_kb"),
    ("quotebank", "entity_kb_aliases"),
    ("quotebank", "unambiguous_mentions"),
    ("quotebank", "entity_embeddings"),
    ("quotebank", "document_embeddings"),
    ("quotebank", "mention_embeddings"),
    ("aida", "entity_kb"),
    ("aida", "unambiguous_mentions"),
    ("aida", "entity_embeddings"),
    ("aida", "document_embeddings"),
    ("aida", "mention_embeddings"),
]


def main() -> int:
    errs = []
    for ds, kind in REQUIRED:
        p = resolve(ds, kind, required=False)
        if p is None:
            errs.append(f"missing {ds}/{kind}")
        else:
            print(f"OK  {ds}/{kind}")
    eigen = list((ROOT / "artifacts/from_scratch").glob("*/Eigen*_live_weigen.pkl"))
    mgenre = [
        ROOT / "score_cache/raw/genre_context_scores_qb.pkl",
        ROOT / "score_cache/raw/genre_context_scores_aida.pkl",
    ]
    if not eigen:
        errs.append("missing artifacts/from_scratch/*/Eigen*_live_weigen.pkl")
    else:
        print(f"OK  {len(eigen)} Eigen pickles")
    for p in mgenre:
        if not p.exists():
            errs.append(f"missing {p.relative_to(ROOT)}")
        else:
            print(f"OK  {p.relative_to(ROOT)}")
    if errs:
        print("FAIL")
        for e in errs:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
