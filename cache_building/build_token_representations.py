#!/usr/bin/env python3
"""Build TokenSet / Statement / NS / PR representation caches.

Historical sources:
  - quotebank_el/get_entity_representation_caches.py
  - quotebank_el/representation.py

These feed the Spark/parquet scoring path. Main Table-2 heuristics use
``entity_kb.pkl`` from ``build_entity_kb.py`` instead.

Usage:
  PYTHONPATH=cache_building/original/quotebank_el \\
  python cache_building/build_token_representations.py \\
      --dump artifacts/cache_build/wikidata_subgraph.json.gz \\
      --labels-dir artifacts/cache_build/entity_metadata \\
      --out-dir artifacts/cache_build/token_representations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cache_building" / "original" / "quotebank_el"))

import representation as rep  # noqa: E402
from helpers import load_pickle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--redirect-xml", default=None)
    ap.add_argument("--wp-ranks", default=None, help="TSV for PRCache.from_ranks_file")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    wikidata_labels = load_pickle(labels_dir / "wikidata_labels.pkl")
    wikidata_aliases = load_pickle(labels_dir / "wikidata_aliases.pkl")
    wikidata_descriptions = load_pickle(labels_dir / "wikidata_descriptions.pkl")

    print("TokenSetRepresentationCache.from_dump …")
    ts = rep.TokenSetRepresentationCache.from_dump(
        args.dump,
        wikidata_labels=wikidata_labels,
        wikidata_aliases=wikidata_aliases,
        wikidata_descriptions=wikidata_descriptions,
    )
    print("StatementRepresentationCache.from_dump …")
    st = rep.StatementRepresentationCache.from_dump(args.dump)
    print("NSCache.from_dump …")
    ns = rep.NSCache.from_dump(args.dump)

    pr = None
    if args.wp_ranks:
        print("PRCache.from_ranks_file …")
        pr = rep.PRCache.from_ranks_file(args.wp_ranks)

    if args.redirect_xml:
        print("Resolving redirects …")
        rr = rep.RedirectResolver(args.redirect_xml)
        ts = rr.resolve_redirects(ts)
        st = rr.resolve_redirects(st)
        ns = rr.resolve_redirects(ns)
        if pr is not None:
            pr = rr.resolve_redirects(pr)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts.to_pickle(out / "ts_cache.pkl")
    st.to_pickle(out / "s_cache.pkl")
    ns.to_pickle(out / "ns_cache.pkl")
    if pr is not None:
        pr.to_pickle(out / "prwp_cache.pkl")
    print(f"wrote representation caches → {out}")


if __name__ == "__main__":
    main()
