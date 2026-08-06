#!/usr/bin/env python3
"""Extract English labels, aliases, and descriptions from a Wikidata dump.

Historical sources:
  - utils/aida_aliases.py, utils/aida_wikicache.py

Writes ``wikidata_{labels,aliases,descriptions}.pkl`` keyed by integer QID
(without the ``Q`` prefix), matching ``representation.py``.

Usage:
  python cache_building/extract_entity_metadata.py \\
      --dump /path/to/wikidata-20211101-all.json.gz \\
      --qids artifacts/cache_build/candidate_qids.pkl \\
      --out-dir artifacts/cache_build/entity_metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from cache_building.io_utils import iter_wikidata_dump, load_pickle, qtoi, save_pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--qids", default=None, help="Optional pickle set[str] to filter")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    keep = None
    if args.qids:
        keep = {qtoi(q) for q in load_pickle(args.qids)}

    labels, aliases, descs = {}, {}, {}
    for ent in tqdm(iter_wikidata_dump(args.dump), desc="labels"):
        qid = ent.get("id")
        if not qid or not qid.startswith("Q"):
            continue
        qi = qtoi(qid)
        if keep is not None and qi not in keep:
            continue
        try:
            if "en" in ent.get("labels", {}):
                labels[qi] = ent["labels"]["en"]["value"]
        except Exception:
            pass
        try:
            if "en" in ent.get("aliases", {}):
                aliases[qi] = [a["value"] for a in ent["aliases"]["en"]]
        except Exception:
            pass
        try:
            if "en" in ent.get("descriptions", {}):
                descs[qi] = ent["descriptions"]["en"]["value"]
        except Exception:
            pass

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_pickle(labels, out / "wikidata_labels.pkl")
    save_pickle(aliases, out / "wikidata_aliases.pkl")
    save_pickle(descs, out / "wikidata_descriptions.pkl")
    print(f"labels={len(labels)} aliases={len(aliases)} descs={len(descs)} → {out}")


if __name__ == "__main__":
    main()
