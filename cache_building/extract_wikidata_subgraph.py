#!/usr/bin/env python3
"""Extract a candidate-QID subgraph from a full Wikidata JSON dump.

Historical sources:
  - quotebank_el/extract_quotebank_subset.py
  - utils/extract_quotebank_subset.py

Usage:
  python cache_building/extract_wikidata_subgraph.py \\
      --dump /path/to/wikidata-20211101-all.json.gz \\
      --qids artifacts/cache_build/candidate_qids.pkl \\
      --out artifacts/cache_build/wikidata_subgraph.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json

from tqdm import tqdm

from cache_building.io_utils import load_pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Full Wikidata JSON dump (.json.gz)")
    ap.add_argument("--qids", required=True, help="Pickle of set[str] QIDs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qids = set(load_pickle(args.qids))
    total = len(qids)
    print(f"looking for {total} QIDs in {args.dump}")

    written = 0
    with gzip.open(args.dump, "rb") as fin, gzip.open(args.out, "wb") as fout:
        pbar = tqdm(fin)
        lines = []
        for raw in pbar:
            pbar.set_postfix({"found": f"{(1 - len(qids) / total) * 100:.2f}%"})
            line = raw.decode("utf-8")
            payload = line[:-2] if line.endswith(",\n") else line.strip().rstrip(",")
            if not payload or payload in ("[", "]"):
                continue
            try:
                ent = json.loads(payload)
            except json.JSONDecodeError:
                continue
            qid = ent.get("id")
            if qid not in qids:
                continue
            qids.remove(qid)
            lines.append(raw if raw.endswith(b"\n") else raw + b"\n")
            written += 1
            if not qids:
                break
        fout.writelines(lines)

    print(f"wrote {written} entities → {args.out}")
    if qids:
        print(f"WARNING: {len(qids)} QIDs not found in dump")


if __name__ == "__main__":
    main()
