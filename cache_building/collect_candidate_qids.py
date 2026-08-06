#!/usr/bin/env python3
"""Collect candidate QIDs from NELight dataset JSON files.

Historical sources:
  - speaker-disambiguation-quotebank/QB_disamb.ipynb (Spark → qids_for_cache.pkl)
  - utils/populate_aida_cache.py

Usage:
  python cache_building/collect_candidate_qids.py \\
      --data data/Quotebank/data.json \\
      --out artifacts/cache_build/candidate_qids.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import load_json, save_pickle


def collect_qids(data: list) -> set[str]:
    qids: set[str] = set()
    for article in data:
        for name in article.get("names", []):
            for qid in name.get("ids", []):
                if isinstance(qid, str) and qid.startswith("Q"):
                    qids.add(qid)
    return qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="NELight data.json (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qids: set[str] = set()
    for path in args.data:
        data = load_json(path)
        part = collect_qids(data)
        print(f"{path}: {len(part)} QIDs")
        qids |= part
    print(f"total unique QIDs: {len(qids)}")
    save_pickle(qids, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
