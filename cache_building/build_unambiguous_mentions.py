#!/usr/bin/env python3
"""Build the unambiguous-mentions cache used by EEIScore / CSSVE.

Historical source: utils/unambiguous_entities.py

Format (critical for EEIScore):
  {articleID: [[qid], [qid], …]}  # one singleton list per unambiguous mention

The scorer reads ``cache[aid][0]`` (first unambiguous mention only).

Usage:
  python cache_building/build_unambiguous_mentions.py \\
      --data data/Quotebank/data.json \\
      --out artifacts/cache_build/unambiguous_mentions_quotebank.pkl
"""

from __future__ import annotations

import argparse

from io_utils import load_json, save_pickle


def build_unambiguous(data: list) -> dict:
    cache = {}
    for article in data:
        aid = article["articleID"]
        cache[aid] = []
        for name in article.get("names", []):
            ids = name.get("ids", [])
            if len(ids) == 1:
                cache[aid].append(ids)  # keep as [qid] singleton list
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load_json(args.data)
    cache = build_unambiguous(data)
    n = sum(len(v) for v in cache.values())
    save_pickle(cache, args.out)
    print(f"articles={len(cache)} unambiguous_mentions={n} → {args.out}")


if __name__ == "__main__":
    main()
