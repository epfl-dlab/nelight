#!/usr/bin/env python3
"""Build the unambiguous-mentions cache used by EEIScore / CSSVE.

Format: ``{articleID: [[qid], [qid], …]}``. The published scorer reads
``cache[aid][0]`` (first unambiguous mention only; see REPRODUCIBILITY.md).
"""

from __future__ import annotations

import argparse

from cache_building.io_utils import load_json, save_pickle


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
