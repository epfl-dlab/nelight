#!/usr/bin/env python3
"""Reconstruct Table 5 coarse buckets from gt_annotation.json (= research gt_new).

Fine null labels (No QID / Impossible / Not listed / Not a person) are
paper-reported: the full checked annotation MDs were not archived.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/from_scratch/quotebank/table5_gt_distribution.json"
PAPER = ROOT / "paper/tables/paper_tables.json"
GT = ROOT / "data/Quotebank/gt_annotation.json"
DATA = ROOT / "data/Quotebank/data.json"

PAPER_FINE = {
    "Ambiguous / No correct QID in Wikidata": 151,
    "Ambiguous / Impossible": 37,
    "Ambiguous / Correct QID not listed": 24,
    "Ambiguous / Not a person": 22,
}


def reconstruct() -> dict:
    data = json.load(open(DATA))
    gt = json.load(open(GT))
    paper_rows = json.load(open(PAPER))["table5_gt_distribution"]["rows"]
    total = sum(len(a["names"]) for a in data)
    n_ann = sum(len(v) for v in gt.values())
    n_gold = sum(1 for a in gt.values() for g in a.values() if g is not None)
    n_null = sum(1 for a in gt.values() for g in a.values() if g is None)
    n_unamb = total - n_ann
    n_amb_data = sum(1 for a in data for n in a["names"] if len(n["ids"]) > 1)

    rows = [
        {"category": "Unambiguous", "mentions": n_unamb, "pct": round(100 * n_unamb / total, 1)},
        {
            "category": "Ambiguous / Gold entity exists",
            "mentions": n_gold,
            "pct": round(100 * n_gold / total, 1),
        },
    ]
    for cat, n in PAPER_FINE.items():
        rows.append({"category": cat, "mentions": n, "pct": round(100 * n / total, 1)})
    rows.append({"category": "Total", "mentions": total, "pct": 100.0})

    paper_by = {r["category"]: r["mentions"] for r in paper_rows}
    match = (
        total == paper_by["Total"]
        and n_null == sum(PAPER_FINE.values())
        and abs(n_unamb - paper_by["Unambiguous"]) <= 2
        and abs(n_gold - paper_by["Ambiguous / Gold entity exists"]) <= 2
        and n_ann == n_amb_data
    )
    out = {
        "rows": rows,
        "paper": paper_rows,
        "reconstructed": {
            "total": total,
            "unambiguous": n_unamb,
            "gold_entity_exists": n_gold,
            "null_gold": n_null,
            "ambiguous_annotated": n_ann,
        },
        "deltas_vs_paper": {
            r["category"]: r["mentions"] - paper_by[r["category"]] for r in rows
        },
        "match": match,
        "notes": [
            "Coarse buckets from data/Quotebank/gt_annotation.json.",
            "Fine null split is paper-reported (annotation MDs not archived).",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    return out


def main():
    out = reconstruct()
    print(json.dumps({"reconstructed": out["reconstructed"], "match": out["match"]}, indent=2))


if __name__ == "__main__":
    main()
