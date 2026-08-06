#!/usr/bin/env python3
"""Build the entity knowledge-base cache used by IScore / popularity scorers.

Historical sources (stitched):
  1. utils/populate_aida_cache.py — raw claims as QID ints / strings
  2. utils/process_cache.py — resolve item values to English labels + description
  3. utils/first_paragraphs.py (+ Embedding calculation.ipynb) — Wikipedia lead
  4. utils/get_ns_np.py, wd_pagerank.py, wp_pagerank.py — centrality fields
  5. Embedding calculation.ipynb — early API builder; dump path is canonical

Output (``entity_kb.pkl``; historically ``wikicache.pkl`` / ``ultimate_wikicache.pkl``):
  {"Q42": {"description": [...], "occupation": [...], "first_paragraph": "...",
           "n_sitelinks": [...], "n_statements": [...], "pagerank": [...], ...}}

Usage:
  python cache_building/build_entity_kb.py \\
      --dump artifacts/cache_build/wikidata_subgraph.json.gz \\
      --labels-dir artifacts/cache_build/entity_metadata \\
      --qids artifacts/cache_build/candidate_qids.pkl \\
      --out artifacts/cache_build/entity_kb.pkl
"""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cache_building.io_utils import iter_wikidata_dump, itoq, load_pickle, qtoi, save_pickle


def extract_raw_claims(ent: dict) -> dict[int, list]:
    """populate_aida_cache.py logic: property-id → list of item-ids / strings."""
    out: dict[int, list] = {}
    for prop, values in ent.get("claims", {}).items():
        if not prop.startswith("P"):
            continue
        pi = int(prop[1:])
        bucket: list = []
        for snak in values:
            mainsnak = snak.get("mainsnak", {})
            if "datavalue" not in mainsnak:
                continue
            datatype = mainsnak.get("datatype")
            dv = mainsnak["datavalue"]
            if datatype == "wikibase-item":
                if "value" in dv:
                    bucket.append(int(dv["value"]["id"][1:]))
            elif datatype == "string":
                if "value" in dv:
                    bucket.append(dv["value"])
        if bucket:
            out[pi] = bucket
    return out


def resolve_to_labels(
    raw: dict[int, dict[int, list]],
    labels: dict[int, str],
    descs: dict[int, str],
    prop_labels: dict[int, str] | None = None,
) -> dict[str, dict[str, list]]:
    """process_cache.py + Embedding calculation prop-label rename."""
    cache: dict[str, dict[str, list]] = {}
    for qi, props in tqdm(raw.items(), desc="resolve labels"):
        qid = itoq(qi)
        entry: dict[str, list] = {}
        for pi, vals in props.items():
            key = prop_labels.get(pi, f"P{pi}") if prop_labels else f"P{pi}"
            resolved = []
            for v in vals:
                if isinstance(v, int):
                    if v in labels:
                        resolved.append(labels[v])
                else:
                    resolved.append(v)
            if resolved:
                entry[key] = resolved
        entry["description"] = [descs[qi]] if qi in descs else [""]
        entry["n_statements"] = [len(props)]
        cache[qid] = entry
    return cache


def attach_ns(cache: dict, dump_path: str) -> None:
    for ent in tqdm(iter_wikidata_dump(dump_path), desc="n_sitelinks"):
        qid = ent.get("id")
        if qid in cache:
            cache[qid]["n_sitelinks"] = [len(ent.get("sitelinks", {}))]


def attach_first_paragraphs(
    cache: dict,
    qid_pid_path: str,
    first_paragraphs_path: str,
) -> None:
    """first_paragraphs.py / Embedding calculation cells 44–45."""
    qid_pid: dict[str, int] = {}
    with bz2.open(qid_pid_path, "rb") as f:
        for line in tqdm(f, desc="qid↔pid"):
            m = json.loads(line)
            qid = m.get("Qid") or m.get("qid")
            if qid in cache:
                qid_pid[qid] = int(m["page_id"])
    pid_qid = {pid: qid for qid, pid in qid_pid.items()}
    with bz2.open(first_paragraphs_path, "rb") as f:
        for line in tqdm(f, desc="first_paragraphs"):
            p = json.loads(line)
            pid = int(p["page_id"])
            if pid in pid_qid:
                cache[pid_qid[pid]]["first_paragraph"] = p["first_paragraph"]


def attach_ranks(cache: dict, ranks_path: str, key: str) -> None:
    """wd_pagerank.py / wp_pagerank.py — TSV `qid\\trank` (qid without Q)."""
    with open(ranks_path) as f:
        # skip header if present
        first = f.readline()
        if "\t" in first and first.strip().split("\t")[0].isdigit():
            qid, rank = first.strip().split("\t")
            q = "Q" + qid
            if q in cache:
                cache[q][key] = [float(rank)]
        for line in tqdm(f, desc=key):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qid, rank = parts[0], parts[1]
            q = qid if qid.startswith("Q") else "Q" + qid
            if q in cache:
                cache[q][key] = [float(rank)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Subgraph or full Wikidata JSON dump")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--qids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prop-labels", default=None, help="Optional pickle {int_pid: en_label}")
    ap.add_argument("--qid-pid", default=None)
    ap.add_argument("--first-paragraphs", default=None)
    ap.add_argument("--wp-ranks", default=None, help="Wikipedia pagerank TSV → pagerank")
    ap.add_argument("--wd-ranks", default=None, help="Wikidata pagerank TSV → pagerank_wd")
    args = ap.parse_args()

    qids = {qtoi(q) for q in load_pickle(args.qids)}
    labels = load_pickle(Path(args.labels_dir) / "wikidata_labels.pkl")
    descs = load_pickle(Path(args.labels_dir) / "wikidata_descriptions.pkl")
    prop_labels = load_pickle(args.prop_labels) if args.prop_labels else None

    raw: dict[int, dict[int, list]] = {}
    for ent in tqdm(iter_wikidata_dump(args.dump), desc="raw claims"):
        qid = ent.get("id")
        if not qid or not qid.startswith("Q"):
            continue
        qi = qtoi(qid)
        if qi not in qids:
            continue
        raw[qi] = extract_raw_claims(ent)

    cache = resolve_to_labels(raw, labels, descs, prop_labels)
    attach_ns(cache, args.dump)

    if args.qid_pid and args.first_paragraphs:
        attach_first_paragraphs(cache, args.qid_pid, args.first_paragraphs)
    if args.wp_ranks:
        attach_ranks(cache, args.wp_ranks, "pagerank")
    if args.wd_ranks:
        attach_ranks(cache, args.wd_ranks, "pagerank_wd")

    save_pickle(cache, args.out)
    print(f"wrote {len(cache)} entities → {args.out}")


if __name__ == "__main__":
    main()
