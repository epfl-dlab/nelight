#!/usr/bin/env python3
"""Quick faithfulness / wiring audit (no Wikidata dump, no GPU required)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runlib"))

from cache_paths import CANONICAL, ensure_canonical_symlinks, resolve  # noqa: E402


def check_caches() -> list[str]:
    errs = []
    created = ensure_canonical_symlinks()
    if created:
        print(f"created {len(created)} canonical symlinks")
    required = [
        ("quotebank", "entity_kb"),
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
    for ds, kind in required:
        p = resolve(ds, kind, required=False)
        if p is None:
            errs.append(f"missing {ds}/{kind}")
        else:
            print(f"OK  {ds}/{kind} → {p}")
    return errs


def check_unambiguous_format() -> list[str]:
    errs = []
    for ds in ("quotebank", "aida"):
        p = resolve(ds, "unambiguous_mentions")
        obj = pickle.load(open(p, "rb"))
        aid = next(iter(obj))
        sample = obj[aid]
        if not isinstance(sample, list):
            errs.append(f"{ds}: unamb value not list")
            continue
        if sample and not isinstance(sample[0], list):
            errs.append(f"{ds}: unamb must be list-of-lists, got flat {type(sample[0])}")
        else:
            print(f"OK  {ds} unambiguous format list-of-lists (n_articles={len(obj)})")
    return errs


def check_scripts_import() -> list[str]:
    errs = []
    for mod in [
        "scripts.reproduce_tables",
        "scripts.reproduce_paper_from_scratch",
        "scripts.reproduce_all_paper_tables",
        "scripts.reproduce_remaining_tables",
    ]:
        # compile only — full import of reproduce_* executes helpers via exec
        path = ROOT / (mod.replace(".", "/") + ".py")
        try:
            compile(path.read_text(), str(path), "exec")
            print(f"OK  compile {path.name}")
        except Exception as e:
            errs.append(f"compile {path.name}: {e}")
    for path in (ROOT / "cache_building").glob("*.py"):
        try:
            compile(path.read_text(), str(path), "exec")
        except Exception as e:
            errs.append(f"compile {path.name}: {e}")
    print(f"OK  cache_building/*.py compile")
    return errs


def check_paper_tables_json() -> list[str]:
    import json

    errs = []
    p = json.load(open(ROOT / "paper/tables/paper_tables.json"))
    for key in [
        "table1_dataset_stats",
        "table2_p_at_1",
        "table3_aida_entity_types_p_at_1",
        "table4_error_analysis",
        "table5_gt_distribution",
        "table6_iscore_ablation",
        "table7_context_size",
        "table8_mgenre_context",
        "table9_tie_breakers",
        "table10_inference_times",
        "table11_mrr",
    ]:
        if key not in p:
            errs.append(f"paper_tables.json missing {key}")
        else:
            print(f"OK  paper_tables.json has {key}")
    # Table 9 must have numeric rows now
    t9 = p["table9_tie_breakers"]
    if "rows" not in t9 or len(t9["rows"]) < 4:
        errs.append("table9 rows incomplete")
    return errs


def main():
    errs: list[str] = []
    print("=== caches ===")
    errs += check_caches()
    print("\n=== unambiguous format ===")
    errs += check_unambiguous_format()
    print("\n=== scripts ===")
    errs += check_scripts_import()
    print("\n=== paper tables JSON ===")
    errs += check_paper_tables_json()
    print("\n=== summary ===")
    if errs:
        print(f"FAIL ({len(errs)} issues)")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
