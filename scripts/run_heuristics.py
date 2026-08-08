#!/usr/bin/env python3
"""Recompute popularity / IScore-family / CSE-family from shipped caches.

Writes score dicts under artifacts/from_scratch/ (recomputed-from-caches, not
Wikidata-from-scratch). Eigen/mGENRE are merged later by merge_paper_scores.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from runlib.cache_paths import resolve as resolve_cache
from runlib.eval import (
    assign_unambiguous,
    flatten_gt,
    load_json,
    load_pickle,
    merge_gt,
    normalize_scores,
    precision_at_one_aida,
    precision_at_one_qb,
    same_score_rank_ensemble,
    save_pickle,
    transform_scores,
    weighted_sum,
)
from runlib.scoring.centrality import WikidataCentralityScorer
from runlib.scoring.semantic import (
    EntityContentSimilarityScorer,
    EntityEntitySimilarityScorer,
)


def build_unambiguous_cache(data) -> dict:
    """Paper format: list of singleton id-lists per article ``[[qid], ...]``."""
    cache = {}
    for article in data:
        u = [list(n["ids"]) for n in article["names"] if len(n["ids"]) == 1]
        cache[article["articleID"]] = u
    return cache


def load_unambiguous_cache(dataset: str, data, path: Path | None = None) -> dict:
    if path is not None:
        return load_pickle(path)
    if dataset in ("quotebank", "aida"):
        p = resolve_cache(dataset, "unambiguous_mentions", required=False)
        if p is not None:
            return load_pickle(p)
    return build_unambiguous_cache(data)


def run_dataset(
    dataset: str,
    out_dir: Path,
    with_embeddings: bool,
    *,
    data_path: Path | None = None,
    entity_kb_path: Path | None = None,
    unambiguous_path: Path | None = None,
    protocol: str | None = None,
    easy_path: Path | None = None,
    hard_path: Path | None = None,
    overall_path: Path | None = None,
):
    emb_cache = content_emb = sent_emb = None
    easy = hard = overall = None

    if data_path is not None:
        data = load_json(data_path)
        if entity_kb_path is None:
            raise SystemExit("--entity-kb is required with --data")
        wiki_cache = load_pickle(entity_kb_path)
        protocol = protocol or "aida"
        if easy_path:
            easy = load_json(easy_path)
        if hard_path:
            hard = load_json(hard_path)
        if overall_path:
            overall = load_json(overall_path)
        elif easy is not None and hard is not None:
            overall = merge_gt(easy, hard)
    elif dataset == "quotebank":
        data = load_json(ROOT / "data/Quotebank/data.json")
        easy = load_json(ROOT / "data/Quotebank/easy.json")
        hard = load_json(ROOT / "data/Quotebank/hard.json")
        overall = merge_gt(easy, hard)
        wiki_cache = load_pickle(resolve_cache("quotebank", "entity_kb"))
        protocol = "qb"
        if with_embeddings:
            emb_cache = load_pickle(resolve_cache("quotebank", "entity_embeddings"))
            content_emb = load_pickle(resolve_cache("quotebank", "document_embeddings"))
            sent_emb = load_pickle(resolve_cache("quotebank", "mention_embeddings"))
    else:
        data = load_json(ROOT / "data/AIDA/data.json")
        easy = load_json(ROOT / "data/AIDA/easy.json")
        hard = load_json(ROOT / "data/AIDA/hard.json")
        overall = load_json(ROOT / "data/AIDA/overall.json")
        wiki_cache = load_pickle(resolve_cache("aida", "entity_kb"))
        protocol = "aida"
        if with_embeddings:
            ep = resolve_cache("aida", "entity_embeddings", required=False)
            dp = resolve_cache("aida", "document_embeddings", required=False)
            mp = resolve_cache("aida", "mention_embeddings", required=False)
            if ep:
                emb_cache = load_pickle(ep)
            if dp:
                content_emb = load_pickle(dp)
            if mp:
                sent_emb = load_pickle(mp)

    if protocol == "quotebank":
        protocol = "qb"
    if protocol not in ("qb", "aida"):
        raise SystemExit(f"unknown protocol {protocol!r}")

    print(f"[{dataset}] articles={len(data)} entity_kb={len(wiki_cache)} protocol={protocol}", flush=True)
    unamb = load_unambiguous_cache(dataset, data, path=unambiguous_path)

    # Drive method names on WikidataCentralityScorer
    methods_raw = {}
    for key in ("LQID", "NP", "NS", "PRWP", "PRWD"):
        print(f"[{dataset}] scoring {key}...", flush=True)
        methods_raw[key] = normalize_scores(
            WikidataCentralityScorer(key, wiki_cache=wiki_cache).score_all(data)
        )

    print(f"[{dataset}] scoring IScore/NIScore...", flush=True)
    methods_raw["IScore"] = normalize_scores(
        EntityContentSimilarityScorer(
            "iscore", wiki_cache=wiki_cache, stem=True, props_to_avoid=["first_paragraph"]
        ).score_all(data)
    )
    methods_raw["NIScore"] = normalize_scores(
        EntityContentSimilarityScorer(
            "iscore_narrow", wiki_cache=wiki_cache, stem=True, props_to_avoid=["first_paragraph"]
        ).score_all(data)
    )

    print(f"[{dataset}] scoring EEIScore...", flush=True)
    methods_raw["EEIScore"] = normalize_scores(
        EntityEntitySimilarityScorer(
            "eeiscore", wiki_cache=wiki_cache, unambiguous_cache=unamb
        ).score_all(data)
    )

    if with_embeddings and emb_cache is not None and content_emb is not None:
        print(f"[{dataset}] scoring CSE/NCSE/CSSVE...", flush=True)
        methods_raw["CSE"] = normalize_scores(
            EntityContentSimilarityScorer(
                "cse",
                wiki_cache=wiki_cache,
                embeddings_cache=emb_cache,
                content_embeddings_cache=content_emb,
            ).score_all(data)
        )
        if sent_emb is not None:
            methods_raw["NCSE"] = normalize_scores(
                EntityContentSimilarityScorer(
                    "ncse",
                    wiki_cache=wiki_cache,
                    embeddings_cache=emb_cache,
                    content_embeddings_cache=content_emb,
                    sentence_embeddings_cache=sent_emb,
                ).score_all(data)
            )
        methods_raw["CSSVE"] = normalize_scores(
            EntityEntitySimilarityScorer(
                "cssve",
                wiki_cache=wiki_cache,
                embeddings_cache=emb_cache,
                unambiguous_cache=unamb,
            ).score_all(data)
        )

    ds_out = out_dir / dataset
    for name, scores in methods_raw.items():
        save_pickle(scores, ds_out / f"{name}.pkl")

    lqid, ns, np_scores, prwp = (
        methods_raw["LQID"],
        methods_raw["NS"],
        methods_raw["NP"],
        methods_raw["PRWP"],
    )

    def tiebreak(scores, *pops):
        out = scores
        for p in pops:
            out = same_score_rank_ensemble(out, p, data)
        return out

    if protocol == "qb":
        ranked = {
            "LQID": methods_raw["LQID"],
            "NP": tiebreak(methods_raw["NP"], ns, lqid),
            "NS": tiebreak(methods_raw["NS"], lqid),
            "PRWD": tiebreak(methods_raw["PRWD"], ns, lqid),
            "PRWP": tiebreak(methods_raw["PRWP"], lqid),
            "IScore": tiebreak(methods_raw["IScore"], np_scores, lqid),
            "NIScore": tiebreak(methods_raw["NIScore"], ns, lqid),
            "EEIScore": tiebreak(methods_raw["EEIScore"], prwp, lqid),
        }
        ui = weighted_sum(
            [methods_raw["IScore"], methods_raw["NIScore"], methods_raw["EEIScore"]],
            [1.0, 1.0, 1.0],
        )
        ranked["UIScore"] = tiebreak(ui, prwp, lqid)
    else:
        ranked = {
            k: methods_raw[k]
            for k in ("LQID", "NP", "NS", "PRWD", "PRWP", "IScore", "NIScore", "EEIScore")
        }
        ranked["UIScore"] = weighted_sum(
            [methods_raw["IScore"], methods_raw["NIScore"], methods_raw["EEIScore"]],
            [0.9, 0.0, 1.0],
        )

    if "CSE" in methods_raw and "NCSE" in methods_raw and "CSSVE" in methods_raw:
        cse_t = transform_scores(methods_raw["CSE"], lambda x: 0.5 * (x + 1.0))
        if protocol == "qb":
            ncse_t = transform_scores(
                methods_raw["NCSE"], lambda x: (x + 1.0) / np.sum(x + 1.0)
            )
            cssve_t = transform_scores(
                methods_raw["CSSVE"], lambda x: (x + 1.0) / np.sum(x + 1.0)
            )
            ucse = weighted_sum([cse_t, ncse_t, cssve_t], [0.45, 0.9, 0.2])
            ranked["UCSE"] = tiebreak(ucse, prwp, lqid)
            ranked["CSE"] = tiebreak(methods_raw["CSE"], ns, lqid)
            ranked["NCSE"] = tiebreak(methods_raw["NCSE"], ns, lqid)
            ranked["CSSVE"] = tiebreak(methods_raw["CSSVE"], ns, lqid)
        else:
            ncse_t = transform_scores(methods_raw["NCSE"], lambda x: 0.5 * (x + 1.0))
            cssve_t = transform_scores(
                methods_raw["CSSVE"], lambda x: (x + 1.0) / np.sum(x + 1.0)
            )
            ranked["UCSE"] = weighted_sum([cse_t, ncse_t, cssve_t], [0.0, 1.0, 1.0])
            ranked["CSE"] = methods_raw["CSE"]
            ranked["NCSE"] = methods_raw["NCSE"]
            ranked["CSSVE"] = methods_raw["CSSVE"]

    if protocol == "aida":
        for k in list(ranked):
            ranked[k] = assign_unambiguous(ranked[k], data)

    results = {}
    splits = []
    if easy is not None:
        splits.append(("easy", easy))
    if hard is not None:
        splits.append(("hard", hard))
    if overall is not None:
        splits.append(("overall", overall))
    for split_name, gt in splits:
        items = flatten_gt(gt)
        fn = precision_at_one_qb if protocol == "qb" else precision_at_one_aida
        results[split_name] = {m: fn(items, sc) for m, sc in ranked.items()}

    methods = [
        "LQID", "NP", "NS", "PRWD", "PRWP",
        "IScore", "NIScore", "EEIScore", "UIScore",
        "CSE", "NCSE", "CSSVE", "UCSE",
    ]
    if results:
        cols = [s for s, _ in splits]
        print(f"\n=== {dataset} recomputed P@1 ===")
        print(f"{'Method':12s}" + "".join(f" {c:>8s}" for c in cols))
        for m in methods:
            if m not in ranked:
                continue
            print(f"{m:12s}" + "".join(f" {results[c][m]:8.3f}" for c in cols))

    prev_path = ds_out / "ranked_scores.pkl"
    if prev_path.exists():
        prev = load_pickle(prev_path)
        for key in ("Eigen", "Eigen (IScore)", "Eigen_IScore", "mGENRE"):
            if key in prev and key not in ranked:
                ranked[key] = prev[key]
    if "Eigen (IScore)" not in ranked and "Eigen_IScore" in ranked:
        ranked["Eigen (IScore)"] = ranked["Eigen_IScore"]

    save_pickle(ranked, ds_out / "ranked_scores.pkl")
    with open(ds_out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{dataset}] wrote {ds_out}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["quotebank", "aida", "both", "custom"], default="both"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--entity-kb", type=Path)
    parser.add_argument("--unambiguous", type=Path)
    parser.add_argument("--protocol", choices=["quotebank", "aida"], default="aida")
    parser.add_argument("--easy", type=Path)
    parser.add_argument("--hard", type=Path)
    parser.add_argument("--overall", type=Path)
    parser.add_argument("--name", default="custom")
    parser.add_argument("--with-embeddings", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "artifacts/from_scratch"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.data is not None or args.dataset == "custom":
        if args.data is None or args.entity_kb is None:
            raise SystemExit("custom run requires --data and --entity-kb")
        all_results = {
            args.name: run_dataset(
                args.name,
                out,
                with_embeddings=False,
                data_path=args.data,
                entity_kb_path=args.entity_kb,
                unambiguous_path=args.unambiguous,
                protocol=args.protocol,
                easy_path=args.easy,
                hard_path=args.hard,
                overall_path=args.overall,
            )
        }
    else:
        datasets = ["quotebank", "aida"] if args.dataset == "both" else [args.dataset]
        all_results = {
            ds: run_dataset(ds, out, with_embeddings=args.with_embeddings)
            for ds in datasets
        }
    with open(out / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
