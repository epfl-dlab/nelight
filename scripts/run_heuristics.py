#!/usr/bin/env python3
"""Run NELight heuristic scorers from caches (no precomputed score pickles).

Recomputes popularity, IScore/NIScore/EEIScore (and optionally CSE family) on
Quotebank and/or AIDA, builds UIScore/UCSE, evaluates against paper splits, and
writes fresh score dicts under artifacts/from_scratch/.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.stats as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runlib"))

from cache_paths import ensure_canonical_symlinks, resolve as resolve_cache  # noqa: E402
from scoring.centrality import WikidataCentralityScorer  # noqa: E402
from scoring.semantic import (  # noqa: E402
    EntityContentSimilarityScorer,
    EntityEntitySimilarityScorer,
)

ensure_canonical_symlinks()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _to_torch(obj):
    """Coerce embedding caches to torch tensors (caches may store numpy)."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.asarray(obj))
    if isinstance(obj, dict):
        return {k: _to_torch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_torch(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_torch(v) for v in obj)
    return obj


def lowercase_keys(scores: dict) -> dict:
    return {
        aid: {n.lower(): np.asarray(v, dtype=np.float64) for n, v in ns.items()}
        for aid, ns in scores.items()
    }


def build_unambiguous_cache(data, wiki_cache=None) -> dict:
    """Build unambiguous cache in the research-code format.

    ``utils/unambiguous_entities.py`` stores a *list of singleton id-lists*
    per article: ``[[qid], [qid], ...]``.  ``EntityEntitySimilarityScorer``
    then reads ``cache[aid][0]`` — i.e. only the **first** unambiguous
    mention — which is what produced the paper EEIScore / CSSVE dumps.
    Do not collapse into a single flat QID list.
    """
    cache = {}
    for article in data:
        u = []
        for name in article["names"]:
            if len(name["ids"]) == 1:
                if wiki_cache is not None and name["ids"][0] not in wiki_cache:
                    continue
                u.append(list(name["ids"]))
        cache[article["articleID"]] = u
    return cache


def load_unambiguous_cache(dataset: str, data) -> dict:
    """Prefer shipped unambiguous-mentions pickle; else rebuild in paper format."""
    p = resolve_cache(dataset, "unambiguous_mentions", required=False)
    if p is not None:
        print(f"[{dataset}] using unambiguous mentions cache {p}", flush=True)
        return load_pickle(p)
    print(f"[{dataset}] rebuilding unambiguous mentions (paper list-of-lists format)", flush=True)
    return build_unambiguous_cache(data, wiki_cache=None)


def same_score_rank_ensemble(primary: dict, secondary: dict, data: list) -> dict:
    out = {}
    for article in data:
        aid = article["articleID"]
        if aid not in primary:
            continue
        out[aid] = {}
        for name in article["names"]:
            if len(name["ids"]) <= 1:
                continue
            n = name["name"].lower()
            if n not in primary.get(aid, {}) or n not in secondary.get(aid, {}):
                continue
            scores = np.asarray(primary[aid][n], dtype=np.float64)
            other = np.asarray(secondary[aid][n], dtype=np.float64)
            ranks = ss.rankdata(scores, method="min").astype(np.float64)
            for i in range(1, len(scores) + 1):
                mask = ranks == i
                if mask.sum() > 1:
                    ranks[mask] = ranks[mask] + ss.rankdata(other[mask], method="min") - 1
            out[aid][n] = ranks
    return out


def weighted_sum(score_dicts, weights):
    out = {}
    for aid, name_scores in score_dicts[0].items():
        out[aid] = {}
        for name, arr in name_scores.items():
            total = weights[0] * np.asarray(arr, dtype=np.float64)
            for sc, w in zip(score_dicts[1:], weights[1:]):
                if w == 0:
                    continue
                total = total + w * np.asarray(sc[aid][name], dtype=np.float64)
            out[aid][name] = total
    return out


def transform_scores(scores, fn):
    return {
        aid: {n: fn(np.asarray(v, dtype=np.float64)) for n, v in ns.items()}
        for aid, ns in scores.items()
    }


def flatten_gt(gt):
    return [(a, n.lower(), g) for a, ns in gt.items() for n, g in ns.items()]


def merge_gt(*gts):
    out = {}
    for gt in gts:
        for a, ns in gt.items():
            out.setdefault(a, {}).update({n.lower(): g for n, g in ns.items()})
    return out


def precision_at_one_qb(gt_items, scores):
    c = t = 0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        if aid in scores and name in scores[aid] and np.asarray(scores[aid][name]).size:
            c += int(np.argmax(scores[aid][name]) == gold)
            t += 1
    return c / t if t else float("nan")


def precision_at_one_aida(gt_items, scores):
    c = 0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        if aid in scores and name in scores[aid] and np.asarray(scores[aid][name]).size:
            c += int(np.argmax(scores[aid][name]) == gold)
    return c / len(gt_items) if gt_items else float("nan")


def assign_unambiguous(scores, data):
    out = {aid: {n: np.array(a, copy=True) for n, a in ns.items()} for aid, ns in scores.items()}
    for article in data:
        aid = article["articleID"]
        for name in article["names"]:
            n = name["name"].lower()
            if len(name["ids"]) == 1:
                out.setdefault(aid, {})[n] = np.array([1.0])
    return out


def normalize_name_keys_in_raw(scores, data):
    """Ensure scorer outputs use lowercased mention keys."""
    return lowercase_keys(scores)


def run_dataset(dataset: str, out_dir: Path, with_embeddings: bool):
    if dataset == "quotebank":
        data = load_json(ROOT / "data/Quotebank/data.json")
        easy = load_json(ROOT / "data/Quotebank/easy.json")
        hard = load_json(ROOT / "data/Quotebank/hard.json")
        overall = merge_gt(easy, hard)
        wiki_cache = load_pickle(resolve_cache("quotebank", "entity_kb"))
        protocol = "qb"
        emb_cache = None
        content_emb = None
        sent_emb = None
        if with_embeddings:
            print(f"[{dataset}] loading embedding caches (torch-coerced)...", flush=True)
            emb_cache = _to_torch(load_pickle(resolve_cache("quotebank", "entity_embeddings")))
            content_emb = _to_torch(load_pickle(resolve_cache("quotebank", "document_embeddings")))
            sent_emb = _to_torch(load_pickle(resolve_cache("quotebank", "mention_embeddings")))
    else:
        data = load_json(ROOT / "data/AIDA/data.json")
        easy = load_json(ROOT / "data/AIDA/easy.json")
        hard = load_json(ROOT / "data/AIDA/hard.json")
        overall = load_json(ROOT / "data/AIDA/overall.json")
        wiki_cache = load_pickle(resolve_cache("aida", "entity_kb"))
        protocol = "aida"
        emb_cache = content_emb = sent_emb = None
        if with_embeddings:
            print(f"[{dataset}] loading embedding caches (torch-coerced)...", flush=True)
            ep = resolve_cache("aida", "entity_embeddings", required=False)
            dp = resolve_cache("aida", "document_embeddings", required=False)
            mp = resolve_cache("aida", "mention_embeddings", required=False)
            if ep:
                emb_cache = _to_torch(load_pickle(ep))
            if dp:
                content_emb = _to_torch(load_pickle(dp))
            if mp:
                sent_emb = _to_torch(load_pickle(mp))
            if emb_cache is None:
                print(
                    f"[{dataset}] AIDA embedding caches not available; skipping CSE family",
                    flush=True,
                )

    print(f"[{dataset}] articles={len(data)} entity_kb={len(wiki_cache)}", flush=True)
    unamb = load_unambiguous_cache(dataset, data)

    # Popularity (method names on WikidataCentralityScorer)
    centrality_map = {
        "LQID": "local_qid",
        "NP": "n_statements",
        "NS": "n_sitelinks",
        "PRWP": "pagerank",
        "PRWD": "pagerank_wd",
    }
    methods_raw = {}
    for key, method in centrality_map.items():
        print(f"[{dataset}] scoring {key} ({method})...", flush=True)
        scorer = WikidataCentralityScorer(method, wiki_cache=wiki_cache)
        methods_raw[key] = normalize_name_keys_in_raw(scorer.score_all(data), data)

    # IScore family (stemmed, avoid first_paragraph alone — paper best = D+S with stemming)
    print(f"[{dataset}] scoring IScore/NIScore...", flush=True)
    iscore = EntityContentSimilarityScorer(
        "iscore", wiki_cache=wiki_cache, stem=True, props_to_avoid=["first_paragraph"]
    )
    niscore = EntityContentSimilarityScorer(
        "iscore_narrow", wiki_cache=wiki_cache, stem=True, props_to_avoid=["first_paragraph"]
    )
    methods_raw["IScore"] = normalize_name_keys_in_raw(iscore.score_all(data), data)
    methods_raw["NIScore"] = normalize_name_keys_in_raw(niscore.score_all(data), data)

    print(f"[{dataset}] scoring EEIScore...", flush=True)
    eeiscore = EntityEntitySimilarityScorer(
        "matching_attributes", wiki_cache=wiki_cache, unambiguous_cache=unamb
    )
    methods_raw["EEIScore"] = normalize_name_keys_in_raw(eeiscore.score_all(data), data)

    if with_embeddings and emb_cache is not None and content_emb is not None:
        print(f"[{dataset}] scoring CSE/NCSE/CSSVE...", flush=True)
        cse = EntityContentSimilarityScorer(
            "paragraph_or_props",
            wiki_cache=wiki_cache,
            embeddings_cache=emb_cache,
            content_embeddings_cache=content_emb,
        )
        methods_raw["CSE"] = normalize_name_keys_in_raw(cse.score_all(data), data)
        if sent_emb is not None:
            ncse = EntityContentSimilarityScorer(
                "paragraph_or_props_narrow",
                wiki_cache=wiki_cache,
                embeddings_cache=emb_cache,
                content_embeddings_cache=content_emb,
                sentence_embeddings_cache=sent_emb,
            )
            methods_raw["NCSE"] = normalize_name_keys_in_raw(ncse.score_all(data), data)
        cssve = EntityEntitySimilarityScorer(
            "matching_attributes_emb",
            wiki_cache=wiki_cache,
            embeddings_cache=emb_cache,
            unambiguous_cache=unamb,
        )
        methods_raw["CSSVE"] = normalize_name_keys_in_raw(cssve.score_all(data), data)

    # Persist raw scores
    ds_out = out_dir / dataset
    for name, scores in methods_raw.items():
        save_pickle(scores, ds_out / f"{name}.pkl")

    # Composites + tie-breaks
    # Quotebank: §6.1 + App. E.3 Table 9 (method-specific TBs, then LQID).
    # AIDA: paper dumps are raw scores; numpy argmax (no popularity TB) matches
    # Table 2 exactly for popularity + I/NI/EEI/UI.
    lqid = methods_raw["LQID"]
    ns = methods_raw["NS"]
    np_scores = methods_raw["NP"]
    prwp = methods_raw["PRWP"]

    def tiebreak(scores, *pops):
        out = scores
        for p in pops:
            out = same_score_rank_ensemble(out, p, data)
        return out

    if dataset == "quotebank":
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
        ui_w = [1.0, 1.0, 1.0]
        ui = weighted_sum(
            [methods_raw["IScore"], methods_raw["NIScore"], methods_raw["EEIScore"]],
            ui_w,
        )
        ranked["UIScore"] = tiebreak(ui, prwp, lqid)
    else:
        ranked = {k: methods_raw[k] for k in [
            "LQID", "NP", "NS", "PRWD", "PRWP", "IScore", "NIScore", "EEIScore"
        ]}
        ui = weighted_sum(
            [methods_raw["IScore"], methods_raw["NIScore"], methods_raw["EEIScore"]],
            [0.9, 0.0, 1.0],
        )
        ranked["UIScore"] = ui

    if "CSE" in methods_raw and "NCSE" in methods_raw and "CSSVE" in methods_raw:
        cse_t = transform_scores(methods_raw["CSE"], lambda x: 0.5 * (x + 1.0))
        if dataset == "quotebank":
            # Table-2 UCSE claim uses Laplacian on NCSE (literal §4.4 half/half → 0.894)
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
            # AIDA UCSE: half-shift CSE/NCSE + Laplacian CSSVE, weights (0,1,1)
            ncse_t = transform_scores(methods_raw["NCSE"], lambda x: 0.5 * (x + 1.0))
            cssve_t = transform_scores(
                methods_raw["CSSVE"], lambda x: (x + 1.0) / np.sum(x + 1.0)
            )
            ucse = weighted_sum([cse_t, ncse_t, cssve_t], [0.0, 1.0, 1.0])
            ranked["UCSE"] = ucse
            ranked["CSE"] = methods_raw["CSE"]
            ranked["NCSE"] = methods_raw["NCSE"]
            ranked["CSSVE"] = methods_raw["CSSVE"]

    if protocol == "aida":
        for k in list(ranked):
            ranked[k] = assign_unambiguous(ranked[k], data)

    # Evaluate
    results = {}
    for split_name, gt in [("easy", easy), ("hard", hard), ("overall", overall)]:
        items = flatten_gt(gt)
        results[split_name] = {}
        for method, scores in ranked.items():
            if protocol == "qb":
                p = precision_at_one_qb(items, scores)
            else:
                p = precision_at_one_aida(items, scores)
            results[split_name][method] = p

    print(f"\n=== {dataset} from-scratch P@1 ===")
    methods = [
        "LQID", "NP", "NS", "PRWD", "PRWP",
        "IScore", "NIScore", "EEIScore", "UIScore",
        "CSE", "NCSE", "CSSVE", "UCSE",
    ]
    print(f"{'Method':12s} {'Easy':>7s} {'Hard':>7s} {'Overall':>8s}")
    for m in methods:
        if m not in ranked:
            continue
        print(
            f"{m:12s} "
            f"{results['easy'][m]:7.3f} "
            f"{results['hard'][m]:7.3f} "
            f"{results['overall'][m]:8.3f}"
        )

    save_pickle(ranked, ds_out / "ranked_scores.pkl")
    with open(ds_out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{dataset}] wrote {ds_out}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["quotebank", "aida", "both"], default="both")
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also recompute CSE/NCSE/CSSVE/UCSE from embedding caches",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts/from_scratch"),
        help="Output directory",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = ["quotebank", "aida"] if args.dataset == "both" else [args.dataset]
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds, out, with_embeddings=args.with_embeddings)
    with open(out / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
