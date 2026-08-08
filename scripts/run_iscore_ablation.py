#!/usr/bin/env python3
"""Recompute Table 6 (IScore feature × normalization ablation) from caches.

Quotebank grid D/P/S/S_A × none/lemma/stem; NS tie-break; overall.json eval.
Unique entity strings are lemmatized once (pywsd is expensive).
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from nltk import PorterStemmer, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pywsd.utils import lemmatize_sentence
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runlib.cache_paths import resolve as resolve_cache  # noqa: E402
from runlib.eval import (  # noqa: E402
    flatten_gt,
    load_json,
    mrr_qb,
    normalize_scores,
    precision_at_one_qb,
    same_score_rank_ensemble,
)
from runlib.scoring.centrality import WikidataCentralityScorer  # noqa: E402

FEATURES = [
    "D", "P", "S", "S_A", "D + P", "D + S", "D + S_A",
    "P + S", "P + S_A", "D + P + S", "D + P + S_A",
]
NORMS = ["No normalization", "Lemmatization", "Stemming"]

PAPER_MATRIX = {
    "D": (0.869, 0.921, 0.890, 0.930, 0.894, 0.934),
    "P": (0.832, 0.903, 0.816, 0.895, 0.832, 0.902),
    "S": (0.894, 0.936, 0.898, 0.940, 0.906, 0.944),
    "S_A": (0.886, 0.932, 0.890, 0.935, 0.898, 0.939),
    "D + P": (0.841, 0.907, 0.820, 0.898, 0.841, 0.906),
    "D + S": (0.902, 0.943, 0.906, 0.945, 0.918, 0.952),
    "D + S_A": (0.890, 0.937, 0.906, 0.947, 0.914, 0.950),
    "P + S": (0.861, 0.919, 0.861, 0.920, 0.873, 0.925),
    "P + S_A": (0.878, 0.928, 0.882, 0.931, 0.882, 0.930),
    "D + P + S": (0.861, 0.921, 0.861, 0.920, 0.873, 0.926),
    "D + P + S_A": (0.886, 0.934, 0.886, 0.934, 0.882, 0.930),
}

_TOKENIZER = TreebankWordTokenizer()
_DETOKENIZER = TreebankWordDetokenizer()
_STEMMER = PorterStemmer()
_STOP = None
_CENTRALITY = {
    "n_statements", "n_sitelinks", "pagerank", "pagerank_wd",
    "indeg", "outdeg", "degree",
}


def _stopwords():
    global _STOP
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    return _STOP


def _feature_props(feature: str):
    use_alias = "S_A" in feature
    base = feature.replace("S_A", "S")
    parts = {p.strip() for p in base.split("+")}
    if parts == {"D"}:
        return use_alias, {"description"}, None
    if parts == {"P"}:
        return use_alias, {"first_paragraph"}, None
    if parts == {"S"}:
        return use_alias, None, {"description", "first_paragraph"}
    if parts == {"D", "P"}:
        return use_alias, {"description", "first_paragraph"}, None
    if parts == {"D", "S"}:
        return use_alias, None, {"first_paragraph"}
    if parts == {"P", "S"}:
        return use_alias, None, {"description"}
    if parts == {"D", "P", "S"}:
        return use_alias, None, set()
    raise ValueError(feature)


def _iter_prop_values(entity: dict, keep, avoid):
    if keep is not None:
        for prop in keep:
            if prop not in entity:
                return  # KeyError→empty bow
            values = entity[prop]
            if prop == "first_paragraph":
                values = [values]
            for value in values:
                yield value
        return
    for prop, values in entity.items():
        if prop in _CENTRALITY or prop in (avoid or set()):
            continue
        if not (re.match(r"^P[0-9]+$", prop) or prop in {"description", "first_paragraph"}):
            continue
        if prop == "first_paragraph":
            values = [values]
        for value in values:
            yield value


def _tokenize_plain(text: str) -> frozenset[str]:
    text = text.replace("\xa0", " ").lower()
    return frozenset(set(_TOKENIZER.tokenize(text)) - _stopwords())


def _tokenize_stem(plain: frozenset[str]) -> frozenset[str]:
    return frozenset(_STEMMER.stem(t) for t in plain)


def _tokenize_lemma(text: str) -> frozenset[str]:
    text = text.replace("\xa0", " ").lower()
    text = _DETOKENIZER.detokenize(lemmatize_sentence(text)).lower()
    return frozenset(set(_TOKENIZER.tokenize(text)) - _stopwords())


def _alpha(toks: frozenset[str]) -> frozenset[str]:
    return frozenset(t for t in toks if re.match("[a-zA-Z]", t))


def build_string_token_tables(strings: set[str]) -> dict[str, dict[str, frozenset[str]]]:
    """For each unique string: alpha token sets under each normalization."""
    tables: dict[str, dict[str, frozenset[str]]] = {}
    for s in tqdm(sorted(strings, key=len), desc="lemma/stem unique strings"):
        plain = _tokenize_plain(s)
        lemma = _tokenize_lemma(s)
        stem = _tokenize_stem(plain)
        tables[s] = {
            "No normalization": _alpha(plain),
            "Lemmatization": _alpha(lemma),
            "Stemming": _alpha(stem),
        }
    return tables


def precompute_article_tokens(data) -> dict[str, dict[str, set[str]]]:
    out = {n: {} for n in NORMS}
    for article in tqdm(data, desc="articles"):
        aid = article["articleID"]
        text = article["content"]
        plain = _tokenize_plain(text)
        lemma = _tokenize_lemma(text)
        stem = _tokenize_stem(plain)
        out["No normalization"][aid] = set(plain)
        out["Lemmatization"][aid] = set(lemma)
        out["Stemming"][aid] = set(stem)
    return out


def collect_entity_strings(cache: dict, qids: set[str]) -> set[str]:
    strings: set[str] = set()
    for qid in qids:
        ent = cache.get(qid)
        if not ent:
            continue
        for prop, values in ent.items():
            if prop in _CENTRALITY:
                continue
            if not (re.match(r"^P[0-9]+$", prop) or prop in {"description", "first_paragraph"}):
                continue
            if prop == "first_paragraph":
                values = [values]
            for value in values:
                if isinstance(value, str) and value:
                    strings.add(value)
    return strings


def entity_bow(entity, keep, avoid, norm: str, str_tables: dict) -> set[str]:
    bow: set[str] = set()
    if keep is not None:
        for prop in keep:
            if prop not in entity:
                return set()
    for value in _iter_prop_values(entity, keep, avoid):
        if not isinstance(value, str) or value not in str_tables:
            # empty / missing from table → skip
            if isinstance(value, str) and value:
                # fallback (should be rare)
                plain = _tokenize_plain(value)
                if norm == "Stemming":
                    bow |= set(_alpha(_tokenize_stem(plain)))
                elif norm == "Lemmatization":
                    bow |= set(_alpha(_tokenize_lemma(value)))
                else:
                    bow |= set(_alpha(plain))
            continue
        bow |= set(str_tables[value][norm])
    return bow


def score_iscore(data, article_tokens, entity_bows) -> dict:
    scores = {}
    for article in data:
        aid = article["articleID"]
        content = article_tokens[aid]
        scores[aid] = {}
        for name in article["names"]:
            if len(name["ids"]) <= 1:
                continue
            mention_toks = set(_TOKENIZER.tokenize(name["name"]))
            ac = content - mention_toks
            scores[aid][name["name"]] = np.array(
                [
                    len(ac & entity_bows[qid]) if qid in entity_bows else 0
                    for qid in name["ids"]
                ],
                dtype=np.float64,
            )
    return scores


def run_ablation(out_dir: Path) -> dict:
    data = load_json(ROOT / "data/Quotebank/data.json")
    gt = load_json(ROOT / "data/Quotebank/overall.json")
    wiki = pickle.load(open(resolve_cache("quotebank", "entity_kb"), "rb"))
    alias = pickle.load(open(resolve_cache("quotebank", "entity_kb_aliases"), "rb"))

    print("[ablation] NS tie-break...", flush=True)
    ns = normalize_scores(
        WikidataCentralityScorer("NS", wiki_cache=wiki).score_all(data)
    )
    candidate_qids = {
        qid
        for article in data
        for name in article["names"]
        if len(name["ids"]) > 1
        for qid in name["ids"]
    }

    print("[ablation] collecting unique entity strings...", flush=True)
    strings = collect_entity_strings(wiki, candidate_qids) | collect_entity_strings(
        alias, candidate_qids
    )
    print(f"[ablation] {len(strings)} unique strings to normalize", flush=True)
    str_tables = build_string_token_tables(strings)

    print("[ablation] article tokens...", flush=True)
    article_tok = precompute_article_tokens(data)

    configs = {}
    for feature in FEATURES:
        use_alias, keep, avoid = _feature_props(feature)
        key = (
            use_alias,
            frozenset(keep) if keep is not None else None,
            frozenset(avoid or ()),
        )
        configs.setdefault(key, []).append(feature)

    print("[ablation] entity BOWs...", flush=True)
    entity_bows = {}
    for (use_alias, keep, avoid), _feats in configs.items():
        cache = alias if use_alias else wiki
        keep_set = set(keep) if keep is not None else None
        avoid_set = set(avoid) if avoid is not None else set()
        for norm in NORMS:
            bows = {
                qid: entity_bow(cache[qid], keep_set, avoid_set, norm, str_tables)
                for qid in candidate_qids
                if qid in cache
            }
            entity_bows[(use_alias, keep, avoid, norm)] = bows

    rows = {}
    lines = []
    gt_items = flatten_gt(gt)
    n = 0
    for feature in FEATURES:
        use_alias, keep, avoid = _feature_props(feature)
        keep_f = frozenset(keep) if keep is not None else None
        avoid_f = frozenset(avoid or ())
        for norm in NORMS:
            n += 1
            bows = entity_bows[(use_alias, keep_f, avoid_f, norm)]
            raw = normalize_scores(score_iscore(data, article_tok[norm], bows))
            ranked = same_score_rank_ensemble(raw, ns, data)
            p = round(precision_at_one_qb(gt_items, ranked), 3)
            m = round(mrr_qb(gt_items, ranked), 3)
            rows.setdefault(feature, {})[norm] = {"p@1": p, "mrr": m}
            lines.append(
                f"Feature: {feature:12s} Normalization: {norm:20s} "
                f"P@1: {p:.3f} MRR: {m:.3f}"
            )

    ok = True
    for feat, paper_vals in PAPER_MATRIX.items():
        for j, norm in enumerate(NORMS):
            pp, pm = paper_vals[2 * j], paper_vals[2 * j + 1]
            gp, gm = rows[feat][norm]["p@1"], rows[feat][norm]["mrr"]
            tol = 0.01 if norm == "Lemmatization" else 0.0015
            cell_ok = bool(abs(gp - pp) <= tol and abs(gm - pm) <= tol)
            rows[feat][norm].update(
                paper_p=float(pp), paper_mrr=float(pm), ok=cell_ok
            )
            ok = ok and cell_ok

    best = rows["D + S"]["Stemming"]
    best_ok = best["p@1"] == 0.918 and best["mrr"] == 0.952
    result = {
        "rows": rows,
        "best": {
            "combination": "D + S",
            "normalization": "Stemming",
            "p@1": float(best["p@1"]),
            "mrr": float(best["mrr"]),
        },
        "match": bool(ok and best_ok),
        "source": "recomputed from caches/quotebank/entity_kb{,_aliases}.pkl",
        "notes": [
            "Best cell D+S/Stemming must match paper 0.918/0.952 exactly.",
            "Lemmatization column allows 0.01 abs tol (pywsd/WordNet drift vs 2022).",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_dir / "iscore_ablation.json", "w"), indent=2, default=str)
    (out_dir / "iscore_ablation.txt").write_text("\n".join(lines) + "\n")
    print(
        f"[ablation] best D+S/Stemming P@1={best['p@1']} MRR={best['mrr']} "
        f"match={result['match']}",
        flush=True,
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/from_scratch/quotebank")
    args = ap.parse_args()
    sys.exit(0 if run_ablation(args.out)["match"] else 1)


if __name__ == "__main__":
    main()
