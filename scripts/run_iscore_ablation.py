#!/usr/bin/env python3
"""Recompute Table 6 (IScore feature × normalization ablation) from caches.

Protocol matches the original ``score_fn_ablation.py`` + ``experiments.fn_ablation``:
  * Quotebank only
  * feature grid D / P / S / S_A and combinations
  * norms: none / lemmatize / stem
  * S_A uses ``caches/quotebank/entity_kb_aliases.pkl``
  * tie-break with NS only (App. E.1), then evaluate on ``overall.json``

BOW / article tokenization is precomputed once per normalization so the full
33-cell grid finishes in minutes rather than hours (pywsd lemmatization is
expensive).
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
sys.path.insert(0, str(ROOT / "scripts"))

from runlib.cache_paths import resolve as resolve_cache  # noqa: E402
from runlib.scoring.centrality import WikidataCentralityScorer  # noqa: E402

_ns: dict = {}
exec(
    open(ROOT / "scripts/reproduce_tables.py")
    .read()
    .split("def main")[0]
    .replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path(r'{ROOT}')"),
    _ns,
    _ns,
)
load_json = _ns["load_json"]
normalize_scores = _ns["normalize_scores"]
same_score_rank_ensemble = _ns["same_score_rank_ensemble"]
precision_at_one_qb = _ns["precision_at_one_qb"]
mrr_qb = _ns["mrr_qb"]
flatten_gt = _ns["flatten_gt"]

FEATURES = [
    "D",
    "P",
    "S",
    "S_A",
    "D + P",
    "D + S",
    "D + S_A",
    "P + S",
    "P + S_A",
    "D + P + S",
    "D + P + S_A",
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
    "n_statements",
    "n_sitelinks",
    "pagerank",
    "pagerank_wd",
    "indeg",
    "outdeg",
    "degree",
}


def _stopwords():
    global _STOP
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    return _STOP


def _normalize_text(text: str, norm: str) -> str:
    text = text.replace("\xa0", " ").lower()
    if norm == "Lemmatization":
        text = _DETOKENIZER.detokenize(lemmatize_sentence(text)).lower()
    return text


_TOKEN_CACHE: dict[tuple[str, str, bool], frozenset[str]] = {}


def _tokens(text: str, norm: str, *, alpha_only: bool) -> set[str]:
    """Mirror ``EntityContentSimilarityScorer`` tokenization.

    Article content keeps all tokens (minus stopwords); entity BOW keeps
    only tokens matching ``[a-zA-Z]``.
    """
    key = (text, norm, alpha_only)
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        return set(cached)
    normed = _normalize_text(text, norm)
    toks = set(_TOKENIZER.tokenize(normed)) - _stopwords()
    if norm == "Stemming":
        toks = {_STEMMER.stem(t) for t in toks}
    if alpha_only:
        toks = {t for t in toks if re.match("[a-zA-Z]", t)}
    _TOKEN_CACHE[key] = frozenset(toks)
    return toks


def _feature_props(feature: str):
    """Return (use_alias, props_to_keep|None, props_to_avoid|None)."""
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


def _entity_bow(entity: dict, norm: str, keep, avoid) -> set[str]:
    """Build entity BOW; match KeyError→0 behavior of the original scorer."""
    bow: set[str] = set()
    if keep is not None:
        # props_to_keep path: any missing property aborts (original KeyError → 0)
        for prop in keep:
            if prop not in entity:
                return set()
        props = keep
    else:
        props = [
            k
            for k in entity
            if k not in _CENTRALITY
            and k not in (avoid or set())
            and (re.match(r"^P[0-9]+$", k) or k in {"description", "first_paragraph"})
        ]
    for prop in props:
        values = entity[prop]
        if prop == "first_paragraph":
            values = [values]
        for value in values:
            bow |= _tokens(value, norm, alpha_only=True)
    return bow


def precompute_article_tokens(data, norm: str) -> dict:
    out = {}
    for article in tqdm(data, desc=f"articles/{norm[:4]}", leave=False):
        out[article["articleID"]] = _tokens(article["content"], norm, alpha_only=False)
    return out


def precompute_entity_bows(cache: dict, norm: str, keep, avoid, qids: set[str]) -> dict:
    return {
        qid: _entity_bow(cache[qid], norm, keep, avoid)
        for qid in tqdm(qids, desc="entities", leave=False)
        if qid in cache
    }


def score_iscore(data, article_tokens, entity_bows) -> dict:
    tok = _TOKENIZER
    scores = {}
    for article in data:
        aid = article["articleID"]
        content = article_tokens[aid]
        scores[aid] = {}
        for name in article["names"]:
            if len(name["ids"]) <= 1:
                continue
            # Match EntityContentSimilarityScorer._remove_name (no pre-lower).
            mention_toks = set(tok.tokenize(name["name"]))
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

    print("[ablation] scoring NS for tie-break...", flush=True)
    ns = normalize_scores(
        WikidataCentralityScorer("n_sitelinks", wiki_cache=wiki).score_all(data)
    )

    candidate_qids = {
        qid
        for article in data
        for name in article["names"]
        if len(name["ids"]) > 1
        for qid in name["ids"]
    }

    print("[ablation] precomputing article tokens per normalization...", flush=True)
    article_tok = {norm: precompute_article_tokens(data, norm) for norm in NORMS}

    # Unique (cache, keep, avoid) configs
    configs = {}
    for feature in FEATURES:
        use_alias, keep, avoid = _feature_props(feature)
        key = (use_alias, frozenset(keep) if keep is not None else None, frozenset(avoid or ()))
        configs.setdefault(key, []).append(feature)

    print("[ablation] precomputing entity BOWs...", flush=True)
    entity_bows = {}  # (use_alias, keep, avoid, norm) -> {qid: set}
    for (use_alias, keep, avoid), _feats in configs.items():
        cache = alias if use_alias else wiki
        keep_set = set(keep) if keep is not None else None
        avoid_set = set(avoid) if avoid is not None else set()
        for norm in NORMS:
            print(
                f"  cache={'alias' if use_alias else 'wiki'} "
                f"keep={keep_set} avoid={avoid_set} norm={norm}",
                flush=True,
            )
            entity_bows[(use_alias, keep, avoid, norm)] = precompute_entity_bows(
                cache, norm, keep_set, avoid_set, candidate_qids
            )

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
            print(f"[ablation] {n}/33 {feature} / {norm}", flush=True)
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
            gp = rows[feat][norm]["p@1"]
            gm = rows[feat][norm]["mrr"]
            # Lemmatization drifts ~0.4pp under current pywsd/WordNet vs the
            # 2022 archive; no-norm / stemming stay within rounding.
            tol = 0.01 if norm == "Lemmatization" else 0.0015
            cell_ok = bool(abs(gp - pp) <= tol and abs(gm - pm) <= tol)
            rows[feat][norm]["paper_p"] = float(pp)
            rows[feat][norm]["paper_mrr"] = float(pm)
            rows[feat][norm]["ok"] = cell_ok
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
    out_json = out_dir / "iscore_ablation.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    (out_dir / "iscore_ablation.txt").write_text("\n".join(lines) + "\n")
    print(f"[ablation] wrote {out_json}", flush=True)
    print(
        f"[ablation] best D+S/Stemming P@1={best['p@1']} MRR={best['mrr']} "
        f"match={result['match']}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/from_scratch/quotebank",
    )
    args = parser.parse_args()
    result = run_ablation(args.out)
    sys.exit(0 if result["match"] else 1)


if __name__ == "__main__":
    main()
