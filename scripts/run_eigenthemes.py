#!/usr/bin/env python3
"""Run Eigenthemes (weigen) from scratch and convert to NELight score dicts.

Builds optional IScore-weighted inputs, runs computeScores against
deepwalk_wikidata.pickle, converts to {articleID: {name: scores}}, and
evaluates Table-2 P@1 (Quotebank: NS→LQID TB; AIDA: no popularity TB).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "from_scratch"


def resolve_eigen_dir() -> Path:
    """Find an Eigenthemes tree with DeepWalk embeddings.

    Search order:
      1. ``$NELIGHT_EIGENTHEMES``
      2. ``workspace/eigenthemes`` (symlink or checkout)
    """
    candidates = []
    env = os.environ.get("NELIGHT_EIGENTHEMES")
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "workspace" / "eigenthemes")
    for cand in candidates:
        if (cand / "unsupervised_el.py").is_file() and (
            cand / "embeddings" / "deepwalk_wikidata.pickle"
        ).is_file():
            return cand.resolve()
    return (ROOT / "workspace" / "eigenthemes").resolve()


EIGEN_DIR = resolve_eigen_dir()
EMB = EIGEN_DIR / "embeddings" / "deepwalk_wikidata.pickle"

from runlib.eval import (  # noqa: E402
    assign_unambiguous,
    flatten_gt,
    load_json,
    normalize_scores,
    precision_at_one_aida,
    precision_at_one_qb,
    same_score_rank_ensemble,
)


def strip_idx(name: str) -> str:
    return re.sub(r"\d+$", "", name)


def load_pk(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pk(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def reweight_eigen_json(eigen_data: dict, score_dict: dict, dataset: str) -> dict:
    """Replace candidate prominence weights with NELight scores (NS / IScore)."""
    data = load_json(ROOT / f"data/{'Quotebank' if dataset == 'quotebank' else 'AIDA'}/data.json")
    # Ordered NELight mentions per (aid, surface): list of (nelight_name, ids)
    idmap: dict[tuple, list] = {}
    for article in data:
        aid = article["articleID"]
        for name in article["names"]:
            surf = strip_idx(name["name"].lower()) if dataset == "aida" else name["name"].lower()
            idmap.setdefault((aid, surf), []).append((name["name"].lower(), name["ids"]))

    out = copy.deepcopy(eigen_data)
    occ_e: dict[tuple, int] = {}
    for aid, items in out.items():
        for item in items:
            ment = item["mention"].lower()
            key = (aid, ment)
            idx = occ_e.get(key, 0)
            occ_e[key] = idx + 1
            lst = idmap.get(key, [])
            if idx >= len(lst):
                continue
            nl_name, ids = lst[idx]
            sc = None
            if aid in score_dict:
                for k, v in score_dict[aid].items():
                    if k.lower() == nl_name:
                        sc = np.asarray(v, dtype=float)
                        break
            if sc is None or len(sc) != len(ids):
                continue
            q2s = {q: float(s) for q, s in zip(ids, sc)}
            new_cands = []
            for c in item["candidates"]:
                qid = c[0]
                new_cands.append([qid, q2s.get(qid, 0.0), c[2] if len(c) > 2 else ment])
            new_cands.sort(key=lambda x: (x[1], int(str(x[0])[1:])), reverse=True)
            item["candidates"] = new_cands
    return out


def _patch_legacy_deps():
    """Compatibility shims for Eigenthemes (NumPy 1.x / older scikit-learn)."""
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, typ)
    # scikit-learn ≥1.6 renamed force_all_finite → ensure_all_finite
    import sklearn.utils.validation as skval

    if not getattr(skval.check_array, "_nelight_force_all_finite_patch", False):
        _orig = skval.check_array

        def _check_array(*args, **kwargs):
            if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return _orig(*args, **kwargs)

        _check_array._nelight_force_all_finite_patch = True  # type: ignore[attr-defined]
        skval.check_array = _check_array  # type: ignore[assignment]


def run_weigen(eigen_json_path: Path, tag: str):
    """Execute Eigenthemes computeScores; return raw dict with weigen rows."""
    sys.path.insert(0, str(EIGEN_DIR))
    os_chdir = Path.cwd()
    _patch_legacy_deps()
    os.chdir(EIGEN_DIR)
    try:
        src = open("unsupervised_el.py").read().split("\ndatasets =")[0]
        src = src.replace("time.clock()", "time.perf_counter()")
        ns: dict = {}
        exec(src, ns)
        import utils as eigen_utils

        data = json.load(open(eigen_json_path))
        vectors = eigen_utils.loadWikipedia2VecVectors(str(EMB))
        ferr = open(os.devnull, "w")
        params = {
            "weight": True,
            "meanCenter": False,
            "embeddingType": "deepwalk",
            "numCands": 20,
            "ncomp": 10,
        }
        (
            key,
            cand_names,
            degree_baseline,
            avg_baseline,
            wavg_baseline,
            eigen,
            weigen,
            labels,
            mention2QueryId,
            queryId2Mention,
            tpca,
            twpca,
        ) = ns["computeScores"](data, tag, vectors, ferr, params)
        return {
            "key": key,
            "cand_names": cand_names,
            "eigen": eigen,
            "weigen": weigen,
            "labels": labels,
            "mention2QueryId": mention2QueryId,
            "queryId2Mention": queryId2Mention,
        }
    finally:
        os.chdir(os_chdir)


def convert_to_nelight(raw: dict, dataset: str) -> dict:
    data = load_json(ROOT / f"data/{'Quotebank' if dataset == 'quotebank' else 'AIDA'}/data.json")
    m2q = raw["mention2QueryId"]
    q2m = {}
    for key, val in m2q.items():
        try:
            qid = val[0] if len(val) == 2 else val
        except TypeError:
            qid = val
        if qid != -1:
            q2m[int(qid)] = key
    per = {}
    for key_s, cand, score in zip(raw["key"], raw["cand_names"], raw["weigen"]):
        qid = int(key_s.split(":")[-1])
        if qid not in q2m:
            continue
        per.setdefault(q2m[qid], {})[cand] = float(score)

    by_surf: dict[tuple, list] = {}
    for mkey, sc in per.items():
        by_surf.setdefault((mkey[0], mkey[1].lower()), []).append((mkey[2], sc))
    for k in by_surf:
        by_surf[k].sort(key=lambda x: x[0])

    out: dict = {}
    for article in data:
        aid = article["articleID"]
        out[aid] = {}
        occ: dict[str, int] = {}
        for name in article["names"]:
            n = name["name"]
            nl = n.lower()
            ids = name["ids"]
            surface = strip_idx(nl) if dataset == "aida" else nl
            idx = occ.get(surface, 0)
            occ[surface] = idx + 1
            lst = by_surf.get((aid, surface), [])
            sc = lst[idx][1] if idx < len(lst) else None
            if sc is None and lst:
                sc = max(lst, key=lambda x: sum(1 for q in ids if q in x[1]))[1]
            if sc is None:
                out[aid][nl] = np.zeros(len(ids), dtype=float)
            else:
                out[aid][nl] = np.array([sc.get(q, 0.0) for q in ids], dtype=float)
    return out


def eval_triple(dataset: str, scores: dict, tb_chain=None):
    ds = "Quotebank" if dataset == "quotebank" else "AIDA"
    data = load_json(ROOT / f"data/{ds}/data.json")
    easy = load_json(ROOT / f"data/{ds}/easy.json")
    hard = load_json(ROOT / f"data/{ds}/hard.json")
    sc = normalize_scores(scores)
    if tb_chain:
        for tb in tb_chain:
            sc = same_score_rank_ensemble(sc, tb, data)
    if dataset == "quotebank":
        overall = {}
        for g in (easy, hard):
            for aid, names in g.items():
                overall.setdefault(aid, {}).update(names)
        fn = precision_at_one_qb
    else:
        overall = load_json(ROOT / "data/AIDA/overall.json")
        sc = assign_unambiguous(sc, data)
        fn = precision_at_one_aida
    return tuple(fn(flatten_gt(gt), sc) for gt in (easy, hard, overall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["quotebank", "aida", "both"], default="both")
    ap.add_argument("--variant", choices=["ns", "iscore", "both"], default="both",
                    help="ns=paper Eigen (degree on AIDA / NS on QB); iscore=Eigen(IScore)")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="Reuse existing eigen_raw_*.pkl for ns variant")
    args = ap.parse_args()

    if not (EIGEN_DIR / "unsupervised_el.py").is_file() or not EMB.exists():
        raise SystemExit(
            "Eigenthemes tree not found.\n"
            "Expected DeepWalk embeddings + unsupervised_el.py under one of:\n"
            "  $NELIGHT_EIGENTHEMES\n"
            "  workspace/eigenthemes\n"
            "Example:\n"
            "  export NELIGHT_EIGENTHEMES=/path/to/eigenthemes\n"
            "  # or: ln -sfn /path/to/eigenthemes workspace/eigenthemes\n"
            "See REPRODUCIBILITY.md (Live Eigenthemes)."
        )
    print(f"Using Eigenthemes tree: {EIGEN_DIR}", flush=True)

    pop = load_pk(ROOT / "scores/popularity_scores.pkl")
    datasets = ["quotebank", "aida"] if args.dataset == "both" else [args.dataset]
    variants = ["ns", "iscore"] if args.variant == "both" else [args.variant]

    for dataset in datasets:
        tag = "quotebank_test_complete.json" if dataset == "quotebank" else "aida_test_complete.json"
        base_json = EIGEN_DIR / "data" / tag
        if not base_json.exists():
            raise SystemExit(
                f"Missing {base_json}. Eigenthemes candidate JSON lists are not "
                "shipped in this repo; place them under the Eigenthemes data/ dir."
            )
        eigen_base = json.load(open(base_json))
        ds_out = OUT / dataset
        ds_out.mkdir(parents=True, exist_ok=True)

        for variant in variants:
            print(f"\n=== {dataset} / {variant} ===", flush=True)
            if variant == "ns":
                json_path = base_json
                raw_path = OUT / f"eigen_raw_{tag}.pkl"
                if args.reuse_raw and raw_path.exists():
                    raw = load_pk(raw_path)
                    print(f"reused {raw_path}", flush=True)
                else:
                    raw = run_weigen(json_path, tag)
                    save_pk(raw, raw_path)
                    print(f"wrote {raw_path}", flush=True)
                scores = convert_to_nelight(raw, dataset)
                out_name = "Eigen_live_weigen.pkl"
                method = "Eigen"
            else:
                # Paper Eigen(IScore) inputs (int(IScore) prominence). Rebuilding from
                # current FS IScore does not reproduce historical integer weights;
                # use the candidate JSONs under the Eigenthemes data/ dir.
                hist_name = (
                    "quotebank_iscore_test_complete.json"
                    if dataset == "quotebank"
                    else "aida_iscore_final_test_complete.json"
                )
                json_path = EIGEN_DIR / "data" / hist_name
                if not json_path.exists():
                    print(f"missing {hist_name}; cannot run Eigen(IScore)")
                    continue
                raw = run_weigen(json_path, hist_name)
                save_pk(raw, OUT / f"eigen_raw_{hist_name}.pkl")
                if dataset == "aida":
                    # Mentions already use japan0-style names matching NELight.
                    data = load_json(ROOT / "data/AIDA/data.json")
                    m2q = raw["mention2QueryId"]
                    q2m = {}
                    for key, val in m2q.items():
                        try:
                            qid = val[0] if len(val) == 2 else val
                        except TypeError:
                            qid = val
                        if qid != -1:
                            q2m[int(qid)] = key
                    per = {}
                    for key_s, cand, score in zip(
                        raw["key"], raw["cand_names"], raw["weigen"]
                    ):
                        qid = int(key_s.split(":")[-1])
                        if qid in q2m:
                            per.setdefault(q2m[qid], {})[cand] = float(score)
                    scores = {}
                    for article in data:
                        aid = article["articleID"]
                        scores[aid] = {}
                        for name in article["names"]:
                            nl = name["name"].lower()
                            ids = name["ids"]
                            sc = None
                            for mkey, s in per.items():
                                if mkey[0] == aid and mkey[1].lower() == nl:
                                    sc = s
                                    break
                            scores[aid][nl] = (
                                np.zeros(len(ids), dtype=float)
                                if sc is None
                                else np.array([sc.get(q, 0.0) for q in ids], dtype=float)
                            )
                else:
                    scores = convert_to_nelight(raw, dataset)
                # Notebook fill: missing/zero eigen → NS
                pop = load_pk(ROOT / "scores/popularity_scores.pkl")
                ns = normalize_scores(pop["qb" if dataset == "quotebank" else "aida"]["ns"])
                data = load_json(
                    ROOT / f"data/{'Quotebank' if dataset == 'quotebank' else 'AIDA'}/data.json"
                )
                for article in data:
                    aid = article["articleID"]
                    for name in article["names"]:
                        nl = name["name"].lower()
                        scores.setdefault(aid, {})
                        arr = scores[aid].get(nl)
                        if arr is None or np.allclose(arr, 0):
                            if aid in ns:
                                for k, v in ns[aid].items():
                                    if k.lower() == nl:
                                        scores[aid][nl] = np.asarray(v, dtype=float)
                                        break
                out_name = "Eigen_IScore_live_weigen.pkl"
                method = "Eigen (IScore)"

            save_pk(scores, ds_out / out_name)

            # Merge into ranked_scores.pkl used by table scripts.
            ranked_path = ds_out / "ranked_scores.pkl"
            ranked = load_pk(ranked_path) if ranked_path.exists() else {}
            if dataset == "quotebank":
                ns = normalize_scores(pop["qb"]["ns"])
                lqid = normalize_scores(pop["qb"]["lqid"])
                # Paper Appendix E.3: Eigen* → NS → LQID
                ranked_scores = scores
                for tb in (ns, lqid):
                    ranked_scores = same_score_rank_ensemble(
                        normalize_scores(ranked_scores), tb,
                        load_json(ROOT / "data/Quotebank/data.json"),
                    )
                ranked[method] = ranked_scores
                metrics = eval_triple(dataset, scores, [ns, lqid])
            else:
                data_a = load_json(ROOT / "data/AIDA/data.json")
                ranked[method] = assign_unambiguous(normalize_scores(scores), data_a)
                metrics = eval_triple(dataset, scores)
            if method == "Eigen (IScore)":
                ranked["Eigen_IScore"] = ranked[method]
            save_pk(ranked, ranked_path)
            print(f"merged {method} into {ranked_path}", flush=True)
            print(f"{method} P@1 e/h/o = {metrics[0]:.3f}/{metrics[1]:.3f}/{metrics[2]:.3f}")


if __name__ == "__main__":
    main()
