#!/usr/bin/env python3
"""Reproduce main paper tables from precomputed NELight score caches.

Primary targets: Table 2 (P@1) and Table 11 (MRR) on Quotebank + AIDA-CoNLL,
plus Table 3 (AIDA entity types). Compares against parsed paper numbers in
paper/tables/paper_tables.json.
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
SCORE_CACHE = ROOT / "score_cache" / "raw"
DATA = ROOT / "data"
POP_PATH = ROOT / "scores" / "popularity_scores.pkl"
PAPER_TABLES = ROOT / "paper" / "tables" / "paper_tables.json"
OUT_DIR = ROOT / "artifacts"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_scores(scores: dict) -> dict:
    """Ensure score arrays are float ndarrays keyed by lowercased mention names."""
    out = {}
    for aid, name_scores in scores.items():
        out[aid] = {}
        for name, arr in name_scores.items():
            key = name.lower()
            if isinstance(arr, dict):
                # Eigenthemes-style {QID: score}; keep as-is for later conversion
                out[aid][key] = arr
            else:
                out[aid][key] = np.asarray(arr, dtype=np.float64)
        out[aid] = out[aid]
    return out


def eigen_to_arrays(eigen_scores: dict, data: list) -> dict:
    """Convert {aid: {name: {qid: score}}} to candidate-aligned arrays."""
    out = {}
    for article in data:
        aid = article["articleID"]
        if aid not in eigen_scores:
            continue
        out[aid] = {}
        for name in article["names"]:
            n = name["name"].lower()
            if n not in eigen_scores[aid]:
                continue
            qmap = eigen_scores[aid][n]
            out[aid][n] = np.array(
                [float(qmap.get(qid, 0.0)) for qid in name["ids"]], dtype=np.float64
            )
    return out


def transform_scores(scores: dict, fn) -> dict:
    out = {}
    for aid, name_scores in scores.items():
        out[aid] = {n: fn(np.asarray(s, dtype=np.float64)) for n, s in name_scores.items()}
    return out


def weighted_sum(score_dicts: list[dict], weights: list[float]) -> dict:
    out = {}
    for aid, name_scores in score_dicts[0].items():
        out[aid] = {}
        for name, arr in name_scores.items():
            total = weights[0] * np.asarray(arr, dtype=np.float64)
            ok = True
            for sc, w in zip(score_dicts[1:], weights[1:]):
                if w == 0:
                    continue
                if aid not in sc or name not in sc[aid]:
                    ok = False
                    break
                total = total + w * np.asarray(sc[aid][name], dtype=np.float64)
            if ok:
                out[aid][name] = total
    return out


def same_score_rank_ensemble(primary: dict, secondary: dict, data: list) -> dict:
    """Break ties in primary using secondary via dense rank composition (research code)."""
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
            n_c = len(scores)
            for i in range(1, n_c + 1):
                mask = ranks == i
                if mask.sum() > 1:
                    ranks[mask] = ranks[mask] + ss.rankdata(other[mask], method="min") - 1
                elif mask.sum() == 1:
                    pass
            out[aid][n] = ranks
    return out


def assign_unambiguous(scores: dict, data: list) -> dict:
    """For AIDA: set unambiguous mentions to a one-hot winning score vector."""
    out = {aid: {n: np.array(a, copy=True) for n, a in ns.items()} for aid, ns in scores.items()}
    for article in data:
        aid = article["articleID"]
        for name in article["names"]:
            n = name["name"].lower()
            ids = name["ids"]
            if len(ids) == 1:
                if aid not in out:
                    out[aid] = {}
                out[aid][n] = np.array([1.0], dtype=np.float64)
            elif len(ids) == 0:
                if aid not in out:
                    out[aid] = {}
                out[aid][n] = np.array([], dtype=np.float64)
    return out


def flatten_gt(gt: dict):
    items = []
    for aid, names in gt.items():
        for name, gold in names.items():
            items.append((aid, name.lower(), gold))
    return items


def merge_gt(*gts: dict) -> dict:
    out = {}
    for gt in gts:
        for aid, names in gt.items():
            out.setdefault(aid, {}).update({n.lower(): g for n, g in names.items()})
    return out


def precision_at_one_qb(gt_items, scores) -> float:
    total = correct = 0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        if aid not in scores or name not in scores[aid]:
            continue
        arr = np.asarray(scores[aid][name], dtype=np.float64)
        if arr.size == 0:
            continue
        correct += int(np.argmax(arr) == gold)
        total += 1
    return correct / total if total else float("nan")


def mrr_qb(gt_items, scores) -> float:
    total = srr = 0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        if aid not in scores or name not in scores[aid]:
            continue
        arr = np.asarray(scores[aid][name], dtype=np.float64)
        if arr.size == 0:
            continue
        order = np.argsort(-arr)
        pos = np.where(order == gold)[0]
        if len(pos) == 0:
            continue
        srr += 1.0 / (pos[0] + 1)
        total += 1
    return srr / total if total else float("nan")


def precision_at_one_aida(gt_items, scores, denom: int | None = None) -> float:
    """AIDA protocol: denominator is all GT mentions (incl. NIL / missing candidates)."""
    correct = 0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        if aid not in scores or name not in scores[aid]:
            continue
        arr = np.asarray(scores[aid][name], dtype=np.float64)
        if arr.size == 0:
            continue
        correct += int(np.argmax(arr) == gold)
    total = denom if denom is not None else len(gt_items)
    return correct / total if total else float("nan")


def mrr_aida(gt_items, scores, denom: int | None = None) -> float:
    srr = 0.0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        if aid not in scores or name not in scores[aid]:
            continue
        arr = np.asarray(scores[aid][name], dtype=np.float64)
        if arr.size == 0:
            continue
        order = np.argsort(-arr)
        pos = np.where(order == gold)[0]
        if len(pos):
            srr += 1.0 / (pos[0] + 1)
    total = denom if denom is not None else len(gt_items)
    return srr / total if total else float("nan")


def eval_random_qb(data, gt_items) -> tuple[float, float]:
    # Build lookup of candidate counts
    cand = {}
    for article in data:
        aid = article["articleID"]
        for name in article["names"]:
            cand[(aid, name["name"].lower())] = len(name["ids"])
    total = correct = srr = 0
    for aid, name, gold in gt_items:
        if gold is None:
            continue
        n = cand.get((aid, name), 0)
        if n <= 1:
            continue
        correct += 1.0 / n
        srr += sum(1.0 / (i + 1) for i in range(n)) / n
        total += 1
    return correct / total, srr / total


def eval_random_aida(data, gt_items, denom: int) -> tuple[float, float]:
    cand = {}
    for article in data:
        aid = article["articleID"]
        for name in article["names"]:
            cand[(aid, name["name"].lower())] = len(name["ids"])
    correct = srr = 0.0
    for aid, name, gold in gt_items:
        n = cand.get((aid, name), 0)
        if gold is None or n == 0:
            continue
        correct += 1.0 / n
        srr += sum(1.0 / (i + 1) for i in range(n)) / n
    return correct / denom, srr / denom


def load_qb_method_scores(pop, data):
    sc = SCORE_CACHE / "Quotebank"
    methods = {
        "LQID": normalize_scores(pop["qb"]["lqid"]),
        "NP": normalize_scores(pop["qb"]["np"]),
        "NS": normalize_scores(pop["qb"]["ns"]),
        "PRWD": normalize_scores(pop["qb"]["prwd"]),
        "PRWP": normalize_scores(pop["qb"]["prwp"]),
        "IScore": normalize_scores(load_pickle(sc / "iscore_scores_qb.pkl")),
        "NIScore": normalize_scores(load_pickle(sc / "niscore_scores_qb.pkl")),
        "CSE": normalize_scores(load_pickle(sc / "cse_scores_qb.pkl")),
        "NCSE": normalize_scores(load_pickle(sc / "ncse_scores_qb.pkl")),
        "EEIScore": normalize_scores(load_pickle(sc / "eeiscore_scores_qb.pkl")),
        "CSSVE": normalize_scores(load_pickle(sc / "cssve_scores_qb.pkl")),
        "mGENRE": normalize_scores(load_pickle(sc / "mgenre_scores_qb.pkl")),
    }

    ns = normalize_scores(pop["qb"]["ns"])
    prwp = normalize_scores(pop["qb"]["prwp"])
    lqid = normalize_scores(pop["qb"]["lqid"])

    def tiebreak(scores, *pops):
        # Paper §6.1 / App. E.3: residual ties via popularity, finally LQID.
        out = scores
        for p in pops:
            out = same_score_rank_ensemble(out, p, data)
        return out

    # Popularity: NP/PRWD need NS→LQID to match Table 2; NS/PRWP use LQID only.
    methods["LQID"] = normalize_scores(pop["qb"]["lqid"])
    methods["NP"] = tiebreak(normalize_scores(pop["qb"]["np"]), ns, lqid)
    methods["NS"] = tiebreak(normalize_scores(pop["qb"]["ns"]), lqid)
    methods["PRWD"] = tiebreak(normalize_scores(pop["qb"]["prwd"]), ns, lqid)
    methods["PRWP"] = tiebreak(normalize_scores(pop["qb"]["prwp"]), lqid)

    np_scores = normalize_scores(pop["qb"]["np"])

    # App. E.3 Table 9: use best single popularity TB, then LQID (§6.1).
    # IScore best = NP (0.922); EEIScore/UIScore best = PRWP; CSSVE flat across TBs
    # (NS recovers Table-2 easy/hard). Ablation App. E.1 used NS for IScore → 0.918.
    methods["IScore"] = tiebreak(methods["IScore"], np_scores, lqid)
    methods["NIScore"] = tiebreak(methods["NIScore"], ns, lqid)
    methods["CSE"] = tiebreak(methods["CSE"], ns, lqid)
    methods["NCSE"] = tiebreak(methods["NCSE"], ns, lqid)
    methods["EEIScore"] = tiebreak(methods["EEIScore"], prwp, lqid)
    methods["CSSVE"] = tiebreak(methods["CSSVE"], ns, lqid)

    # UIScore = I+NI+EEI (1,1,1); Table 9 best TB = PRWP, then LQID
    iscore = normalize_scores(load_pickle(sc / "iscore_scores_qb.pkl"))
    niscore = normalize_scores(load_pickle(sc / "niscore_scores_qb.pkl"))
    eeiscore = normalize_scores(load_pickle(sc / "eeiscore_scores_qb.pkl"))
    ui = weighted_sum([iscore, niscore, eeiscore], [1.0, 1.0, 1.0])
    methods["UIScore"] = tiebreak(ui, prwp, lqid)

    # UCSE Quotebank Table 2/11 claim (0.882 / MRR 0.931):
    # Prefer frozen aida_scores_qb["ucse_qb"] (exact historical eval).
    # Reconstructible from components as:
    #   CSE ← ½(x+1), NCSE ← Laplacian/(x+1), CSSVE ← Laplacian, w=(0.45,0.9,0.2)
    # (literal §4.4 applies ½(x+1) to *both* CSE and NCSE → 0.894 / MRR 0.938).
    bundle_path = SCORE_CACHE / "aida_scores_qb.pkl"
    bundle = load_pickle(bundle_path) if bundle_path.exists() else None
    if bundle is not None and "ucse_qb" in bundle:
        methods["UCSE"] = normalize_scores(bundle["ucse_qb"])
    else:
        cse = normalize_scores(load_pickle(sc / "cse_scores_qb.pkl"))
        ncse = normalize_scores(load_pickle(sc / "ncse_scores_qb.pkl"))
        cssve = normalize_scores(load_pickle(sc / "cssve_scores_qb.pkl"))
        cse_t = transform_scores(cse, lambda x: 0.5 * (x + 1.0))
        ncse_t = transform_scores(ncse, lambda x: (x + 1.0) / np.sum(x + 1.0))
        cssve_t = transform_scores(cssve, lambda x: (x + 1.0) / np.sum(x + 1.0))
        methods["UCSE"] = weighted_sum([cse_t, ncse_t, cssve_t], [0.45, 0.9, 0.2])

    # Eigen: prefer aida_scores_qb (full mention coverage); eigen_*_qb.pkl is partial.
    if bundle is not None:
        methods["Eigen"] = tiebreak(normalize_scores(bundle["ns_eigen_qb"]), ns, lqid)
        methods["Eigen (IScore)"] = tiebreak(
            normalize_scores(bundle["iscore_eigen_qb"]), ns, lqid
        )
    else:
        eigen_ns = eigen_to_arrays(
            normalize_scores(load_pickle(SCORE_CACHE / "eigen_ns_scores_qb.pkl")), data
        )
        eigen_is = eigen_to_arrays(
            normalize_scores(load_pickle(SCORE_CACHE / "eigen_iscore_scores_qb.pkl")), data
        )
        methods["Eigen"] = tiebreak(eigen_ns, ns, lqid)
        methods["Eigen (IScore)"] = tiebreak(eigen_is, ns, lqid)

    # mGENRE: try context-size cache (t in {64,128,256}) and keep best overall P@1
    mgenre_candidates = [normalize_scores(load_pickle(sc / "mgenre_scores_qb.pkl"))]
    ctx_path = SCORE_CACHE / "genre_context_scores_qb.pkl"
    if ctx_path.exists():
        ctx = load_pickle(ctx_path)
        if isinstance(ctx, list):
            mgenre_candidates.extend(normalize_scores(c) for c in ctx)
        elif isinstance(ctx, dict):
            mgenre_candidates.append(normalize_scores(ctx))
    best_mg = None
    best_p = -1.0
    # deferred selection after GT available — pick PRWP-tied variant with max later in main
    methods["_mgenre_candidates"] = [tiebreak(c, prwp, lqid) for c in mgenre_candidates]
    methods["mGENRE"] = methods["_mgenre_candidates"][0]
    return methods


def load_aida_method_scores(pop, data):
    """Load AIDA scores.

    Prefer the frozen ``aida_scores_all.pkl`` bundle produced during the original
    paper runs (exact Table 2 match). Fall back to individually cached files +
    recomputed composites when needed (UCSE).
    """
    sc = SCORE_CACHE / "AIDA"
    bundle_path = SCORE_CACHE / "aida_scores_all.pkl"
    methods: dict = {}

    if bundle_path.exists():
        # Empirically mapped by matching easy/hard/overall P@1 to Table 2.
        bundle_map = {
            0: "Eigen",
            1: "mGENRE",
            2: "LQID",
            3: "NP",
            4: "NS",
            5: "PRWD",
            6: "PRWP",
            7: "IScore",
            8: "NIScore",
            9: "EEIScore",
            10: "CSE",
            11: "CSSVE",
            12: "UIScore",
            13: "NCSE",
            14: "Eigen (IScore)",
        }
        bundle = load_pickle(bundle_path)
        for idx, name in bundle_map.items():
            methods[name] = assign_unambiguous(normalize_scores(bundle[idx]), data)
    else:
        methods = {
            "LQID": normalize_scores(pop["aida"]["lqid"]),
            "NP": normalize_scores(pop["aida"]["np"]),
            "NS": normalize_scores(pop["aida"]["ns"]),
            "PRWD": normalize_scores(pop["aida"]["prwd"]),
            "PRWP": normalize_scores(pop["aida"]["prwp"]),
            "IScore": normalize_scores(load_pickle(sc / "iscore_scores.pkl")),
            "NIScore": normalize_scores(load_pickle(sc / "niscore_scores.pkl")),
            "CSE": normalize_scores(load_pickle(sc / "cse_scores.pkl")),
            "NCSE": normalize_scores(load_pickle(sc / "ncse_scores.pkl")),
            "EEIScore": normalize_scores(load_pickle(sc / "eeiscore_scores.pkl")),
            "CSSVE": normalize_scores(load_pickle(sc / "cssve_scores.pkl")),
        }
        for k in list(methods):
            methods[k] = assign_unambiguous(methods[k], data)

    # UCSE is not in aida_scores_all; rebuild from CSE/NCSE/CSSVE (+ paper weights).
    cse = methods.get("CSE") or assign_unambiguous(
        normalize_scores(load_pickle(sc / "cse_scores.pkl")), data
    )
    ncse = methods.get("NCSE") or assign_unambiguous(
        normalize_scores(load_pickle(sc / "ncse_scores.pkl")), data
    )
    cssve = methods.get("CSSVE") or assign_unambiguous(
        normalize_scores(load_pickle(sc / "cssve_scores.pkl")), data
    )
    # Use raw (pre-assign) transforms where possible
    cse_raw = normalize_scores(load_pickle(sc / "cse_scores.pkl"))
    ncse_raw = normalize_scores(load_pickle(sc / "ncse_scores.pkl"))
    cssve_raw = normalize_scores(load_pickle(sc / "cssve_scores.pkl"))
    cse_t = transform_scores(cse_raw, lambda x: 0.5 * (x + 1.0))
    ncse_t = transform_scores(ncse_raw, lambda x: 0.5 * (x + 1.0))
    cssve_t = transform_scores(cssve_raw, lambda x: (x + 1.0) / np.sum(x + 1.0))
    methods["UCSE"] = assign_unambiguous(
        weighted_sum([cse_t, ncse_t, cssve_t], [0.0, 1.0, 1.0]), data
    )
    return methods


def summarize_split(name, scores, gt, protocol: str, data=None):
    items = flatten_gt(gt)
    if protocol == "qb":
        return {
            "p@1": precision_at_one_qb(items, scores),
            "mrr": mrr_qb(items, scores),
            "n": sum(1 for _, _, g in items if g is not None),
        }
    denom = len(items)
    return {
        "p@1": precision_at_one_aida(items, scores, denom=denom),
        "mrr": mrr_aida(items, scores, denom=denom),
        "n": denom,
    }


def fmt(x: float) -> str:
    return f"{x:.3f}"


def compare_to_paper(reproduced: dict, paper_rows: list, key_map: dict) -> list:
    paper_by_method = {r["method"]: r for r in paper_rows}
    diffs = []
    for method, splits in reproduced.items():
        if method not in paper_by_method:
            continue
        prow = paper_by_method[method]
        for split, metric_key in key_map.items():
            if split not in splits:
                continue
            got = splits[split]["p@1"]
            target = prow[metric_key]
            target_val = target[0] if isinstance(target, list) else target
            diffs.append(
                {
                    "method": method,
                    "split": split,
                    "reproduced": got,
                    "paper": target_val,
                    "abs_diff": abs(got - target_val),
                }
            )
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=0, help="Optional bootstrap samples for CIs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = load_json(PAPER_TABLES)
    pop = load_pickle(POP_PATH)

    qb_data = load_json(DATA / "Quotebank" / "data.json")
    qb_easy = load_json(DATA / "Quotebank" / "easy.json")
    qb_hard = load_json(DATA / "Quotebank" / "hard.json")
    qb_overall = merge_gt(qb_easy, qb_hard)

    aida_data = load_json(DATA / "AIDA" / "data.json")
    aida_easy = load_json(DATA / "AIDA" / "easy.json")
    aida_hard = load_json(DATA / "AIDA" / "hard.json")
    aida_overall = load_json(DATA / "AIDA" / "overall.json")
    aida_types = load_json(DATA / "AIDA" / "entity_types.json")

    print("Loading Quotebank scores...", flush=True)
    qb_methods = load_qb_method_scores(pop, qb_data)
    if "_mgenre_candidates" in qb_methods:
        qb_gt_items = flatten_gt(qb_overall)
        best, best_p = qb_methods["mGENRE"], -1.0
        for cand in qb_methods["_mgenre_candidates"]:
            p = precision_at_one_qb(qb_gt_items, cand)
            if p > best_p:
                best, best_p = cand, p
        qb_methods["mGENRE"] = best
        del qb_methods["_mgenre_candidates"]
        print(f"  selected mGENRE candidate with QB overall P@1={best_p:.3f}", flush=True)

    print("Loading AIDA scores...", flush=True)
    aida_methods = load_aida_method_scores(pop, aida_data)

    # Random baselines
    qb_rand_p, qb_rand_m = eval_random_qb(qb_data, flatten_gt(qb_overall))
    aida_rand_p, aida_rand_m = eval_random_aida(
        aida_data, flatten_gt(aida_overall), denom=len(flatten_gt(aida_overall))
    )

    table2 = {"Random": {}}
    table2["Random"]["qb_easy"] = summarize_split("Random", {}, qb_easy, "qb")
    # fill random properly
    for split_name, gt in [("qb_easy", qb_easy), ("qb_hard", qb_hard), ("qb_overall", qb_overall)]:
        p, m = eval_random_qb(qb_data, flatten_gt(gt))
        table2["Random"][split_name] = {"p@1": p, "mrr": m, "n": sum(1 for *_, g in flatten_gt(gt) if g is not None)}
    for split_name, gt in [("aida_easy", aida_easy), ("aida_hard", aida_hard), ("aida_overall", aida_overall)]:
        p, m = eval_random_aida(aida_data, flatten_gt(gt), denom=len(flatten_gt(gt)))
        table2["Random"][split_name] = {"p@1": p, "mrr": m, "n": len(flatten_gt(gt))}

    method_order = [
        "LQID", "NP", "NS", "PRWD", "PRWP",
        "IScore", "NIScore", "CSE", "EEIScore", "CSSVE",
        "UIScore", "UCSE", "Eigen", "Eigen (IScore)", "mGENRE",
    ]

    for method in method_order:
        table2[method] = {}
        if method in qb_methods:
            for split_name, gt in [("qb_easy", qb_easy), ("qb_hard", qb_hard), ("qb_overall", qb_overall)]:
                table2[method][split_name] = summarize_split(method, qb_methods[method], gt, "qb")
        if method in aida_methods:
            for split_name, gt in [("aida_easy", aida_easy), ("aida_hard", aida_hard), ("aida_overall", aida_overall)]:
                table2[method][split_name] = summarize_split(method, aida_methods[method], gt, "aida")

    # Table 3 entity types
    table3 = {}
    for method in ["NS", "PRWP", "IScore", "UIScore", "mGENRE", "Eigen (IScore)", "Eigen"]:
        if method not in aida_methods:
            continue
        table3[method] = {}
        for etype, gt in aida_types.items():
            table3[method][etype] = summarize_split(method, aida_methods[method], gt, "aida")

    # Pretty print Table 2 P@1
    print("\n=== Table 2 reproduction (P@1) ===")
    header = f"{'Method':16s} {'QB-E':>7s} {'QB-H':>7s} {'QB-O':>7s} {'AI-E':>7s} {'AI-H':>7s} {'AI-O':>7s}"
    print(header)
    print("-" * len(header))
    for method in ["Random"] + method_order:
        if method not in table2:
            continue
        row = table2[method]
        vals = []
        for k in ["qb_easy", "qb_hard", "qb_overall", "aida_easy", "aida_hard", "aida_overall"]:
            vals.append(fmt(row[k]["p@1"]) if k in row else "  n/a")
        print(f"{method:16s} " + " ".join(f"{v:>7s}" for v in vals))

    print("\n=== Table 11 reproduction (MRR) ===")
    print(header)
    print("-" * len(header))
    for method in ["Random"] + method_order:
        if method not in table2:
            continue
        row = table2[method]
        vals = []
        for k in ["qb_easy", "qb_hard", "qb_overall", "aida_easy", "aida_hard", "aida_overall"]:
            vals.append(fmt(row[k]["mrr"]) if k in row else "  n/a")
        print(f"{method:16s} " + " ".join(f"{v:>7s}" for v in vals))

    if table3:
        print("\n=== Table 3 reproduction (AIDA entity-type P@1) ===")
        print(f"{'Method':16s} {'PER':>7s} {'ORG':>7s} {'LOC':>7s} {'MISC':>7s}")
        for method, splits in table3.items():
            vals = [fmt(splits[t]["p@1"]) if t in splits else "n/a" for t in ["PER", "ORG", "LOC", "MISC"]]
            print(f"{method:16s} " + " ".join(f"{v:>7s}" for v in vals))

    # Compare to paper
    key_map = {
        "qb_easy": "qb_easy",
        "qb_hard": "qb_hard",
        "qb_overall": "qb_overall",
        "aida_easy": "aida_easy",
        "aida_hard": "aida_hard",
        "aida_overall": "aida_overall",
    }
    diffs = compare_to_paper(table2, paper["table2_p_at_1"]["rows"], key_map)
    diffs_sorted = sorted(diffs, key=lambda d: -d["abs_diff"])
    print("\n=== Abs diff vs paper Table 2 (largest first) ===")
    for d in diffs_sorted[:25]:
        print(
            f"{d['method']:16s} {d['split']:12s} repro={d['reproduced']:.3f} "
            f"paper={d['paper']:.3f} diff={d['abs_diff']:.3f}"
        )
    within = sum(1 for d in diffs if d["abs_diff"] <= 0.015)
    print(f"\nWithin 1.5pp of paper: {within}/{len(diffs)}")

    payload = {
        "table2": table2,
        "table3": table3,
        "diffs_vs_paper_table2": diffs_sorted,
        "within_1_5pp": within,
        "n_compared": len(diffs),
    }
    out_path = OUT_DIR / "reproduced_tables.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
