#!/usr/bin/env python3
"""Reproduce paper Tables 3, 6, 7, 8, 11 from from-scratch / validated caches."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "artifacts/from_scratch"
OUT = ROOT / "artifacts" / "remaining_tables.json"

_ns: dict = {}
exec(
    open(ROOT / "scripts/reproduce_tables.py")
    .read()
    .split("def main")[0]
    .replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path(r'{ROOT}')"),
    _ns,
    _ns,
)
globals().update({k: v for k, v in _ns.items() if callable(v) or k.isupper()})


def load_pk(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def qb_overall_gt():
    easy = load_json(ROOT / "data/Quotebank/easy.json")
    hard = load_json(ROOT / "data/Quotebank/hard.json")
    ov = {}
    for g in (easy, hard):
        for a, ns in g.items():
            ov.setdefault(a, {}).update({n.lower(): v for n, v in ns.items()})
    return ov


def main():
    paper = load_json(ROOT / "paper/tables/paper_tables.json")
    data_q = load_json(ROOT / "data/Quotebank/data.json")
    data_a = load_json(ROOT / "data/AIDA/data.json")
    ov_q = qb_overall_gt()
    results = {}

    # ---- Table 3 ----
    sys.path.insert(0, str(ROOT / "scripts"))
    import reproduce_paper_from_scratch as r2

    ranked_a = r2.aida_methods()
    types = load_json(ROOT / "data/AIDA/entity_types.json")
    t3 = {}
    for method in ["NS", "PRWP", "IScore", "UIScore", "mGENRE", "Eigen (IScore)", "Eigen"]:
        sc = ranked_a[method]
        t3[method] = {}
        for etype, gt in types.items():
            items = flatten_gt(gt)
            t3[method][etype] = round(precision_at_one_aida(items, sc), 3)
    results["table3"] = t3
    print("=== Table 3 (AIDA entity types P@1) ===")
    for m, row in t3.items():
        print(f"{m:16s} " + " ".join(f"{k}={v:.3f}" for k, v in row.items()))

    # ---- Table 6 (IScore ablation; precomputed matrix from paper runs) ----
    abl_txt = (ROOT / "results/fn_ablation_results.txt").read_text()
    best_line = [ln for ln in abl_txt.splitlines() if "D + S" in ln and "Stemming" in ln][0]
    results["table6_best"] = best_line.strip()
    print("\n=== Table 6 best cell ===")
    print(best_line.strip())
    assert "0.918" in best_line and "0.952" in best_line

    # ---- Table 7 (context size) ----
    ns = normalize_scores(load_pk(FS / "quotebank/NS.pkl"))
    lqid = normalize_scores(load_pk(FS / "quotebank/LQID.pkl"))
    np_s = normalize_scores(load_pk(FS / "quotebank/NP.pkl"))
    cse = normalize_scores(load_pk(ROOT / "score_cache/raw/cse_scores_qb.pkl"))
    ncse = normalize_scores(load_pk(ROOT / "score_cache/raw/ncse_scores_qb.pkl"))
    iscore = normalize_scores(load_pk(FS / "quotebank/IScore.pkl"))
    niscore = normalize_scores(load_pk(FS / "quotebank/NIScore.pkl"))

    def tb(sc, *pops):
        out = sc
        for p in pops:
            out = same_score_rank_ensemble(out, p, data_q)
        return out

    t7 = {
        "CSE Narrow": (
            precision_at_one_qb(flatten_gt(ov_q), tb(ncse, ns, lqid)),
            mrr_qb(flatten_gt(ov_q), tb(ncse, ns, lqid)),
        ),
        "CSE Entire": (
            precision_at_one_qb(flatten_gt(ov_q), tb(cse, ns, lqid)),
            mrr_qb(flatten_gt(ov_q), tb(cse, ns, lqid)),
        ),
        "CSE Ensemble": (
            precision_at_one_qb(
                flatten_gt(ov_q), tb(weighted_sum([cse, ncse], [1.0, 1.0]), ns, lqid)
            ),
            mrr_qb(
                flatten_gt(ov_q), tb(weighted_sum([cse, ncse], [1.0, 1.0]), ns, lqid)
            ),
        ),
        # App. E ablation TB = NS (not Table-2 NP) → 0.918
        "IScore Narrow": (
            precision_at_one_qb(flatten_gt(ov_q), tb(niscore, ns, lqid)),
            mrr_qb(flatten_gt(ov_q), tb(niscore, ns, lqid)),
        ),
        "IScore Entire": (
            precision_at_one_qb(flatten_gt(ov_q), tb(iscore, ns, lqid)),
            mrr_qb(flatten_gt(ov_q), tb(iscore, ns, lqid)),
        ),
        "IScore Ensemble": (
            precision_at_one_qb(
                flatten_gt(ov_q),
                tb(weighted_sum([iscore, niscore], [1.0, 1.0]), ns, lqid),
            ),
            mrr_qb(
                flatten_gt(ov_q),
                tb(weighted_sum([iscore, niscore], [1.0, 1.0]), ns, lqid),
            ),
        ),
    }
    results["table7"] = {k: {"p@1": round(v[0], 3), "mrr": round(v[1], 3)} for k, v in t7.items()}
    print("\n=== Table 7 (context size) ===")
    for k, v in results["table7"].items():
        print(f"{k:18s} P@1={v['p@1']:.3f} MRR={v['mrr']:.3f}")

    # ---- Table 8 (mGENRE context) ----
    t8 = []
    for i, t in enumerate([64, 128, 256]):
        row = {"t": t}
        for ds, proto in [("quotebank", "qb"), ("aida", "aida")]:
            for cand in [
                FS / ds / f"mGENRE_from_raw_t{t}.pkl",
                FS / ds / f"mGENRE_t{t}.pkl",
            ]:
                if cand.exists():
                    sc = normalize_scores(load_pk(cand))
                    break
            else:
                # fall back to genre_context dumps
                ctx = load_pk(ROOT / f"score_cache/raw/genre_context_scores_{'qb' if ds=='quotebank' else 'aida'}.pkl")
                sc = normalize_scores(ctx[i])
            if ds == "aida":
                sc = assign_unambiguous(sc, data_a)
                gt = load_json(ROOT / "data/AIDA/overall.json")
                row[f"{proto}_p"] = round(precision_at_one_aida(flatten_gt(gt), sc), 3)
                row[f"{proto}_mrr"] = round(mrr_aida(flatten_gt(gt), sc), 3)
            else:
                row[f"{proto}_p"] = round(precision_at_one_qb(flatten_gt(ov_q), sc), 3)
                row[f"{proto}_mrr"] = round(mrr_qb(flatten_gt(ov_q), sc), 3)
        t8.append(row)
    results["table8"] = t8
    print("\n=== Table 8 (mGENRE context) ===")
    for r in t8:
        print(
            f"t={r['t']}: QB {r['qb_p']:.3f}/{r['qb_mrr']:.3f}  "
            f"AIDA {r['aida_p']:.3f}/{r['aida_mrr']:.3f}"
        )

    # ---- Table 11 MRR (from FS ranked) ----
    ranked_q = {k: normalize_scores(v) for k, v in load_pk(FS / "quotebank/ranked_scores.pkl").items()}
    easy_q = load_json(ROOT / "data/Quotebank/easy.json")
    hard_q = load_json(ROOT / "data/Quotebank/hard.json")
    easy_a = load_json(ROOT / "data/AIDA/easy.json")
    hard_a = load_json(ROOT / "data/AIDA/hard.json")
    ov_a = load_json(ROOT / "data/AIDA/overall.json")
    t11 = {}
    methods = [
        "LQID", "NP", "NS", "PRWD", "PRWP", "IScore", "NIScore", "CSE",
        "EEIScore", "CSSVE", "UIScore", "UCSE", "Eigen", "Eigen (IScore)", "mGENRE",
    ]
    for m in methods:
        t11[m] = {
            "qb": [
                round(mrr_qb(flatten_gt(easy_q), ranked_q[m]), 3),
                round(mrr_qb(flatten_gt(hard_q), ranked_q[m]), 3),
                round(mrr_qb(flatten_gt(ov_q), ranked_q[m]), 3),
            ],
            "aida": [
                round(mrr_aida(flatten_gt(easy_a), ranked_a[m]), 3),
                round(mrr_aida(flatten_gt(hard_a), ranked_a[m]), 3),
                round(mrr_aida(flatten_gt(ov_a), ranked_a[m]), 3),
            ],
        }
    results["table11"] = t11
    print("\n=== Table 11 (MRR overall) ===")
    for m in methods:
        print(f"{m:16s} QB={t11[m]['qb'][2]:.3f} AIDA={t11[m]['aida'][2]:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
