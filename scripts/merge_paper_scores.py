#!/usr/bin/env python3
"""Merge in-repo Eigen + mGENRE scores into artifacts/from_scratch ranked bundles.

Called after ``run_heuristics.py``. Sources (all shipped in the repo / LFS):

- Eigen: ``artifacts/from_scratch/{ds}/Eigen*_live_weigen.pkl``
  (fallback: Quotebank ``score_cache/raw/eigen_*_scores_qb.pkl``)
- mGENRE: ``mGENRE_best.pkl`` written by ``convert_mgenre_raw.py``
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "artifacts/from_scratch"
SC = ROOT / "score_cache/raw"


def load_pk(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pk(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def normalize(scores: dict) -> dict:
    return {
        aid: {
            n.lower(): (
                v
                if isinstance(v, dict)
                else np.asarray(v, dtype=np.float64)
            )
            for n, v in ns.items()
        }
        for aid, ns in scores.items()
    }


def eigen_dict_to_arrays(eigen_scores: dict, data: list) -> dict:
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
            if isinstance(qmap, dict):
                out[aid][n] = np.array(
                    [float(qmap.get(qid, 0.0)) for qid in name["ids"]],
                    dtype=np.float64,
                )
            else:
                out[aid][n] = np.asarray(qmap, dtype=np.float64)
    return out


def load_json(path: Path):
    import json

    with open(path) as f:
        return json.load(f)


def merge_dataset(dataset: str) -> None:
    ds_out = FS / dataset
    ranked_path = ds_out / "ranked_scores.pkl"
    ranked = load_pk(ranked_path) if ranked_path.exists() else {}

    # --- Eigen ---
    eigen_map = {
        "Eigen": [
            ds_out / "Eigen_live_weigen.pkl",
            ds_out / "Eigen.pkl",
        ],
        "Eigen (IScore)": [
            ds_out / "Eigen_IScore_live_weigen.pkl",
            ds_out / "Eigen_IScore.pkl",
        ],
    }
    data = load_json(
        ROOT / ("data/Quotebank/data.json" if dataset == "quotebank" else "data/AIDA/data.json")
    )
    for key, cands in eigen_map.items():
        if key in ranked:
            continue
        for path in cands:
            if not path.exists():
                continue
            raw = load_pk(path)
            sample = next(iter(next(iter(raw.values())).values()))
            if isinstance(sample, dict):
                ranked[key] = eigen_dict_to_arrays(raw, data)
            else:
                ranked[key] = normalize(raw)
            print(f"[{dataset}] Eigen ← {path.name}", flush=True)
            break
        else:
            if dataset == "quotebank" and key == "Eigen":
                # Legacy NS-eigen dump
                p = SC / "eigen_ns_scores_qb.pkl"
                if p.exists():
                    ranked[key] = eigen_dict_to_arrays(load_pk(p), data)
                    print(f"[{dataset}] Eigen ← {p.name}", flush=True)
            if dataset == "quotebank" and key == "Eigen (IScore)":
                p = SC / "eigen_iscore_scores_qb.pkl"
                if p.exists():
                    ranked[key] = eigen_dict_to_arrays(load_pk(p), data)
                    print(f"[{dataset}] Eigen (IScore) ← {p.name}", flush=True)

    if "Eigen_IScore" in ranked and "Eigen (IScore)" not in ranked:
        ranked["Eigen (IScore)"] = ranked["Eigen_IScore"]

    # --- mGENRE ---
    if "mGENRE" not in ranked:
        for name in ("mGENRE_best.pkl", "mGENRE_t128.pkl", "mGENRE_t256.pkl"):
            path = ds_out / name
            if path.exists():
                ranked["mGENRE"] = normalize(load_pk(path))
                print(f"[{dataset}] mGENRE ← {name}", flush=True)
                break

    save_pk(ranked, ranked_path)
    print(f"[{dataset}] ranked methods: {sorted(ranked)}", flush=True)


def main():
    for ds in ("quotebank", "aida"):
        merge_dataset(ds)


if __name__ == "__main__":
    main()
