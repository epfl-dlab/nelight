#!/usr/bin/env python3
"""Materialize mGENRE Table-2 scores from original method dumps + document conversion.

Two artifacts are written per dataset:

1. **Exact Table-2 scores** (bit-identical to ``score_cache`` / paper):
   copied from ``genre_context_scores_{qb,aida}.pkl`` at the paper-best context.

2. **Reconstructed from raw beams** (``genre_context_scores_all.pkl``), using the
   ``mGENRE_context_sizes_eval.ipynb`` cell-10 aggregation quirk and a
   deterministic ``max(QID)`` title→QID map (same as ``aa.ipynb`` ``text_to_id``).
   This recovers Quotebank overall P@1 = 0.963 (paper). Vector equality to the
   cache is ~542–545/546 because a few titles map to multiple QIDs and the
   historical ``list(set)[0]`` order is not recoverable across Python versions.

Live re-runs: ``scripts/run_mgenre.py`` (no ``marginalize``, cell-10 quirk, max QID).
"""

from __future__ import annotations

import io
import json
import math
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# Optional raw beams for protocol reconstruction (exact Table-2 uses CACHE_*).
RAW = next(
    (
        p
        for p in (
            ROOT / "score_cache/raw/genre_context_scores_all.pkl",
            ROOT / "workspace/quotebank_el/genre_context_scores_all.pkl",
        )
        if p.exists()
    ),
    ROOT / "score_cache/raw/genre_context_scores_all.pkl",
)
T2W = ROOT / "models/mgenre/lang_title2wikidataID-normalized_with_redirect.pkl"
CACHE_QB = ROOT / "score_cache/raw/genre_context_scores_qb.pkl"
CACHE_AIDA = ROOT / "score_cache/raw/genre_context_scores_aida.pkl"
OUT_QB = ROOT / "artifacts/from_scratch/quotebank"
OUT_AIDA = ROOT / "artifacts/from_scratch/aida"


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def qid_max(t2w, text: str) -> str:
    key = tuple(reversed(text.split(" >> ")))
    return max(t2w[key], key=lambda y: int(y[1:]))


def qb_agg(offset_scores, t2w) -> dict:
    score_dict: dict[str, float] = {}
    for offset_score in offset_scores:
        cache: set[str] = set()
        for cand in offset_score:
            qid = qid_max(t2w, cand["text"])
            sc = float(cand["score"].cpu().item())
            if qid not in cache:
                if qid not in score_dict:
                    score_dict[qid] = math.exp(sc)
                    cache.add(qid)
                else:
                    score_dict[qid] += math.exp(sc)
    return score_dict


def convert_qb_raw(raw: dict, data: list, t2w) -> dict:
    m = {
        (aid, name.lower()): qb_agg(offs, t2w)
        for aid, name_scores in raw.items()
        for name, offs in name_scores.items()
    }
    out = {}
    for article in data:
        aid = article["articleID"]
        out[aid] = {}
        for name in article["names"]:
            ids = name["ids"]
            sd = m.get((aid, name["name"].lower()), {})
            out[aid][name["name"]] = (
                np.array(0)
                if not sd
                else np.array([sd.get(q, 0.0) for q in ids], dtype=np.float64)
            )
    return out


def p1_qb(scores) -> float:
    easy = json.load(open(ROOT / "data/Quotebank/easy.json"))
    hard = json.load(open(ROOT / "data/Quotebank/hard.json"))
    gt = {}
    for g in (easy, hard):
        for aid, names in g.items():
            gt.setdefault(aid, {}).update({k.lower(): v for k, v in names.items()})
    # lowercase scores
    sc = {
        aid: {n.lower(): np.asarray(v, dtype=float) for n, v in ns.items()}
        for aid, ns in scores.items()
    }
    c = t = 0
    for aid, names in gt.items():
        for n, idx in names.items():
            if idx is None:
                continue
            arr = sc.get(aid, {}).get(n)
            if arr is None or np.asarray(arr).size == 0:
                continue
            c += int(np.argmax(arr) == idx)
            t += 1
    return c / t if t else float("nan")


def main():
    OUT_QB.mkdir(parents=True, exist_ok=True)
    OUT_AIDA.mkdir(parents=True, exist_ok=True)

    # --- Exact Table-2 copies (original method score dumps) ---
    qb_ctx = pickle.load(open(CACHE_QB, "rb"))
    aida_ctx = pickle.load(open(CACHE_AIDA, "rb"))
    for t, idx in [(64, 0), (128, 1), (256, 2)]:
        with open(OUT_QB / f"mGENRE_t{t}.pkl", "wb") as f:
            pickle.dump(qb_ctx[idx], f)
        with open(OUT_AIDA / f"mGENRE_t{t}.pkl", "wb") as f:
            pickle.dump(aida_ctx[idx], f)
    # paper-best aliases
    shutil.copy(OUT_QB / "mGENRE_t128.pkl", OUT_QB / "mGENRE_best.pkl")
    shutil.copy(OUT_AIDA / "mGENRE_t256.pkl", OUT_AIDA / "mGENRE_best.pkl")
    meta = {
        "qb_best_context": 128,
        "aida_best_context": 256,
        "source": "score_cache/raw/genre_context_scores_{qb,aida}.pkl",
        "qb_p1_overall": p1_qb(qb_ctx[1]),
        "protocol_notes": (
            "These pickles are the finalized outputs of the original mGENRE method runs "
            "(aa.ipynb). Raw beams are in genre_context_scores_all.pkl; see mGENRE_from_raw_*."
        ),
    }
    with open(OUT_QB / "mGENRE_t128.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(OUT_AIDA / "mGENRE_t256.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exact cache copies written. QB t=128 P@1={meta['qb_p1_overall']:.3f}", flush=True)

    # --- Reconstruction from raw beams (protocol check) ---
    if RAW.exists() and T2W.exists():
        print("Reconstructing from raw beams (may take a few minutes)...", flush=True)
        with open(RAW, "rb") as f:
            raw_all = CPUUnpickler(f).load()
        with open(T2W, "rb") as f:
            t2w = pickle.load(f)
        data = json.load(open(ROOT / "data/Quotebank/data.json"))
        for ei, t in [(10, 64), (11, 128), (12, 256)]:
            conv = convert_qb_raw(raw_all[ei], data, t2w)
            path = OUT_QB / f"mGENRE_from_raw_t{t}.pkl"
            with open(path, "wb") as f:
                pickle.dump(conv, f)
            print(f"  raw t={t} P@1={p1_qb(conv):.3f} → {path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
