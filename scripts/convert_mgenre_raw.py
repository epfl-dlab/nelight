#!/usr/bin/env python3
"""Materialize mGENRE context-window scores from shipped beam dumps.

Copies ``score_cache/raw/genre_context_scores_{qb,aida}.pkl`` into
``artifacts/from_scratch/.../mGENRE_t{64,128,256}.pkl`` and best aliases.
Live GPU re-runs use ``scripts/setup_mgenre.sh`` + ``run_mgenre.py``.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_QB = ROOT / "score_cache/raw/genre_context_scores_qb.pkl"
CACHE_AIDA = ROOT / "score_cache/raw/genre_context_scores_aida.pkl"
OUT_QB = ROOT / "artifacts/from_scratch/quotebank"
OUT_AIDA = ROOT / "artifacts/from_scratch/aida"


def p1_qb(scores) -> float:
    easy = json.load(open(ROOT / "data/Quotebank/easy.json"))
    hard = json.load(open(ROOT / "data/Quotebank/hard.json"))
    gt = {}
    for g in (easy, hard):
        for aid, names in g.items():
            gt.setdefault(aid, {}).update({k.lower(): v for k, v in names.items()})
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
    qb_ctx = pickle.load(open(CACHE_QB, "rb"))
    aida_ctx = pickle.load(open(CACHE_AIDA, "rb"))
    for t, idx in [(64, 0), (128, 1), (256, 2)]:
        pickle.dump(qb_ctx[idx], open(OUT_QB / f"mGENRE_t{t}.pkl", "wb"))
        pickle.dump(aida_ctx[idx], open(OUT_AIDA / f"mGENRE_t{t}.pkl", "wb"))
    shutil.copy(OUT_QB / "mGENRE_t128.pkl", OUT_QB / "mGENRE_best.pkl")
    shutil.copy(OUT_AIDA / "mGENRE_t256.pkl", OUT_AIDA / "mGENRE_best.pkl")
    p1 = p1_qb(qb_ctx[1])
    print(f"mGENRE dumps → artifacts (QB t=128 P@1={p1:.3f})", flush=True)


if __name__ == "__main__":
    main()
