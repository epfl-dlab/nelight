#!/usr/bin/env python3
"""Evaluate artifacts/from_scratch scores against paper Table 2."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse helpers from reproduce_tables without running main
_ns = {}
exec(
    open(ROOT / "scripts/reproduce_tables.py")
    .read()
    .split("def main")[0]
    .replace(
        "ROOT = Path(__file__).resolve().parents[1]",
        f"ROOT = Path(r'{ROOT}')",
    ),
    _ns,
)
for k, v in _ns.items():
    if callable(v) or k in {"ROOT", "SCORE_CACHE"}:
        globals()[k] = v

PAPER = {
    "quotebank": {
        "LQID": (0.828, 0.238, 0.727),
        "NP": (0.921, 0.143, 0.788),
        "NS": (1.0, 0.0, 0.829),
        "PRWD": (0.768, 0.214, 0.673),
        "PRWP": (0.926, 0.333, 0.824),
        "IScore": (0.956, 0.762, 0.922),
        "NIScore": (0.966, 0.571, 0.898),  # paper overall 0.851 is inconsistent typo
        "EEIScore": (0.951, 0.69, 0.906),
        "UIScore": (0.966, 0.833, 0.943),
        "mGENRE": (0.995, 0.81, 0.963),
    },
    "aida": {
        "LQID": (0.856, 0.259, 0.554),
        "NP": (0.856, 0.19, 0.536),
        "NS": (0.908, 0.275, 0.588),
        "PRWD": (0.838, 0.155, 0.517),
        "PRWP": (0.938, 0.282, 0.607),
        "IScore": (0.863, 0.549, 0.632),
        "NIScore": (0.851, 0.407, 0.562),
        "EEIScore": (0.815, 0.382, 0.562),
        "UIScore": (0.833, 0.577, 0.621),
        "mGENRE": (0.925, 0.61, 0.682),
    },
}


def eval_scores(dataset: str, scores: dict) -> dict:
    data = load_json(ROOT / f"data/{'Quotebank' if dataset == 'quotebank' else 'AIDA'}/data.json")
    easy = load_json(ROOT / f"data/{'Quotebank' if dataset == 'quotebank' else 'AIDA'}/easy.json")
    hard = load_json(ROOT / f"data/{'Quotebank' if dataset == 'quotebank' else 'AIDA'}/hard.json")
    if dataset == "quotebank":
        overall = merge_gt(easy, hard) if "merge_gt" in dir() else load_json(ROOT / "data/Quotebank/overall.json")
        try:
            overall = merge_gt(easy, hard)
        except Exception:
            overall = load_json(ROOT / "data/Quotebank/overall.json")
        sc = normalize_scores(scores)
        p_fn = precision_at_one_qb
    else:
        overall = load_json(ROOT / "data/AIDA/overall.json")
        sc = assign_unambiguous(normalize_scores(scores), data)
        p_fn = precision_at_one_aida
    out = {}
    for split, gt in [("easy", easy), ("hard", hard), ("overall", overall)]:
        out[split] = p_fn(flatten_gt(gt), sc)
    return out


def main():
    fs = ROOT / "artifacts/from_scratch"
    rows = []
    for dataset in ("quotebank", "aida"):
        metrics_path = fs / dataset / "metrics.json"
        if metrics_path.exists():
            metrics = json.load(open(metrics_path))
            for method, paper in PAPER[dataset].items():
                if method == "mGENRE":
                    continue
                if method not in metrics["overall"]:
                    continue
                got = (
                    metrics["easy"][method],
                    metrics["hard"][method],
                    metrics["overall"][method],
                )
                rows.append((dataset, method, paper, got))

        # mGENRE pickles
        ctx = 128 if dataset == "quotebank" else 256
        mg = fs / dataset / f"mGENRE_t{ctx}.pkl"
        if mg.exists():
            scores = pickle.load(open(mg, "rb"))
            got_d = eval_scores(dataset, scores)
            got = (got_d["easy"], got_d["hard"], got_d["overall"])
            rows.append((dataset, "mGENRE", PAPER[dataset]["mGENRE"], got))

    print(f"{'DS':10s} {'Method':12s} {'Paper E/H/O':22s} {'Got E/H/O':22s} {'ΔO':>7s}")
    for ds, method, paper, got in rows:
        d = got[2] - paper[2]
        print(
            f"{ds:10s} {method:12s} "
            f"{paper[0]:.3f}/{paper[1]:.3f}/{paper[2]:.3f}   "
            f"{got[0]:.3f}/{got[1]:.3f}/{got[2]:.3f}   "
            f"{d:+.3f}"
        )


if __name__ == "__main__":
    # merge_gt may exist from exec
    if "merge_gt" not in globals():
        def merge_gt(a, b):
            out = {k: dict(v) for k, v in a.items()}
            for k, v in b.items():
                out.setdefault(k, {}).update(v)
            return out
    main()
