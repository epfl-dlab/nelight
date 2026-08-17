"""Shared evaluation helpers for paper-table reproduction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.stats as ss

ROOT = Path(__file__).resolve().parents[1]


def data_dir(dataset: str) -> Path:
    return ROOT / ("data/Quotebank" if dataset == "quotebank" else "data/AIDA")


def load_articles(dataset: str):
    return load_json(data_dir(dataset) / "data.json")


def load_gt(dataset: str, split: str):
    return load_json(data_dir(dataset) / f"{split}.json")


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


def normalize_scores(scores: dict) -> dict:
    out = {}
    for aid, name_scores in scores.items():
        out[aid] = {}
        for name, arr in name_scores.items():
            key = name.lower()
            if isinstance(arr, dict):
                out[aid][key] = arr
            else:
                out[aid][key] = np.asarray(arr, dtype=np.float64)
    return out


def transform_scores(scores: dict, fn) -> dict:
    return {
        aid: {n: fn(np.asarray(s, dtype=np.float64)) for n, s in name_scores.items()}
        for aid, name_scores in scores.items()
    }


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
    """Break ties in primary using secondary via dense rank composition."""
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


def assign_unambiguous(scores: dict, data: list) -> dict:
    out = {
        aid: {n: np.array(a, copy=True) for n, a in ns.items()}
        for aid, ns in scores.items()
    }
    for article in data:
        aid = article["articleID"]
        for name in article["names"]:
            n = name["name"].lower()
            ids = name["ids"]
            if len(ids) == 1:
                out.setdefault(aid, {})[n] = np.array([1.0], dtype=np.float64)
            elif len(ids) == 0:
                out.setdefault(aid, {})[n] = np.array([], dtype=np.float64)
    return out


def flatten_gt(gt: dict):
    return [(aid, name.lower(), gold) for aid, names in gt.items() for name, gold in names.items()]


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


def approx_eq(a, b, tol=0.002) -> bool:
    return abs(float(a) - float(b)) <= tol
