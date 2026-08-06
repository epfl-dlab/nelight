#!/usr/bin/env python3
"""Run mGENRE with the exact protocol recovered from aa.ipynb / mGENRE_context_sizes_eval.

Critical details (must match published score caches):
  * Do **not** pass marginalize=True — raw beams keep ``{text, score}``.
  * Map ``title >> lang`` → QID with ``max(ids, key=lambda y: int(y[1:]))``.
  * Quotebank: sum across offsets using the cell-10 ``cache.add`` quirk
    (first offset: first beam hit only; later offsets: sum all beam hypos for known QIDs).
  * AIDA: one mention → first beam hit per QID, then align to candidate list.
  * Context: at most ``t`` mBART tokens on each side; mention marked ``[START]``/``[END]``.
  * beam=10; missing candidates score 0; scores are ``exp(log-likelihood)``.

Paper-best: Quotebank t=128, AIDA t=256.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/mgenre/fairseq_multilingual_entity_disambiguation"
DEFAULT_TRIE = ROOT / "models/mgenre/titles_lang_all105_marisa_trie_with_redirect.pkl"
DEFAULT_T2W = ROOT / "models/mgenre/lang_title2wikidataID-normalized_with_redirect.pkl"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def detokenize(toks):
    return "▁".join(toks).replace("▁▁", " ").replace("▁", "")


def qid_max(t2w, text: str) -> str:
    """Deterministic title→QID map (same rule as aa.ipynb text_to_id)."""
    key = tuple(reversed(text.split(" >> ")))
    return max(t2w[key], key=lambda y: int(y[1:]))


def qb_agg_from_beams(offset_beams, t2w) -> dict:
    """Cell-10 aggregation including cache.add quirk."""
    score_dict: dict[str, float] = {}
    for beam in offset_beams:
        cache: set[str] = set()
        for cand in beam:
            try:
                qid = qid_max(t2w, cand["text"])
            except Exception:
                continue
            sc = float(cand["score"].item() if hasattr(cand["score"], "item") else cand["score"])
            if qid not in cache:
                if qid not in score_dict:
                    score_dict[qid] = math.exp(sc)
                    cache.add(qid)
                else:
                    score_dict[qid] += math.exp(sc)
    return score_dict


def aida_agg_from_beam(beam, t2w) -> dict:
    """First occurrence per QID (AIDA single-mention conversion)."""
    score_dict: dict[str, float] = {}
    for cand in beam:
        try:
            qid = qid_max(t2w, cand["text"])
        except Exception:
            continue
        if qid in score_dict:
            continue
        sc = float(cand["score"].item() if hasattr(cand["score"], "item") else cand["score"])
        score_dict[qid] = math.exp(sc)
    return score_dict


def build_sentence(content_words, offset, context: int, tokenizer) -> str:
    start, end = offset[0], offset[1]
    lc = " ".join(content_words[:start])
    mention = " ".join(content_words[start:end])
    rc = " ".join(content_words[end:])
    lc_t = detokenize(tokenizer.tokenize(lc)[-context:]) if lc else ""
    rc_t = detokenize(tokenizer.tokenize(rc)[:context]) if rc else ""
    return f"{lc_t} [START] {mention} [END] {rc_t}".strip()


def run_sample(model, trie, sentence: str):
    """Beam search without marginalize / text_to_id — raw {text, score} hypos."""
    import torch

    with torch.no_grad():
        out = model.sample(
            [sentence],
            prefix_allowed_tokens_fn=lambda batch_id, sent: [
                e
                for e in trie.get(sent.tolist())
                if e < len(model.task.target_dictionary)
            ],
            beam=10,
            # NO marginalize, NO text_to_id — matches genre_context_scores_all.pkl
        )
    return out[0]


def run(dataset: str, context: int, device: str, limit: int | None, out: Path):
    import torch
    from transformers import MBartTokenizer
    from genre.fairseq_model import mGENRE
    from genre.trie import MarisaTrie, Trie

    sys.modules["__main__"].MarisaTrie = MarisaTrie
    sys.modules["__main__"].Trie = Trie

    if dataset == "quotebank":
        data = load_json(ROOT / "data/Quotebank/data.json")
        gt = load_json(ROOT / "data/Quotebank/overall.json")
        content_split = lambda c: c.replace("\xa0", " ").split(" ")
    else:
        data = load_json(ROOT / "data/AIDA/data.json")
        gt = load_json(ROOT / "data/AIDA/overall.json")
        content_split = lambda c: c.split()

    print("Loading assets...", flush=True)
    t0 = time.time()
    with open(DEFAULT_T2W, "rb") as f:
        t2w = pickle.load(f)
    with open(DEFAULT_TRIE, "rb") as f:
        trie = pickle.load(f)
    model = mGENRE.from_pretrained(str(DEFAULT_MODEL)).eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
        print(f"device={device} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = "cpu"
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    print(f"assets ready in {time.time() - t0:.1f}s", flush=True)

    scores = {}
    if out.exists():
        with open(out, "rb") as f:
            scores = pickle.load(f)
        print(f"resuming ({len(scores)} articles)", flush=True)

    articles = list(data)
    if limit:
        articles = articles[:limit]

    n_calls = 0
    t_infer = 0.0
    for article in tqdm(articles, desc=f"mGENRE-{dataset}-t{context}"):
        aid = article["articleID"]
        if aid in scores and scores[aid]:
            continue
        scores[aid] = {}
        content_words = content_split(article["content"])
        for name in article["names"]:
            nkey = name["name"]
            ids = name["ids"]
            gt_l = {k.lower(): v for k, v in gt.get(aid, {}).items()}
            if len(ids) <= 1 or gt_l.get(nkey.lower()) is None:
                scores[aid][nkey] = np.array(0)
                continue

            offsets = name.get("offsets") or [[0, min(1, len(content_words))]]
            beams = []
            for offset in offsets:
                sentence = build_sentence(content_words, offset, context, tokenizer)
                try:
                    t1 = time.time()
                    beams.append(run_sample(model, trie, sentence))
                    t_infer += time.time() - t1
                    n_calls += 1
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"warn {aid}/{nkey}@{offset}: {exc}")

            if dataset == "quotebank":
                sd = qb_agg_from_beams(beams, t2w)
            else:
                # AIDA: one offset typical; if several, sum first-hit dicts
                sd = {}
                for beam in beams:
                    for q, sc in aida_agg_from_beam(beam, t2w).items():
                        sd[q] = sd.get(q, 0.0) + sc

            scores[aid][nkey] = np.array([sd.get(q, 0.0) for q in ids], dtype=np.float64)

        save_pickle(scores, out)

    meta = {
        "dataset": dataset,
        "context": context,
        "n_articles": len(articles),
        "n_model_calls": n_calls,
        "infer_seconds": round(t_infer, 2),
        "sec_per_call": round(t_infer / n_calls, 3) if n_calls else None,
        "protocol": "no-marginalize; max-QID; QB cell10 quirk",
        "out": str(out),
    }
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["quotebank", "aida"], required=True)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    context = args.context or (128 if args.dataset == "quotebank" else 256)
    out = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts/from_scratch" / args.dataset / f"mGENRE_t{context}.pkl"
    )
    for path in (DEFAULT_MODEL / "model.pt", DEFAULT_TRIE, DEFAULT_T2W):
        if not path.exists():
            print(f"Missing {path}", file=sys.stderr)
            sys.exit(1)
    run(args.dataset, context, args.device, args.limit, out)


if __name__ == "__main__":
    main()
