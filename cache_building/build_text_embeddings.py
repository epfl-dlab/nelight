#!/usr/bin/env python3
"""Build BART text embeddings for entities, documents, and mention contexts.

Historical sources:
  - utils/wikidata_embeddings.py   → entity_embeddings
  - utils/content_embeddings.py    → document_embeddings
  - utils/sentence_embeddings.py   → mention_embeddings

Uses ``facebook/bart-base``. Entity caches store mask-mean-pooled vectors
``[n_values, 1, H]`` (scorers already mean over the token axis).

Usage:
  python cache_building/build_text_embeddings.py entity \\
      --entity-kb caches/aida/entity_kb.pkl \\
      --out caches/aida/entity_embeddings.pkl

  python cache_building/build_text_embeddings.py document \\
      --data data/AIDA/data.json \\
      --out caches/aida/document_embeddings.pkl

  python cache_building/build_text_embeddings.py mention \\
      --data data/AIDA/data.json \\
      --out caches/aida/mention_embeddings.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BartModel, BartTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cache_building" / "original" / "utils"))
sys.path.insert(0, str(ROOT / "runlib"))

from io_utils import load_json, load_pickle, save_pickle  # noqa: E402

SKIP_PROPS = {"n_sitelinks", "pagerank", "pagerank_wd", "n_statements"}


def load_model(device: str):
    print(f"Loading facebook/bart-base on {device} …", flush=True)
    tok = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartModel.from_pretrained("facebook/bart-base").to(device).eval()
    return tok, model


@torch.no_grad()
def embed_texts(tok, model, texts, device: str, max_length: int = 1024):
    """Return last_hidden_state for a text batch (padded), CPU tensor."""
    if not texts:
        return None
    inputs = tok(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    out = model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
    ).last_hidden_state
    return out.detach().cpu()


@torch.no_grad()
def embed_texts_pooled(tok, model, texts, device: str, max_length: int = 1024):
    """Mean-pool token states with the attention mask → ``[n, 1, H]`` on CPU."""
    if not texts:
        return None
    inputs = tok(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    mask = inputs["attention_mask"].to(device)
    out = model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=mask,
    ).last_hidden_state
    weights = mask.unsqueeze(-1).to(out.dtype)
    pooled = (out * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return pooled.unsqueeze(1).detach().cpu().contiguous()


def build_entity(entity_kb_path, out, device, batch_size: int = 64):
    """Embed entity KB properties as compact pooled ``[n_values, 1, H]`` tensors."""
    cache = load_pickle(entity_kb_path)
    tok, model = load_model(device)
    emb = {qid: {} for qid in cache}

    singles = []  # (qid, prop, text)
    multis = []  # (qid, prop, texts)
    for qid, prop_dict in cache.items():
        for prop, value_list in prop_dict.items():
            if prop in SKIP_PROPS:
                continue
            if isinstance(value_list, str):
                value_list = [value_list]
            if not value_list:
                continue
            texts = [v if isinstance(v, str) else str(v) for v in value_list]
            if len(texts) == 1:
                singles.append((qid, prop, texts[0]))
            else:
                multis.append((qid, prop, texts))

    for i in tqdm(range(0, len(singles), batch_size), desc="entity singles"):
        chunk = singles[i : i + batch_size]
        pooled = embed_texts_pooled(tok, model, [t for _, _, t in chunk], device)
        for j, (qid, prop, _) in enumerate(chunk):
            emb[qid][prop] = pooled[j : j + 1].clone()

    for qid, prop, texts in tqdm(multis, desc="entity multis"):
        e = embed_texts_pooled(tok, model, texts, device)
        if e is not None:
            emb[qid][prop] = e

    del model, tok
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    out = Path(out)
    n_props = sum(len(v) for v in emb.values())
    print(f"pickling {len(emb)} entities / {n_props} props → {out} …", flush=True)
    save_pickle(emb, out)
    print(f"wrote {out} ({len(emb)} entities)", flush=True)


def build_document(data_path, out, device):
    """Embed full articles; store pooled ``[1, 1, H]`` (scorers mean over tokens)."""
    data = load_json(data_path)
    tok, model = load_model(device)
    emb = {}
    for article in tqdm(data, desc="document embeddings"):
        e = embed_texts_pooled(tok, model, [article["content"]], device)
        emb[article["articleID"]] = e
    save_pickle(emb, out)
    print(f"wrote {out}")


def build_mention(data_path, out, device):
    """Embed mention sentences; store pooled ``[n_sents, 1, H]``."""
    from processing import sentences_with_name  # type: ignore

    data = load_json(data_path)
    tok, model = load_model(device)
    emb = {}
    for article in tqdm(data, desc="mention embeddings"):
        aid = article["articleID"]
        emb[aid] = {}
        for name in article.get("names", []):
            try:
                sents = sentences_with_name(name, article["content"])
            except Exception:
                continue
            if not sents:
                continue
            e = embed_texts_pooled(tok, model, sents, device)
            emb[aid][name["name"]] = e
    save_pickle(emb, out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["entity", "document", "mention"])
    ap.add_argument("--entity-kb", default=None, help="entity_kb.pkl (for kind=entity)")
    ap.add_argument("--wikicache", default=None, help="Deprecated alias of --entity-kb")
    ap.add_argument("--data", default=None, help="NELight data.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.kind == "entity":
        kb = args.entity_kb or args.wikicache
        assert kb, "--entity-kb required"
        build_entity(kb, args.out, args.device, batch_size=args.batch_size)
    elif args.kind == "document":
        assert args.data, "--data required"
        build_document(args.data, args.out, args.device)
    else:
        assert args.data, "--data required"
        build_mention(args.data, args.out, args.device)


if __name__ == "__main__":
    main()
