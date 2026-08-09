# Caches

Large files under `caches/{quotebank,aida}/` (stored with Git LFS).

| File | Used for |
|---|---|
| `entity_kb.pkl` | Popularity and text-overlap heuristics |
| `entity_kb_aliases.pkl` | Quotebank IScore ablation (Table 6) |
| `unambiguous_mentions.pkl` | EEIScore / CSSVE |
| `entity_embeddings.pkl` | CSE / CSSVE |
| `document_embeddings.pkl` | CSE |
| `mention_embeddings.pkl` | NCSE |

What is recomputed vs shipped: **[REPRODUCIBILITY.md](../REPRODUCIBILITY.md)**.
Rebuild from a Wikidata dump: `cache_building/README.md`.
