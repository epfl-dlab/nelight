# Caches

Canonical files live under `caches/{quotebank,aida}/` (tracked with Git LFS).

| Path | Role |
|---|---|
| `*/entity_kb.pkl` | Wikidata attributes + centrality (I/NI/EEIScore, NS, NP, PR*, LQID) |
| `quotebank/entity_kb_aliases.pkl` | Alias-augmented KB (IScore ablations) |
| `*/entity_embeddings.pkl` | BART entity property embeddings (CSE, CSSVE) |
| `*/document_embeddings.pkl` | BART full-article embeddings (CSE) |
| `*/mention_embeddings.pkl` | BART mention-sentence embeddings (NCSE) |
| `*/unambiguous_mentions.pkl` | Per-article unambiguous QIDs (EEIScore, CSSVE) |

Path resolution: `runlib/cache_paths.py`. Rebuild from a Wikidata dump:
`cache_building/README.md`.
