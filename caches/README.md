# NELight caches

Canonical names live under `caches/{quotebank,aida}/` as symlinks to the
legacy Drive filenames (multi-GB artifacts are not duplicated).

| Canonical path | Role | Scorers |
|---|---|---|
| `quotebank/entity_kb.pkl` | Wikidata attributes + centrality | I/NI/EEIScore, NS, NP, PR*, LQID |
| `quotebank/entity_kb_aliases.pkl` | Alias-augmented KB (ablations) | IScore ablations |
| `quotebank/entity_embeddings.pkl` | BART entity property embeddings | CSE, CSSVE |
| `quotebank/document_embeddings.pkl` | BART full-article embeddings | CSE |
| `quotebank/mention_embeddings.pkl` | BART mention-sentence embeddings | NCSE |
| `quotebank/unambiguous_mentions.pkl` | Per-article unambiguous QIDs | EEIScore, CSSVE |
| `aida/entity_kb.pkl` | AIDA Wikidata KB | AIDA heuristics |
| `aida/unambiguous_mentions.pkl` | AIDA unambiguous QIDs | AIDA EEI/CSSVE |

Resolution is centralized in `runlib/cache_paths.py` (canonical first, then
legacy). Call `ensure_canonical_symlinks()` after unpacking Drive assets.

Legacy directories kept for provenance: `Quotebank/`, `aida_raw/`,
`unambiguous_cache.pkl`, `workspace_caches/`.
