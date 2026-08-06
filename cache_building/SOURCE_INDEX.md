# Source map

| Reconstructed script | Historical source(s) | Canonical output |
|---|---|---|
| `collect_candidate_qids.py` | `QB_disamb.ipynb`; `populate_aida_cache.py` | `candidate_qids.pkl` |
| `extract_wikidata_subgraph.py` | `extract_quotebank_subset.py` | `wikidata_subgraph.json.gz` |
| `extract_entity_metadata.py` | `aida_aliases.py`, `aida_wikicache.py` | `entity_metadata/wikidata_*.pkl` |
| `build_entity_kb.py` | `populate_aida_cache.py` + `process_cache.py` + `first_paragraphs.py` + pagerank helpers + Embedding calculation.ipynb | `entity_kb.pkl` |
| `build_unambiguous_mentions.py` | `unambiguous_entities.py` | `unambiguous_mentions_*.pkl` |
| `build_text_embeddings.py` | `wikidata_embeddings.py`, `content_embeddings.py`, `sentence_embeddings.py` | `*_embeddings.pkl` |
| `build_token_representations.py` | `get_entity_representation_caches.py`, `representation.py` | `token_representations/` |

Legacy filename aliases: `wikicache.pkl` ≡ `entity_kb.pkl`,
`unambiguous_cache.pkl` ≡ `unambiguous_mentions.pkl`,
`wikidata_embeddings.pkl` ≡ `entity_embeddings.pkl`,
`content_embeddings.pkl` ≡ `document_embeddings.pkl`,
`sentence_embeddings.pkl` ≡ `mention_embeddings.pkl`.
