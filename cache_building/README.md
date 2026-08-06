# Building NELight caches from a Wikidata dump

The 2021-11-01 Wikidata dump used in the paper is **not** bundled here.
This directory reconstructs the builders. Unmodified originals (with their
historical absolute paths) live under `original/`.

Canonical cache names are documented in [`../caches/README.md`](../caches/README.md).

## Pipeline

```text
data/{Quotebank,AIDA}/data.json
        │
        ▼
collect_candidate_qids.py          → candidate_qids.pkl
        │
        ▼
extract_wikidata_subgraph.py       → wikidata_subgraph.json.gz
        │
        ├──────────────────────────┐
        ▼                          ▼
extract_entity_metadata.py     build_token_representations.py
  entity_metadata/               (optional Spark path)
        │
        ▼
build_entity_kb.py                 → entity_kb.pkl
        │
        ├─→ build_unambiguous_mentions.py → unambiguous_mentions_*.pkl
        │
        └─→ build_text_embeddings.py
              entity_embeddings.pkl
              document_embeddings.pkl
              mention_embeddings.pkl
```

## Run (when a dump is available)

```bash
DUMP=/path/to/wikidata-YYYYMMDD-all.json.gz bash cache_building/run_pipeline.sh
```

Optional environment variables for first paragraphs / PageRank (used by
`run_pipeline.sh` if you wire them in):

```bash
export QID_PID=/path/to/qid_pid_mapping.json.bz2
export FIRST_PARAGRAPHS=/path/to/first_paragraphs.jsonl.bz2
export WP_RANKS=/path/to/wikipedia.ranks   # → entity field pagerank (PRWP)
export WD_RANKS=/path/to/wikidata.ranks    # → entity field pagerank_wd (PRWD)
```

**PageRank sources used in the paper**

- **PRWP (`pagerank`)** — Andreas Thalhammer’s public Wikipedia PageRank
  (2021), e.g.
  `https://danker.s3.amazonaws.com/2021-11-15.allwiki.links.rank.bz2`
  ([index](https://danker.s3.amazonaws.com/index.html)).
- **PRWD (`pagerank_wd`)** — Wikidata PageRank computed with
  [danker](https://github.com/athalhammer/danker).

Rank files are TSV `qid_number<TAB>score` (no `Q` prefix). Pass them to
`build_entity_kb.py` as `--wp-ranks` / `--wd-ranks`. More detail:
`REPRODUCIBILITY.md` (Wikipedia and Wikidata PageRank).

`build_text_embeddings.py` stores mask-mean-pooled vectors (`[n, 1, H]`);
scorers already average over the token axis.

## Source map

See [`SOURCE_INDEX.md`](SOURCE_INDEX.md).
