# Building caches from Wikidata

The Wikidata dump used in the paper is not in this repo. Use these scripts to
rebuild entity databases (and optional text embeddings) for Quotebank, AIDA, or
your own data. A rebuilt KB will **not** bit-match paper numbers unless you use
the same dump era and PageRank assets (paper tooling references
`wikidata-20211101-all.json.gz`).

## Steps

1. Collect candidate Wikidata ids from your `data.json`
2. Extract a subgraph + labels from a Wikidata dump
3. Build `entity_kb.pkl` (optionally attach PageRank)
4. Build unambiguous-mention lists
5. Optionally embed text with BART (`build_text_embeddings.py`)

## Quick run

Needs [uv](https://docs.astral.sh/uv/) and a
[Wikidata JSON dump](https://dumps.wikimedia.org/wikidatawiki/entities/)
(`*-all.json.gz`).

```bash
DUMP=/path/to/wikidata-YYYYMMDD-all.json.gz bash cache_building/run_pipeline.sh
```

Defaults read `data/Quotebank/data.json` and `data/AIDA/data.json`. For one
custom file, set both inputs to the same path:

```bash
DUMP=… DATA_QB=/path/to/data.json DATA_AIDA=/path/to/data.json OUT=artifacts/my_cache \
  bash cache_building/run_pipeline.sh
```

## Optional inputs

```bash
export WP_RANKS=/path/to/wikipedia.ranks   # Wikipedia PageRank → PRWP
export WD_RANKS=/path/to/wikidata.ranks    # Wikidata PageRank → PRWD
export QID_PID=/path/to/qid_pid_mapping.json.bz2
export FIRST_PARAGRAPHS=/path/to/first_paragraphs.jsonl.bz2
```

PageRank files are plain text maps from entity id to rank (one score per
line / id). Provide them via `WP_RANKS` / `WD_RANKS` if you want PRWP / PRWD.

Text embeddings need `uv sync --extra from-scratch` and (preferably) a GPU.
