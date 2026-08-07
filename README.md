# NELight

Code and data for
**[Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)**
(Čuljak et al., NAACL SRW 2022).

## Reproduce the paper tables

Needs [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/)
(large data files).

```bash
git clone https://github.com/epfl-dlab/nelight.git
cd nelight
git lfs install && git lfs pull
bash scripts/reproduce_all.sh
```

Recomputes heuristics from the shipped caches, then rebuilds Tables 1–11.
No GPU. Writes `artifacts/all_paper_tables.json`.

More detail, re-running methods, and known typos in the PDF:
**[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

## Use on your own data

Build an entity database from Wikidata, then score mentions with the paper’s
popularity and text-overlap heuristics (not the embedding-based ones).

**1. Format.** A JSON list of articles:

```json
[
  {
    "articleID": "doc-1",
    "content": "… full document text …",
    "names": [
      {
        "name": "paris0",
        "ids": ["Q90", "Q142"],
        "offsets": [[12, 17]]
      }
    ]
  }
]
```

Each `name` should be unique within the article. `ids` are Wikidata candidate
ids. `offsets` are word spans in `content` (split on spaces).

**2. Build caches** (needs a
[Wikidata dump](https://dumps.wikimedia.org/wikidatawiki/entities/)):

```bash
uv sync --extra from-scratch
export DUMP=/path/to/wikidata-YYYYMMDD-all.json.gz
export DATA=/path/to/your/data.json
export OUT=artifacts/my_cache

uv run python cache_building/collect_candidate_qids.py \
  --data "$DATA" --out "$OUT/candidate_qids.pkl"
uv run python cache_building/extract_wikidata_subgraph.py \
  --dump "$DUMP" --qids "$OUT/candidate_qids.pkl" \
  --out "$OUT/wikidata_subgraph.json.gz"
uv run python cache_building/extract_entity_metadata.py \
  --dump "$DUMP" --qids "$OUT/candidate_qids.pkl" \
  --out-dir "$OUT/entity_metadata"
uv run python cache_building/build_entity_kb.py \
  --dump "$OUT/wikidata_subgraph.json.gz" \
  --labels-dir "$OUT/entity_metadata" \
  --qids "$OUT/candidate_qids.pkl" \
  --out "$OUT/entity_kb.pkl"
uv run python cache_building/build_unambiguous_mentions.py \
  --data "$DATA" --out "$OUT/unambiguous_mentions.pkl"
```

Or: `DUMP=… DATA_QB="$DATA" DATA_AIDA="$DATA" OUT=… bash cache_building/run_pipeline.sh`  
(PageRank and related options: `cache_building/README.md`.)

**3. Score:**

```bash
uv run --extra from-scratch python scripts/run_heuristics.py \
  --data "$DATA" \
  --entity-kb "$OUT/entity_kb.pkl" \
  --unambiguous "$OUT/unambiguous_mentions.pkl" \
  --protocol aida \
  --name mydata \
  --out artifacts/my_run
```

Optional gold labels for accuracy: `--easy` / `--hard` / `--overall`
(JSON maps `{articleID: {mention: gold_candidate_index}}`).

## What’s in the repo

| Path | What it is |
|---|---|
| `data/` | Quotebank and AIDA evaluation sets |
| `score_cache/` | Saved method scores from the paper runs |
| `caches/` | Entity databases and text embeddings (Git LFS) |
| `scripts/` | Table builders and method runners |
| `cache_building/` | Rebuild caches from a Wikidata dump |
| `paper/` | PDF and machine-readable table numbers |
