# NELight

Code and data for
**[Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)**
(Čuljak et al., NAACL SRW 2022).

## Reproduce paper tables

Needs [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).

```bash
git clone https://github.com/epfl-dlab/nelight.git
cd nelight
git lfs install && git lfs pull
bash scripts/reproduce_all.sh
```

Recomputes heuristics from shipped `caches/`, merges in-repo Eigen/mGENRE
scores, rebuilds Tables 1–11 → `artifacts/all_paper_tables.json`. No GPU.
This is **not** a Wikidata-dump rebuild — see
**[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** for provenance, PDF typos,
paper/code mismatches, and what cannot be rebuilt.

Scorers use the paper method names (`LQID`, `NS`, `cse`, `eeiscore`, …).

## Use on your own data

**1. Format** — JSON list of articles:

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

`offsets` should be word spans. The scripts assume that document content consists of whitespace-joined tokens. This is an artifact of ensuring compatibility with Quotebank's article metadata format.  

**2. Build an entity KB** from a
[Wikidata dump](https://dumps.wikimedia.org/wikidatawiki/entities/) — see
`cache_building/README.md`, or:

```bash
uv sync --extra from-scratch
DUMP=… DATA_QB=… DATA_AIDA=… OUT=artifacts/my_cache \
  bash cache_building/run_pipeline.sh
```

**3. Score** popularity / text-overlap heuristics:

```bash
uv run --extra from-scratch python scripts/run_heuristics.py \
  --dataset custom --data /path/to/data.json \
  --entity-kb artifacts/my_cache/entity_kb.pkl --name my_run
```
