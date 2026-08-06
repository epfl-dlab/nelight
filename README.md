# NELight

Code and evaluation for
**[Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)**
(Čuljak et al., NAACL SRW 2022).

## Rebuild the paper tables

The repo includes the score files from the original experiments. To recompute
Tables 1–11 (including AIDA entity types and the appendix):

```bash
git lfs install && git lfs pull
pip install -r requirements-repro.txt
bash scripts/reproduce_all.sh
```

No GPU. Details, how to re-run each method yourself, installs, and known typos
in the PDF: **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

## Quickstart: heuristics on your data

Build entity caches from a Wikidata dump, then run popularity + I/NI/EEIScore +
UIScore on your own mentions.

### 1. Format the input

One JSON list of articles. Each mention lists candidate Wikidata QIDs (and
offsets in the document text):

```json
[
  {
    "articleID": "doc-1",
    "content": "full document text …",
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

`name` is a unique mention key within the article (paper data often appends an
index). Gold labels for evaluation are optional (see `--easy` / `--hard` /
`--overall` below); scoring only needs `data.json`.

### 2. Build caches

Needs a [Wikidata JSON dump](https://dumps.wikimedia.org/wikidatawiki/entities/)
(`*-all.json.gz`). Optional PageRank files improve PRWP / PRWD (see
`cache_building/README.md`).

```bash
pip install -r requirements-from-scratch.txt
export PYTHONPATH=cache_building${PYTHONPATH:+:$PYTHONPATH}
export DUMP=/path/to/wikidata-YYYYMMDD-all.json.gz
export DATA=/path/to/your/data.json
export OUT=artifacts/my_cache

python3 cache_building/collect_candidate_qids.py \
  --data "$DATA" --out "$OUT/candidate_qids.pkl"

python3 cache_building/extract_wikidata_subgraph.py \
  --dump "$DUMP" --qids "$OUT/candidate_qids.pkl" \
  --out "$OUT/wikidata_subgraph.json.gz"

python3 cache_building/extract_entity_metadata.py \
  --dump "$DUMP" --qids "$OUT/candidate_qids.pkl" \
  --out-dir "$OUT/entity_metadata"

# Optional: WP_RANKS=… WD_RANKS=… FIRST_PARAGRAPHS=… QID_PID=…
python3 cache_building/build_entity_kb.py \
  --dump "$OUT/wikidata_subgraph.json.gz" \
  --labels-dir "$OUT/entity_metadata" \
  --qids "$OUT/candidate_qids.pkl" \
  --out "$OUT/entity_kb.pkl"

python3 cache_building/build_unambiguous_mentions.py \
  --data "$DATA" --out "$OUT/unambiguous_mentions.pkl"
```

Or point the bundled pipeline at your file (it still takes two `--data` slots,
pass the same path twice):

```bash
DUMP=… DATA_QB="$DATA" DATA_AIDA="$DATA" OUT=artifacts/my_cache \
  bash cache_building/run_pipeline.sh
```

### 3. Run heuristics

```bash
PYTHONPATH=runlib python3 scripts/run_heuristics.py \
  --data "$DATA" \
  --entity-kb "$OUT/entity_kb.pkl" \
  --unambiguous "$OUT/unambiguous_mentions.pkl" \
  --protocol aida \
  --name mydata \
  --out artifacts/my_run
```

Writes `artifacts/my_run/mydata/{LQID,NP,NS,PRWD,PRWP,IScore,NIScore,EEIScore}.pkl`
and `ranked_scores.pkl`. `--protocol aida` uses raw argmax (good default);
`--protocol quotebank` applies the paper’s popularity tie-breaks.

If you have gold maps `{articleID: {mention: gold_candidate_index}}`:

```bash
PYTHONPATH=runlib python3 scripts/run_heuristics.py \
  --data "$DATA" --entity-kb "$OUT/entity_kb.pkl" \
  --unambiguous "$OUT/unambiguous_mentions.pkl" \
  --protocol aida --name mydata \
  --easy /path/to/easy.json --hard /path/to/hard.json \
  --out artifacts/my_run
```

## Re-running methods (optional)

| Method | Command | Install |
|---|---|---|
| Popularity, I/NI/EEIScore, UIScore | `PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both` | `requirements-from-scratch.txt` |
| CSE / NCSE / CSSVE / UCSE | same, with `--with-embeddings` | + embedding caches via LFS |
| mGENRE (use saved scores) | `python3 scripts/convert_mgenre_raw.py` | `requirements-repro.txt` |
| mGENRE (run the model again) | `python scripts/run_mgenre.py --dataset quotebank --context 128` | see REPRODUCIBILITY.md §5 |
| Eigenthemes | `python3 scripts/run_eigenthemes.py --dataset both --reuse-raw` | see REPRODUCIBILITY.md §6 |

| Path | Contents |
|---|---|
| `paper/tables/paper_tables.json` | Numbers as printed in the paper |
| `artifacts/all_paper_tables.json` | What the scripts produce |
| `caches/{quotebank,aida}/` | Entity KB and BART embeddings (Git LFS) |
| `score_cache/raw/` | Saved per-method scores |
| `cache_building/` | Rebuild caches from a Wikidata dump |
