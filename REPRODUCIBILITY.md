# Reproducing *Strong Heuristics for Named Entity Linking*

[Čuljak et al., NAACL SRW 2022](https://aclanthology.org/2022.naacl-srw.30/)

## Quick start

```bash
bash scripts/reproduce_all.sh
# or:
python3 scripts/reproduce_all_paper_tables.py
```

Writes `artifacts/all_paper_tables.json` (Tables 1–11). Needs `numpy` / `scipy` and the shipped `score_cache/`, `scores/`, `caches/`, and `data/` trees.

From-scratch heuristics (entity KB + optional BART caches):

```bash
PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both --with-embeddings
python3 scripts/reproduce_paper_from_scratch.py
```

## Known paper errors

These are **print/dump mismatches**, not method bugs. Scripts treat the
**Reproduced** column as the target.

| Issue | Printed | Reproduced | Notes |
|---|---|---|---|
| Quotebank NIScore overall (Table 2) | 0.851 | **0.898** | Impossible given easy/hard 0.966/0.571 on a 203+42 split |
| AIDA NIScore overall (Table 2) | 0.562 | **0.589** | Easy/hard match the paper; overall likely copied from EEIScore’s 0.562 |
| AIDA Eigen easy (Table 2) | 0.859 | **0.858** | Overall **0.617** exact |
| Table 11 AIDA MRR (CSE, EEIScore, CSSVE, UCSE, NIScore) | see PDF | dump-derived | Same dumps match Table 2 P@1; printed MRR rows are scrambled |
| Table 8 / 11 AIDA mGENRE MRR | 0.720 / 0.730 (hard/overall) | **0.743 / 0.736** | P@1 matches; printed MRR undercounts the dumps |
| Table 9 UIScore+PRWP | 0.942 | **0.943** | Same as Table 2 UIScore overall |

## Other notes
Exact Table-2 numbers use `score_cache/raw/genre_context_scores_*.pkl`.
Details and cache builders: `caches/README.md`, `cache_building/README.md`.
Parsed paper cells: `paper/tables/paper_tables.json`.

## Layout

```
data/            Quotebank + AIDA evaluation JSON
scores/          popularity + IScore ablation pickles
score_cache/raw/ frozen method score dumps
caches/          entity KB, unambiguous mentions, BART embeddings (Git LFS)
scripts/         table reproduction + from-scratch runners
runlib/          importable scorers + cache path resolution
cache_building/  rebuild caches from a Wikidata dump
paper/           PDFs + parsed tables
artifacts/       script outputs
```
