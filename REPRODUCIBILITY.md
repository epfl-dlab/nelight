# Reproducing *Strong Heuristics for Named Entity Linking*

[Čuljak et al., NAACL SRW 2022](https://aclanthology.org/2022.naacl-srw.30/)

This document covers **dump-based** table reproduction (exact paper numbers,
aside from documented print errors) and **from-scratch** re-runs of every
method family.

## 1. Setup

```bash
git clone git@github.com:epfl-dlab/nelight.git
cd nelight
git lfs install
git lfs pull
```

### Dependencies by path

| Path | Install | Extra assets |
|---|---|---|
| Dump-based Tables 1–11 | `pip install -r requirements-repro.txt` | shipped `score_cache/`, `scores/`, `data/`, `caches/` (LFS) |
| Heuristics + CSE family | `pip install -r requirements-from-scratch.txt` then `python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"` | shipped `caches/` |
| Rebuild BART embeddings | same as heuristics | GPU recommended |
| mGENRE live | §5 below | checkpoint + trie + title map under `models/mgenre/` |
| Eigenthemes live | §6 below | DeepWalk pickle + candidate JSONs under `workspace/eigenthemes/` |

Python **3.9+** is fine for dump-based / heuristics. Live mGENRE needs the
GENRE fairseq fork (typically Python 3.8–3.10).

## 2. Dump-based: all paper tables (recommended)

Exact Table-2 / appendix numbers from frozen method dumps (no GPU):

```bash
pip install -r requirements-repro.txt
bash scripts/reproduce_all.sh
# or:
python3 scripts/reproduce_all_paper_tables.py
```

Writes `artifacts/all_paper_tables.json`. Coverage:

| Table | Content |
|---|---|
| 1 | Dataset difficulty splits |
| 2 | Main P@1 |
| 3 | **Type-based** AIDA P@1 (PER/ORG/LOC/MISC) |
| 4 | UIScore error analysis (count; categories qualitative) |
| 5 | Quotebank GT ambiguity distribution |
| 6 | IScore feature/normalization ablation |
| 7 | CSE / IScore narrow vs entire vs ensemble |
| 8 | mGENRE context windows |
| 9 | Popularity tie-breakers |
| 10 | Inference times (paper hardware, reported as-is) |
| 11 | MRR companion to Table 2 |

Cross-checks: `scripts/reproduce_tables.py`, `scripts/reproduce_remaining_tables.py`,
`scripts/reproduce_paper_from_scratch.py` (uses from-scratch artifacts when present,
else dumps).

## 3. From-scratch: popularity + I/NI/EEIScore + UIScore

```bash
pip install -r requirements-from-scratch.txt
# NLTK data (once):
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both
python3 scripts/reproduce_paper_from_scratch.py
```

Uses `caches/{quotebank,aida}/entity_kb.pkl` and `unambiguous_mentions.pkl`.
**Quotebank** applies method-specific popularity TB → LQID; **AIDA** uses raw
`argmax` (no popularity TB). Outputs: `artifacts/from_scratch/{quotebank,aida}/`.

## 4. From-scratch: CSE / NCSE / CSSVE / UCSE

```bash
PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both --with-embeddings
```

Needs `caches/*/entity_embeddings.pkl`, `document_embeddings.pkl`,
`mention_embeddings.pkl` (Git LFS).

**Paper-exact AIDA CSE family** in Table 2 still uses `score_cache/raw/AIDA/*`
dumps inside `reproduce_paper_from_scratch.py` (reconstructed pooled BART caches
match CSE/UCSE; CSSVE can drift ~1pp). Quotebank embedding scores match from
shipped caches.

### Rebuild BART caches (optional)

```bash
pip install -r requirements-from-scratch.txt
python cache_building/build_text_embeddings.py document \
  --data data/AIDA/data.json --out caches/aida/document_embeddings.pkl --device cuda:0
python cache_building/build_text_embeddings.py mention \
  --data data/AIDA/data.json --out caches/aida/mention_embeddings.pkl --device cuda:0
python cache_building/build_text_embeddings.py entity \
  --entity-kb caches/aida/entity_kb.pkl --out caches/aida/entity_embeddings.pkl --device cuda:0
```

Entity/document/mention builders store **mask-mean-pooled** `[n, 1, H]` vectors
(`facebook/bart-base`). Full Wikidata-dump rebuild: `cache_building/README.md`.

## 5. From-scratch: mGENRE

### Exact Table-2 numbers (dumps / raw beams)

No GPU. Uses finalized dumps; optionally reconstructs Quotebank from raw beams:

```bash
pip install -r requirements-repro.txt
# Optional for raw-beam check: place title map at
#   models/mgenre/lang_title2wikidataID-normalized_with_redirect.pkl
python3 scripts/convert_mgenre_raw.py
```

Paper-best contexts: Quotebank **t=128** (P@1 0.963), AIDA **t=256** (P@1 0.682).
Sources: `score_cache/raw/genre_context_scores_{qb,aida}.pkl` and optional
`score_cache/raw/genre_context_scores_all.pkl`.

### Live GPU re-run

**Protocol** (from `aa.ipynb` / Appendix B.3):

1. Mark mention with `[START]` / `[END]`; keep at most `t` mBART tokens on each side.
2. Constrained beam search, **`beam=10`**, **no** `marginalize`.
3. Map `title >> lang` → QID with `max(ids, key=lambda y: int(y[1:]))`.
4. Score = `exp(log-likelihood)`; missing candidates → 0.
5. Quotebank: sum per-offset scores with the notebook cell-10 `cache.add` quirk.

**Install fairseq + GENRE** (prefix-constraint fork required):

```bash
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq && pip install --editable ./ && cd ..
git clone https://github.com/facebookresearch/GENRE.git
# GENRE's `genre/` package must be importable (pip install -e GENRE or PYTHONPATH).
pip install -r requirements-mgenre.txt
```

Point `FAIRSEQ_ROOT` at the fairseq checkout if needed:

```bash
export FAIRSEQ_ROOT=/path/to/fairseq
source scripts/mgenre_env.sh   # optional; prepends FAIRSEQ_ROOT to PYTHONPATH
```

**Download assets** into `models/mgenre/`:

```bash
mkdir -p models/mgenre && cd models/mgenre
wget https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz
tar -xzf fairseq_multilingual_entity_disambiguation.tar.gz
wget https://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl
wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
cd ../..
```

Loading the title→QID map needs tens of GB of RAM.

**Run:**

```bash
python scripts/run_mgenre.py --dataset quotebank --context 128 --device cuda:0
python scripts/run_mgenre.py --dataset aida --context 256 --device cuda:0
# or: bash scripts/run_mgenre_pipeline.sh
python3 scripts/eval_from_scratch.py
```

Live beams are not bit-identical to the 2022 stack (fairseq/CUDA), but argmax
agreement is near-complete; prefer dumps / `convert_mgenre_raw.py` for exact
paper cells.

## 6. From-scratch: Eigenthemes

Needs `workspace/eigenthemes/` (DeepWalk embeddings + candidate JSON inputs from
the original research tree) and a working Eigenthemes Python env (numpy/scipy/
sklearn; see that tree’s README).

```bash
python3 scripts/run_eigenthemes.py --dataset both --variant both --reuse-raw
```

- **Eigen**: weigen, `meanCenter=False`, `numCands=20`, `ncomp=10`.  
  Quotebank: NS-weighted JSON + `NS→LQID`. AIDA: degree JSON, raw argmax.
- **Eigen (IScore)**: same weigen on shipped `*_iscore_*_test_complete.json`,
  then NS fill for missing embeddings.

## 7. Protocol summary

**Quotebank** — popularity TB → LQID (`NP→LQID` for IScore; `PRWP→LQID` for
EEIScore/UIScore; `NS→LQID` otherwise).  
**UIScore** = IScore+NIScore+EEIScore `(1,1,1)`.  
**UCSE** (0.882): CSE←½(x+1), NCSE←Laplacian, CSSVE←Laplacian, w=`(0.45,0.9,0.2)`.

**AIDA** — raw scores, numpy `argmax`.  
**UIScore** `(0.9,0,1)`. **UCSE** = ½(NCSE+1)+Laplacian(CSSVE), w=`(0,1,1)`.

## 8. Known paper errors

Print/dump mismatches (not method bugs). Scripts target the **Reproduced** column.

| Issue | Printed | Reproduced | Notes |
|---|---|---|---|
| Quotebank NIScore overall (Table 2) | 0.851 | **0.898** | Impossible given easy/hard 0.966/0.571 on 203+42 |
| AIDA NIScore overall (Table 2) | 0.562 | **0.589** | Easy/hard match; overall likely copied from EEIScore |
| AIDA Eigen easy (Table 2) | 0.859 | **0.858** | Overall **0.617** exact |
| Table 11 AIDA MRR (CSE, EEIScore, CSSVE, UCSE, NIScore) | see PDF | dump-derived | Same dumps match Table 2 P@1 |
| Table 8 / 11 AIDA mGENRE MRR | 0.720 / 0.730 | **0.743 / 0.736** | P@1 matches; printed MRR undercounts |
| Table 9 UIScore+PRWP | 0.942 | **0.943** | Same as Table 2 UIScore overall |

Also: Table 4 categories are qualitative (error **count** 14 matches); Table 5
is paper annotation arithmetic; Table 10 is paper hardware (reported as-is).
Table 3 matches P@1 point estimates (not bootstrap CI half-widths).

## 9. Layout

```
data/                 Quotebank + AIDA evaluation JSON (+ entity_types.json)
scores/               popularity + IScore ablation pickles
score_cache/raw/      frozen method score dumps (incl. mGENRE)
caches/               entity KB, unambiguous mentions, BART embeddings (Git LFS)
scripts/              table reproduction + from-scratch runners
runlib/               importable scorers + cache path resolution
cache_building/       rebuild caches from a Wikidata dump
models/mgenre/        mGENRE checkpoint + trie + title map (download; gitignored)
workspace/eigenthemes/ DeepWalk + Eigen JSON inputs (optional; gitignored)
paper/                PDFs + parsed tables
artifacts/            script outputs
```
